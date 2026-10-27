"""Plain-text progress for a reconstruction run.

Messages are normal log lines (no carriage returns, no progress-bar glyphs) so
they stay readable in Windows PowerShell and in log.txt.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("video2mesh")

STAGE_NAMES = (
    "frame extraction",
    "feature extraction",
    "matching",
    "mapping",
    "undistort",
    "dense",
    "mesh",
    "export",
)

STAGE_HINTS = {
    "frame extraction": (
        "Check that the file is a video ffmpeg or OpenCV can open. "
        "Install ffmpeg, or try a lower --fps."
    ),
    "feature extraction": (
        "Confirm COLMAP starts with `colmap feature_extractor -h`. "
        "On a CUDA build with no usable GPU, rerun with --gpu off."
    ),
    "matching": (
        "Sequential matching is slow on long clips. "
        "Retry with --quality fast or a smaller --max-frames."
    ),
    "mapping": (
        "Move more slowly, keep textured surfaces in frame, and prefer sideways motion. "
        "A shorter test is --quality fast --fps 1 --max-frames 120."
    ),
    "undistort": (
        "Undistort is optional. If it keeps failing, rerun with --no-dense to mesh the sparse cloud."
    ),
    "dense": (
        "Patch-match stereo needs a CUDA build of COLMAP. "
        "Without CUDA the CPU stereo fallback is used. --no-dense meshes the sparse points only."
    ),
    "mesh": "Try --poisson-trim 0 or --mesh bpa if the cloud is thin.",
    "export": "Check that the output folder is writable and the disk is not full.",
}

_PROCESSED = re.compile(r"Processed file \[(\d+)\s*/\s*(\d+)\]", re.IGNORECASE)
_MATCH_BLOCK = re.compile(r"Matching block \[(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_REGISTER = re.compile(r"Registering image #(\d+)", re.IGNORECASE)
_ELAPSED = re.compile(r"Elapsed time:\s*([0-9.]+)\s*\[minutes\]", re.IGNORECASE)
_FEATURES = re.compile(r"Features:\s*(\d+)", re.IGNORECASE)


@dataclass
class StageRecord:
    name: str
    elapsed: float
    detail: str


class RunProgress:
    """Numbered stages and the wall-clock summary for one reconstruct call."""

    def __init__(self, stages: tuple[str, ...] = STAGE_NAMES) -> None:
        self.stages = stages
        self.total = len(stages)
        self.index = {name: number for number, name in enumerate(stages, start=1)}
        self.records: list[StageRecord] = []
        self.current: str | None = None
        self.started = time.perf_counter()
        self._stage_started = self.started

    def start(self, name: str) -> None:
        if name not in self.index:
            raise KeyError(name)
        self.current = name
        self._stage_started = time.perf_counter()
        logger.info("")
        logger.info("=== Stage %d of %d: %s ===", self.index[name], self.total, name)

    def finish(self, detail: str = "") -> None:
        if self.current is None:
            return
        elapsed = time.perf_counter() - self._stage_started
        self.records.append(StageRecord(self.current, elapsed, detail))
        suffix = f" ({detail})" if detail else ""
        logger.info("--- %s finished in %s%s", self.current, format_duration(elapsed), suffix)
        self.current = None

    def skip(self, name: str, reason: str) -> None:
        self.start(name)
        logger.info("%s", reason)
        self.finish("skipped")

    def failure_message(self, exc: BaseException) -> str:
        """Stage name, the underlying error, and one next step."""
        if self.current is None:
            if not self.records:
                where = "Failed during setup, before frame extraction."
                hint = "Install COLMAP and ffmpeg, or set COLMAP_EXE and FFMPEG_EXE. See the README."
            else:
                done = {record.name for record in self.records}
                upcoming = next((name for name in self.stages if name not in done), None)
                if upcoming is None:
                    where = f"Failed after {self.records[-1].name}."
                    hint = ""
                else:
                    where = (
                        f"Failed after {self.records[-1].name}, before stage "
                        f"{self.index[upcoming]} of {self.total} ({upcoming})."
                    )
                    hint = STAGE_HINTS.get(upcoming, "")
        else:
            elapsed = time.perf_counter() - self._stage_started
            where = (
                f"Stage {self.index[self.current]} of {self.total} "
                f"({self.current}) failed after {format_duration(elapsed)}."
            )
            hint = STAGE_HINTS.get(self.current, "")
        parts = [where, str(exc).strip()]
        if hint and hint not in str(exc):
            parts.append("Next: " + hint)
        return "\n".join(parts)

    def render_summary(self, lines: list[str]) -> str:
        total = time.perf_counter() - self.started
        rows = [f"video2mesh finished in {format_duration(total)}"]
        width = max((len(record.name) for record in self.records), default=0)
        for record in self.records:
            detail = f"  {record.detail}" if record.detail else ""
            rows.append(
                f"  {record.name:<{width}}  {format_duration(record.elapsed):>8}{detail}"
            )
        rows.extend(lines)
        return "\n".join(rows)


def call_with_heartbeat(label: str, fn, interval: float = 20.0):
    """Run ``fn`` and log a timestamp while it produces no other output."""
    if interval <= 0:
        return fn()
    stop = threading.Event()
    started = time.monotonic()

    def _pulse() -> None:
        while not stop.wait(interval):
            logger.info(
                "still working on %s (%s elapsed, no new output)",
                label,
                format_duration(time.monotonic() - started),
            )

    thread = threading.Thread(target=_pulse, name="video2mesh-heartbeat", daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)


def format_duration(seconds: float) -> str:
    """Short wall-clock duration, such as 4.2s or 1m 05s."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def should_log_tick(done: int, total: int | None) -> bool:
    """True for the first item, the last item, and about 20 updates in between."""
    if done <= 0:
        return False
    if total is None or total <= 0:
        return done == 1 or done % 10 == 0
    if done == 1 or done >= total:
        return True
    step = 1 if total <= 20 else max(1, total // 20)
    return done % step == 0


def colmap_progress_line(line: str) -> str | None:
    """Turn a useful COLMAP log line into a short progress message.

    Returns None for routine lines. Those still go to log.txt at DEBUG.
    """
    text = line.strip()
    if not text:
        return None
    match = _PROCESSED.search(text)
    if match:
        return f"processed image {match.group(1)}/{match.group(2)}"
    match = _MATCH_BLOCK.search(text)
    if match:
        return f"matching block {match.group(1)}/{match.group(2)}"
    match = _REGISTER.search(text)
    if match:
        return "registering image #" + match.group(1)
    match = _FEATURES.search(text)
    if match:
        return f"features: {match.group(1)}"
    match = _ELAPSED.search(text)
    if match:
        return f"COLMAP elapsed {match.group(1)} min"
    if "ERROR" in text or text.startswith("E") and "Error" in text:
        return text
    if "WARNING" in text or "Failed" in text or "fail" in text[:40].lower():
        return text
    return None
