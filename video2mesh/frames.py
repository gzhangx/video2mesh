"""Extract a temporally ordered JPEG sequence from a video."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video2mesh.deps import FFMPEG_INSTALL_HELP, command_argv, find_ffmpeg, run_command
from video2mesh.errors import ReconstructionError

logger = logging.getLogger("video2mesh")


@dataclass
class ExtractedFrame:
    filename: str
    video_timestamp_sec: float
    source_frame_index: int
    blur_score: float
    width: int
    height: int


@dataclass
class FrameExtraction:
    frames_dir: Path
    frames: list[ExtractedFrame]
    video_fps: float
    extractor: str

    def manifest(self) -> dict:
        return {
            "video_fps": self.video_fps,
            "extractor": self.extractor,
            "frames": [frame.__dict__ for frame in self.frames],
        }

    def write_manifest(self, path: Path) -> None:
        path.write_text(json.dumps(self.manifest(), indent=2), encoding="utf-8")


def extract_frames(
    video: Path,
    frames_dir: Path,
    *,
    fps: float,
    max_frames: int,
    blur_threshold: float,
    max_image_size: int,
    ffmpeg_exe: str | None,
    use_ffmpeg: bool = True,
) -> FrameExtraction:
    """Write ``frame_######.jpg`` files and drop extras, then optionally blurry ones.

    ffmpeg is used when it is on PATH because it handles more codecs. OpenCV is
    the fallback and is also used when ffmpeg fails. Frame numbers stay
    contiguous so COLMAP sequential matching follows time order.
    """
    video = Path(video)
    if not video.is_file():
        raise ReconstructionError(f"Input video does not exist: {video}")
    frames_dir = Path(frames_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta = _video_metadata(video)
    if meta is None and find_ffmpeg(ffmpeg_exe) is None:
        raise ReconstructionError(FFMPEG_INSTALL_HELP.strip())

    video_fps = meta[0] if meta else 0.0
    frame_count = meta[1] if meta else 0
    effective_fps = _effective_fps(fps, video_fps, frame_count, max_frames)
    logger.info(
        "Extracting frames at %.3f fps (video %.3f fps, %s frames), cap %d",
        effective_fps,
        video_fps,
        frame_count if frame_count else "unknown",
        max_frames,
    )

    ffmpeg = find_ffmpeg(ffmpeg_exe) if use_ffmpeg else None
    extractor = ""
    if ffmpeg is not None:
        try:
            _extract_ffmpeg(ffmpeg, video, frames_dir, effective_fps)
            extractor = "ffmpeg"
        except ReconstructionError as exc:
            logger.warning("ffmpeg extraction failed (%s). Trying OpenCV.", exc)
            _clear_jpegs(frames_dir)
    if not extractor:
        if meta is None:
            raise ReconstructionError(
                "OpenCV could not open the video and ffmpeg did not produce frames.\n"
                + FFMPEG_INSTALL_HELP.strip()
            )
        _extract_opencv(video, frames_dir, video_fps, effective_fps, max_frames)
        extractor = "opencv"

    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(paths) < 3:
        raise ReconstructionError(
            f"Only {len(paths)} frame(s) were extracted from {video.name}. "
            "Need at least 3. Check that the file is a readable video and that "
            "--fps is not far above the video frame rate."
        )
    if len(paths) > max_frames:
        paths = _subsample_paths(paths, max_frames)
        _delete_unselected(frames_dir, paths)

    records = _score_resize_and_renumber(
        paths,
        frames_dir,
        effective_fps=effective_fps,
        video_fps=video_fps,
        blur_threshold=blur_threshold,
        max_image_size=max_image_size,
    )
    if len(records) < 3:
        raise ReconstructionError(
            f"Only {len(records)} frame(s) left after blur filtering "
            f"(threshold {blur_threshold}). Lower --blur-threshold or recapture "
            "with a faster shutter."
        )
    logger.info("Kept %d frames in %s using %s", len(records), frames_dir, extractor)
    return FrameExtraction(frames_dir, records, video_fps, extractor)


def blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian. Small values mean a blurry frame."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def uniform_indices(count: int, keep: int) -> list[int]:
    """Indices that keep the first and last item and spread the rest evenly."""
    if keep >= count:
        return list(range(count))
    if keep <= 1:
        return [0]
    raw = [int(round(i * (count - 1) / (keep - 1))) for i in range(keep)]
    # Rounding can collide on short sequences. Fill forward to stay unique and sorted.
    chosen: list[int] = []
    for index in raw:
        if chosen and index <= chosen[-1]:
            index = chosen[-1] + 1
        if index >= count:
            break
        chosen.append(index)
    return chosen


def _video_metadata(video: Path) -> tuple[float, int] | None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or count <= 0:
        return None
    return fps, count


def _effective_fps(requested: float, video_fps: float, frame_count: int, max_frames: int) -> float:
    effective = requested
    if video_fps > 0:
        # The fps filter duplicates frames when asked to run faster than the source.
        effective = min(effective, video_fps)
    if video_fps > 0 and frame_count > 0 and max_frames > 0:
        duration = frame_count / video_fps
        if duration > 0:
            effective = min(effective, max_frames / duration)
    return max(effective, 1e-3)


def _extract_ffmpeg(ffmpeg: str, video: Path, frames_dir: Path, fps: float) -> None:
    pattern = str(frames_dir / "frame_%06d.jpg")
    run_command(
        command_argv(
            ffmpeg,
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"fps={fps:.6f}",
                "-q:v",
                "2",
                pattern,
            ],
        )
    )
    if not any(frames_dir.glob("frame_*.jpg")):
        raise ReconstructionError("ffmpeg exited successfully but wrote no frames.")


def _extract_opencv(
    video: Path,
    frames_dir: Path,
    video_fps: float,
    target_fps: float,
    max_frames: int,
) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ReconstructionError(f"OpenCV could not open {video}")
    step = 1
    if video_fps > 0 and target_fps < video_fps:
        step = max(1, int(round(video_fps / target_fps)))
    written = 0
    index = 0
    while written < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            written += 1
            path = frames_dir / f"frame_{written:06d}.jpg"
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                capture.release()
                raise ReconstructionError(f"Failed to write {path}")
        index += 1
    capture.release()


def _subsample_paths(paths: list[Path], keep: int) -> list[Path]:
    return [paths[index] for index in uniform_indices(len(paths), keep)]


def _delete_unselected(frames_dir: Path, selected: list[Path]) -> None:
    keep = {path.resolve() for path in selected}
    for path in frames_dir.glob("frame_*.jpg"):
        if path.resolve() not in keep:
            path.unlink()


def _score_resize_and_renumber(
    paths: list[Path],
    frames_dir: Path,
    *,
    effective_fps: float,
    video_fps: float,
    blur_threshold: float,
    max_image_size: int,
) -> list[ExtractedFrame]:
    staged: list[tuple[Path, float, int]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Skipping unreadable frame %s", path.name)
            path.unlink(missing_ok=True)
            continue
        score = blur_score(image)
        # frame_000001 is the first extracted sample. Its timestamp is 0 when
        # ffmpeg's fps filter starts at the beginning of the file.
        ordinal = _ordinal(path)
        timestamp = (ordinal - 1) / effective_fps if effective_fps > 0 else 0.0
        source_index = int(round(timestamp * video_fps)) if video_fps > 0 else ordinal - 1
        if blur_threshold > 0 and score < blur_threshold:
            logger.info("Dropping blurry %s (score %.1f < %.1f)", path.name, score, blur_threshold)
            path.unlink()
            continue
        resized = resize_max_side(image, max_image_size)
        if resized.shape[:2] != image.shape[:2]:
            if not cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise ReconstructionError(f"Failed to rewrite {path}")
        staged.append((path, score, source_index))

    # Two-step rename so we never overwrite a file we still need.
    temp_paths: list[tuple[Path, float, int, float]] = []
    for index, (path, score, source_index) in enumerate(staged, start=1):
        ordinal = _ordinal(path)
        timestamp = (ordinal - 1) / effective_fps if effective_fps > 0 else 0.0
        temp = frames_dir / f"__ren_{index:06d}.jpg"
        path.rename(temp)
        temp_paths.append((temp, score, source_index, timestamp))

    records: list[ExtractedFrame] = []
    for index, (temp, score, source_index, timestamp) in enumerate(temp_paths, start=1):
        final = frames_dir / f"frame_{index:06d}.jpg"
        temp.rename(final)
        image = cv2.imread(str(final), cv2.IMREAD_COLOR)
        if image is None:
            raise ReconstructionError(f"Frame disappeared during rename: {final}")
        height, width = image.shape[:2]
        records.append(
            ExtractedFrame(
                filename=final.name,
                video_timestamp_sec=timestamp,
                source_frame_index=source_index,
                blur_score=score,
                width=width,
                height=height,
            )
        )
    return records


def _ordinal(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return 1
    return int(digits)


def _clear_jpegs(folder: Path) -> None:
    for path in folder.glob("*.jpg"):
        path.unlink()
