"""Locate ffmpeg and COLMAP, and run their command-line tools."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

from video2mesh.errors import MissingDependencyError, ReconstructionError

logger = logging.getLogger("video2mesh")

COLMAP_INSTALL_HELP = """\
COLMAP was not found.

Windows:
  1. Open https://github.com/colmap/colmap/releases and download a Windows zip.
     - Choose the CUDA build if you have an NVIDIA GPU (this enables patch-match
       stereo, which is the highest-quality dense step).
     - Choose the no-CUDA build otherwise. Sparse reconstruction still works,
       and video2mesh falls back to CPU stereo for the dense cloud.
  2. Unzip it, for example to C:\\tools\\COLMAP.
  3. Add the folder that contains colmap.exe (or colmap.bat) to your PATH,
     or set the COLMAP_EXE environment variable to the full path of that file.
     PowerShell, for the current terminal only:
       $env:COLMAP_EXE = "C:\\tools\\COLMAP\\colmap.bat"
       $env:Path = "C:\\tools\\COLMAP;" + $env:Path

Linux:
  sudo apt install colmap
  or build from source: https://colmap.github.io/install.html

macOS:
  brew install colmap

Optional alternative: pip install pycolmap
  and run with --backend pycolmap. The COLMAP executable is the path this
  tool is tested with, and it is the easier install on Windows.
"""

FFMPEG_INSTALL_HELP = """\
ffmpeg was not found, and OpenCV could not read this video either.

Windows:
  winget install Gyan.FFmpeg
  or download the essentials build from https://www.gyan.dev/ffmpeg/builds/
  and add its bin folder to PATH.
  You can also set FFMPEG_EXE to the full path of ffmpeg.exe.

Linux:  sudo apt install ffmpeg
macOS:  brew install ffmpeg
"""


def find_executable(name: str, override: str | None, env_var: str) -> str | None:
    """Resolve a tool path from a CLI flag, an environment variable, or PATH."""
    if override:
        candidate = Path(override)
        if not candidate.is_file():
            raise MissingDependencyError(
                f"{env_var} override does not point at a file: {override}"
            )
        return str(candidate)
    env_value = os.environ.get(env_var)
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_file():
            raise MissingDependencyError(
                f"{env_var} is set but is not a file: {env_value}"
            )
        return str(candidate)
    found = shutil.which(name)
    return found


def find_colmap(override: str | None = None) -> str | None:
    return find_executable("colmap", override, "COLMAP_EXE")


def find_ffmpeg(override: str | None = None) -> str | None:
    return find_executable("ffmpeg", override, "FFMPEG_EXE")


def require_colmap(override: str | None = None) -> str:
    exe = find_colmap(override)
    if exe is None:
        raise MissingDependencyError(COLMAP_INSTALL_HELP.strip())
    return exe


def command_argv(exe: str, args: list[str]) -> list[str]:
    """Build an argv list. Windows batch wrappers need ``cmd /c``."""
    if os.name == "nt" and exe.lower().endswith((".bat", ".cmd")):
        return ["cmd", "/c", exe, *args]
    return [exe, *args]


def colmap_version_text(exe: str) -> str:
    try:
        proc = subprocess.run(
            command_argv(exe, ["-h"]),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(could not run {exe}: {exc})"
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in text.splitlines():
        if "COLMAP" in line:
            return line.strip()
    return "COLMAP"


def colmap_has_cuda(exe: str) -> bool:
    """Return whether this COLMAP binary can run CUDA patch-match stereo.

    ``patch_match_stereo -h`` exits with an error on no-CUDA builds, and the
    banner of other commands says ``with CUDA`` or ``without CUDA``.
    """
    try:
        proc = subprocess.run(
            command_argv(exe, ["feature_extractor", "-h"]),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    banner = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "without CUDA" in banner:
        return False
    if "with CUDA" in banner:
        return True
    try:
        probe = subprocess.run(
            command_argv(exe, ["patch_match_stereo", "-h"]),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    blob = (probe.stdout or "") + "\n" + (probe.stderr or "")
    if "requires CUDA" in blob or "without CUDA" in blob:
        return False
    return probe.returncode == 0


def decide_gpu(exe: str, mode: str) -> bool:
    """Resolve ``--gpu auto|on|off`` to a boolean for this COLMAP build."""
    available = colmap_has_cuda(exe)
    if mode == "on":
        if not available:
            raise MissingDependencyError(
                "GPU was requested (--gpu on) but this COLMAP build reports no CUDA.\n"
                "Install the Windows CUDA release of COLMAP, or run with --gpu off.\n"
                "Without CUDA, video2mesh still reconstructs cameras and uses its\n"
                "CPU stereo fallback for the dense point cloud."
            )
        return True
    if mode == "off":
        return False
    return available


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    echo: str = "debug",
) -> str:
    """Run a command, capture its combined log, and raise on a non-zero exit.

    ``echo`` is ``debug`` (default), ``info``, or ``silent``. Output is always
    returned so a failure can include the tail of the log.
    """
    logger.debug("Running: %s", " ".join(argv))
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise MissingDependencyError(
            f"Could not start {argv[0]!r}. Is it installed and on PATH?"
        ) from exc
    assert proc.stdout is not None
    tail: deque[str] = deque(maxlen=40)
    chunks: list[str] = []
    log_fn = {"debug": logger.debug, "info": logger.info, "silent": None}[echo]
    for line in proc.stdout:
        line = line.rstrip()
        chunks.append(line)
        tail.append(line)
        if log_fn is not None and line:
            log_fn(line)
    code = proc.wait()
    output = "\n".join(chunks)
    if code != 0:
        preview = "\n".join(tail) if tail else "(no output)"
        raise ReconstructionError(
            f"Command failed with exit code {code}: {argv[0]}\n{preview}"
        )
    return output
