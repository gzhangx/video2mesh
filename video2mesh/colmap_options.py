"""Map logical COLMAP options onto the flags a given binary actually accepts.

COLMAP 4 renamed several SIFT-wide settings. ``max_image_size``, ``use_gpu``,
``num_threads``, and ``gpu_index`` moved from ``SiftExtraction`` to
``FeatureExtraction``, and the matching equivalents moved from ``SiftMatching``
to ``FeatureMatching``. SIFT-only knobs (``max_num_features``, ``max_ratio``,
``cross_check``, and so on) stayed put. Mapper, undistort, patch-match, fusion,
and model conversion kept the names this pipeline already used.

Each subcommand's ``-h`` text is parsed once. A logical option is written with
the first candidate that appears there. If none of the candidates exist, the
option is omitted and a warning is logged instead of failing the whole run
with ``unrecognised option``.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

from video2mesh.deps import command_argv
from video2mesh.errors import ReconstructionError

logger = logging.getLogger("video2mesh")

_OPTION_RE = re.compile(r"--([A-Za-z][A-Za-z0-9_.]*)")

# Prefer the COLMAP 4 name, then the COLMAP 3 name. The help scan picks
# whichever of these the installed binary still prints.
FEATURE_MAX_IMAGE_SIZE = ("FeatureExtraction.max_image_size", "SiftExtraction.max_image_size")
FEATURE_USE_GPU = ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu")
FEATURE_NUM_THREADS = ("FeatureExtraction.num_threads", "SiftExtraction.num_threads")
FEATURE_GPU_INDEX = ("FeatureExtraction.gpu_index", "SiftExtraction.gpu_index")
FEATURE_MAX_NUM_FEATURES = ("SiftExtraction.max_num_features", "FeatureExtraction.max_num_features")

MATCH_USE_GPU = ("FeatureMatching.use_gpu", "SiftMatching.use_gpu")
MATCH_NUM_THREADS = ("FeatureMatching.num_threads", "SiftMatching.num_threads")
MATCH_GPU_INDEX = ("FeatureMatching.gpu_index", "SiftMatching.gpu_index")
MATCH_GUIDED = ("FeatureMatching.guided_matching", "SiftMatching.guided_matching")
MATCH_MAX_NUM_MATCHES = ("FeatureMatching.max_num_matches", "SiftMatching.max_num_matches")

# ``images`` became ``frames`` inside a few bundle-adjustment option names.
MAPPER_BA_GLOBAL_IMAGES_RATIO = ("Mapper.ba_global_frames_ratio", "Mapper.ba_global_images_ratio")
MAPPER_BA_GLOBAL_IMAGES_FREQ = ("Mapper.ba_global_frames_freq", "Mapper.ba_global_images_freq")
MAPPER_SNAPSHOT_IMAGES_FREQ = ("Mapper.snapshot_frames_freq", "Mapper.snapshot_images_freq")
MAPPER_BA_MIN_RESIDUALS_THREADS = (
    "Mapper.ba_min_num_residuals_for_cpu_multi_threading",
    "Mapper.ba_min_num_residuals_for_multi_threading",
)

# Stereo fusion's input selector has always been a top-level flag in the
# documented CLI. Keep a prefixed alias in case a build nests it.
STEREO_INPUT_TYPE = ("input_type", "StereoFusion.input_type")


@dataclass(frozen=True)
class ColmapOption:
    """One value plus the flag names that have meant that value."""

    logical: str
    candidates: tuple[str, ...]
    value: object


def option(logical: str, candidates: str | tuple[str, ...], value: object) -> ColmapOption:
    if isinstance(candidates, str):
        names = (candidates,)
    else:
        names = tuple(candidates)
    if not names:
        raise ValueError(f"Option {logical!r} has no candidate flags")
    return ColmapOption(logical, names, value)


def parse_colmap_help(text: str) -> set[str]:
    """Return option names from ``colmap <command> -h`` output.

    Only lines that look like option declarations are scanned, so a mention of
    a flag inside a sentence does not count as support.
    """
    found: set[str] = set()
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("-"):
            continue
        for match in _OPTION_RE.finditer(stripped):
            found.add(match.group(1))
    return found


def format_colmap_value(value: object) -> str:
    """Render a value the way COLMAP's parser accepts it.

    Boolean words such as ``true`` are not reliable: some builds read them as
    zero. ``0`` and ``1`` work on COLMAP 3 and 4.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def choose_flag(available: set[str], candidates: tuple[str, ...]) -> str | None:
    """First candidate that this binary's help text lists."""
    for name in candidates:
        if name in available:
            return name
    return None


class ColmapCli:
    """Cached ``-h`` lookup for one ``colmap`` executable."""

    def __init__(self, exe: str, help_text: dict[str, str] | None = None) -> None:
        self.exe = exe
        self._help_text = dict(help_text or {})
        self._flags: dict[str, set[str]] = {}

    def flags(self, subcommand: str) -> set[str]:
        if subcommand not in self._flags:
            text = self._help_text.get(subcommand)
            if text is None:
                text = load_colmap_help(self.exe, subcommand)
                self._help_text[subcommand] = text
            self._flags[subcommand] = parse_colmap_help(text)
        return self._flags[subcommand]

    def add(self, argv: list[str], subcommand: str, options: list[ColmapOption]) -> None:
        """Append ``--flag value`` pairs for the options this subcommand supports."""
        available = self.flags(subcommand)
        if not available:
            logger.warning(
                "Could not read options from `colmap %s -h`. "
                "Passing the preferred flag name for each option.",
                subcommand,
            )
        for item in options:
            flag = item.candidates[0] if not available else choose_flag(available, item.candidates)
            if flag is None:
                logger.warning(
                    "COLMAP %s has no flag for %s (tried %s). Skipping that option.",
                    subcommand,
                    item.logical,
                    ", ".join(f"--{name}" for name in item.candidates),
                )
                continue
            if available and flag != item.candidates[0]:
                logger.info(
                    "COLMAP %s: using --%s for %s",
                    subcommand,
                    flag,
                    item.logical,
                )
            argv.extend([f"--{flag}", format_colmap_value(item.value)])


def load_colmap_help(exe: str, subcommand: str) -> str:
    """Run ``colmap <subcommand> -h`` and return stdout plus stderr.

    A non-zero exit is still usable. No-CUDA builds reject ``patch_match_stereo
    -h`` before printing options, and the caller decides what to do with an
    empty flag set.
    """
    try:
        proc = subprocess.run(
            command_argv(exe, [subcommand, "-h"]),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReconstructionError(f"Could not run `{exe} {subcommand} -h`: {exc}") from exc
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def apply_pycolmap_extraction(options: object, *, max_image_size: int, max_num_features: int, use_gpu: bool) -> list[str]:
    """Set extraction fields on either a COLMAP 3 or COLMAP 4 pycolmap object.

    COLMAP 4 keeps ``max_num_features`` on the nested SIFT options and moves
    image size and the GPU switch to the parent. Older bindings store all three
    on one flat object, or on ``options.sift``. Returns the attribute paths
    that were written.
    """
    applied: list[str] = []
    _set_first(options, ("max_image_size",), max_image_size, applied, "")
    _set_first(options, ("use_gpu",), use_gpu, applied, "")
    _set_first(options, ("max_num_features",), max_num_features, applied, "")
    sift = getattr(options, "sift", None)
    if sift is not None:
        _set_first(sift, ("max_image_size",), max_image_size, applied, "sift")
        _set_first(sift, ("use_gpu",), use_gpu, applied, "sift")
        _set_first(sift, ("max_num_features",), max_num_features, applied, "sift")
    if not any(name.endswith("max_image_size") for name in applied):
        logger.warning(
            "This pycolmap build has no max_image_size field on FeatureExtractionOptions."
        )
    if not any(name.endswith("use_gpu") for name in applied):
        logger.warning("This pycolmap build has no use_gpu field on FeatureExtractionOptions.")
    return applied


def apply_pycolmap_matching(options: object, *, use_gpu: bool) -> list[str]:
    """Set the GPU switch on COLMAP 4's parent options or COLMAP 3's SIFT options."""
    applied: list[str] = []
    _set_first(options, ("use_gpu",), use_gpu, applied, "")
    sift = getattr(options, "sift", None)
    if sift is not None:
        _set_first(sift, ("use_gpu",), use_gpu, applied, "sift")
    if not applied:
        logger.warning("This pycolmap build has no use_gpu field on FeatureMatchingOptions.")
    return applied


def _set_first(obj: object, names: tuple[str, ...], value: object, applied: list[str], prefix: str) -> bool:
    for name in names:
        if not hasattr(obj, name):
            continue
        try:
            setattr(obj, name, value)
        except Exception:
            logger.debug("Could not set pycolmap option %s%s", prefix + "." if prefix else "", name)
            continue
        applied.append(f"{prefix}.{name}" if prefix else name)
        return True
    return False
