"""Quality presets and the options for one reconstruction run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Tuned for video: short baselines, one moving camera, and a mapper that
# should still succeed on a brief clip (COLMAP's own min_model_size is 10).
QUALITY_PRESETS: dict[str, dict] = {
    "fast": {
        "max_image_size": 800,
        "max_num_features": 4096,
        "sequential_overlap": 10,
        "sift_max_image_size": 800,
        "poisson_depth": 7,
        "init_min_num_inliers": 40,
        "init_min_tri_angle": 8.0,
        "stereo_num_disparities": 96,
        "stereo_stride": 4,
    },
    "medium": {
        "max_image_size": 1600,
        "max_num_features": 8192,
        "sequential_overlap": 12,
        "sift_max_image_size": 1600,
        "poisson_depth": 8,
        "init_min_num_inliers": 60,
        "init_min_tri_angle": 8.0,
        "stereo_num_disparities": 128,
        "stereo_stride": 3,
    },
    "high": {
        "max_image_size": 2400,
        "max_num_features": 16384,
        "sequential_overlap": 20,
        "sift_max_image_size": 2400,
        "poisson_depth": 10,
        "init_min_num_inliers": 80,
        "init_min_tri_angle": 10.0,
        "stereo_num_disparities": 160,
        "stereo_stride": 2,
    },
}

CAMERA_MODELS = (
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
)

SFM_TIPS = """\
Things that usually help:
  - Move slowly so neighboring frames overlap a lot
  - Keep textured surfaces in frame (blank walls, sky, and mirrors are hard)
  - Prefer sideways motion; turning in place does not give depth
  - Drop blurry frames with --blur-threshold (try 40 to 80)
  - Retry a shorter test with --quality fast --fps 1 --max-frames 120
  - Wide lenses often need --camera-model OPENCV
"""


@dataclass
class ReconstructConfig:
    """All knobs for ``video2mesh reconstruct``."""

    video: Path
    output: Path
    fps: float = 2.0
    max_frames: int = 300
    blur_threshold: float = 0.0
    quality: str = "medium"
    dense: bool = True
    mesh_method: str = "poisson"
    camera_model: str = "SIMPLE_RADIAL"
    max_image_size: int | None = None
    overwrite: bool = False
    poisson_trim: float = 0.05
    poisson_depth: int | None = None
    gpu: str = "auto"
    backend: str = "auto"
    colmap_exe: str | None = None
    ffmpeg_exe: str | None = None
    log_level: str = "INFO"
    view: bool = False
    preset: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.video = Path(self.video)
        self.output = Path(self.output)
        if self.quality not in QUALITY_PRESETS:
            known = ", ".join(QUALITY_PRESETS)
            raise ValueError(f"Unknown quality {self.quality!r}. Choose one of: {known}")
        if self.camera_model not in CAMERA_MODELS:
            known = ", ".join(CAMERA_MODELS)
            raise ValueError(f"Unknown camera model {self.camera_model!r}. Choose one of: {known}")
        if self.mesh_method not in {"poisson", "bpa", "both"}:
            raise ValueError("--mesh must be poisson, bpa, or both")
        if self.gpu not in {"auto", "on", "off"}:
            raise ValueError("--gpu must be auto, on, or off")
        if self.backend not in {"auto", "cli", "pycolmap"}:
            raise ValueError("--backend must be auto, cli, or pycolmap")
        if self.fps <= 0:
            raise ValueError("--fps must be positive")
        if self.max_frames < 3:
            raise ValueError("--max-frames must be at least 3")
        if not 0.0 <= self.poisson_trim < 0.9:
            raise ValueError("--poisson-trim must be in [0, 0.9)")
        if self.poisson_depth is not None and self.poisson_depth < 4:
            raise ValueError("--poisson-depth must be at least 4")
        preset = dict(QUALITY_PRESETS[self.quality])
        if self.max_image_size is not None:
            if self.max_image_size < 160:
                raise ValueError("--max-image-size must be at least 160")
            preset["max_image_size"] = self.max_image_size
            preset["sift_max_image_size"] = self.max_image_size
        if self.poisson_depth is not None:
            preset["poisson_depth"] = self.poisson_depth
        self.preset = preset

    @property
    def effective_max_image_size(self) -> int:
        return int(self.preset["max_image_size"])

    @property
    def effective_poisson_depth(self) -> int:
        return int(self.preset["poisson_depth"])
