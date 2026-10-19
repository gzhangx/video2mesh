"""Expected failures raised by the video2mesh pipeline."""


class Video2MeshError(Exception):
    """Base class for errors the CLI should print without a traceback."""


class MissingDependencyError(Video2MeshError):
    """ffmpeg, COLMAP, or a required Python package is not available."""


class ReconstructionError(Video2MeshError):
    """The video could not be turned into a usable reconstruction."""
