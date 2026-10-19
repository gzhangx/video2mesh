"""Recover a scene mesh and per-frame camera poses from a video.

The pipeline extracts frames, runs COLMAP structure-from-motion, builds a
dense cloud when it can (CUDA patch-match, otherwise a CPU stereo fallback),
and meshes the cloud with Open3D.
"""

__version__ = "0.1.0"
