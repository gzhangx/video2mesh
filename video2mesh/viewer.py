"""Open3D viewer: mesh (or point cloud) plus a frustum for every camera."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from video2mesh.errors import ReconstructionError


def load_scene(
    output_dir: Path,
    *,
    frustum_scale: float = 0.0,
    prefer_point_cloud: bool = False,
) -> list[o3d.geometry.Geometry]:
    """Build the geometries the viewer draws. This does not open a window."""
    output_dir = Path(output_dir)
    cameras_path = output_dir / "cameras.json"
    if not cameras_path.is_file():
        raise ReconstructionError(
            f"No cameras.json in {output_dir}. Run video2mesh reconstruct first."
        )
    payload = json.loads(cameras_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    geometries: list[o3d.geometry.Geometry] = []

    mesh_path = output_dir / "mesh.ply"
    cloud_path = output_dir / "pointcloud.ply"
    if prefer_point_cloud or not mesh_path.is_file():
        if not cloud_path.is_file():
            raise ReconstructionError(f"No mesh.ply or pointcloud.ply in {output_dir}")
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        geometries.append(cloud)
        bounds = np.asarray(cloud.points)
    else:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        geometries.append(mesh)
        bounds = np.asarray(mesh.vertices)

    scale = frustum_scale
    if scale <= 0:
        scale = _auto_scale(bounds, frames)
    for index, frame in enumerate(frames):
        color = _frustum_color(index, max(len(frames) - 1, 1))
        geometries.append(frustum_lines(frame, scale=scale, color=color))
    return geometries


def view_reconstruction(
    output_dir: Path,
    *,
    frustum_scale: float = 0.0,
    prefer_point_cloud: bool = False,
) -> None:
    """Open an interactive window. Requires a display."""
    if not _has_display():
        raise ReconstructionError(
            "No display is available, so the Open3D window cannot open.\n"
            "On Windows, run this from a normal desktop session:\n"
            f"  video2mesh view {output_dir}\n"
            "The mesh and cameras.json can still be opened in MeshLab or Blender."
        )
    geometries = load_scene(
        output_dir,
        frustum_scale=frustum_scale,
        prefer_point_cloud=prefer_point_cloud,
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="video2mesh",
        width=1280,
        height=720,
        mesh_show_back_face=True,
    )


def frustum_lines(frame: dict, *, scale: float, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """Camera pyramid from the center through the four image corners at ``scale`` depth."""
    intrinsics = frame["intrinsics"]
    fx = float(intrinsics.get("fx", intrinsics.get("f")))
    fy = float(intrinsics.get("fy", fx))
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    width = float(frame["width"])
    height = float(frame["height"])
    rotation = _rotation_from_frame(frame)
    center = np.asarray(frame["camera_center_world"], dtype=np.float64)

    pixels = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    )
    corners_cam = []
    for u, v in pixels:
        direction = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        direction = direction / np.linalg.norm(direction) * scale
        corners_cam.append(direction)
    corners_cam_arr = np.stack(corners_cam)
    # X_cam = R X_world + t, and C = -R^T t, so X_world = R^T X_cam + C.
    corners_world = (rotation.T @ corners_cam_arr.T).T + center
    points = np.vstack([center.reshape(1, 3), corners_world])
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(lines), 1)))
    return line_set


def _rotation_from_frame(frame: dict) -> np.ndarray:
    matrix = np.asarray(frame["world_to_camera"], dtype=np.float64)
    if matrix.shape == (3, 4):
        return matrix[:, :3]
    from video2mesh.colmap_io import qvec_to_rotmat

    return qvec_to_rotmat(np.asarray(frame["rotation_quaternion_wxyz"], dtype=np.float64))


def _auto_scale(bounds: np.ndarray, frames: list[dict]) -> float:
    chunks = []
    if bounds is not None and len(bounds) > 0:
        chunks.append(bounds)
    if frames:
        chunks.append(np.array([frame["camera_center_world"] for frame in frames], dtype=np.float64))
    if not chunks:
        return 0.2
    stacked = np.vstack(chunks)
    diagonal = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    return max(diagonal * 0.08, 1e-3)


def _frustum_color(index: int, last: int) -> tuple[float, float, float]:
    mix = index / last
    return (0.15 + 0.8 * mix, 0.35, 0.85 - 0.6 * mix)


def _has_display() -> bool:
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
