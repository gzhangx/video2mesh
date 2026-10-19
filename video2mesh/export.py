"""Write meshes, the point cloud, and camera poses."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from video2mesh import __version__
from video2mesh.colmap_io import Reconstruction, named_intrinsics
from video2mesh.errors import ReconstructionError
from video2mesh.frames import ExtractedFrame

SCALE_NOTE = (
    "Monocular video has no absolute scale. Positions are in the COLMAP frame "
    "and are defined only up to a similarity transform (rotation, translation, and scale)."
)
CONVENTION = (
    "COLMAP world-to-camera: X_cam = R X_world + t. "
    "rotation_quaternion_wxyz is (qw, qx, qy, qz) and builds R. "
    "camera_center_world is C = -R^T t."
)


def write_point_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise ReconstructionError(f"Failed to write point cloud {path}")


def write_mesh(ply_path: Path, obj_path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_normals=True):
        raise ReconstructionError(f"Failed to write {ply_path}")
    # OBJ has no triangle-normal record. Clear them for the write so Open3D
    # does not print a warning, then restore them on the in-memory mesh.
    saved_normals = None
    if mesh.has_triangle_normals():
        saved_normals = np.asarray(mesh.triangle_normals).copy()
        mesh.triangle_normals = o3d.utility.Vector3dVector()
    try:
        wrote = o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_normals=True)
    finally:
        if saved_normals is not None and len(saved_normals):
            mesh.triangle_normals = o3d.utility.Vector3dVector(saved_normals)
    if not wrote:
        raise ReconstructionError(f"Failed to write {obj_path}")


def camera_records(
    reconstruction: Reconstruction,
    extracted: list[ExtractedFrame],
) -> list[dict]:
    """One JSON-ready record per registered frame, in filename order."""
    by_name = {frame.filename: frame for frame in extracted}
    records = []
    for pose in reconstruction.ordered_frames():
        camera = reconstruction.cameras.get(pose.camera_id)
        if camera is None:
            continue
        rotation = pose.rotation
        translation = pose.tvec
        center = pose.center
        world_to_camera = np.concatenate([rotation, translation.reshape(3, 1)], axis=1)
        intrinsics = named_intrinsics(camera.model, camera.params)
        meta = by_name.get(Path(pose.name).name)
        records.append(
            {
                "filename": Path(pose.name).name,
                "frame_index": _frame_index(pose.name),
                "video_timestamp_sec": None if meta is None else meta.video_timestamp_sec,
                "source_frame_index": None if meta is None else meta.source_frame_index,
                "blur_score": None if meta is None else meta.blur_score,
                "width": camera.width,
                "height": camera.height,
                "camera_model": camera.model,
                "intrinsics": intrinsics,
                "rotation_quaternion_wxyz": [float(v) for v in pose.qvec],
                "translation": [float(v) for v in translation],
                "camera_center_world": [float(v) for v in center],
                "world_to_camera": world_to_camera.tolist(),
                "image_id": pose.id,
                "camera_id": pose.camera_id,
            }
        )
    return records


def write_cameras(
    json_path: Path,
    csv_path: Path,
    *,
    source_video: Path,
    reconstruction: Reconstruction,
    extracted: list[ExtractedFrame],
    sparse_dir: Path,
    dense_method: str,
) -> list[dict]:
    records = camera_records(reconstruction, extracted)
    extracted_names = {frame.filename for frame in extracted}
    registered_names = {record["filename"] for record in records}
    payload = {
        "video2mesh_version": __version__,
        "source_video": str(source_video),
        "scale_note": SCALE_NOTE,
        "coordinate_convention": CONVENTION,
        "sparse_model": str(sparse_dir),
        "dense_method": dense_method,
        "num_frames_extracted": len(extracted),
        "num_frames_registered": len(records),
        "unregistered_frames": sorted(extracted_names - registered_names),
        "frames": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, records)
    return records


def _write_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "filename",
        "frame_index",
        "video_timestamp_sec",
        "source_frame_index",
        "blur_score",
        "width",
        "height",
        "camera_model",
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "qw",
        "qx",
        "qy",
        "qz",
        "tx",
        "ty",
        "tz",
        "cx_world",
        "cy_world",
        "cz_world",
        "image_id",
        "camera_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            intrinsics = record["intrinsics"]
            quat = record["rotation_quaternion_wxyz"]
            translation = record["translation"]
            center = record["camera_center_world"]
            writer.writerow(
                {
                    "filename": record["filename"],
                    "frame_index": record["frame_index"],
                    "video_timestamp_sec": _blank(record["video_timestamp_sec"]),
                    "source_frame_index": _blank(record["source_frame_index"]),
                    "blur_score": _blank(record["blur_score"]),
                    "width": record["width"],
                    "height": record["height"],
                    "camera_model": record["camera_model"],
                    "fx": intrinsics.get("fx", ""),
                    "fy": intrinsics.get("fy", ""),
                    "cx": intrinsics.get("cx", ""),
                    "cy": intrinsics.get("cy", ""),
                    "k1": intrinsics.get("k1", ""),
                    "k2": intrinsics.get("k2", ""),
                    "p1": intrinsics.get("p1", ""),
                    "p2": intrinsics.get("p2", ""),
                    "qw": quat[0],
                    "qx": quat[1],
                    "qy": quat[2],
                    "qz": quat[3],
                    "tx": translation[0],
                    "ty": translation[1],
                    "tz": translation[2],
                    "cx_world": center[0],
                    "cy_world": center[1],
                    "cz_world": center[2],
                    "image_id": record["image_id"],
                    "camera_id": record["camera_id"],
                }
            )


def _frame_index(name: str) -> int:
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    return int(digits) if digits else 0


def _blank(value):
    return "" if value is None else value
