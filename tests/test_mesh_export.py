"""Meshing, pose export, and frustum geometry without COLMAP."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from video2mesh.colmap_io import Camera, FramePose, Point3D, Reconstruction
from video2mesh.export import write_cameras
from video2mesh.frames import ExtractedFrame
from video2mesh.mesh import orient_and_mesh
from video2mesh.viewmath import image_plane_corners
from video2mesh.viewer import frustum_lines, load_scene


def _sphere_cloud(n: int = 1200) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    colors = np.clip(points * 0.5 + 0.5, 0.0, 1.0)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def test_poisson_mesh_has_triangles():
    meshes = orient_and_mesh(
        _sphere_cloud(),
        np.array([[0.0, 0.0, 4.0]]),
        method="poisson",
        poisson_depth=6,
        poisson_trim=0.02,
    )
    mesh = meshes["mesh"]
    assert len(mesh.vertices) > 50
    assert len(mesh.triangles) > 50
    assert mesh.has_vertex_colors()


def test_camera_json_and_csv(tmp_path: Path):
    reconstruction = Reconstruction(
        cameras={
            1: Camera(
                id=1,
                model="OPENCV",
                width=640,
                height=480,
                params=np.array([600, 610, 320, 240, 0.01, -0.02, 0.0, 0.0]),
            )
        },
        frames={
            1: FramePose(
                id=1,
                qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                tvec=np.array([0.0, 0.0, 4.0]),
                camera_id=1,
                name="frame_000001.jpg",
            )
        },
        points=[
            Point3D(
                id=1,
                xyz=np.zeros(3),
                rgb=np.zeros(3, dtype=np.uint8),
                error=0.1,
            )
        ],
    )
    extracted = [
        ExtractedFrame(
            filename="frame_000001.jpg",
            video_timestamp_sec=0.5,
            source_frame_index=12,
            blur_score=88.0,
            width=640,
            height=480,
        )
    ]
    records = write_cameras(
        tmp_path / "cameras.json",
        tmp_path / "cameras.csv",
        source_video=Path("room.mp4"),
        reconstruction=reconstruction,
        extracted=extracted,
        sparse_dir=tmp_path / "sparse" / "0",
        dense_method="sparse",
    )
    assert records[0]["intrinsics"]["fx"] == 600
    assert records[0]["intrinsics"]["k1"] == 0.01
    np.testing.assert_allclose(records[0]["camera_center_world"], [0, 0, -4])
    payload = json.loads((tmp_path / "cameras.json").read_text(encoding="utf-8"))
    assert payload["frames"][0]["rotation_quaternion_wxyz"] == [1, 0, 0, 0]
    assert payload["frames"][0]["video_timestamp_sec"] == 0.5
    csv_text = (tmp_path / "cameras.csv").read_text(encoding="utf-8")
    assert "frame_000001.jpg" in csv_text
    assert "cx_world" in csv_text


def test_frustum_and_viewer_scene(tmp_path: Path):
    frame = {
        "filename": "frame_000001.jpg",
        "width": 640,
        "height": 480,
        "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
        "rotation_quaternion_wxyz": [1, 0, 0, 0],
        "translation": [0, 0, 2],
        "camera_center_world": [0, 0, -2],
        "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2]],
    }
    lines = frustum_lines(frame, scale=0.5, color=(1.0, 0.2, 0.2))
    assert len(lines.points) == 5
    assert len(lines.lines) == 8
    np.testing.assert_allclose(np.asarray(lines.points)[0], [0, 0, -2])
    np.testing.assert_allclose(np.asarray(lines.points)[1:], image_plane_corners(frame, 0.5))

    cloud = _sphere_cloud(80)
    assert o3d.io.write_point_cloud(str(tmp_path / "pointcloud.ply"), cloud)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    assert o3d.io.write_triangle_mesh(str(tmp_path / "mesh.ply"), mesh)
    payload = {
        "frames": [frame],
    }
    (tmp_path / "cameras.json").write_text(json.dumps(payload), encoding="utf-8")
    geometries = load_scene(tmp_path, frustum_scale=0.3)
    assert len(geometries) == 2
