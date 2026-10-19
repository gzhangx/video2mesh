"""Pose math and COLMAP model parsing."""

import struct
from pathlib import Path

import numpy as np

from video2mesh.colmap_io import qvec_to_rotmat, read_model


def test_identity_quaternion_and_camera_center():
    rotation = qvec_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-8)
    # X_cam = X_world + t with t = (0, 0, 5) places the camera at (0, 0, -5).
    center = -rotation.T @ np.array([0.0, 0.0, 5.0])
    np.testing.assert_allclose(center, [0.0, 0.0, -5.0])


def test_180_degree_rotation_about_x():
    rotation = qvec_to_rotmat(np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(rotation, np.diag([1.0, -1.0, -1.0]), atol=1e-8)


def test_read_text_model(tmp_path: Path):
    (tmp_path / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 640 480 700 710 320 240\n",
        encoding="utf-8",
    )
    (tmp_path / "images.txt").write_text(
        "# images\n"
        "1 1 0 0 0 0 0 5 1 frame_000001.jpg\n"
        "\n"
        "2 0 1 0 0 0 0 0 1 frame_000002.jpg\n"
        "0 0 1\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "# points\n7 1.5 -2 3 10 20 30 0.25 1 0\n",
        encoding="utf-8",
    )
    model = read_model(tmp_path)
    assert set(model.cameras) == {1}
    assert model.cameras[1].model == "PINHOLE"
    assert len(model.frames) == 2
    first = model.frames[1]
    np.testing.assert_allclose(first.center, [0.0, 0.0, -5.0])
    assert model.frames[2].name == "frame_000002.jpg"
    assert len(model.points) == 1
    np.testing.assert_allclose(model.points[0].xyz, [1.5, -2.0, 3.0])


def test_read_binary_model_roundtrip(tmp_path: Path):
    # PINHOLE is model id 1 and has 4 parameters.
    with (tmp_path / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 3, 1, 640, 480))
        handle.write(struct.pack("<dddd", 500.0, 500.0, 320.0, 240.0))
    name = b"frame_000003.jpg\x00"
    with (tmp_path / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<I", 9))
        handle.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
        handle.write(struct.pack("<ddd", 0.0, 0.0, 2.0))
        handle.write(struct.pack("<I", 3))
        handle.write(name)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<ddq", 10.0, 11.0, 4))
    with (tmp_path / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", 4))
        handle.write(struct.pack("<ddd", 0.0, 1.0, 2.0))
        handle.write(struct.pack("<BBB", 8, 9, 10))
        handle.write(struct.pack("<d", 0.1))
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<II", 9, 0))

    model = read_model(tmp_path)
    camera = model.cameras[3]
    assert camera.model == "PINHOLE"
    assert camera.width == 640
    np.testing.assert_allclose(camera.params, [500, 500, 320, 240])
    frame = model.frames[9]
    assert frame.name == "frame_000003.jpg"
    np.testing.assert_allclose(frame.center, [0, 0, -2])
    assert model.points[0].id == 4
    assert model.points[0].rgb.tolist() == [8, 9, 10]
