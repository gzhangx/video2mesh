"""Projection and viewer key handling. These tests do not open a window."""

import json
from pathlib import Path

import numpy as np
import pytest

from video2mesh.cli import main
from video2mesh.colmap_io import read_model
from video2mesh.errors import ReconstructionError
from video2mesh.viewmath import (
    ORBIT_STEP_DEG,
    ZOOM_IN,
    KeyEffect,
    ViewState,
    apply_key_effect,
    dolly_camera,
    image_plane_corners,
    image_plane_mesh,
    interpret_key,
    key_token,
    load_overlay,
    orbit_camera,
    pose_from_model,
    project_points,
    render_overlay,
    starting_frame_index,
)


def _frame(**overrides):
    frame = {
        "filename": "frame_000001.jpg",
        "width": 100,
        "height": 80,
        "camera_model": "SIMPLE_PINHOLE",
        "intrinsics": {"f": 50.0, "fx": 50.0, "fy": 50.0, "cx": 50.0, "cy": 40.0},
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
        "camera_center_world": [0.0, 0.0, 0.0],
        "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    }
    frame.update(overrides)
    return frame


def test_project_optical_axis_hits_the_principal_point():
    uv, depth, valid = project_points(np.array([[0.0, 0.0, 10.0]]), _frame())
    assert valid[0]
    assert depth[0] == pytest.approx(10.0)
    np.testing.assert_allclose(uv[0], [50.0, 40.0])


def test_project_uses_pose_and_scales_intrinsics_to_the_bitmap():
    frame = _frame(
        translation=[0.0, 0.0, 2.0],
        camera_center_world=[0.0, 0.0, -2.0],
        world_to_camera=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2]],
    )
    uv, depth, valid = project_points(np.array([[0.0, 0.0, 0.0]]), frame, image_size=(50, 40))
    assert valid[0]
    assert depth[0] == pytest.approx(2.0)
    # Half-size bitmap: the principal point moves from (50, 40) to (25, 20).
    np.testing.assert_allclose(uv[0], [25.0, 20.0])


def test_project_simple_radial_and_points_behind_the_camera():
    frame = _frame(
        camera_model="SIMPLE_RADIAL",
        intrinsics={"f": 100.0, "fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0, "k1": 0.2},
        width=200,
        height=160,
    )
    uv, _depth, valid = project_points(np.array([[1.0, 0.0, 10.0]]), frame)
    # xn = 0.1, r2 = 0.01, factor = 1.002, u = 100 * 0.1002 + 50.
    assert valid[0]
    assert uv[0, 0] == pytest.approx(60.02)
    assert uv[0, 1] == pytest.approx(40.0)

    pinhole = _frame(
        camera_model="PINHOLE",
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0},
        width=200,
        height=160,
    )
    plain, _, _ = project_points(np.array([[1.0, 0.0, 10.0]]), pinhole)
    assert plain[0, 0] == pytest.approx(60.0)
    assert uv[0, 0] != pytest.approx(plain[0, 0])

    _uv, _depth, behind = project_points(np.array([[0.0, 0.0, -1.0]]), frame)
    assert not behind[0]


def test_overlay_paints_the_projected_pixel():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    uv = np.array([[10.0, 20.0]])
    depth = np.array([5.0])
    valid = np.array([True])
    rgb = np.array([[1.0, 0.0, 0.0]])
    painted = render_overlay(image, uv, depth, valid, rgb)
    assert tuple(painted[20, 10]) == (255, 0, 0)
    hidden = render_overlay(image, uv, depth, valid, rgb, hide_points=True)
    assert tuple(hidden[20, 10]) == (0, 0, 0)


def test_image_plane_is_the_sensor_rectangle():
    frame = _frame(
        width=100,
        height=100,
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
    )
    corners = image_plane_corners(frame, depth=2.0)
    np.testing.assert_allclose(corners[0], [-1.0, -1.0, 2.0])
    np.testing.assert_allclose(corners[1], [1.0, -1.0, 2.0])
    np.testing.assert_allclose(corners[2], [1.0, 1.0, 2.0])
    assert np.allclose(corners[:, 2], 2.0)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    mesh = image_plane_mesh(frame, rgb, depth=2.0)
    assert len(mesh.triangles) == 4
    assert len(mesh.triangle_uvs) == 12
    np.testing.assert_allclose(np.asarray(mesh.vertices)[0], [-1.0, -1.0, 2.0])


def test_orbit_yaw_pitch_zoom_and_pose_matrix():
    eye = np.array([0.0, 0.0, 5.0])
    center = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    right, _ = orbit_camera(eye, center, up, yaw_deg=ORBIT_STEP_DEG, pitch_deg=0)
    left, _ = orbit_camera(eye, center, up, yaw_deg=-ORBIT_STEP_DEG, pitch_deg=0)
    raised, raised_up = orbit_camera(eye, center, up, yaw_deg=0, pitch_deg=ORBIT_STEP_DEG)
    assert right[0] > 0
    assert left[0] < 0
    assert raised[1] > 0
    # A second pitch step keeps moving up, and a view already on the pole stays put.
    higher, _ = orbit_camera(raised, center, raised_up, 0, ORBIT_STEP_DEG)
    assert higher[1] > raised[1]
    pole, _ = orbit_camera(np.array([0.0, 5.0, 0.0]), center, up, 0, 80)
    np.testing.assert_allclose(pole, [0.0, 5.0, 0.0], atol=1e-6)

    closer = dolly_camera(eye, center, ZOOM_IN)
    assert np.linalg.norm(closer) < np.linalg.norm(eye)

    look = np.array([0.0, 0.0, -1.0])
    right_axis = np.cross(look, up)
    model = np.eye(4)
    model[:3, 0] = right_axis
    model[:3, 1] = up
    model[:3, 2] = -look
    model[:3, 3] = eye
    got_eye, forward, got_up = pose_from_model(model)
    np.testing.assert_allclose(got_eye, eye)
    np.testing.assert_allclose(forward, look)
    np.testing.assert_allclose(got_up, up)


def test_key_map_and_frame_wrap():
    assert interpret_key("w").pitch_deg == ORBIT_STEP_DEG
    assert interpret_key("s").pitch_deg == -ORBIT_STEP_DEG
    assert interpret_key("a").yaw_deg == -ORBIT_STEP_DEG
    assert interpret_key("d").yaw_deg == ORBIT_STEP_DEG
    assert interpret_key("q").zoom_factor == ZOOM_IN
    assert interpret_key("e").zoom_factor > 1
    assert interpret_key("r").reset
    assert interpret_key("left").frame_delta == -1
    assert interpret_key("right").frame_delta == 1
    assert interpret_key("n").frame_delta == 1
    assert interpret_key("p").frame_delta == -1
    assert interpret_key("[").frame_delta == -1
    assert interpret_key("]").frame_delta == 1
    assert interpret_key("f").toggle_photo
    assert interpret_key("h").toggle_points
    assert interpret_key("c").toggle_color
    assert interpret_key("g").toggle_source
    assert interpret_key("x") is None
    assert interpret_key(None) is None

    class _Key:
        name = "LEFT"

    assert key_token(_Key()) == "left"
    assert key_token(ord("W")) == "w"
    assert key_token(119) == "w"  # Open3D KeyName.W
    assert interpret_key(key_token(263)).frame_delta == -1  # KeyName.LEFT
    assert interpret_key(key_token(264)).frame_delta == 1  # KeyName.RIGHT
    assert interpret_key(key_token(91)).frame_delta == -1  # KeyName.LEFT_BRACKET
    assert interpret_key(key_token(ord("D"))).yaw_deg == ORBIT_STEP_DEG

    state = ViewState()
    state = apply_key_effect(state, KeyEffect(frame_delta=-1), 4)
    assert state.frame_index == 3
    state = apply_key_effect(state, interpret_key("f"), 4)
    state = apply_key_effect(state, interpret_key("h"), 4)
    state = apply_key_effect(state, interpret_key("c"), 4)
    state = apply_key_effect(state, interpret_key("g"), 4)
    assert state.show_photo
    assert not state.show_points
    assert state.color_mode == "depth"
    assert state.point_source == "cloud"
    # A repeated key is just another call, which is what a held key does.
    state = apply_key_effect(state, interpret_key("c"), 4)
    assert state.color_mode == "rgb"

    assert starting_frame_index(2, 5) == 1
    with pytest.raises(ReconstructionError):
        starting_frame_index(0, 5)
    with pytest.raises(ReconstructionError):
        starting_frame_index(6, 5)


def test_view_help_lists_the_keys(capsys):
    try:
        code = main(["view", "--help"])
    except SystemExit as exc:
        code = exc.code
    assert code == 0
    text = capsys.readouterr().out
    assert "--frame" in text
    assert "W / S" in text
    assert "Left Right" in text
    assert "out/frames" in text


def test_overlay_uses_named_sparse_model_and_ignores_colmap4_files(tmp_path: Path):
    sparse3 = tmp_path / "sparse" / "3"
    sparse0 = tmp_path / "sparse" / "0"
    sparse3.mkdir(parents=True)
    sparse0.mkdir()
    (sparse3 / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 SIMPLE_PINHOLE 100 80 50 50 40\n",
        encoding="utf-8",
    )
    (sparse3 / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "1 1 0 0 0 0 0 2 1 frame_000001.jpg\n\n",
        encoding="utf-8",
    )
    (sparse3 / "points3D.txt").write_text(
        "# 3D point list\n"
        "1 0.25 0 4 10 20 30 0.2\n",
        encoding="utf-8",
    )
    (sparse3 / "frames.bin").write_bytes(b"not-a-frame-model")
    (sparse3 / "rigs.bin").write_bytes(b"not-a-rig-model")
    (sparse3 / "frames.txt").write_text("# COLMAP 4 frames\n", encoding="utf-8")
    (sparse3 / "rigs.txt").write_text("# COLMAP 4 rigs\n", encoding="utf-8")
    # A different model that must not win just because it is sparse/0.
    (sparse0 / "cameras.txt").write_text("1 PINHOLE 10 10 1 1 5 5\n", encoding="utf-8")
    (sparse0 / "points3D.txt").write_text(
        "1 9 9 9 1 1 1 1\n2 8 8 8 1 1 1 1\n",
        encoding="utf-8",
    )
    payload = {
        "sparse_model": "D:\\work\\cur\\video2mesh\\out\\sparse\\3",
        "frames": [
            {
                "filename": "frame_000001.jpg",
                "width": 100,
                "height": 80,
                "camera_model": "SIMPLE_PINHOLE",
                "intrinsics": {"f": 50, "cx": 50, "cy": 40},
                "camera_center_world": [0, 0, -2],
                "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2]],
            }
        ],
    }
    (tmp_path / "cameras.json").write_text(json.dumps(payload), encoding="utf-8")
    model = read_model(sparse3)
    assert len(model.points) == 1
    assert model.frames[1].name == "frame_000001.jpg"
    overlay = load_overlay(tmp_path, payload)
    assert overlay.sparse_dir == sparse3
    np.testing.assert_allclose(overlay.sparse_xyz[0], [0.25, 0, 4])
    np.testing.assert_allclose(overlay.sparse_rgb[0], [10 / 255, 20 / 255, 30 / 255])
    uv, _depth, valid = project_points(overlay.sparse_xyz, overlay.frames[0])
    assert valid[0]
    # Camera at z=-2 looking along +Z. Point z=4 is 6 units in front.
    # u = 50 * (0.25 / 6) + 50.
    assert uv[0, 0] == pytest.approx(50.0 + 50.0 * 0.25 / 6.0)
