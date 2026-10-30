"""Projection and viewer key handling. These tests do not open a window."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from video2mesh.cli import main
from video2mesh.colmap_io import read_model
from video2mesh.errors import ReconstructionError
from video2mesh.viewmath import (
    FLY_STEP_FRACTION,
    ORBIT_STEP_DEG,
    ZOOM_IN,
    KeyEffect,
    ViewState,
    apply_key_effect,
    background_plane_depth,
    apply_follow_offset,
    camera_view_pose,
    follow_offset,
    follow_offset_is_plain,
    dolly_camera,
    fly_camera,
    fly_distance,
    frustum_visible,
    half_turn,
    marker_axes,
    marker_mesh,
    undistort_bgr,
    world_to_camera_matrix,
    SurfacePicker,
    pick_ray,
    image_plane_corners,
    image_plane_mesh,
    interpret_key,
    key_token,
    look_distance,
    pinhole_matrix,
    vertical_fov_degrees,
    load_overlay,
    orbit_camera,
    pose_from_model,
    project_points,
    render_overlay,
    resolve_sparse_dir,
    roll_camera,
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
    verts = np.asarray(mesh.vertices)
    np.testing.assert_allclose(verts[0], [-1.0, -1.0, 2.0])
    # Top-left of the photo samples the top of the bitmap (texture v=1).
    uvs = np.asarray(mesh.triangle_uvs)
    np.testing.assert_allclose(uvs[0], [0.0, 1.0])
    np.testing.assert_allclose(uvs[1], [1.0, 0.0])
    # The scene-facing copy is a sliver further along the optical axis.
    np.testing.assert_allclose(verts[4] - verts[0], [0.0, 0.0, 0.002])


def test_camera_view_looks_through_the_frame_upright():
    eye, look, up = camera_view_pose(_frame(), distance=3.0)
    np.testing.assert_allclose(eye, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(look, [0.0, 0.0, 3.0])
    # OpenCV Y is down, so image-up is camera -Y.
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0])
    # height 80, fy 50 -> 2 * atan(0.8)
    assert vertical_fov_degrees(_frame()) == pytest.approx(np.rad2deg(2.0 * np.arctan(0.8)))
    matrix, width, height = pinhole_matrix(_frame())
    assert (width, height) == (100.0, 80.0)
    np.testing.assert_allclose(matrix, [[50.0, 0.0, 50.0], [0.0, 50.0, 40.0], [0.0, 0.0, 1.0]])

    shifted = _frame(camera_center_world=[1.0, 2.0, 3.0])
    eye, look, up = camera_view_pose(shifted, distance=4.0)
    np.testing.assert_allclose(eye, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(look, [1.0, 2.0, 7.0])
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0])
    # A point ahead of the camera is used; a point behind falls back.
    assert look_distance(_frame(), np.array([0.0, 0.0, 5.0]), minimum=1.0) == pytest.approx(5.0)
    assert look_distance(_frame(), np.array([0.0, 0.0, -2.0]), minimum=1.5) == pytest.approx(1.5)
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, -1.0]])
    assert background_plane_depth(_frame(), points, minimum=0.2) == pytest.approx(5.1)
    assert background_plane_depth(_frame(), np.zeros((0, 3)), minimum=0.4) == pytest.approx(0.4)


def test_image_mode_keeps_only_the_current_frustum():
    assert frustum_visible(0, 1, False)
    assert frustum_visible(2, 1, False)
    assert frustum_visible(1, 1, True)
    assert not frustum_visible(0, 1, True)
    assert not frustum_visible(2, 1, True)


def test_pinhole_extrinsic_matches_the_upright_view():
    frame = _frame(
        world_to_camera=[[0, -1, 0, 1], [0, 0, -1, 2], [1, 0, 0, 3]],
    )
    del frame["camera_center_world"]
    extrinsic = world_to_camera_matrix(frame)
    # Open3D turns a world-to-camera matrix into its camera the same way.
    model = np.linalg.inv(extrinsic) @ np.diag([1.0, -1.0, -1.0, 1.0])
    eye, look, up = camera_view_pose(frame, distance=4.0)
    forward = (look - eye) / np.linalg.norm(look - eye)
    np.testing.assert_allclose(model[:3, 3], eye)
    np.testing.assert_allclose(model[:3, 1], up)
    np.testing.assert_allclose(-model[:3, 2], forward)


def test_undistort_leaves_a_pinhole_bitmap_unchanged():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[1, 2] = (9, 8, 7)
    frame = _frame(width=10, height=8, intrinsics={"f": 8.0, "fx": 8.0, "fy": 8.0, "cx": 5.0, "cy": 4.0})
    np.testing.assert_array_equal(undistort_bgr(image, frame), image)
    radial = _frame(
        width=10,
        height=8,
        camera_model="SIMPLE_RADIAL",
        intrinsics={"f": 8.0, "fx": 8.0, "fy": 8.0, "cx": 5.0, "cy": 4.0, "k1": 0.0},
    )
    np.testing.assert_array_equal(undistort_bgr(image, radial), image)


def test_follow_offset_carries_zoom_onto_the_next_camera():
    home = _frame()
    distance = 10.0
    eye, look, up = camera_view_pose(home, distance)
    assert follow_offset_is_plain(follow_offset(home, eye, look, up, distance))
    zoomed_eye = look + 0.4 * (eye - look)
    offset = follow_offset(home, zoomed_eye, look, up, distance)
    assert not follow_offset_is_plain(offset)
    moved = _frame(camera_center_world=[4.0, 1.0, 0.0])
    got_eye, got_look, got_up = apply_follow_offset(moved, 20.0, offset)
    straight_eye, straight_look, straight_up = camera_view_pose(moved, 20.0)
    np.testing.assert_allclose(got_look, straight_look)
    np.testing.assert_allclose(got_up, straight_up)
    np.testing.assert_allclose(got_eye, straight_look + 0.4 * (straight_eye - straight_look))


def test_pick_ray_rejects_the_infinite_far_plane():
    ray = pick_ray(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.2, 4.9]))
    assert ray is not None
    origin, direction = ray
    np.testing.assert_allclose(origin, [0.0, 0.0, 5.0])
    np.testing.assert_allclose(direction, [0.0, 0.2, -0.1])
    assert pick_ray(np.array([0.0, 0.0, 5.0]), np.array([np.nan, 0.0, -np.inf])) is None


def test_marker_sits_on_the_clicked_surface_point():
    triangle = o3d.geometry.TriangleMesh()
    triangle.vertices = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangle.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    picker = SurfacePicker(triangle, snap_distance=0.1)
    hit = picker.hit(np.array([0.2, 0.2, 1.0]), np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(hit, [0.2, 0.2, 0.0], atol=1e-5)
    assert picker.hit(np.array([2.0, 2.0, 1.0]), np.array([0.0, 0.0, -1.0])) is None
    assert picker.hit(np.array([0.2, 0.2, 1.0]), np.zeros(3)) is None

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0], [0.0, 0.01, 5.0], [1.0, 0.0, 1.0]])
    points = SurfacePicker(cloud, snap_distance=0.05)
    np.testing.assert_allclose(points.hit(np.zeros(3), np.array([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0])
    assert points.hit(np.zeros(3), np.array([0.0, 1.0, 0.0])) is None

    cube = marker_mesh("cube", 2.0)
    center = np.asarray(cube.vertices).mean(axis=0)
    np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-6)
    extent = np.asarray(cube.vertices).max(axis=0) - np.asarray(cube.vertices).min(axis=0)
    np.testing.assert_allclose(extent, [2.0, 2.0, 2.0])
    sphere = marker_axes(4.0)
    ends = np.asarray(sphere.points)
    np.testing.assert_allclose(ends[1], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(ends[2], [0.0, 4.0, 0.0])
    np.testing.assert_allclose(ends[3], [0.0, 0.0, 4.0])
    ball = marker_mesh("sphere", 2.0)
    radius = np.linalg.norm(np.asarray(ball.vertices), axis=1)
    np.testing.assert_allclose(radius, np.full(len(radius), 1.0), atol=1e-5)


def test_half_turn_swaps_opposite_corners():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    image[3, 5] = (4, 5, 6)
    turned = half_turn(image)
    assert turned[3, 5].tolist() == [1, 2, 3]
    assert turned[0, 0].tolist() == [4, 5, 6]
    assert turned.flags["C_CONTIGUOUS"]


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
    # Q/E zoom moves only the eye. The look-at point is the caller's center.
    np.testing.assert_allclose(closer, center + (eye - center) * ZOOM_IN)

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


def test_roll_and_fly_move_the_third_axis_and_both_points():
    eye = np.array([0.0, 0.0, 5.0])
    center = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    # Looking toward -Z, camera-right is +X. Positive roll tilts up that way.
    rolled_eye, rolled_up = roll_camera(eye, center, up, ORBIT_STEP_DEG)
    np.testing.assert_allclose(rolled_eye, eye)
    assert rolled_up[0] > 0
    assert rolled_up[1] > 0
    assert abs(float(np.dot(rolled_up, center - eye))) < 1e-8
    assert np.linalg.norm(rolled_up) == pytest.approx(1.0)
    opposite_eye, opposite_up = roll_camera(eye, center, up, -ORBIT_STEP_DEG)
    np.testing.assert_allclose(opposite_eye, eye)
    assert opposite_up[0] < 0
    _, quarter = roll_camera(eye, center, up, 90.0)
    np.testing.assert_allclose(quarter, [1.0, 0.0, 0.0], atol=1e-6)

    forward_eye, forward_center = fly_camera(eye, center, 2.0)
    np.testing.assert_allclose(forward_eye, [0.0, 0.0, 3.0])
    np.testing.assert_allclose(forward_center, [0.0, 0.0, -2.0])
    assert np.linalg.norm(forward_center - forward_eye) == pytest.approx(5.0)
    back_eye, back_center = fly_camera(eye, center, -1.0)
    np.testing.assert_allclose(back_eye, [0.0, 0.0, 6.0])
    np.testing.assert_allclose(back_center, [0.0, 0.0, 1.0])
    same_eye, same_center = fly_camera(eye, eye.copy(), 3.0)
    np.testing.assert_allclose(same_eye, eye)
    np.testing.assert_allclose(same_center, eye)
    assert fly_distance(10.0, 1) == pytest.approx(FLY_STEP_FRACTION * 10.0)
    assert fly_distance(10.0, -1) == pytest.approx(-FLY_STEP_FRACTION * 10.0)


def test_key_map_and_frame_wrap():
    assert interpret_key("w").pitch_deg == ORBIT_STEP_DEG
    assert interpret_key("s").pitch_deg == -ORBIT_STEP_DEG
    assert interpret_key("a").yaw_deg == -ORBIT_STEP_DEG
    assert interpret_key("d").yaw_deg == ORBIT_STEP_DEG
    assert interpret_key("z").roll_deg == ORBIT_STEP_DEG
    assert interpret_key("c").roll_deg == -ORBIT_STEP_DEG
    assert not interpret_key("c").toggle_color
    assert interpret_key("v").toggle_color
    assert interpret_key("q").zoom_factor == ZOOM_IN
    assert interpret_key("e").zoom_factor > 1
    assert interpret_key("r").reset
    assert interpret_key("up").fly_sign == 1
    assert interpret_key("down").fly_sign == -1
    assert interpret_key("left").frame_delta == -1
    assert interpret_key("right").frame_delta == 1
    assert interpret_key("left").fly_sign == 0
    assert interpret_key("n").frame_delta == 1
    assert interpret_key("p").frame_delta == -1
    assert interpret_key("[").frame_delta == -1
    assert interpret_key("]").frame_delta == 1
    assert interpret_key("f").toggle_photo
    assert interpret_key("h").toggle_points
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
    assert interpret_key(key_token(265)).fly_sign == 1  # KeyName.UP
    assert interpret_key(key_token(266)).fly_sign == -1  # KeyName.DOWN
    assert interpret_key(key_token(91)).frame_delta == -1  # KeyName.LEFT_BRACKET
    assert interpret_key(key_token(ord("D"))).yaw_deg == ORBIT_STEP_DEG
    assert interpret_key(key_token(ord("Z"))).roll_deg == ORBIT_STEP_DEG
    assert interpret_key(key_token(ord("C"))).roll_deg == -ORBIT_STEP_DEG
    assert interpret_key(key_token(ord("V"))).toggle_color

    state = ViewState()
    state = apply_key_effect(state, KeyEffect(frame_delta=-1), 4)
    assert state.frame_index == 3
    state = apply_key_effect(state, interpret_key("f"), 4)
    state = apply_key_effect(state, interpret_key("h"), 4)
    state = apply_key_effect(state, interpret_key("v"), 4)
    state = apply_key_effect(state, interpret_key("g"), 4)
    assert state.show_photo
    assert not state.show_points
    assert state.color_mode == "depth"
    assert state.point_source == "cloud"
    # A repeated key is just another call, which is what a held key does.
    state = apply_key_effect(state, interpret_key("v"), 4)
    assert state.color_mode == "rgb"
    # Roll and fly do not change the overlay state.
    rolled = apply_key_effect(state, interpret_key("z"), 4)
    flown = apply_key_effect(state, interpret_key("up"), 4)
    assert rolled == state
    assert flown == state

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
    assert "Z / C" in text
    assert "Up / Down" in text
    assert "Left / Right" in text
    color_line = next(line for line in text.splitlines() if line.strip().startswith("V "))
    assert "color" in color_line
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
        "sparse_model": "Z:\\__video2mesh_does_not_exist__\\sparse\\3",
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


def _write_cameras_txt(model: Path) -> None:
    model.mkdir(parents=True, exist_ok=True)
    (model / "cameras.txt").write_text("1 PINHOLE 10 10 1 1 5 5\n", encoding="utf-8")


def test_resolve_sparse_dir_prefers_local_folder_over_an_existing_absolute_path(tmp_path: Path):
    output = tmp_path / "out"
    local = output / "sparse" / "3"
    _write_cameras_txt(local)
    (local / "points3D.txt").write_text("1 0 0 1 1 1 1 1\n", encoding="utf-8")
    absolute = tmp_path / "real_drive" / "sparse" / "3"
    _write_cameras_txt(absolute)
    (absolute / "points3D.txt").write_text(
        "1 9 9 9 1 1 1 1\n2 8 8 8 1 1 1 1\n",
        encoding="utf-8",
    )
    chosen = resolve_sparse_dir(output, str(absolute))
    assert chosen == local


def test_resolve_sparse_dir_falls_back_to_the_absolute_path(tmp_path: Path):
    output = tmp_path / "moved"
    output.mkdir()
    absolute = tmp_path / "original" / "sparse" / "3"
    _write_cameras_txt(absolute)
    chosen = resolve_sparse_dir(output, str(absolute))
    assert chosen == absolute
    missing = resolve_sparse_dir(output, "Z:\\__video2mesh_does_not_exist__\\sparse\\3")
    assert missing is None
