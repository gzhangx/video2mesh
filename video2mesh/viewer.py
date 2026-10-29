"""Open3D viewer: mesh (or point cloud), camera frustums, and a frame overlay."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from video2mesh.errors import ReconstructionError
from video2mesh.viewmath import (
    VIEW_HELP,
    OverlayData,
    ViewState,
    KeyEffect,
    apply_key_effect,
    background_plane_depth,
    camera_view_pose,
    dolly_camera,
    fit_display,
    fly_camera,
    fly_distance,
    frame_caption,
    image_plane_corners,
    image_plane_mesh,
    interpret_key,
    key_token,
    load_frame_bgr,
    load_overlay,
    look_distance,
    orbit_camera,
    pinhole_matrix,
    pose_from_model,
    roll_camera,
    project_points,
    render_overlay,
    starting_frame_index,
)


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
    frame_number: int = 1,
) -> None:
    """Open an interactive window. Requires a display.

    ``frame_number`` is 1-based, matching ``view --frame``.
    """
    output_dir = Path(output_dir)
    geometries = load_scene(
        output_dir,
        frustum_scale=frustum_scale,
        prefer_point_cloud=prefer_point_cloud,
    )
    payload = json.loads((output_dir / "cameras.json").read_text(encoding="utf-8"))
    overlay = load_overlay(output_dir, payload)
    index = starting_frame_index(frame_number, len(overlay.frames))
    if not _has_display():
        raise ReconstructionError(
            "No display is available, so the Open3D window cannot open.\n"
            "On Windows, run this from a normal desktop session:\n"
            f"  python -m video2mesh view {output_dir}\n"
            "The mesh and cameras.json can still be opened in MeshLab or Blender."
        )
    bounds = _bounds_from_scene(geometries, overlay.frames)
    scale = frustum_scale if frustum_scale > 0 else _auto_scale(bounds, overlay.frames)
    point_source = "sparse" if len(overlay.sparse_xyz) else "cloud"
    print(f"Viewing {output_dir}")
    print(_overlay_banner(overlay))
    print(VIEW_HELP)
    print()
    _ReconstructionWindow(
        output_dir,
        geometries,
        overlay,
        scale=scale,
        state=ViewState(frame_index=index, point_source=point_source),
    ).run()


def frustum_lines(frame: dict, *, scale: float, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """Camera pyramid from the center to the four corners of the image plane.

    The far face is the flat sensor rectangle at depth ``scale``, the same
    quad the frame photo is drawn on.
    """
    center = np.asarray(frame["camera_center_world"], dtype=np.float64)
    corners_world = image_plane_corners(frame, scale)
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


class _ReconstructionWindow:
    """3D scene plus a side panel of the current frame with points projected on it."""

    def __init__(
        self,
        output_dir: Path,
        geometries: list[o3d.geometry.Geometry],
        overlay: OverlayData,
        *,
        scale: float,
        state: ViewState,
    ) -> None:
        self.output_dir = output_dir
        self.geometries = geometries
        self.overlay = overlay
        self.scale = scale
        self.state = state
        self._panel_width = 480
        self._bgr_cache: dict[int, tuple[np.ndarray, bool]] = {}
        self._checkbox_guard = False

    def run(self) -> None:
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering

        self._gui = gui
        self._rendering = rendering
        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("video2mesh", 1440, 860)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background(np.array([0.07, 0.08, 0.09, 1.0], dtype=np.float32))
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.label = gui.Label(" ")
        self.label.text_color = gui.Color(1, 1, 1)
        self.show_image = gui.Checkbox("Show image")
        self.show_image.checked = self.state.show_photo
        self.show_image.tooltip = (
            "Draw this frame's photo on its image plane and look through the camera"
        )
        self.show_image.set_on_checked(self._on_show_image)
        self.image = gui.ImageWidget()
        self._lit = rendering.MaterialRecord()
        self._lit.shader = "defaultLit"
        self._points = rendering.MaterialRecord()
        self._points.shader = "defaultUnlit"
        self._points.point_size = 3.0
        self._lines = rendering.MaterialRecord()
        self._lines.shader = "unlitLine"
        self._lines.line_width = 2.0
        self._lines.base_color = [1.0, 1.0, 1.0, 1.0]
        self._photo_mat = rendering.MaterialRecord()
        # Open3D 0.20's defaultUnlitTransparency shader has no srgbColor uniform,
        # and the renderer always sets one, which aborts the frame.
        self._photo_mat.shader = "defaultUnlit"
        self._photo_mat.base_color = [1.0, 1.0, 1.0, 1.0]
        model = self.geometries[0]
        material = self._points if isinstance(model, o3d.geometry.PointCloud) else self._lit
        self.scene.scene.add_geometry("model", model, material)
        for index, frame in enumerate(self.overlay.frames):
            self._put_frustum(index, selected=index == self.state.frame_index)
        bounds = _bounds_from_scene(self.geometries, self.overlay.frames)
        center = bounds.mean(axis=0) if len(bounds) else np.zeros(3)
        if len(bounds):
            low = bounds.min(axis=0)
            high = np.maximum(bounds.max(axis=0), low + 1e-3)
            box = o3d.geometry.AxisAlignedBoundingBox(low, high)
        else:
            box = o3d.geometry.AxisAlignedBoundingBox(np.zeros(3), np.ones(3))
        self._reset_bounds = box
        self._reset_center = center.astype(np.float32)
        self.scene.setup_camera(60.0, box, self._reset_center)
        self.window.add_child(self.scene)
        self.window.add_child(self.show_image)
        self.window.add_child(self.label)
        self.window.add_child(self.image)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self._refresh_overlay()
        app.run()

    def _on_layout(self, _ctx) -> None:
        gui = self._gui
        rect = self.window.content_rect
        panel = max(280, min(520, rect.width // 3))
        em = int(self.window.theme.font_size)
        check_h = em + 12
        label_h = max(64, em * 3)
        x = rect.get_right() - panel
        self.scene.frame = gui.Rect(rect.x, rect.y, max(1, rect.width - panel), rect.height)
        self.show_image.frame = gui.Rect(x + 8, rect.y + 4, max(1, panel - 8), check_h)
        self.label.frame = gui.Rect(x + 8, rect.y + 4 + check_h, max(1, panel - 16), label_h)
        image_top = 4 + check_h + label_h
        self.image.frame = gui.Rect(x, rect.y + image_top, panel, max(1, rect.height - image_top))
        if abs(panel - self._panel_width) > 8:
            self._panel_width = panel
            self._refresh_overlay()

    def _on_key(self, event) -> bool:
        if event.type != self._gui.KeyEvent.Type.DOWN:
            return False
        effect = interpret_key(key_token(event.key))
        if effect is None:
            return False
        if effect.reset:
            self.scene.setup_camera(60.0, self._reset_bounds, self._reset_center)
        elif effect.zoom_factor is not None:
            eye, center, up = self._pose()
            eye = dolly_camera(eye, center, effect.zoom_factor)
            self.scene.look_at(center.astype(np.float32), eye.astype(np.float32), up.astype(np.float32))
        elif effect.yaw_deg or effect.pitch_deg:
            eye, center, up = self._pose()
            eye, up = orbit_camera(eye, center, up, effect.yaw_deg, effect.pitch_deg)
            self.scene.look_at(center.astype(np.float32), eye.astype(np.float32), up.astype(np.float32))
        elif effect.roll_deg:
            eye, center, up = self._pose()
            _eye, up = roll_camera(eye, center, up, effect.roll_deg)
            self.scene.look_at(center.astype(np.float32), eye.astype(np.float32), up.astype(np.float32))
        elif effect.fly_sign:
            eye, center, up = self._pose()
            extent = np.asarray(self._reset_bounds.get_extent(), dtype=np.float64)
            distance = fly_distance(float(np.linalg.norm(extent)), effect.fly_sign)
            eye, center = fly_camera(eye, center, distance)
            center32 = center.astype(np.float32)
            self.scene.look_at(center32, eye.astype(np.float32), up.astype(np.float32))
            # look_at leaves the orbit pivot where it was. Fly moves both.
            self.scene.center_of_rotation = center32
        overlay_changed = any(
            (
                effect.frame_delta,
                effect.toggle_photo,
                effect.toggle_points,
                effect.toggle_color,
                effect.toggle_source,
            )
        )
        if overlay_changed:
            previous = self.state.frame_index
            self.state = apply_key_effect(self.state, effect, len(self.overlay.frames))
            turned_on = effect.toggle_photo and self.state.show_photo
            frame_moved = self.state.frame_index != previous
            self._commit_overlay(previous, aim=self.state.show_photo and (turned_on or frame_moved))
        self.window.post_redraw()
        return True

    def _on_show_image(self, checked: bool) -> None:
        if self._checkbox_guard or checked == self.state.show_photo:
            return
        previous = self.state.frame_index
        self.state = apply_key_effect(
            self.state,
            KeyEffect(toggle_photo=True),
            len(self.overlay.frames),
        )
        self._commit_overlay(previous, aim=self.state.show_photo)
        self.window.post_redraw()

    def _commit_overlay(self, previous: int, *, aim: bool) -> None:
        if self.state.frame_index != previous:
            self._put_frustum(previous, selected=False)
            self._put_frustum(self.state.frame_index, selected=True)
        self._sync_show_image()
        self._refresh_overlay()
        if aim:
            self._look_through_current_frame()

    def _sync_show_image(self) -> None:
        self._checkbox_guard = True
        try:
            self.show_image.checked = self.state.show_photo
        finally:
            self._checkbox_guard = False

    def _look_through_current_frame(self) -> None:
        """Match the viewport to this frame, so the photo fills the camera frame."""
        frame = self.overlay.frames[self.state.frame_index]
        extent = float(np.linalg.norm(np.asarray(self._reset_bounds.get_extent(), dtype=np.float64)))
        minimum = max(extent * 0.35, self.scale * 4.0, 1e-2)
        target = np.asarray(self._reset_center, dtype=np.float64)
        distance = look_distance(frame, target, minimum)
        eye, look, up = camera_view_pose(frame, distance)
        look32 = look.astype(np.float32)
        self.scene.look_at(look32, eye.astype(np.float32), up.astype(np.float32))
        # look_at leaves the orbit pivot where it was.
        self.scene.center_of_rotation = look32
        near = min(max(self.scale * 0.02, 1e-4), self.scale * 0.25)
        plane = self._photo_depth(frame)
        far = max(extent * 20.0, distance * 8.0, plane * 3.0, self.scale * 20.0, near * 1000.0)
        matrix, width, height = pinhole_matrix(frame)
        # Intrinsics map the image rectangle onto the view, principal point included.
        self.scene.scene.camera.set_projection(matrix, near, far, width, height)

    def _pose(self):
        eye, _forward, up = pose_from_model(self.scene.scene.camera.get_model_matrix())
        center = np.asarray(self.scene.center_of_rotation, dtype=np.float64)
        return eye, center, up

    def _refresh_overlay(self) -> None:
        frame = self.overlay.frames[self.state.frame_index]
        bgr, found = self._frame_image(self.state.frame_index, frame)
        shown = fit_display(bgr, self._panel_width)
        xyz, rgb = self._active_points()
        uv, depth, valid = project_points(xyz, frame, (shown.shape[1], shown.shape[0]))
        picture = render_overlay(
            shown,
            uv,
            depth,
            valid,
            rgb,
            color_mode=self.state.color_mode,
            hide_points=not self.state.show_points,
        )
        self.image.update_image(o3d.geometry.Image(np.ascontiguousarray(picture)))
        visible = int(np.count_nonzero(valid)) if self.state.show_points else 0
        self.label.text = frame_caption(
            self.state,
            frame,
            len(self.overlay.frames),
            visible,
            image_found=found,
        )
        self._update_photo(frame, bgr)
        self.window.title = f"video2mesh  {self.state.frame_index + 1}/{len(self.overlay.frames)}  {frame.get('filename', '')}"

    def _frame_image(self, index: int, frame: dict):
        cached = self._bgr_cache.get(index)
        if cached is None:
            cached = load_frame_bgr(self.output_dir, frame)
            self._bgr_cache[index] = cached
        return cached

    def _active_points(self):
        if self.state.point_source == "cloud" and len(self.overlay.cloud_xyz):
            return self.overlay.cloud_xyz, self.overlay.cloud_rgb
        if self.state.point_source == "sparse" and len(self.overlay.sparse_xyz):
            return self.overlay.sparse_xyz, self.overlay.sparse_rgb
        if len(self.overlay.sparse_xyz):
            return self.overlay.sparse_xyz, self.overlay.sparse_rgb
        return self.overlay.cloud_xyz, self.overlay.cloud_rgb

    def _update_photo(self, frame: dict, bgr: np.ndarray) -> None:
        name = "frame-photo"
        scene = self.scene.scene
        if not self.state.show_photo:
            if scene.has_geometry(name):
                scene.show_geometry(name, False)
            return
        rgb = cv2_rgb(bgr)
        mesh = image_plane_mesh(frame, rgb, self._photo_depth(frame))
        if scene.has_geometry(name):
            scene.remove_geometry(name)
        self._photo_mat.albedo_img = o3d.geometry.Image(np.ascontiguousarray(rgb))
        scene.add_geometry(name, mesh, self._photo_mat)

    def _photo_depth(self, frame: dict) -> float:
        return background_plane_depth(frame, _box_corners(self._reset_bounds), self.scale)

    def _put_frustum(self, index: int, *, selected: bool) -> None:
        frame = self.overlay.frames[index]
        color = (1.0, 0.82, 0.15) if selected else _frustum_color(index, max(len(self.overlay.frames) - 1, 1))
        lines = frustum_lines(frame, scale=self.scale, color=color)
        name = f"frustum-{index}"
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)
        self.scene.scene.add_geometry(name, lines, self._lines)


def _box_corners(box: o3d.geometry.AxisAlignedBoundingBox) -> np.ndarray:
    low = np.asarray(box.get_min_bound(), dtype=np.float64)
    high = np.asarray(box.get_max_bound(), dtype=np.float64)
    xs, ys, zs = np.meshgrid(
        (low[0], high[0]),
        (low[1], high[1]),
        (low[2], high[2]),
        indexing="ij",
    )
    return np.stack((xs.ravel(), ys.ravel(), zs.ravel()), axis=1)


def cv2_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _overlay_banner(overlay: OverlayData) -> str:
    if overlay.sparse_dir is not None and len(overlay.sparse_xyz):
        where = overlay.sparse_dir.name
        return f"Overlay: {len(overlay.sparse_xyz)} sparse points from sparse/{where}."
    if len(overlay.cloud_xyz):
        return f"Overlay: sparse model has no points; using {len(overlay.cloud_xyz)} points from pointcloud.ply."
    return "Overlay: no sparse points and no pointcloud.ply. The side panel shows frames only."


def _bounds_from_scene(geometries: list[o3d.geometry.Geometry], frames: list[dict]) -> np.ndarray:
    chunks = []
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.PointCloud):
            points = np.asarray(geometry.points)
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            points = np.asarray(geometry.vertices)
        else:
            continue
        if len(points):
            chunks.append(points)
    if frames:
        chunks.append(np.array([frame["camera_center_world"] for frame in frames], dtype=np.float64))
    if not chunks:
        return np.zeros((1, 3))
    return np.vstack(chunks)
