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
    frustum_visible,
    image_plane_corners,
    undistort_bgr,
    image_plane_mesh,
    interpret_key,
    key_token,
    load_frame_bgr,
    load_overlay,
    look_distance,
    marker_axes,
    marker_mesh,
    orbit_camera,
    pick_ray,
    pinhole_matrix,
    world_to_camera_matrix,
    pose_from_model,
    roll_camera,
    project_points,
    render_overlay,
    starting_frame_index,
    SurfacePicker,
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
        self.show_features = False
        self.mesh_alpha = 0.35
        self._placing = False
        self._eat_mouse = False
        self._axis_guard = False
        self._position: np.ndarray | None = None
        self._marker_kind_shown = ""

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
        self.features = gui.Checkbox("Feature points")
        self.features.checked = False
        self.features.enabled = len(self.overlay.sparse_xyz) > 0
        self.features.tooltip = "Show the COLMAP sparse points so they can be checked against the photo"
        self.features.set_on_checked(self._on_features)
        self.alpha_label = gui.Label(self._alpha_caption())
        self.alpha_label.text_color = gui.Color(1, 1, 1)
        self.alpha_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.alpha_slider.set_limits(0.0, 1.0)
        self.alpha_slider.double_value = self.mesh_alpha
        self.alpha_slider.set_on_value_changed(self._on_mesh_alpha)
        self.shape = gui.Combobox()
        self.shape.add_item("Cube")
        self.shape.add_item("Sphere")
        self.shape.selected_index = 0
        self.shape.tooltip = "Shape to attach to the mesh"
        self.shape.set_on_selection_changed(self._on_shape)
        self.place = gui.Button("Place")
        self.place.tooltip = "Click Place, then a point on the mesh. Place again to attach somewhere else."
        self.place.set_on_clicked(self._on_place)
        self.marker_label = gui.Label("Place, then click the mesh")
        self.marker_label.text_color = gui.Color(1, 1, 1)
        self.x_label, self.x_edit = self._axis_field("X", gui.Color(1.0, 0.35, 0.35), self._on_axis_x)
        self.y_label, self.y_edit = self._axis_field("Y", gui.Color(0.35, 0.9, 0.35), self._on_axis_y)
        self.z_label, self.z_edit = self._axis_field("Z", gui.Color(0.45, 0.55, 1.0), self._on_axis_z)
        self.image = gui.ImageWidget()
        self._lit = rendering.MaterialRecord()
        self._lit.shader = "defaultLit"
        # Lit transparency. defaultUnlitTransparency has no srgbColor uniform,
        # and Open3D 0.20 aborts when it sets one.
        self._lit_ghost = rendering.MaterialRecord()
        self._lit_ghost.shader = "defaultLitTransparency"
        self._lit_ghost.base_color = [1.0, 1.0, 1.0, self.mesh_alpha]
        self._lit_ghost.base_roughness = 0.7
        self._lit_ghost.base_metallic = 0.0
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
        self._feature_mat = rendering.MaterialRecord()
        self._feature_mat.shader = "defaultUnlit"
        self._feature_mat.point_size = 5.0
        self._feature_mat.base_color = [1.0, 1.0, 1.0, 1.0]
        self._marker_mat = rendering.MaterialRecord()
        self._marker_mat.shader = "defaultLit"
        self._marker_mat.base_color = [0.93, 0.45, 0.12, 1.0]
        model = self.geometries[0]
        material = self._points if isinstance(model, o3d.geometry.PointCloud) else self._lit
        self.scene.scene.add_geometry("model", model, material)
        self._add_feature_points()
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
        extent = float(np.linalg.norm(np.asarray(box.get_extent(), dtype=np.float64)))
        self._marker_size = max(extent * 0.03, self.scale * 0.35, 1e-3)
        self._picker = SurfacePicker(
            model,
            snap_distance=max(self._marker_size * 1.5, extent * 0.02, 1e-4),
        )
        low = np.asarray(box.get_min_bound(), dtype=np.float64)
        high = np.asarray(box.get_max_bound(), dtype=np.float64)
        pad = max(extent, 1.0)
        for edit in (self.x_edit, self.y_edit, self.z_edit):
            edit.set_limits(float(low.min() - pad), float(high.max() + pad))
        self.scene.set_on_mouse(self._on_mouse)
        self.window.add_child(self.scene)
        self.window.add_child(self.show_image)
        self.window.add_child(self.features)
        self.window.add_child(self.alpha_label)
        self.window.add_child(self.alpha_slider)
        self.window.add_child(self.shape)
        self.window.add_child(self.place)
        self.window.add_child(self.marker_label)
        self.window.add_child(self.x_label)
        self.window.add_child(self.x_edit)
        self.window.add_child(self.y_label)
        self.window.add_child(self.y_edit)
        self.window.add_child(self.z_label)
        self.window.add_child(self.z_edit)
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
        inner_w = max(1, panel - 16)
        left = x + 8
        self.scene.frame = gui.Rect(rect.x, rect.y, max(1, rect.width - panel), rect.height)
        rows = (
            ((self.show_image,), "full"),
            ((self.features,), "full"),
            ((self.alpha_label,), "full"),
            ((self.alpha_slider,), "full"),
            ((self.shape, self.place), "split"),
            ((self.marker_label,), "full"),
            ((self.x_label, self.x_edit), "axis"),
            ((self.y_label, self.y_edit), "axis"),
            ((self.z_label, self.z_edit), "axis"),
        )
        top = rect.y + 4
        for widgets, kind in rows:
            self._place_row(widgets, kind, left, top, inner_w, check_h, em)
            top += check_h
        text_top = top - rect.y
        self.label.frame = gui.Rect(left, rect.y + text_top, inner_w, label_h)
        image_top = text_top + label_h
        self.image.frame = gui.Rect(x, rect.y + image_top, panel, max(1, rect.height - image_top))
        if abs(panel - self._panel_width) > 8:
            self._panel_width = panel
            self._refresh_overlay()

    def _place_row(self, widgets, kind: str, left: int, top: int, inner_w: int, height: int, em: int) -> None:
        gui = self._gui
        if kind == "split":
            gap = 8
            half = max(1, (inner_w - gap) // 2)
            widgets[0].frame = gui.Rect(left, top, half, height)
            widgets[1].frame = gui.Rect(left + half + gap, top, max(1, inner_w - half - gap), height)
            return
        if kind == "axis":
            label_w = max(28, em * 2)
            widgets[0].frame = gui.Rect(left, top, label_w, height)
            widgets[1].frame = gui.Rect(left + label_w, top, max(1, inner_w - label_w), height)
            return
        widgets[0].frame = gui.Rect(left, top, inner_w, height)

    def _axis_field(self, caption: str, color, callback):
        label = self._gui.Label(caption)
        label.text_color = color
        edit = self._gui.NumberEdit(self._gui.NumberEdit.DOUBLE)
        edit.decimal_precision = 4
        edit.double_value = 0.0
        edit.enabled = False
        edit.tooltip = f"Slide the shape along {caption}"
        edit.set_on_value_changed(callback)
        return label, edit

    def _on_mouse(self, event):
        gui = self._gui
        if self._eat_mouse:
            if event.type == gui.MouseEvent.Type.BUTTON_UP:
                self._eat_mouse = False
            return gui.Widget.EventCallbackResult.HANDLED
        if not self._placing:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type != gui.MouseEvent.Type.BUTTON_DOWN or not event.is_button_down(gui.MouseButton.LEFT):
            return gui.Widget.EventCallbackResult.IGNORED
        hit = self._pick_surface(event.x, event.y)
        if hit is None:
            self.marker_label.text = "No mesh there. Click the surface."
            return gui.Widget.EventCallbackResult.IGNORED
        self._eat_mouse = True
        self._placing = False
        self.place.text = "Place"
        self._set_position(hit)
        self.marker_label.text = "Attached. Edit X, Y, or Z to slide it."
        self.window.post_redraw()
        return gui.Widget.EventCallbackResult.HANDLED

    def _pick_surface(self, x: float, y: float) -> np.ndarray | None:
        frame = self.scene.frame
        width = float(frame.width)
        height = float(frame.height)
        if width < 1.0 or height < 1.0:
            return None
        px, py = float(x), float(y)
        if not (0.0 <= px < width and 0.0 <= py < height):
            px -= float(frame.x)
            py -= float(frame.y)
        if not (0.0 <= px < width and 0.0 <= py < height):
            return None
        camera = self.scene.scene.camera
        eye, _forward, _up = pose_from_model(camera.get_model_matrix())
        # Depth 1 is the far plane, which unprojects to infinity and misses.
        aim = np.asarray(camera.unproject(px, py, 0.0, width, height), dtype=np.float64).reshape(3)
        ray = pick_ray(eye, aim)
        if ray is None:
            return None
        origin, direction = ray
        return self._picker.hit(origin, direction)

    def _on_place(self) -> None:
        self._placing = not self._placing
        if self._placing:
            self.place.text = "Click mesh"
            self.marker_label.text = "Click a point on the mesh"
        else:
            self.place.text = "Place"
            self.marker_label.text = (
                "Place, then click the mesh" if self._position is None else "Attached. Edit X, Y, or Z to slide it."
            )

    def _on_shape(self, _text: str, _index: int) -> None:
        if self._position is None:
            return
        self._rebuild_marker()
        self.window.post_redraw()

    def _on_axis_x(self, value: float) -> None:
        self._on_axis(0, value)

    def _on_axis_y(self, value: float) -> None:
        self._on_axis(1, value)

    def _on_axis_z(self, value: float) -> None:
        self._on_axis(2, value)

    def _on_axis(self, index: int, value: float) -> None:
        if self._axis_guard or self._position is None:
            return
        self._position[index] = float(value)
        self._apply_marker_transform()
        self.window.post_redraw()

    def _set_position(self, position: np.ndarray) -> None:
        self._position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self._axis_guard = True
        try:
            self.x_edit.double_value = float(self._position[0])
            self.y_edit.double_value = float(self._position[1])
            self.z_edit.double_value = float(self._position[2])
        finally:
            self._axis_guard = False
        for edit in (self.x_edit, self.y_edit, self.z_edit):
            edit.enabled = True
        self._ensure_marker()
        self._apply_marker_transform()

    def _ensure_marker(self) -> None:
        kind = "sphere" if self.shape.selected_text == "Sphere" else "cube"
        if self.scene.scene.has_geometry("marker") and kind == self._marker_kind_shown:
            return
        self._rebuild_marker()

    def _rebuild_marker(self) -> None:
        scene = self.scene.scene
        for name in ("marker", "marker-axes"):
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        kind = "sphere" if self.shape.selected_text == "Sphere" else "cube"
        self._marker_kind_shown = kind
        scene.add_geometry("marker", marker_mesh(kind, self._marker_size), self._marker_mat)
        scene.add_geometry("marker-axes", marker_axes(self._marker_size * 2.5), self._lines)
        self._apply_marker_transform()

    def _apply_marker_transform(self) -> None:
        if self._position is None:
            return
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = self._position
        scene = self.scene.scene
        for name in ("marker", "marker-axes"):
            if scene.has_geometry(name):
                scene.set_geometry_transform(name, matrix)

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

    def _on_features(self, checked: bool) -> None:
        self.show_features = bool(checked)
        if self.scene.scene.has_geometry("features"):
            self.scene.scene.show_geometry("features", self.show_features)
        self.window.post_redraw()

    def _on_mesh_alpha(self, value: float) -> None:
        self.mesh_alpha = min(1.0, max(0.0, float(value)))
        self._lit_ghost.base_color = [1.0, 1.0, 1.0, self.mesh_alpha]
        self.alpha_label.text = self._alpha_caption()
        if self.state.show_photo:
            self._apply_model_opacity()
        self.window.post_redraw()

    def _alpha_caption(self) -> str:
        return f"Mesh alpha  {self.mesh_alpha:.2f}"

    def _commit_overlay(self, previous: int, *, aim: bool) -> None:
        if self.state.frame_index != previous:
            self._put_frustum(previous, selected=False)
            self._put_frustum(self.state.frame_index, selected=True)
        self._sync_frustum_visibility()
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
        _eye, look, _up = camera_view_pose(frame, distance)
        near = min(max(self.scale * 0.02, 1e-4), self.scale * 0.25)
        plane = self._photo_depth(frame)
        far = max(extent * 20.0, distance * 8.0, plane * 3.0, self.scale * 20.0, near * 1000.0)
        matrix, width, height = pinhole_matrix(frame)
        # setup_camera keeps this pinhole camera across resizes. look_at would
        # replace it with a generic field of view, and the points would miss
        # the photo. The extrinsic is the COLMAP world-to-camera matrix.
        self.scene.setup_camera(
            matrix,
            world_to_camera_matrix(frame),
            int(width),
            int(height),
            self._reset_bounds,
        )
        self.scene.scene.camera.set_projection(matrix, near, far, width, height)
        self.scene.center_of_rotation = look.astype(np.float32)

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
            self._apply_model_opacity()
            if scene.has_geometry(name):
                scene.show_geometry(name, False)
            return
        self._apply_model_opacity()
        rgb = cv2_rgb(undistort_bgr(bgr, frame))
        mesh = image_plane_mesh(frame, rgb, self._photo_depth(frame))
        if scene.has_geometry(name):
            scene.remove_geometry(name)
        self._photo_mat.albedo_img = o3d.geometry.Image(np.ascontiguousarray(rgb))
        scene.add_geometry(name, mesh, self._photo_mat)

    def _apply_model_opacity(self) -> None:
        """Ghost the model while the photo is on so the picture shows through."""
        model = self.geometries[0]
        if self.state.show_photo:
            material = self._lit_ghost
        elif isinstance(model, o3d.geometry.PointCloud):
            material = self._points
        else:
            material = self._lit
        if self.scene.scene.has_geometry("model"):
            self.scene.scene.modify_geometry_material("model", material)

    def _add_feature_points(self) -> None:
        xyz = self.overlay.sparse_xyz
        if len(xyz) == 0:
            return
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        rgb = np.asarray(self.overlay.sparse_rgb, dtype=np.float64)
        if len(rgb) == len(xyz):
            cloud.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))
        self.scene.scene.add_geometry("features", cloud, self._feature_mat)
        self.scene.scene.show_geometry("features", False)

    def _photo_depth(self, frame: dict) -> float:
        return background_plane_depth(frame, _box_corners(self._reset_bounds), self.scale)

    def _sync_frustum_visibility(self) -> None:
        """In image mode, keep only the camera whose photo is showing."""
        scene = self.scene.scene
        current = self.state.frame_index
        image_mode = self.state.show_photo
        for index in range(len(self.overlay.frames)):
            name = f"frustum-{index}"
            if scene.has_geometry(name):
                scene.show_geometry(name, frustum_visible(index, current, image_mode))

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
