"""Projection, orbit math, and key map for the reconstruction viewer.

Nothing in this module opens a window. The Open3D window lives in viewer.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from video2mesh.colmap_io import list_model_dirs, read_points3d_binary, read_points3d_text
from video2mesh.errors import ReconstructionError

ORBIT_STEP_DEG = 4.0
ZOOM_IN = 0.85
ZOOM_OUT = 1.0 / ZOOM_IN
# Fraction of the scene bounding-box diagonal moved by one Up/Down press.
FLY_STEP_FRACTION = 0.04
OVERLAY_POINT_LIMIT = 12000

VIEW_HELP = """\
Keys (held keys repeat):
  W / S       pitch the view up / down around the look-at point
  A / D       yaw the view left / right around the look-at point
  Z / C       roll clockwise / counter-clockwise around the view direction
  Q / E       zoom in / out (eye moves, look-at stays)
  Up / Down   dolly forward / backward along the view (eye and look-at move together)
  R           reset the view
  Left / Right, N / P, or [ / ]    previous / next frame
  F           show this frame's photo as a plane in its frustum
  H           hide or show the projected points
  V           color those points by RGB or by depth
  G           use sparse points or pointcloud.ply
Mouse: left-drag orbits, right-drag pans, wheel zooms.

The side panel shows the original frame from out/frames. Points are projected
with that frame's pose and intrinsics, including lens distortion. Images in
out/dense/images are undistorted, so the overlay does not use them.
""".rstrip()


@dataclass(frozen=True)
class KeyEffect:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    zoom_factor: float | None = None
    fly_sign: int = 0
    reset: bool = False
    frame_delta: int = 0
    toggle_photo: bool = False
    toggle_points: bool = False
    toggle_color: bool = False
    toggle_source: bool = False


@dataclass(frozen=True)
class ViewState:
    frame_index: int = 0
    show_photo: bool = False
    show_points: bool = True
    color_mode: str = "rgb"
    point_source: str = "sparse"


@dataclass
class OverlayData:
    frames: list[dict]
    sparse_dir: Path | None
    sparse_xyz: np.ndarray
    sparse_rgb: np.ndarray
    cloud_xyz: np.ndarray
    cloud_rgb: np.ndarray


# Open3D gui.KeyName integer values that are outside ASCII.
_SPECIAL_KEY_CODES = {263: "left", 264: "right", 265: "up", 266: "down"}


def key_token(key) -> str | None:
    """Normalize an Open3D key code or a letter to a lowercase name.

    ``KeyEvent.key`` is an int. Letters match ``KeyName`` (``W`` is 119, ``w``).
    Arrows are the codes above.
    """
    if isinstance(key, int):
        if key in _SPECIAL_KEY_CODES:
            return _SPECIAL_KEY_CODES[key]
        if 32 <= key < 127:
            return chr(key).lower()
    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    return None


def interpret_key(token: str | None) -> KeyEffect | None:
    """Map one key name to a view change. Unknown keys return None."""
    if not token:
        return None
    aliases = {
        "[": "left_bracket",
        "]": "right_bracket",
    }
    name = aliases.get(token.lower(), token.lower())
    step = ORBIT_STEP_DEG
    table = {
        "w": KeyEffect(pitch_deg=step),
        "s": KeyEffect(pitch_deg=-step),
        "a": KeyEffect(yaw_deg=-step),
        "d": KeyEffect(yaw_deg=step),
        "z": KeyEffect(roll_deg=step),
        "c": KeyEffect(roll_deg=-step),
        "q": KeyEffect(zoom_factor=ZOOM_IN),
        "e": KeyEffect(zoom_factor=ZOOM_OUT),
        "r": KeyEffect(reset=True),
        "up": KeyEffect(fly_sign=1),
        "down": KeyEffect(fly_sign=-1),
        "left": KeyEffect(frame_delta=-1),
        "right": KeyEffect(frame_delta=1),
        "n": KeyEffect(frame_delta=1),
        "p": KeyEffect(frame_delta=-1),
        "left_bracket": KeyEffect(frame_delta=-1),
        "right_bracket": KeyEffect(frame_delta=1),
        "f": KeyEffect(toggle_photo=True),
        "h": KeyEffect(toggle_points=True),
        "v": KeyEffect(toggle_color=True),
        "g": KeyEffect(toggle_source=True),
    }
    return table.get(name)


def apply_key_effect(state: ViewState, effect: KeyEffect, num_frames: int) -> ViewState:
    """Return the frame/overlay state after one key. Orbit and zoom are separate."""
    index = state.frame_index
    if effect.frame_delta and num_frames > 0:
        index = (index + effect.frame_delta) % num_frames
    color_mode = state.color_mode
    if effect.toggle_color:
        color_mode = "depth" if color_mode == "rgb" else "rgb"
    point_source = state.point_source
    if effect.toggle_source:
        point_source = "cloud" if point_source == "sparse" else "sparse"
    return replace(
        state,
        frame_index=index,
        show_photo=not state.show_photo if effect.toggle_photo else state.show_photo,
        show_points=not state.show_points if effect.toggle_points else state.show_points,
        color_mode=color_mode,
        point_source=point_source,
    )


def starting_frame_index(frame_number: int, count: int) -> int:
    """Convert a 1-based --frame value to an index."""
    if count <= 0:
        raise ReconstructionError("cameras.json has no registered frames to view.")
    if frame_number < 1 or frame_number > count:
        raise ReconstructionError(
            f"--frame {frame_number} is outside 1..{count} ({count} registered frames)."
        )
    return frame_number - 1


def rotate_about(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Right-hand rotation of a vector around an axis."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0 or degrees == 0:
        return np.asarray(vector, dtype=np.float64)
    axis = axis / norm
    theta = np.deg2rad(degrees)
    cos = np.cos(theta)
    sin = np.sin(theta)
    vector = np.asarray(vector, dtype=np.float64)
    return vector * cos + np.cross(axis, vector) * sin + axis * np.dot(axis, vector) * (1.0 - cos)


def pose_from_model(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eye, forward, and up from a Filament camera model matrix.

    Open3D's ``look_at`` stores a camera-to-world matrix: column 0 is camera
    right, column 1 is up, column 2 points from the look-at point back toward
    the eye, and column 3 is the eye position.
    """
    model = np.asarray(matrix, dtype=np.float64)
    eye = model[:3, 3].copy()
    up = model[:3, 1].copy()
    forward = -model[:3, 2]
    return eye, forward, up


def orbit_camera(
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Orbit the eye around ``center``.

    Positive yaw moves the camera to its right. Positive pitch raises it.
    Pitch stops before the eye crosses the pole so the view does not flip.
    """
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up_norm = np.linalg.norm(up)
    if up_norm == 0:
        return eye, up
    up = up / up_norm
    offset = eye - center
    dist = np.linalg.norm(offset)
    if dist < 1e-8:
        return eye, up

    def _basis(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        look = -vector
        look_norm = np.linalg.norm(look)
        if look_norm < 1e-8:
            return None
        look = look / look_norm
        right = np.cross(look, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            return None
        right = right / right_norm
        new_up = np.cross(right, look)
        new_up_norm = np.linalg.norm(new_up)
        if new_up_norm < 1e-8:
            return None
        return right, new_up / new_up_norm

    if yaw_deg:
        offset = rotate_about(offset, up, yaw_deg)
    if pitch_deg:
        basis = _basis(offset)
        if basis is not None:
            right, _current_up = basis
            pitched = rotate_about(offset, right, -pitch_deg)
            pitched_dir = pitched / max(np.linalg.norm(pitched), 1e-8)
            if abs(float(np.dot(pitched_dir, up))) <= 0.995:
                offset = pitched
    rebuilt = _basis(offset)
    if rebuilt is None:
        return center + offset, up
    _right, new_up = rebuilt
    return center + offset, new_up


def dolly_camera(eye: np.ndarray, center: np.ndarray, factor: float) -> np.ndarray:
    """Move the eye toward the center. ``factor`` < 1 zooms in.

    Q/E use this. Up/Down use :func:`fly_camera`, which translates the eye
    and the look-at point together.
    """
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return center + (eye - center) * float(factor)


def roll_camera(
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    roll_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the camera around its view direction. Eye and look-at stay put.

    Positive ``roll_deg`` is a right-hand rotation of ``up`` around
    ``center - eye``, so the up vector tilts toward camera-right. Looking
    along the view, that is clockwise. The returned up is unit length and
    perpendicular to the view direction.
    """
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = center - eye
    rolled = rotate_about(up, forward, roll_deg)
    forward_norm = np.linalg.norm(forward)
    if forward_norm > 1e-8:
        forward_hat = forward / forward_norm
        rolled = rolled - forward_hat * float(np.dot(rolled, forward_hat))
    rolled_norm = np.linalg.norm(rolled)
    if rolled_norm < 1e-8:
        up_norm = np.linalg.norm(up)
        if up_norm == 0:
            return eye, up
        return eye, up / up_norm
    return eye, rolled / rolled_norm


def fly_distance(scene_diagonal: float, sign: int) -> float:
    """World-unit step for one Up/Down press. ``sign`` is +1 forward."""
    return float(sign) * FLY_STEP_FRACTION * float(scene_diagonal)


def fly_camera(
    eye: np.ndarray,
    center: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the eye and the look-at point together along the view.

    Positive ``distance`` moves toward the look-at point (forward). The
    distance between them does not change. A zero-length view is left as-is.
    """
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    direction = center - eye
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return eye.copy(), center.copy()
    step = direction / norm * float(distance)
    return eye + step, center + step


def project_points(
    points: np.ndarray,
    frame: dict,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixels.

    Returns ``uv`` (N, 2), camera-frame depth, and a mask of points that are
    in front of the camera and inside the image. Intrinsics are scaled when
    ``image_size`` differs from the frame the camera was calibrated on.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        empty_uv = np.zeros((0, 2), dtype=np.float64)
        empty = np.zeros((0,), dtype=np.float64)
        return empty_uv, empty, np.zeros((0,), dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ReconstructionError(f"Expected Nx3 points, got shape {points.shape}.")
    rotation, center = _camera_pose(frame)
    camera_points = (rotation @ (points - center).T).T
    depth = camera_points[:, 2]
    valid = np.isfinite(depth) & (depth > 1e-6)
    safe_z = np.where(valid, depth, 1.0)
    xn = camera_points[:, 0] / safe_z
    yn = camera_points[:, 1] / safe_z
    xd, yd = _distort(xn, yn, str(frame.get("camera_model", "PINHOLE")), frame.get("intrinsics", {}))
    fx, fy, cx, cy, width, height = _scaled_intrinsics(frame, image_size)
    u = fx * xd + cx
    v = fy * yd + cy
    inside = valid & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    return np.stack([u, v], axis=1), depth, inside


def image_plane_corners(frame: dict, depth: float) -> np.ndarray:
    """World positions of the image's four corners on a plane at ``depth``."""
    fx, fy, cx, cy, _width, _height = _scaled_intrinsics(frame, None)
    width = float(frame["width"])
    height = float(frame["height"])
    pixels = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    )
    rotation, center = _camera_pose(frame)
    corners = []
    for u, v in pixels:
        camera_point = np.array([(u - cx) / fx * depth, (v - cy) / fy * depth, depth])
        corners.append(rotation.T @ camera_point + center)
    return np.stack(corners)


def image_plane_mesh(frame: dict, rgb: np.ndarray, depth: float) -> o3d.geometry.TriangleMesh:
    """Textured quad, drawn from both sides, sitting on the image plane."""
    corners = image_plane_corners(frame, depth)
    vertices = o3d.utility.Vector3dVector(corners)
    # Two windings so the photo is visible from the camera and from behind.
    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = vertices
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    tl, tr, br, bl = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    uvs = [tl, tr, br, tl, br, bl, tl, br, tr, tl, bl, br]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(np.asarray(uvs, dtype=np.float64))
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0, 0, 0])
    texture = np.ascontiguousarray(_fit_rgb(rgb, 1024))
    mesh.textures = [o3d.geometry.Image(texture)]
    return mesh


def render_overlay(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    rgb: np.ndarray,
    *,
    color_mode: str = "rgb",
    hide_points: bool = False,
) -> np.ndarray:
    """Return an RGB image with the visible points painted on top."""
    canvas = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if hide_points or valid.size == 0 or not np.any(valid):
        return canvas
    chosen = np.flatnonzero(valid)
    # Far points first so a nearer point keeps the pixel.
    chosen = chosen[np.argsort(-depth[chosen])]
    colors = _point_colors(rgb[chosen], depth[chosen], color_mode)
    height, width = canvas.shape[:2]
    uu = np.clip(np.rint(uv[chosen, 0]).astype(np.int32), 0, width - 1)
    vv = np.clip(np.rint(uv[chosen, 1]).astype(np.int32), 0, height - 1)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            x = np.clip(uu + du, 0, width - 1)
            y = np.clip(vv + dv, 0, height - 1)
            canvas[y, x] = colors
    return canvas


def frame_caption(
    state: ViewState,
    frame: dict,
    num_frames: int,
    visible: int,
    *,
    image_found: bool,
) -> str:
    name = str(frame.get("filename", "frame"))
    if not state.show_points:
        points = "points hidden"
    else:
        source = "sparse points" if state.point_source == "sparse" else "pointcloud.ply"
        points = f"{visible} points  {state.color_mode}  {source}"
    photo = "photo on" if state.show_photo else "photo off"
    missing = "" if image_found else "  (frame image missing)"
    note = _model_note(str(frame.get("camera_model", "")))
    extra = f"  {note}" if note else ""
    return f"{state.frame_index + 1}/{num_frames}  {name}{missing}\n{points}   {photo}{extra}"


def load_overlay(output_dir: Path, payload: dict | None = None) -> OverlayData:
    """Load poses from cameras.json and points from the matching sparse model.

    ``cameras.json`` may store an absolute path from the machine that ran
    the reconstruction. ``<output>/sparse/<folder name>`` is used when that
    folder exists, so a moved output directory still opens the right model.
    The stored absolute path is the fallback. Extra COLMAP 4 files such as
    ``frames.bin`` and ``rigs.bin`` are ignored.
    """
    output_dir = Path(output_dir)
    if payload is None:
        cameras_path = output_dir / "cameras.json"
        if not cameras_path.is_file():
            raise ReconstructionError(
                f"No cameras.json in {output_dir}. Run video2mesh reconstruct first."
            )
        import json

        payload = json.loads(cameras_path.read_text(encoding="utf-8"))
    frames = list(payload.get("frames") or [])
    if not frames:
        raise ReconstructionError("cameras.json has no registered frames to view.")
    sparse_dir = resolve_sparse_dir(output_dir, payload.get("sparse_model"))
    sparse_xyz, sparse_rgb = _load_sparse_points(sparse_dir)
    cloud_xyz, cloud_rgb = _load_ply(output_dir / "pointcloud.ply")
    sparse_xyz, sparse_rgb = subsample_points(sparse_xyz, sparse_rgb, OVERLAY_POINT_LIMIT)
    cloud_xyz, cloud_rgb = subsample_points(cloud_xyz, cloud_rgb, OVERLAY_POINT_LIMIT)
    return OverlayData(frames, sparse_dir, sparse_xyz, sparse_rgb, cloud_xyz, cloud_rgb)


def resolve_sparse_dir(output_dir: Path, sparse_model: str | None) -> Path | None:
    """Find the sparse model named in cameras.json.

    A moved reconstruction still has ``<output>/sparse/<folder name>``. That
    local folder wins over the absolute path stored in cameras.json, which is
    only used when the local folder is not a model. The folder name is the
    last path segment, so a Windows path still matches on any platform.
    """
    output_dir = Path(output_dir)
    named = _folder_name(sparse_model) if sparse_model else ""
    if named:
        local = output_dir / "sparse" / named
        if _is_model_dir(local):
            return local
    if sparse_model:
        raw = Path(sparse_model)
        for candidate in (raw, output_dir / raw):
            if _is_model_dir(candidate):
                return candidate
    models = list_model_dirs(output_dir / "sparse")
    if named:
        for model in models:
            if model.name == named:
                return model
    if not models:
        return None
    return max(models, key=_points_file_size)


def subsample_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(xyz)
    if count == 0 or count <= limit or limit <= 0:
        return xyz, rgb
    step = int(np.ceil(count / float(limit)))
    return xyz[::step], rgb[::step]


def load_frame_bgr(output_dir: Path, frame: dict, max_side: int = 1280) -> tuple[np.ndarray, bool]:
    """Read ``out/frames/<filename>``. A missing file becomes a blank image."""
    filename = str(frame.get("filename", ""))
    path = output_dir / "frames" / Path(filename).name
    image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
    found = image is not None
    if image is None:
        width = max(int(frame.get("width", 64)), 1)
        height = max(int(frame.get("height", 48)), 1)
        image = np.full((height, width, 3), 32, dtype=np.uint8)
        cv2.putText(
            image,
            "missing " + Path(filename).name,
            (8, min(28, height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image, found


def fit_display(image_bgr: np.ndarray, panel_width: int) -> np.ndarray:
    """Scale a frame so it fits the side panel without changing the aspect ratio."""
    height, width = image_bgr.shape[:2]
    if panel_width <= 0 or width <= panel_width:
        return image_bgr
    scale = panel_width / float(width)
    return cv2.resize(
        image_bgr,
        (panel_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _camera_pose(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(frame.get("world_to_camera", []), dtype=np.float64)
    if matrix.shape == (3, 4):
        rotation = matrix[:, :3]
        translation = matrix[:, 3]
    else:
        from video2mesh.colmap_io import qvec_to_rotmat

        rotation = qvec_to_rotmat(np.asarray(frame["rotation_quaternion_wxyz"], dtype=np.float64))
        translation = np.asarray(frame["translation"], dtype=np.float64)
    if "camera_center_world" in frame:
        center = np.asarray(frame["camera_center_world"], dtype=np.float64)
    else:
        center = -rotation.T @ translation
    return rotation, center


def _scaled_intrinsics(frame: dict, image_size: tuple[int, int] | None):
    intr = frame.get("intrinsics") or {}
    fx = float(intr.get("fx", intr.get("f")))
    fy = float(intr.get("fy", fx))
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    width = float(frame["width"])
    height = float(frame["height"])
    if image_size is not None and width > 0 and height > 0:
        image_width, image_height = image_size
        if image_width != width or image_height != height:
            sx = image_width / width
            sy = image_height / height
            fx *= sx
            cx *= sx
            fy *= sy
            cy *= sy
            width = float(image_width)
            height = float(image_height)
    return fx, fy, cx, cy, width, height


def _distort(xn: np.ndarray, yn: np.ndarray, model: str, intr: dict):
    model = model.upper()
    r2 = xn * xn + yn * yn
    if model in {"", "SIMPLE_PINHOLE", "PINHOLE"}:
        return xn, yn
    if model == "SIMPLE_RADIAL":
        factor = 1.0 + float(intr.get("k1", 0.0)) * r2
        return xn * factor, yn * factor
    if model == "RADIAL":
        r4 = r2 * r2
        factor = 1.0 + float(intr.get("k1", 0.0)) * r2 + float(intr.get("k2", 0.0)) * r4
        return xn * factor, yn * factor
    if model in {"OPENCV", "FULL_OPENCV"}:
        r4 = r2 * r2
        r6 = r4 * r2
        k1 = float(intr.get("k1", 0.0))
        k2 = float(intr.get("k2", 0.0))
        p1 = float(intr.get("p1", 0.0))
        p2 = float(intr.get("p2", 0.0))
        radial = 1.0 + k1 * r2 + k2 * r4
        if model == "FULL_OPENCV":
            k3 = float(intr.get("k3", 0.0))
            numerator = radial + k3 * r6
            denominator = (
                1.0
                + float(intr.get("k4", 0.0)) * r2
                + float(intr.get("k5", 0.0)) * r4
                + float(intr.get("k6", 0.0)) * r6
            )
            radial = np.divide(numerator, denominator, out=np.ones_like(r2), where=np.abs(denominator) > 1e-8)
        x = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        y = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        return x, y
    # Fisheye and FOV are not the plumb-bob model. The pinhole term still
    # lands nearby; the caption says the distortion terms were not applied.
    return xn, yn


def _point_colors(rgb: np.ndarray, depth: np.ndarray, color_mode: str) -> np.ndarray:
    if color_mode == "depth":
        return _depth_colors(depth)
    colors = np.asarray(rgb, dtype=np.float64)
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _depth_colors(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    low = float(np.percentile(depth, 5))
    high = float(np.percentile(depth, 95))
    if high - low < 1e-6:
        span = np.zeros_like(depth)
    else:
        span = np.clip((depth - low) / (high - low), 0.0, 1.0)
    near = np.array([255.0, 210.0, 40.0])
    far = np.array([40.0, 90.0, 230.0])
    mixed = near[None, :] * (1.0 - span[:, None]) + far[None, :] * span[:, None]
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _fit_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image * 255.0 if image.max() <= 1 else image), 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _model_note(model: str) -> str:
    upper = model.upper()
    if "FISHEYE" in upper or upper == "FOV":
        return "pinhole approximation"
    return ""


def _folder_name(sparse_model: str) -> str:
    text = str(sparse_model).replace("\\", "/").rstrip("/")
    if not text:
        return ""
    return text.split("/")[-1]


def _is_model_dir(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and (
        (path / "cameras.bin").is_file() or (path / "cameras.txt").is_file()
    )


def _points_file_size(model: Path) -> int:
    total = 0
    for name in ("points3D.bin", "points3D.txt"):
        path = model / name
        if path.is_file():
            total += path.stat().st_size
    return total


def _load_sparse_points(model_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if model_dir is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    text_path = model_dir / "points3D.txt"
    binary_path = model_dir / "points3D.bin"
    if text_path.is_file():
        points = read_points3d_text(text_path)
    elif binary_path.is_file():
        points = read_points3d_binary(binary_path)
    else:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xyz = np.stack([point.xyz for point in points]).astype(np.float64)
    rgb = np.stack([point.rgb for point in points]).astype(np.float64) / 255.0
    return xyz, rgb


def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.zeros((0, 3)), np.zeros((0, 3))
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float64)
    if len(xyz) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if cloud.has_colors():
        rgb = np.asarray(cloud.colors, dtype=np.float64)
    else:
        rgb = np.full_like(xyz, 0.7)
    return xyz, rgb
