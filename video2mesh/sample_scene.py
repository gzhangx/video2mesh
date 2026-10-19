"""Render a short textured-room clip so the pipeline can be tried without a camera.

The camera orbits inside a box and looks at a few smaller boxes. Every surface
carries a unique high-contrast pattern, which is what COLMAP needs in place of
a real scene. The pinhole focal length matches COLMAP's default prior
(1.2 times the long image side) so bundle adjustment starts close to the truth.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from video2mesh.deps import command_argv, find_ffmpeg


def write_sample_video(
    path: Path,
    *,
    num_frames: int = 16,
    width: int = 640,
    height: int = 480,
    fps: int = 8,
) -> Path:
    """Write an H.264 mp4 (or an MJPG avi if ffmpeg is missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = render_sequence(num_frames=num_frames, width=width, height=height)
    ffmpeg = find_ffmpeg(None)
    if ffmpeg and path.suffix.lower() == ".mp4":
        return _encode_ffmpeg(ffmpeg, frames, path, fps)
    if path.suffix.lower() == ".mp4":
        path = path.with_suffix(".avi")
    return _encode_opencv(frames, path, fps)


def render_sequence(
    *,
    num_frames: int = 16,
    width: int = 640,
    height: int = 480,
) -> list[np.ndarray]:
    textures = [_make_texture(seed) for seed in range(8)]
    boxes = _scene_boxes()
    focal = 1.2 * max(width, height)
    frames = []
    for index in range(num_frames):
        eye, target = _camera_pose(index, num_frames)
        frames.append(render_frame(width, height, eye, target, focal, boxes, textures))
    return frames


def render_frame(
    width: int,
    height: int,
    eye: np.ndarray,
    target: np.ndarray,
    focal: float,
    boxes: list[dict],
    textures: list[np.ndarray],
) -> np.ndarray:
    """Return a BGR uint8 image."""
    directions = _ray_directions(width, height, eye, target, focal)
    best_t = np.full((height, width), np.inf, dtype=np.float64)
    color = np.zeros((height, width, 3), dtype=np.float64)
    light = _normalize(np.array([0.25, 0.85, 0.35], dtype=np.float64))
    for box in boxes:
        hit_t, axis, sign, valid = _intersect(eye, directions, box["bmin"], box["bmax"], box["inside"])
        closer = valid & (hit_t < best_t)
        if not np.any(closer):
            continue
        best_t = np.where(closer, hit_t, best_t)
        hit = eye + hit_t[..., None] * directions
        rel = (hit - box["bmin"]) / np.maximum(box["bmax"] - box["bmin"], 1e-8)
        uu = np.where(axis == 0, rel[..., 2], rel[..., 0])
        vv = np.where(axis == 1, rel[..., 2], rel[..., 1])
        sampled = _sample(textures[box["texture"]], uu, vv)
        normal = np.zeros_like(directions)
        np.put_along_axis(normal, axis[..., None], (-sign)[..., None], axis=-1)
        shade = 0.62 + 0.38 * np.clip(np.sum(normal * light, axis=-1), 0.0, 1.0)
        painted = sampled * shade[..., None]
        color[closer] = painted[closer]
    rgb = np.clip(color, 0, 255).astype(np.uint8)
    return rgb[..., ::-1]


def _scene_boxes() -> list[dict]:
    def box(bmin, bmax, texture, inside):
        return {
            "bmin": np.asarray(bmin, dtype=np.float64),
            "bmax": np.asarray(bmax, dtype=np.float64),
            "texture": texture,
            "inside": inside,
        }

    return [
        box([-1.8, 0.0, -1.8], [1.8, 2.4, 1.8], 0, True),
        box([-0.35, 0.0, -0.35], [0.35, 0.72, 0.35], 1, False),
        box([-0.12, 0.72, -0.12], [0.16, 0.98, 0.18], 2, False),
        box([0.55, 0.0, -0.22], [0.92, 0.42, 0.28], 3, False),
        box([-0.95, 0.0, 0.35], [-0.55, 0.55, 0.78], 4, False),
    ]


def _camera_pose(index: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    start = np.deg2rad(-32.0)
    end = np.deg2rad(32.0)
    if count == 1:
        angle = 0.0
    else:
        angle = start + (end - start) * (index / (count - 1))
    radius = 1.28
    eye = np.array([radius * np.sin(angle), 1.12, radius * np.cos(angle)], dtype=np.float64)
    target = np.array([0.0, 0.48, 0.0], dtype=np.float64)
    return eye, target


def _ray_directions(width: int, height: int, eye: np.ndarray, target: np.ndarray, focal: float) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, np.array([0.0, 1.0, 0.0])))
    up = _normalize(np.cross(right, forward))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    xs = (np.arange(width, dtype=np.float64) - cx) / focal
    ys = (np.arange(height, dtype=np.float64) - cy) / focal
    directions = (
        forward[None, None, :]
        + xs[None, :, None] * right[None, None, :]
        - ys[:, None, None] * up[None, None, :]
    )
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def _intersect(origin, directions, bmin, bmax, inside):
    safe = np.where(np.abs(directions) < 1e-8, np.copysign(1e-8, directions + 1e-12), directions)
    inv = 1.0 / safe
    t0 = (bmin - origin) * inv
    t1 = (bmax - origin) * inv
    tsmall = np.minimum(t0, t1)
    tbig = np.maximum(t0, t1)
    tmin = np.max(tsmall, axis=-1)
    tmax = np.min(tbig, axis=-1)
    axis_enter = np.argmax(tsmall, axis=-1)
    axis_exit = np.argmin(tbig, axis=-1)
    if inside:
        hit_t = tmax
        axis = axis_exit
        valid = (tmax > 1e-4) & (tmax >= tmin)
    else:
        hit_t = tmin
        axis = axis_enter
        valid = (tmin > 1e-4) & (tmin <= tmax)
    sign = np.sign(np.take_along_axis(safe, axis[..., None], axis=-1)[..., 0])
    sign = np.where(sign == 0, 1.0, sign)
    return hit_t, axis, sign, valid


def _sample(texture: np.ndarray, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    ui = np.clip((np.mod(uu, 1.0) * (width - 1)).astype(np.int32), 0, width - 1)
    vi = np.clip((np.mod(vv, 1.0) * (height - 1)).astype(np.int32), 0, height - 1)
    return texture[vi, ui].astype(np.float64)


def _make_texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    image = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    for y in range(0, 256, 32):
        for x in range(0, 256, 32):
            color = rng.integers(0, 255, size=3, dtype=np.uint8)
            patch = image[y : y + 32, x : x + 32]
            image[y : y + 32, x : x + 32] = (0.45 * patch + 0.55 * color).astype(np.uint8)
    image[::16, :, :] = 20
    image[:, ::16, :] = 235
    # A unique mark so neighboring walls are not interchangeable.
    mark = 40 + (seed * 37) % 180
    image[20:70, mark : mark + 18, :] = np.array([240, 30, 30], dtype=np.uint8)
    image[mark % 160 : (mark % 160) + 24, 180:230, :] = np.array([30, 220, 80], dtype=np.uint8)
    return image


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _encode_ffmpeg(ffmpeg: str, frames: list[np.ndarray], path: Path, fps: int) -> Path:
    temp = path.parent / f".{path.stem}_frames"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        for index, frame in enumerate(frames, start=1):
            cv2.imwrite(str(temp / f"frame_{index:06d}.png"), frame)
        subprocess.run(
            command_argv(
                ffmpeg,
                [
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(temp / "frame_%06d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "16",
                    str(path),
                ],
            ),
            check=True,
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return path


def _encode_opencv(frames: list[np.ndarray], path: Path, fps: int) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path
