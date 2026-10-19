"""Read COLMAP models and convert poses to camera centers.

COLMAP stores a world-to-camera transform. A point in the world ``X`` lands in
the camera as ``X_cam = R X + t``, with ``R`` built from a unit quaternion
``(qw, qx, qy, qz)``. The camera center in world coordinates is ``C = -R^T t``.
Monocular video has no absolute scale: distances are only defined up to a
similarity transform.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from video2mesh.errors import ReconstructionError

# model_id -> (name, number of parameters). Matches COLMAP's camera_models.h.
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_NAME_TO_ID = {name: idx for idx, (name, _) in CAMERA_MODELS.items()}

PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k1"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
    "FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    "FOV": ("fx", "fy", "cx", "cy", "omega"),
    "SIMPLE_RADIAL_FISHEYE": ("f", "cx", "cy", "k"),
    "RADIAL_FISHEYE": ("f", "cx", "cy", "k1", "k2"),
    "THIN_PRISM_FISHEYE": (
        "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "s1", "s2", "s3", "s4",
    ),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class FramePose:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def rotation(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def center(self) -> np.ndarray:
        rotation = self.rotation
        return -rotation.T @ self.tvec


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


@dataclass
class Reconstruction:
    cameras: dict[int, Camera] = field(default_factory=dict)
    frames: dict[int, FramePose] = field(default_factory=dict)
    points: list[Point3D] = field(default_factory=list)

    def ordered_frames(self) -> list[FramePose]:
        return sorted(self.frames.values(), key=lambda frame: frame.name)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion ``(qw, qx, qy, qz)`` to a world-to-camera matrix."""
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm(qvec)
    if norm == 0:
        raise ReconstructionError("A camera pose has a zero quaternion.")
    w, x, y, z = qvec / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def named_intrinsics(model: str, params: np.ndarray) -> dict[str, float]:
    """Map COLMAP parameters to names and always include fx, fy, cx, cy when possible."""
    values = [float(v) for v in np.asarray(params, dtype=np.float64).tolist()]
    names = PARAM_NAMES.get(model, ())
    named: dict[str, float] = {}
    if names and len(names) == len(values):
        named = dict(zip(names, values))
    else:
        named = {f"param{i}": value for i, value in enumerate(values)}
    if "f" in named:
        named.setdefault("fx", named["f"])
        named.setdefault("fy", named["f"])
    if model == "SIMPLE_RADIAL_FISHEYE" and "k" in named:
        named.setdefault("k1", named["k"])
    return named


def camera_matrix(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return a 3x3 intrinsics matrix and OpenCV distortion coefficients."""
    named = named_intrinsics(camera.model, camera.params)
    if "fx" not in named or "cx" not in named or "cy" not in named:
        raise ReconstructionError(
            f"Camera {camera.id} model {camera.model} has no focal length or principal point."
        )
    fx = named["fx"]
    fy = named.get("fy", fx)
    matrix = np.array(
        [[fx, 0.0, named["cx"]], [0.0, fy, named["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        dist = [0.0, 0.0, 0.0, 0.0]
    elif camera.model == "SIMPLE_RADIAL":
        dist = [named.get("k1", 0.0), 0.0, 0.0, 0.0]
    elif camera.model == "RADIAL":
        dist = [named.get("k1", 0.0), named.get("k2", 0.0), 0.0, 0.0]
    elif camera.model == "OPENCV":
        dist = [named.get("k1", 0.0), named.get("k2", 0.0), named.get("p1", 0.0), named.get("p2", 0.0)]
    elif camera.model == "FULL_OPENCV":
        dist = [
            named.get("k1", 0.0),
            named.get("k2", 0.0),
            named.get("p1", 0.0),
            named.get("p2", 0.0),
            named.get("k3", 0.0),
            named.get("k4", 0.0),
            named.get("k5", 0.0),
            named.get("k6", 0.0),
        ]
    else:
        # Fisheye and FOV models are not a drop-in OpenCV plumb-bob model.
        # After COLMAP undistortion the camera is PINHOLE and this branch is unused.
        dist = [0.0, 0.0, 0.0, 0.0]
    return matrix, np.asarray(dist, dtype=np.float64)


def read_model(path: Path) -> Reconstruction:
    """Load a COLMAP model directory from text files, or binary if text is absent."""
    path = Path(path)
    if (path / "cameras.txt").is_file() and (path / "images.txt").is_file():
        reconstruction = Reconstruction(
            cameras=read_cameras_text(path / "cameras.txt"),
            frames=read_images_text(path / "images.txt"),
        )
        points_path = path / "points3D.txt"
        if points_path.is_file():
            reconstruction.points = read_points3d_text(points_path)
        return reconstruction
    if (path / "cameras.bin").is_file() and (path / "images.bin").is_file():
        reconstruction = Reconstruction(
            cameras=read_cameras_binary(path / "cameras.bin"),
            frames=read_images_binary(path / "images.bin"),
        )
        points_path = path / "points3D.bin"
        if points_path.is_file():
            reconstruction.points = read_points3d_binary(points_path)
        return reconstruction
    raise ReconstructionError(
        f"No COLMAP model at {path}. Expected cameras.txt/images.txt or the .bin equivalents."
    )


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _content_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = np.array([float(value) for value in parts[4:]], dtype=np.float64)
        cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images_text(path: Path) -> dict[int, FramePose]:
    frames: dict[int, FramePose] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 10:
                raise ReconstructionError(f"Bad images.txt line in {path}: {stripped}")
            frame = FramePose(
                id=int(parts[0]),
                qvec=np.array([float(value) for value in parts[1:5]], dtype=np.float64),
                tvec=np.array([float(value) for value in parts[5:8]], dtype=np.float64),
                camera_id=int(parts[8]),
                name=" ".join(parts[9:]),
            )
            # The following line is the 2D observations. It may be empty.
            handle.readline()
            frames[frame.id] = frame
    return frames


def read_points3d_text(path: Path) -> list[Point3D]:
    points: list[Point3D] = []
    for line in _content_lines(path):
        parts = line.split()
        if len(parts) < 8:
            continue
        points.append(
            Point3D(
                id=int(parts[0]),
                xyz=np.array([float(value) for value in parts[1:4]], dtype=np.float64),
                rgb=np.array([int(value) for value in parts[4:7]], dtype=np.uint8),
                error=float(parts[7]),
            )
        )
    return points


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as handle:
        count = _unpack(handle, "<Q", 8)[0]
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(handle, "<iiQQ", 24)
            if model_id not in CAMERA_MODELS:
                raise ReconstructionError(f"Unknown COLMAP camera model id {model_id} in {path}")
            model, nparams = CAMERA_MODELS[model_id]
            params = np.array(_unpack(handle, "<" + "d" * nparams, 8 * nparams), dtype=np.float64)
            cameras[int(camera_id)] = Camera(int(camera_id), model, int(width), int(height), params)
    return cameras


def read_images_binary(path: Path) -> dict[int, FramePose]:
    frames: dict[int, FramePose] = {}
    with path.open("rb") as handle:
        count = _unpack(handle, "<Q", 8)[0]
        for _ in range(count):
            image_id = _unpack(handle, "<I", 4)[0]
            qvec = np.array(_unpack(handle, "<dddd", 32), dtype=np.float64)
            tvec = np.array(_unpack(handle, "<ddd", 24), dtype=np.float64)
            camera_id = _unpack(handle, "<I", 4)[0]
            name = _read_c_string(handle)
            npoints = _unpack(handle, "<Q", 8)[0]
            handle.seek(24 * npoints, os.SEEK_CUR)
            frames[int(image_id)] = FramePose(
                id=int(image_id),
                qvec=qvec,
                tvec=tvec,
                camera_id=int(camera_id),
                name=name,
            )
    return frames


def read_points3d_binary(path: Path) -> list[Point3D]:
    points: list[Point3D] = []
    with path.open("rb") as handle:
        count = _unpack(handle, "<Q", 8)[0]
        for _ in range(count):
            point_id = _unpack(handle, "<Q", 8)[0]
            xyz = np.array(_unpack(handle, "<ddd", 24), dtype=np.float64)
            rgb = np.array(_unpack(handle, "<BBB", 3), dtype=np.uint8)
            error = _unpack(handle, "<d", 8)[0]
            track_length = _unpack(handle, "<Q", 8)[0]
            handle.seek(8 * track_length, os.SEEK_CUR)
            points.append(
                Point3D(id=int(point_id), xyz=xyz, rgb=rgb, error=float(error))
            )
    return points


def list_model_dirs(sparse_root: Path) -> list[Path]:
    """Return model folders under a mapper output directory, largest-name order not applied."""
    sparse_root = Path(sparse_root)
    if not sparse_root.exists():
        return []
    children = []
    for child in sorted(sparse_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "cameras.bin").is_file() or (child / "cameras.txt").is_file():
            children.append(child)
    if children:
        return children
    if (sparse_root / "cameras.bin").is_file() or (sparse_root / "cameras.txt").is_file():
        return [sparse_root]
    return []


def _content_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _unpack(handle, fmt: str, size: int):
    data = handle.read(size)
    if len(data) != size:
        raise ReconstructionError(
            f"COLMAP binary model ended early (expected {size} bytes, got {len(data)})."
        )
    return struct.unpack(fmt, data)


def _read_c_string(handle) -> str:
    chunks = bytearray()
    while True:
        char = handle.read(1)
        if not char:
            raise ReconstructionError("COLMAP image name was not null-terminated.")
        if char == b"\x00":
            break
        chunks.extend(char)
    return chunks.decode("utf-8", errors="replace")
