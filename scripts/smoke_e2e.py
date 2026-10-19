"""Run video2mesh on a short synthetic video and check the outputs.

This needs ffmpeg (or OpenCV), a COLMAP executable, and the Python dependencies.
It is the end-to-end check; the unit tests under tests/ do not call COLMAP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open3d as o3d

from video2mesh.sample_scene import write_sample_video
from video2mesh.cli import main


def main_smoke() -> int:
    root = Path("sample_data")
    root.mkdir(exist_ok=True)
    video = write_sample_video(
        root / "room.mp4",
        num_frames=16,
        width=640,
        height=480,
        fps=8,
    )
    output = root / "out"
    code = main(
        [
            "reconstruct",
            str(video),
            "-o",
            str(output),
            "--overwrite",
            "--fps",
            "8",
            "--max-frames",
            "16",
            "--quality",
            "fast",
            "--mesh",
            "poisson",
            "--gpu",
            "off",
            "--log-level",
            "INFO",
        ]
    )
    if code != 0:
        print(f"reconstruct exited {code}", file=sys.stderr)
        return code

    cameras = json.loads((output / "cameras.json").read_text(encoding="utf-8"))
    registered = cameras["num_frames_registered"]
    print(f"registered frames: {registered}")
    if registered < 3:
        print("too few registered frames", file=sys.stderr)
        return 1
    frame = cameras["frames"][0]
    for key in (
        "filename",
        "intrinsics",
        "rotation_quaternion_wxyz",
        "translation",
        "camera_center_world",
    ):
        if key not in frame:
            print(f"cameras.json frame missing {key}", file=sys.stderr)
            return 1
    focal = frame["intrinsics"].get("fx") or frame["intrinsics"].get("f")
    if focal is None or float(focal) < 0.5 * float(frame["width"]):
        print(f"focal length looks wrong: {focal}", file=sys.stderr)
        return 1
    sparse_model = Path(cameras["sparse_model"])
    if not (sparse_model / "cameras.txt").is_file() or not (sparse_model / "images.txt").is_file():
        print(f"sparse model has no text files: {sparse_model}", file=sys.stderr)
        return 1

    mesh = o3d.io.read_triangle_mesh(str(output / "mesh.ply"))
    cloud = o3d.io.read_point_cloud(str(output / "pointcloud.ply"))
    print(f"mesh triangles: {len(mesh.triangles)}")
    print(f"cloud points: {len(cloud.points)}")
    if len(mesh.triangles) < 10 or len(cloud.points) < 30:
        print("mesh or cloud is empty", file=sys.stderr)
        return 1
    if not (output / "mesh.obj").is_file():
        print("missing mesh.obj", file=sys.stderr)
        return 1
    if not (output / "cameras.csv").is_file():
        print("missing cameras.csv", file=sys.stderr)
        return 1
    sparse = output / "sparse"
    if not any(sparse.glob("**/cameras.bin")) and not any(sparse.glob("**/cameras.txt")):
        print("missing COLMAP sparse model", file=sys.stderr)
        return 1
    print("smoke ok")
    print(f"mesh: {output / 'mesh.ply'}")
    print(f"cameras: {output / 'cameras.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_smoke())
