"""Command-line entry point for video2mesh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video2mesh import __version__
from video2mesh.config import CAMERA_MODELS, ReconstructConfig
from video2mesh.errors import MissingDependencyError, Video2MeshError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2mesh",
        description=(
            "Turn a video into a 3D mesh and a camera pose for every extracted frame. "
            "Uses COLMAP structure-from-motion and Open3D meshing."
        ),
    )
    parser.add_argument("--version", action="version", version=f"video2mesh {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    reconstruct = sub.add_parser(
        "reconstruct",
        help="Extract frames, reconstruct cameras, and mesh the scene",
    )
    reconstruct.add_argument("video", type=Path, help="Input video file")
    reconstruct.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory for frames, the COLMAP model, mesh, and camera files",
    )
    reconstruct.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="How many frames per second to keep (default: 2). Capped at the video frame rate.",
    )
    reconstruct.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to feed COLMAP (default: 300)",
    )
    reconstruct.add_argument(
        "--blur-threshold",
        type=float,
        default=0.0,
        help=(
            "Drop frames whose Laplacian variance is below this value. "
            "0 disables the filter. 40 to 80 is a reasonable start for 1080p video."
        ),
    )
    reconstruct.add_argument(
        "--quality",
        choices=("fast", "medium", "high"),
        default="medium",
        help="Feature count, image size, and Poisson depth (default: medium)",
    )
    reconstruct.add_argument(
        "--dense",
        dest="dense",
        action="store_true",
        default=True,
        help="Build a dense cloud before meshing (default)",
    )
    reconstruct.add_argument(
        "--no-dense",
        dest="dense",
        action="store_false",
        help="Mesh the sparse COLMAP points only",
    )
    reconstruct.add_argument(
        "--mesh",
        choices=("poisson", "bpa", "both"),
        default="poisson",
        help="poisson, ball pivoting (bpa), or both (default: poisson)",
    )
    reconstruct.add_argument(
        "--camera-model",
        default="SIMPLE_RADIAL",
        choices=CAMERA_MODELS,
        help="COLMAP camera model for the whole video (default: SIMPLE_RADIAL)",
    )
    reconstruct.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Resize frames so the long side is at most this many pixels. Overrides the quality preset.",
    )
    reconstruct.add_argument(
        "--poisson-depth",
        type=int,
        default=None,
        help="Octree depth for Poisson reconstruction. Overrides the quality preset.",
    )
    reconstruct.add_argument(
        "--poisson-trim",
        type=float,
        default=0.05,
        help="Drop this quantile of the lowest-density Poisson vertices (default: 0.05)",
    )
    reconstruct.add_argument(
        "--gpu",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use CUDA inside COLMAP when the binary supports it (default: auto)",
    )
    reconstruct.add_argument(
        "--backend",
        choices=("auto", "cli", "pycolmap"),
        default="auto",
        help="COLMAP executable, the pycolmap package, or whichever is installed (default: auto)",
    )
    reconstruct.add_argument("--colmap-exe", default=None, help="Path to colmap.exe / colmap.bat")
    reconstruct.add_argument("--ffmpeg-exe", default=None, help="Path to ffmpeg.exe")
    reconstruct.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already has files",
    )
    reconstruct.add_argument(
        "--view",
        action="store_true",
        help="Open the mesh and camera frustums when reconstruction finishes",
    )
    reconstruct.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    reconstruct.set_defaults(func=_cmd_reconstruct)

    from video2mesh.viewmath import VIEW_HELP

    view = sub.add_parser(
        "view",
        help="Show the mesh, camera frustums, and a per-frame point overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Open the reconstruction in an Open3D window. "
            "The side panel projects 3D points onto each source frame."
        ),
        epilog=VIEW_HELP,
    )
    view.add_argument("output", type=Path, help="Directory produced by reconstruct")
    view.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="Frustum depth in world units. 0 picks a size from the scene (default).",
    )
    view.add_argument(
        "--point-cloud",
        action="store_true",
        help="Draw pointcloud.ply instead of mesh.ply",
    )
    view.add_argument(
        "--frame",
        type=int,
        default=1,
        help="Registered frame to select first, counting from 1 (default: 1).",
    )
    view.set_defaults(func=_cmd_view)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Video2MeshError as exc:
        # reconstruct() already wrote the stage, the command, and a hint.
        if not getattr(exc, "logged", False):
            print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, MissingDependencyError) else 1
    return 0


def _cmd_reconstruct(args: argparse.Namespace) -> None:
    config = ReconstructConfig(
        video=args.video,
        output=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        blur_threshold=args.blur_threshold,
        quality=args.quality,
        dense=args.dense,
        mesh_method=args.mesh,
        camera_model=args.camera_model,
        max_image_size=args.max_image_size,
        overwrite=args.overwrite,
        poisson_trim=args.poisson_trim,
        poisson_depth=args.poisson_depth,
        gpu=args.gpu,
        backend=args.backend,
        colmap_exe=args.colmap_exe,
        ffmpeg_exe=args.ffmpeg_exe,
        log_level=args.log_level,
        view=args.view,
    )
    from video2mesh.pipeline import reconstruct

    result = reconstruct(config)
    if config.view:
        from video2mesh.viewer import view_reconstruction

        view_reconstruction(result.output_dir)


def _cmd_view(args: argparse.Namespace) -> None:
    from video2mesh.viewer import view_reconstruction

    view_reconstruction(
        args.output,
        frustum_scale=args.scale,
        prefer_point_cloud=args.point_cloud,
        frame_number=args.frame,
    )


if __name__ == "__main__":
    raise SystemExit(main())
