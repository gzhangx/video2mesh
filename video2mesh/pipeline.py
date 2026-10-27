"""Glue frames, SfM, dense fusion, meshing, and export into one run."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video2mesh import __version__
from video2mesh.config import ReconstructConfig
from video2mesh.dense import build_point_cloud
from video2mesh.deps import colmap_version_text, decide_gpu, require_colmap
from video2mesh.errors import ReconstructionError, Video2MeshError
from video2mesh.export import write_cameras, write_mesh, write_point_cloud
from video2mesh.frames import FrameExtraction, extract_frames
from video2mesh.mesh import orient_and_mesh
from video2mesh.progress import RunProgress
from video2mesh.sfm import SfmOutput, run_sfm, undistort_images

logger = logging.getLogger("video2mesh")


@dataclass
class ReconstructResult:
    output_dir: Path
    mesh_ply: Path
    mesh_obj: Path
    pointcloud_ply: Path
    cameras_json: Path
    cameras_csv: Path
    sparse_dir: Path
    num_frames: int
    num_registered: int
    num_points: int
    dense_method: str
    mesh_vertices: int
    mesh_triangles: int


def reconstruct(config: ReconstructConfig) -> ReconstructResult:
    output = _prepare_output(config.output, config.overwrite)
    _setup_logging(output / "log.txt", config.log_level)
    progress = RunProgress()
    try:
        return _reconstruct(config, output, progress)
    except Video2MeshError as exc:
        message = progress.failure_message(exc)
        logger.error("%s", message)
        raised = type(exc)(message)
        raised.logged = True  # type: ignore[attr-defined]
        raise raised from exc


def _reconstruct(
    config: ReconstructConfig,
    output: Path,
    progress: RunProgress,
) -> ReconstructResult:
    logger.info("video2mesh %s", __version__)
    logger.info("Video: %s", config.video)
    logger.info("Output: %s", output)

    use_gpu = False
    colmap_banner = ""
    if config.backend == "pycolmap":
        backend = "pycolmap"
        use_gpu = config.gpu == "on"
        logger.info("Structure-from-motion backend: pycolmap")
    else:
        from video2mesh.deps import find_colmap

        exe = find_colmap(config.colmap_exe)
        if exe or config.backend == "cli":
            exe = require_colmap(config.colmap_exe)
            colmap_banner = colmap_version_text(exe)
            use_gpu = decide_gpu(exe, config.gpu)
            logger.info("%s", colmap_banner)
            logger.info("CUDA patch-match: %s", "yes" if use_gpu else "no")
            backend = "cli"
        else:
            backend = "pycolmap"
            use_gpu = config.gpu == "on"
            logger.info("COLMAP executable not found; using pycolmap")

    (output / "run_config.json").write_text(
        json.dumps(_public_config(config, backend, use_gpu, colmap_banner), indent=2),
        encoding="utf-8",
    )

    progress.start("frame extraction")
    extraction = extract_frames(
        config.video,
        output / "frames",
        fps=config.fps,
        max_frames=config.max_frames,
        blur_threshold=config.blur_threshold,
        max_image_size=config.effective_max_image_size,
        ffmpeg_exe=config.ffmpeg_exe,
    )
    extraction.write_manifest(output / "frames_manifest.json")
    progress.finish(f"{len(extraction.frames)} frames")

    sfm = run_sfm(
        image_dir=extraction.frames_dir,
        workspace=output,
        preset=config.preset,
        camera_model=config.camera_model,
        use_gpu=use_gpu,
        backend=backend,
        colmap_exe=config.colmap_exe,
        progress=progress,
    )

    undistorted = None
    if config.dense:
        progress.start("undistort")
        try:
            undistorted = undistort_images(
                image_dir=extraction.frames_dir,
                model_dir=sfm.model_dir,
                dense_dir=output / "dense",
                colmap_exe=config.colmap_exe,
                max_image_size=config.effective_max_image_size,
                backend=sfm.backend,
            )
        except ReconstructionError as exc:
            logger.warning(
                "Undistortion failed (%s). Dense stereo will use the original frames.",
                exc,
            )
            progress.finish("continued without undistortion")
        else:
            detail = "undistorted images" if undistorted is not None else "no undistorted model"
            progress.finish(detail)
    else:
        progress.skip(
            "undistort",
            "Dense reconstruction is off, so undistortion is skipped.",
        )

    progress.start("dense")
    cloud, dense_method = build_point_cloud(
        sparse=sfm.reconstruction,
        image_dir=extraction.frames_dir,
        workspace=output,
        preset=config.preset,
        use_cuda=use_gpu and config.dense,
        dense=config.dense,
        colmap_exe=config.colmap_exe,
        undistorted_model=undistorted,
        backend=sfm.backend,
    )
    progress.finish(f"{len(cloud.points)} points via {dense_method}")

    pointcloud_path = output / "pointcloud.ply"
    centers = np.array([pose.center for pose in sfm.reconstruction.ordered_frames()])
    progress.start("mesh")
    try:
        meshes = orient_and_mesh(
            cloud,
            centers,
            method=config.mesh_method,
            poisson_depth=config.effective_poisson_depth,
            poisson_trim=config.poisson_trim,
        )
    except ReconstructionError as exc:
        try:
            _write_poses(config, output, pointcloud_path, cloud, sfm, extraction, dense_method)
            note = (
                f"\nCamera poses were still written to {output / 'cameras.json'} "
                f"and the point cloud to {pointcloud_path}."
            )
        except Exception as write_exc:
            note = f"\nCamera poses could not be saved either: {write_exc}"
        raise ReconstructionError(f"{exc}{note}") from exc
    mesh = meshes["mesh"]
    progress.finish(f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    progress.start("export")
    records = _write_poses(config, output, pointcloud_path, cloud, sfm, extraction, dense_method)
    mesh_ply = output / "mesh.ply"
    mesh_obj = output / "mesh.obj"
    write_mesh(mesh_ply, mesh_obj, mesh)
    if "bpa" in meshes:
        write_mesh(output / "mesh_bpa.ply", output / "mesh_bpa.obj", meshes["bpa"])
    progress.finish("mesh.ply, cameras.json")

    result = ReconstructResult(
        output_dir=output,
        mesh_ply=mesh_ply,
        mesh_obj=mesh_obj,
        pointcloud_ply=pointcloud_path,
        cameras_json=output / "cameras.json",
        cameras_csv=output / "cameras.csv",
        sparse_dir=sfm.model_dir,
        num_frames=len(extraction.frames),
        num_registered=len(records),
        num_points=len(cloud.points),
        dense_method=dense_method,
        mesh_vertices=len(mesh.vertices),
        mesh_triangles=len(mesh.triangles),
    )
    logger.info(
        "\n%s",
        progress.render_summary(
            [
                f"frames extracted: {result.num_frames}",
                f"cameras registered: {result.num_registered}",
                f"point cloud: {result.num_points} points ({result.dense_method})",
                (
                    f"mesh: {result.mesh_vertices} vertices, "
                    f"{result.mesh_triangles} triangles"
                ),
                f"mesh.ply: {result.mesh_ply}",
                f"mesh.obj: {result.mesh_obj}",
                f"pointcloud.ply: {result.pointcloud_ply}",
                f"cameras.json: {result.cameras_json}",
                f"cameras.csv: {result.cameras_csv}",
                f"sparse model: {result.sparse_dir}",
                f"log.txt: {output / 'log.txt'}",
            ]
        ),
    )
    return result


def _write_poses(config, output, pointcloud_path, cloud, sfm: SfmOutput, extraction: FrameExtraction, dense_method: str):
    write_point_cloud(pointcloud_path, cloud)
    return write_cameras(
        output / "cameras.json",
        output / "cameras.csv",
        source_video=config.video,
        reconstruction=sfm.reconstruction,
        extracted=extraction.frames,
        sparse_dir=sfm.model_dir,
        dense_method=dense_method,
    )


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ReconstructionError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise ReconstructionError(
                    f"Output directory is not empty: {path}\nPass --overwrite to replace it."
                )
            resolved = path.resolve()
            if resolved == Path(resolved.anchor) or len(resolved.parts) < 2:
                raise ReconstructionError(f"Refusing to delete {resolved}")
            shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _setup_logging(log_file: Path, level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    log = logging.getLogger("video2mesh")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(console)
    log.addHandler(file_handler)


def _public_config(config: ReconstructConfig, backend: str, use_gpu: bool, banner: str) -> dict:
    return {
        "video2mesh_version": __version__,
        "video": str(config.video),
        "output": str(config.output),
        "fps": config.fps,
        "max_frames": config.max_frames,
        "blur_threshold": config.blur_threshold,
        "quality": config.quality,
        "dense": config.dense,
        "mesh_method": config.mesh_method,
        "camera_model": config.camera_model,
        "max_image_size": config.effective_max_image_size,
        "poisson_trim": config.poisson_trim,
        "poisson_depth": config.effective_poisson_depth,
        "gpu": config.gpu,
        "use_gpu": use_gpu,
        "backend": backend,
        "colmap": banner,
    }
