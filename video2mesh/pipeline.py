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
from video2mesh.errors import ReconstructionError
from video2mesh.export import write_cameras, write_mesh, write_point_cloud
from video2mesh.frames import extract_frames
from video2mesh.mesh import orient_and_mesh
from video2mesh.sfm import run_sfm, undistort_images

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

    sfm = run_sfm(
        image_dir=extraction.frames_dir,
        workspace=output,
        preset=config.preset,
        camera_model=config.camera_model,
        use_gpu=use_gpu,
        backend=backend,
        colmap_exe=config.colmap_exe,
    )

    undistorted = None
    if config.dense:
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
            logger.warning("Undistortion failed (%s). Dense stereo will use the original frames.", exc)

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
    pointcloud_path = output / "pointcloud.ply"
    write_point_cloud(pointcloud_path, cloud)

    records = write_cameras(
        output / "cameras.json",
        output / "cameras.csv",
        source_video=config.video,
        reconstruction=sfm.reconstruction,
        extracted=extraction.frames,
        sparse_dir=sfm.model_dir,
        dense_method=dense_method,
    )

    centers = np.array([pose.center for pose in sfm.reconstruction.ordered_frames()])
    try:
        meshes = orient_and_mesh(
            cloud,
            centers,
            method=config.mesh_method,
            poisson_depth=config.effective_poisson_depth,
            poisson_trim=config.poisson_trim,
        )
    except ReconstructionError as exc:
        raise ReconstructionError(
            f"{exc}\nCamera poses were still written to {output / 'cameras.json'} "
            f"and the point cloud to {pointcloud_path}."
        ) from exc

    mesh = meshes["mesh"]
    mesh_ply = output / "mesh.ply"
    mesh_obj = output / "mesh.obj"
    write_mesh(mesh_ply, mesh_obj, mesh)
    if "bpa" in meshes:
        write_mesh(output / "mesh_bpa.ply", output / "mesh_bpa.obj", meshes["bpa"])

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
        "Done. %d/%d frames registered, %d points, mesh %d verts / %d tris. Method: %s",
        result.num_registered,
        result.num_frames,
        result.num_points,
        result.mesh_vertices,
        result.mesh_triangles,
        result.dense_method,
    )
    logger.info("Mesh: %s", mesh_ply)
    logger.info("Cameras: %s", result.cameras_json)
    return result


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
