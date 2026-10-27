"""Structure-from-motion: COLMAP features, sequential matching, and mapping."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

from video2mesh.colmap_io import Reconstruction, list_model_dirs, read_model
from video2mesh.colmap_options import (
    FEATURE_MAX_IMAGE_SIZE,
    FEATURE_MAX_NUM_FEATURES,
    FEATURE_USE_GPU,
    MATCH_USE_GPU,
    ColmapCli,
    apply_pycolmap_extraction,
    apply_pycolmap_matching,
    option,
)
from video2mesh.config import SFM_TIPS
from video2mesh.deps import command_argv, require_colmap, run_command
from video2mesh.errors import MissingDependencyError, ReconstructionError
from video2mesh.progress import RunProgress, call_with_heartbeat

logger = logging.getLogger("video2mesh")


@dataclass
class SfmOutput:
    model_dir: Path
    reconstruction: Reconstruction
    backend: str
    cuda: bool


def run_sfm(
    *,
    image_dir: Path,
    workspace: Path,
    preset: dict,
    camera_model: str,
    use_gpu: bool,
    backend: str,
    colmap_exe: str | None,
    progress: RunProgress | None = None,
) -> SfmOutput:
    """Run sequential SfM and return the largest reconstructed model."""
    database = workspace / "database.db"
    sparse_root = workspace / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)

    chosen = _resolve_backend(backend, colmap_exe)
    logger.info("Structure-from-motion backend: %s", chosen)
    exe: str | None
    if chosen == "cli":
        exe = require_colmap(colmap_exe)
        _run_cli(
            exe,
            image_dir=image_dir,
            database=database,
            sparse_root=sparse_root,
            preset=preset,
            camera_model=camera_model,
            use_gpu=use_gpu,
            progress=progress,
        )
    else:
        exe = None
        _run_pycolmap(
            image_dir=image_dir,
            database=database,
            sparse_root=sparse_root,
            preset=preset,
            camera_model=camera_model,
            use_gpu=use_gpu,
            progress=progress,
        )

    model_dir = _select_largest_model(sparse_root, exe)
    _ensure_text_model(model_dir, exe)
    reconstruction = read_model(model_dir)
    if len(reconstruction.frames) < 3:
        raise ReconstructionError(
            f"COLMAP registered {len(reconstruction.frames)} image(s); at least 3 are required.\n"
            + SFM_TIPS
        )
    if len(reconstruction.points) < 20:
        raise ReconstructionError(
            f"COLMAP triangulated only {len(reconstruction.points)} points, which is not "
            "enough to mesh.\n"
            + SFM_TIPS
        )
    logger.info(
        "Sparse model %s: %d cameras, %d images, %d points",
        model_dir,
        len(reconstruction.cameras),
        len(reconstruction.frames),
        len(reconstruction.points),
    )
    _stage_finish(
        progress,
        f"{len(reconstruction.frames)} images, {len(reconstruction.points)} points",
    )
    _log_intrinsics(reconstruction)
    return SfmOutput(
        model_dir=model_dir,
        reconstruction=reconstruction,
        backend=chosen,
        cuda=use_gpu,
    )


def undistort_images(
    *,
    image_dir: Path,
    model_dir: Path,
    dense_dir: Path,
    colmap_exe: str | None,
    max_image_size: int,
    backend: str,
) -> Path | None:
    """Create a pinhole workspace for dense stereo. Returns the undistorted model dir."""
    if dense_dir.exists():
        # A previous failed attempt should not mix old images into this run.
        import shutil

        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)
    if backend == "cli":
        exe = require_colmap(colmap_exe)
        argv = ["image_undistorter"]
        ColmapCli(exe).add(
            argv,
            "image_undistorter",
            [
                option("image_path", "image_path", image_dir),
                option("input_path", "input_path", model_dir),
                option("output_path", "output_path", dense_dir),
                option("output_type", "output_type", "COLMAP"),
                option("max_image_size", ("max_image_size", "Undistortion.max_image_size"), max_image_size),
            ],
        )
        run_command(
            command_argv(exe, argv),
            echo="progress",
            label="colmap image_undistorter",
        )
    else:
        try:
            import pycolmap
        except ImportError as exc:
            raise MissingDependencyError(
                "pycolmap is required to undistort when the COLMAP executable is not used."
            ) from exc
        if not hasattr(pycolmap, "undistort_images"):
            logger.warning(
                "This pycolmap build has no undistort_images. Dense stereo will use the original frames."
            )
            return None
        logger.info("running pycolmap undistort_images")
        call_with_heartbeat(
            "pycolmap undistort",
            lambda: pycolmap.undistort_images(str(dense_dir), str(model_dir), str(image_dir)),
        )

    undistorted_model = dense_dir / "sparse"
    if not undistorted_model.exists():
        logger.warning("Undistortion did not write %s", undistorted_model)
        return None
    return undistorted_model


def _resolve_backend(backend: str, colmap_exe: str | None) -> str:
    if backend == "cli":
        require_colmap(colmap_exe)
        return "cli"
    if backend == "pycolmap":
        return "pycolmap"
    from video2mesh.deps import find_colmap

    if find_colmap(colmap_exe):
        return "cli"
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            "Neither the COLMAP executable nor pycolmap is available.\n"
            "Install COLMAP and put colmap on PATH (see the README), or pip install pycolmap."
        ) from None
    return "pycolmap"


def _run_cli(
    exe: str,
    *,
    image_dir: Path,
    database: Path,
    sparse_root: Path,
    preset: dict,
    camera_model: str,
    use_gpu: bool,
    progress: RunProgress | None,
) -> None:
    gpu_flag = "1" if use_gpu else "0"
    cli = ColmapCli(exe)
    _stage_start(progress, "feature extraction")
    logger.info("COLMAP feature_extractor (gpu=%s, model=%s)", gpu_flag, camera_model)
    extract_argv = ["feature_extractor"]
    cli.add(
        extract_argv,
        "feature_extractor",
        [
            option("database_path", "database_path", database),
            option("image_path", "image_path", image_dir),
            option("single_camera", "ImageReader.single_camera", "1"),
            option("camera_model", "ImageReader.camera_model", camera_model),
            option("feature_max_image_size", FEATURE_MAX_IMAGE_SIZE, preset["sift_max_image_size"]),
            option("feature_max_num_features", FEATURE_MAX_NUM_FEATURES, preset["max_num_features"]),
            option("feature_use_gpu", FEATURE_USE_GPU, gpu_flag),
        ],
    )
    run_command(
        command_argv(exe, extract_argv),
        echo="progress",
        label="colmap feature_extractor",
    )
    _stage_finish(progress)
    _stage_start(progress, "matching")
    logger.info(
        "COLMAP sequential_matcher (overlap=%s)",
        preset["sequential_overlap"],
    )
    match_argv = ["sequential_matcher"]
    cli.add(
        match_argv,
        "sequential_matcher",
        [
            option("database_path", "database_path", database),
            option("sequential_overlap", "SequentialMatching.overlap", preset["sequential_overlap"]),
            option("quadratic_overlap", "SequentialMatching.quadratic_overlap", "1"),
            option("matching_use_gpu", MATCH_USE_GPU, gpu_flag),
        ],
    )
    run_command(
        command_argv(exe, match_argv),
        echo="progress",
        label="colmap sequential_matcher",
    )
    _stage_finish(progress)
    _stage_start(progress, "mapping")
    logger.info("COLMAP mapper")
    map_argv = ["mapper"]
    cli.add(
        map_argv,
        "mapper",
        [
            option("database_path", "database_path", database),
            option("image_path", "image_path", image_dir),
            option("output_path", "output_path", sparse_root),
            option("min_model_size", "Mapper.min_model_size", "3"),
            option("multiple_models", "Mapper.multiple_models", "1"),
            option("init_min_num_inliers", "Mapper.init_min_num_inliers", preset["init_min_num_inliers"]),
            option("init_min_tri_angle", "Mapper.init_min_tri_angle", preset["init_min_tri_angle"]),
            option("init_max_forward_motion", "Mapper.init_max_forward_motion", "0.98"),
            option("abs_pose_min_num_inliers", "Mapper.abs_pose_min_num_inliers", "15"),
            option("ba_refine_principal_point", "Mapper.ba_refine_principal_point", "0"),
        ],
    )
    run_command(
        command_argv(exe, map_argv),
        echo="progress",
        label="colmap mapper",
    )


def _run_pycolmap(
    *,
    image_dir: Path,
    database: Path,
    sparse_root: Path,
    preset: dict,
    camera_model: str,
    use_gpu: bool,
    progress: RunProgress | None,
) -> None:
    try:
        import pycolmap
    except ImportError as exc:
        raise MissingDependencyError(
            "pycolmap is not installed. Install the COLMAP executable (recommended on Windows)\n"
            "or run: pip install pycolmap"
        ) from exc

    _stage_start(progress, "feature extraction")
    logger.info("pycolmap feature extraction")
    extract_kwargs: dict = {}
    if hasattr(pycolmap, "CameraMode"):
        extract_kwargs["camera_mode"] = pycolmap.CameraMode.SINGLE
    reader = _reader_options(pycolmap, camera_model)
    if reader is not None:
        extract_kwargs["reader_options"] = reader
    extraction = _extraction_options(pycolmap, preset, use_gpu)
    if extraction is not None:
        extract_kwargs["extraction_options"] = extraction
    call_with_heartbeat(
        "pycolmap feature extraction",
        lambda: _invoke(pycolmap.extract_features, str(database), str(image_dir), **extract_kwargs),
    )
    _stage_finish(progress)

    if not hasattr(pycolmap, "match_sequential"):
        raise ReconstructionError(
            "This pycolmap version has no match_sequential. Upgrade pycolmap or use the COLMAP executable."
        )
    _stage_start(progress, "matching")
    logger.info("pycolmap sequential matching")
    match_kwargs: dict = {}
    matching = _matching_options(pycolmap, use_gpu)
    if matching is not None:
        match_kwargs["matching_options"] = matching
    sequential = _sequential_options(pycolmap, int(preset["sequential_overlap"]))
    if sequential is not None:
        match_kwargs["pairing_options"] = sequential
        match_kwargs["sequential_options"] = sequential
    call_with_heartbeat(
        "pycolmap sequential matching",
        lambda: _invoke(pycolmap.match_sequential, str(database), **match_kwargs),
    )
    _stage_finish(progress)

    _stage_start(progress, "mapping")
    logger.info("pycolmap incremental mapping")
    options = _pipeline_options(pycolmap, preset)
    map_kwargs = {"options": options} if options is not None else {}
    reconstructions = call_with_heartbeat(
        "pycolmap incremental mapping",
        lambda: _invoke(
            pycolmap.incremental_mapping,
            str(database),
            str(image_dir),
            str(sparse_root),
            **map_kwargs,
        ),
    )
    if not list_model_dirs(sparse_root) and isinstance(reconstructions, dict):
        for key, reconstruction in reconstructions.items():
            dest = sparse_root / str(key)
            dest.mkdir(parents=True, exist_ok=True)
            _write_reconstruction(reconstruction, dest)


def _select_largest_model(sparse_root: Path, colmap_exe: str | None) -> Path:
    models = list_model_dirs(sparse_root)
    if not models:
        raise ReconstructionError(
            "COLMAP mapping produced no sparse model. The frames may not overlap, "
            "or the scene may not have enough texture.\n"
            + SFM_TIPS
        )
    if len(models) == 1:
        return models[0]

    ranked: list[tuple[int, int, Path]] = []
    for model in models:
        try:
            _ensure_text_model(model, colmap_exe)
            reconstruction = read_model(model)
        except ReconstructionError:
            continue
        ranked.append((len(reconstruction.frames), len(reconstruction.points), model))
    if not ranked:
        raise ReconstructionError("COLMAP wrote model folders that could not be read.\n" + SFM_TIPS)
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = ranked[0][2]
    logger.info(
        "Mapper produced %d models; using %s (%d registered images)",
        len(ranked),
        best.name,
        ranked[0][0],
    )
    return best


def _ensure_text_model(model_dir: Path, colmap_exe: str | None) -> None:
    if (model_dir / "cameras.txt").is_file() and (model_dir / "images.txt").is_file():
        return
    if colmap_exe:
        convert_argv = ["model_converter"]
        ColmapCli(colmap_exe).add(
            convert_argv,
            "model_converter",
            [
                option("input_path", "input_path", model_dir),
                option("output_path", "output_path", model_dir),
                option("output_type", "output_type", "TXT"),
            ],
        )
        run_command(
            command_argv(colmap_exe, convert_argv),
            echo="progress",
            label="colmap model_converter",
        )
        return
    try:
        import pycolmap
    except ImportError:
        return
    if not (model_dir / "cameras.bin").is_file():
        return
    try:
        reconstruction = pycolmap.Reconstruction(str(model_dir))
    except Exception:
        return
    _write_text(reconstruction, model_dir)


def _write_reconstruction(reconstruction, dest: Path) -> None:
    if hasattr(reconstruction, "write"):
        reconstruction.write(str(dest))
    _write_text(reconstruction, dest)


def _write_text(reconstruction, dest: Path) -> None:
    if hasattr(reconstruction, "write_text"):
        reconstruction.write_text(str(dest))
    elif hasattr(reconstruction, "export_text"):
        reconstruction.export_text(str(dest))


def _invoke(fn, *args, **kwargs):
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(*args, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    try:
        return fn(*args, **accepted)
    except TypeError:
        # Some pycolmap versions want options as a positional argument.
        if "options" in kwargs and "options" not in accepted:
            return fn(*args, kwargs["options"])
        raise


def _reader_options(pycolmap, camera_model: str):
    if not hasattr(pycolmap, "ImageReaderOptions"):
        return None
    options = pycolmap.ImageReaderOptions()
    _assign(options, {"camera_model": camera_model, "single_camera": True})
    return options


def _extraction_options(pycolmap, preset: dict, use_gpu: bool):
    if not hasattr(pycolmap, "FeatureExtractionOptions"):
        return None
    options = pycolmap.FeatureExtractionOptions()
    apply_pycolmap_extraction(
        options,
        max_image_size=int(preset["sift_max_image_size"]),
        max_num_features=int(preset["max_num_features"]),
        use_gpu=use_gpu,
    )
    return options


def _matching_options(pycolmap, use_gpu: bool):
    if not hasattr(pycolmap, "FeatureMatchingOptions"):
        return None
    options = pycolmap.FeatureMatchingOptions()
    apply_pycolmap_matching(options, use_gpu=use_gpu)
    return options


def _sequential_options(pycolmap, overlap: int):
    if not hasattr(pycolmap, "SequentialMatchingOptions"):
        return None
    options = pycolmap.SequentialMatchingOptions()
    _assign(options, {"overlap": overlap, "quadratic_overlap": True})
    return options


def _pipeline_options(pycolmap, preset: dict):
    options = None
    if hasattr(pycolmap, "IncrementalPipelineOptions"):
        options = pycolmap.IncrementalPipelineOptions()
    elif hasattr(pycolmap, "IncrementalMapperOptions"):
        options = pycolmap.IncrementalMapperOptions()
    if options is None:
        return None
    values = {
        "min_model_size": 3,
        "multiple_models": True,
        "init_min_num_inliers": int(preset["init_min_num_inliers"]),
        "init_min_tri_angle": float(preset["init_min_tri_angle"]),
        "ba_refine_principal_point": False,
    }
    _assign(options, values)
    mapper = getattr(options, "mapper", None)
    if mapper is not None:
        _assign(mapper, values)
    return options


def _log_intrinsics(reconstruction: Reconstruction) -> None:
    from video2mesh.colmap_io import named_intrinsics

    for camera in reconstruction.cameras.values():
        named = named_intrinsics(camera.model, camera.params)
        logger.info(
            "Camera %s %s %dx%d %s",
            camera.id,
            camera.model,
            camera.width,
            camera.height,
            ", ".join(f"{key}={value:.4g}" for key, value in named.items()),
        )
        fx = named.get("fx")
        fy = named.get("fy", fx)
        if fx and fy and min(fx, fy) > 0 and max(fx, fy) / min(fx, fy) > 1.3:
            logger.warning(
                "fx and fy differ by more than 30%. Unless the lens is anamorphic, "
                "rerun with --camera-model SIMPLE_RADIAL so both axes share one focal length."
            )
        long_side = max(camera.width, camera.height)
        if fx and long_side and fx < 0.25 * long_side:
            logger.warning(
                "Estimated focal length %.1f px is very short for a %dx%d frame. "
                "The clip may need more sideways motion or a stronger texture.",
                fx,
                camera.width,
                camera.height,
            )


def _stage_start(progress: RunProgress | None, name: str) -> None:
    if progress is not None:
        progress.start(name)


def _stage_finish(progress: RunProgress | None, detail: str = "") -> None:
    if progress is not None:
        progress.finish(detail)


def _assign(obj, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            continue
        try:
            setattr(obj, key, value)
        except Exception:
            logger.debug("Could not set pycolmap option %s", key)
