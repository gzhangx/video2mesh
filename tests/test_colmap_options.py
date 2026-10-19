"""COLMAP 3.x and 4.x option names for the same logical setting."""

import logging

import cv2
import numpy as np
import pytest

from video2mesh.colmap_options import (
    FEATURE_MAX_IMAGE_SIZE,
    FEATURE_MAX_NUM_FEATURES,
    FEATURE_USE_GPU,
    MAPPER_BA_GLOBAL_IMAGES_FREQ,
    MAPPER_BA_GLOBAL_IMAGES_RATIO,
    MAPPER_BA_MIN_RESIDUALS_THREADS,
    MAPPER_SNAPSHOT_IMAGES_FREQ,
    MATCH_GUIDED,
    MATCH_MAX_NUM_MATCHES,
    MATCH_USE_GPU,
    STEREO_INPUT_TYPE,
    ColmapCli,
    apply_pycolmap_extraction,
    apply_pycolmap_matching,
    format_colmap_value,
    option,
    parse_colmap_help,
)
from video2mesh.deps import command_argv, find_colmap, run_command

HELP_3 = """
COLMAP 3.9.1 (Commit Unknown on Unknown without CUDA)
  -h [ --help ]
  --database_path arg
  --image_path arg
  --ImageReader.single_camera arg (=0)
  --ImageReader.camera_model arg (=SIMPLE_RADIAL)
  --SiftExtraction.num_threads arg (=-1)
  --SiftExtraction.use_gpu arg (=1)
  --SiftExtraction.gpu_index arg (=-1)
  --SiftExtraction.max_image_size arg (=3200)
  --SiftExtraction.max_num_features arg (=8192)
  --SiftMatching.use_gpu arg (=1)
  --SiftMatching.guided_matching arg (=0)
  --SiftMatching.max_num_matches arg (=32768)
  --SequentialMatching.overlap arg (=10)
  --SequentialMatching.quadratic_overlap arg (=1)
  --Mapper.min_model_size arg (=10)
  --Mapper.multiple_models arg (=1)
  --Mapper.init_min_num_inliers arg (=100)
  --Mapper.init_min_tri_angle arg (=16)
  --Mapper.init_max_forward_motion arg (=0.95)
  --Mapper.abs_pose_min_num_inliers arg (=30)
  --Mapper.ba_refine_principal_point arg (=0)
  --Mapper.ba_global_images_ratio arg (=1.1)
  --Mapper.ba_global_images_freq arg (=500)
  --Mapper.snapshot_images_freq arg (=0)
  --Mapper.ba_min_num_residuals_for_multi_threading arg (=50000)
  --input_path arg
  --output_path arg
  --output_type arg (=COLMAP)
  --max_image_size arg (=-1)
  --workspace_path arg
  --workspace_format arg (=COLMAP)
  --PatchMatchStereo.geom_consistency arg (=0)
  --input_type arg (=geometric)
"""

HELP_4 = """
COLMAP 4.2.1 (Commit abc on 2026-08-01 with CUDA)
  -h [ --help ]
  --database_path arg
  --image_path arg
  --ImageReader.single_camera arg (=0)
  --ImageReader.camera_model arg (=SIMPLE_RADIAL)
  --FeatureExtraction.type arg (=SIFT)
  --FeatureExtraction.max_image_size arg (=3200)
  --FeatureExtraction.num_threads arg (=-1)
  --FeatureExtraction.use_gpu arg (=1)
  --FeatureExtraction.gpu_index arg (=-1)
  --SiftExtraction.max_num_features arg (=8192)
  --FeatureMatching.type arg (=SIFT)
  --FeatureMatching.use_gpu arg (=1)
  --FeatureMatching.guided_matching arg (=0)
  --FeatureMatching.max_num_matches arg (=32768)
  --SiftMatching.max_ratio arg (=0.8)
  --SiftMatching.cross_check arg (=1)
  --SequentialMatching.overlap arg (=10)
  --SequentialMatching.quadratic_overlap arg (=1)
  --Mapper.min_model_size arg (=10)
  --Mapper.multiple_models arg (=1)
  --Mapper.init_min_num_inliers arg (=100)
  --Mapper.init_min_tri_angle arg (=16)
  --Mapper.init_max_forward_motion arg (=0.95)
  --Mapper.abs_pose_min_num_inliers arg (=30)
  --Mapper.ba_refine_principal_point arg (=0)
  --Mapper.ba_global_frames_ratio arg (=1.1)
  --Mapper.ba_global_frames_freq arg (=500)
  --Mapper.snapshot_frames_freq arg (=0)
  --Mapper.ba_min_num_residuals_for_cpu_multi_threading arg (=50000)
  --input_path arg
  --output_path arg
  --output_type arg (=COLMAP)
  --max_image_size arg (=-1)
  --workspace_path arg
  --workspace_format arg (=COLMAP)
  --PatchMatchStereo.geom_consistency arg (=0)
  --input_type arg (=geometric)
"""

# Same flag lists the real subcommands share. The command name only selects
# which cached help blob is consulted.
_COMMANDS = (
    "feature_extractor",
    "sequential_matcher",
    "exhaustive_matcher",
    "mapper",
    "image_undistorter",
    "model_converter",
    "patch_match_stereo",
    "stereo_fusion",
)


def _cli(help_text: str) -> ColmapCli:
    return ColmapCli("colmap", {name: help_text for name in _COMMANDS})


def test_parse_ignores_flags_mentioned_in_prose():
    text = (
        "  --FeatureExtraction.max_image_size arg (=3200)\n"
        "Descriptions may say --SiftExtraction.max_image_size was removed.\n"
    )
    assert parse_colmap_help(text) == {"FeatureExtraction.max_image_size"}


def test_bool_values_are_numeric():
    assert format_colmap_value(True) == "1"
    assert format_colmap_value(False) == "0"


def test_feature_extractor_uses_sift_flags_on_colmap_3():
    argv = ["feature_extractor"]
    _cli(HELP_3).add(
        argv,
        "feature_extractor",
        [
            option("feature_max_image_size", FEATURE_MAX_IMAGE_SIZE, 800),
            option("feature_max_num_features", FEATURE_MAX_NUM_FEATURES, 4096),
            option("feature_use_gpu", FEATURE_USE_GPU, "0"),
        ],
    )
    assert argv == [
        "feature_extractor",
        "--SiftExtraction.max_image_size",
        "800",
        "--SiftExtraction.max_num_features",
        "4096",
        "--SiftExtraction.use_gpu",
        "0",
    ]


def test_feature_extractor_uses_feature_flags_on_colmap_4():
    argv = ["feature_extractor"]
    _cli(HELP_4).add(
        argv,
        "feature_extractor",
        [
            option("feature_max_image_size", FEATURE_MAX_IMAGE_SIZE, 1600),
            option("feature_max_num_features", FEATURE_MAX_NUM_FEATURES, 8192),
            option("feature_use_gpu", FEATURE_USE_GPU, "1"),
        ],
    )
    assert "--FeatureExtraction.max_image_size" in argv
    assert "--FeatureExtraction.use_gpu" in argv
    assert "--SiftExtraction.max_num_features" in argv
    assert "--SiftExtraction.max_image_size" not in argv
    assert "--SiftExtraction.use_gpu" not in argv


def test_matchers_follow_the_same_gpu_rename():
    for command in ("sequential_matcher", "exhaustive_matcher"):
        argv = [command]
        _cli(HELP_3).add(argv, command, [option("matching_use_gpu", MATCH_USE_GPU, "0")])
        assert argv[1:] == ["--SiftMatching.use_gpu", "0"]
        argv = [command]
        _cli(HELP_4).add(
            argv,
            command,
            [
                option("matching_use_gpu", MATCH_USE_GPU, "1"),
                option("guided_matching", MATCH_GUIDED, "0"),
                option("max_num_matches", MATCH_MAX_NUM_MATCHES, 32768),
                option("sequential_overlap", "SequentialMatching.overlap", 10),
            ],
        )
        assert "--FeatureMatching.use_gpu" in argv
        assert "--FeatureMatching.guided_matching" in argv
        assert "--FeatureMatching.max_num_matches" in argv
        assert "--SequentialMatching.overlap" in argv
        assert "--SiftMatching.use_gpu" not in argv


def test_mapper_frame_renames_and_stable_init_flags():
    argv = ["mapper"]
    _cli(HELP_4).add(
        argv,
        "mapper",
        [
            option("min_model_size", "Mapper.min_model_size", "3"),
            option("init_min_num_inliers", "Mapper.init_min_num_inliers", 40),
            option("init_min_tri_angle", "Mapper.init_min_tri_angle", 8.0),
            option("ba_global_images_ratio", MAPPER_BA_GLOBAL_IMAGES_RATIO, "1.1"),
            option("ba_global_images_freq", MAPPER_BA_GLOBAL_IMAGES_FREQ, "500"),
            option("snapshot_images_freq", MAPPER_SNAPSHOT_IMAGES_FREQ, "0"),
            option("ba_min_residuals_threads", MAPPER_BA_MIN_RESIDUALS_THREADS, "50000"),
        ],
    )
    assert "--Mapper.ba_global_frames_ratio" in argv
    assert "--Mapper.ba_global_frames_freq" in argv
    assert "--Mapper.snapshot_frames_freq" in argv
    assert "--Mapper.ba_min_num_residuals_for_cpu_multi_threading" in argv
    assert "--Mapper.min_model_size" in argv
    argv = ["mapper"]
    _cli(HELP_3).add(
        argv,
        "mapper",
        [
            option("ba_global_images_ratio", MAPPER_BA_GLOBAL_IMAGES_RATIO, "1.1"),
            option("ba_global_images_freq", MAPPER_BA_GLOBAL_IMAGES_FREQ, "500"),
            option("snapshot_images_freq", MAPPER_SNAPSHOT_IMAGES_FREQ, "0"),
            option("ba_min_residuals_threads", MAPPER_BA_MIN_RESIDUALS_THREADS, "50000"),
        ],
    )
    assert "--Mapper.ba_global_images_ratio" in argv
    assert "--Mapper.ba_global_images_freq" in argv
    assert "--Mapper.snapshot_images_freq" in argv
    assert "--Mapper.ba_min_num_residuals_for_multi_threading" in argv


def test_dense_and_converter_flags_are_kept_on_both_generations():
    for help_text in (HELP_3, HELP_4):
        cli = _cli(help_text)
        undistort = ["image_undistorter"]
        cli.add(
            undistort,
            "image_undistorter",
            [
                option("output_type", "output_type", "COLMAP"),
                option("max_image_size", ("max_image_size", "Undistortion.max_image_size"), 1600),
            ],
        )
        assert undistort[1:] == ["--output_type", "COLMAP", "--max_image_size", "1600"]
        convert = ["model_converter"]
        cli.add(convert, "model_converter", [option("output_type", "output_type", "TXT")])
        assert "--output_type" in convert
        patch = ["patch_match_stereo"]
        cli.add(
            patch,
            "patch_match_stereo",
            [option("geom_consistency", "PatchMatchStereo.geom_consistency", "1")],
        )
        assert patch[1:] == ["--PatchMatchStereo.geom_consistency", "1"]
        fusion = ["stereo_fusion"]
        cli.add(fusion, "stereo_fusion", [option("input_type", STEREO_INPUT_TYPE, "geometric")])
        assert fusion[1:] == ["--input_type", "geometric"]


def test_missing_flag_is_dropped_with_a_warning(caplog):
    argv = ["feature_extractor"]
    with caplog.at_level(logging.WARNING):
        _cli(HELP_4).add(
            argv,
            "feature_extractor",
            [option("not_a_real_knob", ("Gone.option", "Also.gone"), "1")],
        )
    assert argv == ["feature_extractor"]
    assert "no flag for not_a_real_knob" in caplog.text


def test_pycolmap_extraction_shapes():
    class Sift3:
        max_image_size = 0
        max_num_features = 0
        use_gpu = True

    class Flat3:
        sift = Sift3()

    applied = apply_pycolmap_extraction(Flat3(), max_image_size=800, max_num_features=1000, use_gpu=False)
    assert Flat3.sift.max_image_size == 800
    assert Flat3.sift.max_num_features == 1000
    assert Flat3.sift.use_gpu is False
    assert "sift.max_image_size" in applied

    class Sift4:
        max_num_features = 0

    class Parent4:
        max_image_size = 0
        use_gpu = True
        sift = Sift4()

    parent = Parent4()
    apply_pycolmap_extraction(parent, max_image_size=1600, max_num_features=2048, use_gpu=False)
    assert parent.max_image_size == 1600
    assert parent.use_gpu is False
    assert parent.sift.max_num_features == 2048
    assert not hasattr(parent.sift, "max_image_size") or parent.sift.max_num_features == 2048

    class Match3:
        class _Sift:
            use_gpu = True

        sift = _Sift()

    match = Match3()
    apply_pycolmap_matching(match, use_gpu=False)
    assert match.sift.use_gpu is False

    class Match4:
        use_gpu = True

        class _Sift:
            max_ratio = 0.8

        sift = _Sift()

    match4 = Match4()
    apply_pycolmap_matching(match4, use_gpu=False)
    assert match4.use_gpu is False


def test_installed_colmap_help_selects_a_real_flag():
    exe = find_colmap(None)
    if exe is None:
        pytest.skip("colmap is not installed")
    cli = ColmapCli(exe)
    argv = ["feature_extractor"]
    cli.add(
        argv,
        "feature_extractor",
        [
            option("feature_max_image_size", FEATURE_MAX_IMAGE_SIZE, 640),
            option("feature_use_gpu", FEATURE_USE_GPU, "0"),
            option("feature_max_num_features", FEATURE_MAX_NUM_FEATURES, 512),
        ],
    )
    flags = set(argv[1::2])
    assert flags & {"--FeatureExtraction.max_image_size", "--SiftExtraction.max_image_size"}
    assert flags & {"--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"}
    assert "--SiftExtraction.max_num_features" in flags or "--FeatureExtraction.max_num_features" in flags
    for command in ("sequential_matcher", "exhaustive_matcher", "mapper", "image_undistorter", "model_converter"):
        assert cli.flags(command), f"colmap {command} -h listed no options"


def test_installed_feature_extractor_accepts_mapped_flags(tmp_path):
    exe = find_colmap(None)
    if exe is None:
        pytest.skip("colmap is not installed")
    images = tmp_path / "images"
    images.mkdir()
    image = np.random.default_rng(0).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images / "frame_000001.jpg"), image)
    cli = ColmapCli(exe)
    argv = ["feature_extractor"]
    cli.add(
        argv,
        "feature_extractor",
        [
            option("database_path", "database_path", tmp_path / "database.db"),
            option("image_path", "image_path", images),
            option("single_camera", "ImageReader.single_camera", "1"),
            option("camera_model", "ImageReader.camera_model", "SIMPLE_RADIAL"),
            option("feature_max_image_size", FEATURE_MAX_IMAGE_SIZE, 640),
            option("feature_max_num_features", FEATURE_MAX_NUM_FEATURES, 256),
            option("feature_use_gpu", FEATURE_USE_GPU, "0"),
        ],
    )
    run_command(command_argv(exe, argv), echo="silent")
    assert (tmp_path / "database.db").is_file()
