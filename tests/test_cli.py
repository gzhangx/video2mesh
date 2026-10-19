"""CLI parsing and the synthetic room renderer."""

import numpy as np

from video2mesh.cli import main
from video2mesh.sample_scene import render_frame, render_sequence


def test_help_exits_zero():
    try:
        code = main(["--help"])
    except SystemExit as exc:
        code = exc.code
    assert code == 0


def test_reconstruct_help():
    try:
        code = main(["reconstruct", "--help"])
    except SystemExit as exc:
        code = exc.code
    assert code == 0


def test_missing_video_is_a_clean_error(tmp_path):
    code = main(
        [
            "reconstruct",
            str(tmp_path / "nope.mp4"),
            "-o",
            str(tmp_path / "out"),
            "--max-frames",
            "5",
        ]
    )
    assert code == 1


def test_sample_frame_is_textured():
    frames = render_sequence(num_frames=2, width=160, height=120)
    assert len(frames) == 2
    assert frames[0].shape == (120, 160, 3)
    # A flat image would have tiny variance. The room pattern should not.
    assert float(frames[0].std()) > 20
    assert not np.array_equal(frames[0], frames[1])
    single = render_frame(
        80,
        60,
        np.array([0.0, 1.1, 1.2]),
        np.array([0.0, 0.4, 0.0]),
        96.0,
        [
            {
                "bmin": np.array([-1.0, 0.0, -1.0]),
                "bmax": np.array([1.0, 2.0, 1.0]),
                "texture": 0,
                "inside": True,
            }
        ],
        [np.full((32, 32, 3), 180, dtype=np.uint8)],
    )
    assert single.shape == (60, 80, 3)
