"""Stage banners, COLMAP progress lines, and subprocess heartbeats."""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from video2mesh.deps import run_command
from video2mesh.errors import ReconstructionError
from video2mesh.frames import extract_frames
from video2mesh.progress import (
    RunProgress,
    colmap_progress_line,
    format_duration,
    should_log_tick,
)


def _enable(caplog) -> None:
    logging.getLogger("video2mesh").propagate = True
    caplog.set_level(logging.INFO, logger="video2mesh")


def test_stage_banner_and_failure_hint(caplog):
    _enable(caplog)
    progress = RunProgress()
    progress.start("matching")
    assert any("=== Stage 3 of 8: matching ===" in record.getMessage() for record in caplog.records)
    message = progress.failure_message(ReconstructionError("Command failed with exit code 1: colmap sequential_matcher"))
    assert "Stage 3 of 8 (matching) failed" in message
    assert "colmap sequential_matcher" in message
    assert "Next:" in message
    assert "max-frames" in message


def test_summary_includes_skipped_stage_and_totals():
    progress = RunProgress()
    progress.skip("undistort", "Dense reconstruction is off, so undistortion is skipped.")
    text = progress.render_summary(["frames extracted: 16", "mesh.ply: /tmp/mesh.ply"])
    assert text.startswith("video2mesh finished in ")
    assert "undistort" in text
    assert "skipped" in text
    assert "frames extracted: 16" in text
    assert "mesh.ply: /tmp/mesh.ply" in text


def test_failure_between_stages_names_the_next_one():
    progress = RunProgress()
    progress.start("frame extraction")
    progress.finish("4 frames")
    message = progress.failure_message(RuntimeError("pycolmap is not installed"))
    assert "before stage 2 of 8 (feature extraction)" in message
    assert "Next:" in message
    assert "pycolmap is not installed" in message


def test_setup_failure_before_any_stage():
    message = RunProgress().failure_message(RuntimeError("COLMAP was not found"))
    assert message.startswith("Failed during setup, before frame extraction.")
    assert "COLMAP was not found" in message
    assert "Next:" in message


def test_colmap_progress_lines():
    assert colmap_progress_line("Processed file [42/150]") == "processed image 42/150"
    assert colmap_progress_line("Matching block [3/10, 1/2]") == "matching block 3/10"
    assert colmap_progress_line("Registering image #7 (num_reg_frames=6)") == "registering image #7"
    assert colmap_progress_line("  Features: 2048") == "features: 2048"
    assert colmap_progress_line("Elapsed time: 1.250 [minutes]") == "COLMAP elapsed 1.250 min"
    assert colmap_progress_line("I20260101 something routine") is None
    assert "unrecognised option" in colmap_progress_line("Failed to parse options - unrecognised option")


def test_should_log_tick_covers_ends_and_samples():
    assert should_log_tick(0, 150) is False
    assert should_log_tick(1, 150) is True
    assert should_log_tick(2, 150) is False
    assert should_log_tick(150, 150) is True
    assert format_duration(4.2) == "4.2s"
    assert format_duration(65) == "1m 05s"


def test_failed_command_names_the_argv():
    with pytest.raises(ReconstructionError) as caught:
        run_command(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            echo="silent",
            label="boom",
        )
    text = str(caught.value)
    assert "exit code 3" in text
    assert "sys.exit(3)" in text


def test_quiet_subprocess_emits_a_heartbeat(caplog):
    _enable(caplog)
    run_command(
        [sys.executable, "-c", "import time; time.sleep(1.3)"],
        echo="progress",
        label="unit sleep",
        heartbeat_seconds=0.4,
    )
    assert any("still working on unit sleep" in record.getMessage() for record in caplog.records)
    assert any("running unit sleep" in record.getMessage() for record in caplog.records)


def test_opencv_extract_logs_frame_counts(tmp_path: Path, caplog):
    _enable(caplog)
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (80, 64),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(1)
    for _ in range(4):
        writer.write(rng.integers(0, 255, size=(64, 80, 3), dtype=np.uint8))
    writer.release()

    extraction = extract_frames(
        video,
        tmp_path / "frames",
        fps=10,
        max_frames=10,
        blur_threshold=0,
        max_image_size=200,
        ffmpeg_exe=None,
        use_ffmpeg=False,
    )
    assert len(extraction.frames) == 4
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("extracted ") and "/4" in message for message in messages)
    assert any(message.startswith("scored ") for message in messages)
