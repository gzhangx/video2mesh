"""Frame sampling, blur rejection, and video extraction."""

from pathlib import Path

import cv2
import numpy as np

from video2mesh.frames import blur_score, extract_frames, uniform_indices


def test_uniform_indices_include_endpoints():
    indexes = uniform_indices(10, 4)
    assert indexes[0] == 0
    assert indexes[-1] == 9
    assert indexes == sorted(set(indexes))
    assert len(indexes) == 4


def test_blur_score_is_low_for_a_flat_image():
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    sharp[::2, ::2] = 255
    assert blur_score(flat) < 1.0
    assert blur_score(sharp) > blur_score(flat)


def test_extract_drops_blurry_frames(tmp_path: Path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (80, 64),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for index in range(6):
        if index == 2:
            frame = np.full((64, 80, 3), 90, dtype=np.uint8)
        else:
            frame = rng.integers(0, 255, size=(64, 80, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    extraction = extract_frames(
        video,
        tmp_path / "frames",
        fps=10,
        max_frames=20,
        blur_threshold=5.0,
        max_image_size=200,
        ffmpeg_exe=None,
        use_ffmpeg=False,
    )
    assert len(extraction.frames) == 5
    names = [frame.filename for frame in extraction.frames]
    assert names == [f"frame_{i:06d}.jpg" for i in range(1, 6)]
    assert all(frame.blur_score >= 5.0 for frame in extraction.frames)
    assert (tmp_path / "frames" / "frame_000001.jpg").is_file()
