"""Write a synthetic room video that video2mesh can reconstruct."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video2mesh.sample_scene import write_sample_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("sample_room.mp4"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    path = write_sample_video(
        args.output,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(path)


if __name__ == "__main__":
    main()
