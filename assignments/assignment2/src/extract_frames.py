"""
Step 2: Extract frames from the input video using ffmpeg.
Uses ffmpeg subprocess for reliable codec handling (avoids OpenCV AV1 issues).
Saves JPEGs to data/frames/ as frame_%04d.jpg (1-indexed by ffmpeg).

Naming: frame_0001.jpg = 1st second, frame_0002.jpg = 2nd second, etc.
Timestamp (sec) = int(filename number) - 1  (since ffmpeg 1-indexes)
"""
import subprocess
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "input_video.mp4"
FRAMES_DIR = BASE_DIR / "data" / "frames"
SAMPLE_FPS = 1  # 1 frame per second


def extract_frames():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Cannot find {VIDEO_PATH}. Run download_video.py first.")

    print(f"[info] Extracting frames at {SAMPLE_FPS} fps using ffmpeg...")
    print(f"[info] Source: {VIDEO_PATH}")
    print(f"[info] Output: {FRAMES_DIR}/")

    subprocess.run(
        [
            "ffmpeg",
            "-i", str(VIDEO_PATH),
            "-vf", f"fps={SAMPLE_FPS}",
            "-q:v", "2",          # high quality JPEG
            "-y",                  # overwrite existing
            str(FRAMES_DIR / "frame_%04d.jpg"),
        ],
        check=True,
    )

    count = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    print(f"[done] Saved {count} frames to {FRAMES_DIR}/")


if __name__ == "__main__":
    extract_frames()