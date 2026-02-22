"""
Step 1: Download the source video from YouTube.
Forces H.264 codec (avc1) to avoid AV1 decoding issues in containers.
Saves to data/input_video.mp4
"""
import subprocess
import os
from pathlib import Path

VIDEO_URL = "https://www.youtube.com/watch?v=YcvECxtXoxQ"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "input_video.mp4"


def download_video():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"[exists] Removing old video to re-download with correct codec...")
        OUTPUT_PATH.unlink()

    print(f"[download] Fetching video (H.264) from {VIDEO_URL}")
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[vcodec^=avc1]+bestaudio/best[vcodec^=avc1]/best",
            "--merge-output-format", "mp4",
            "-o", str(OUTPUT_PATH),
            VIDEO_URL,
        ],
        check=True,
    )
    print(f"[done] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    download_video()