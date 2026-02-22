"""
Step 3: Run YOLOv8 car-parts detector on all extracted frames.
Produces outputs/detections.parquet with the required schema.

Frame naming convention (from ffmpeg fps=1):
  frame_0001.jpg = timestamp 0s
  frame_0002.jpg = timestamp 1s
  ...
"""
import os
import re
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
FRAMES_DIR = BASE_DIR / "data" / "frames"
OUTPUT_PATH = BASE_DIR / "outputs" / "detections.parquet"
VIDEO_ID = "YcvECxtXoxQ"

# --- Model config ---
MODEL_PATH = BASE_DIR / "models" / "best.pt"   # fine-tuned weights
FALLBACK_MODEL = "yolov8n.pt"                   # COCO fallback
CONFIDENCE_THRESHOLD = 0.25


def load_model():
    if MODEL_PATH.exists():
        print(f"[model] Loading fine-tuned model from {MODEL_PATH}")
        return YOLO(str(MODEL_PATH))
    else:
        print(f"[model] {MODEL_PATH} not found — falling back to {FALLBACK_MODEL}")
        return YOLO(FALLBACK_MODEL)


def parse_frame_number(filename: str) -> int:
    """Extract frame number from filename like frame_0001.jpg (ffmpeg 1-indexed)."""
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else -1


def detect_all_frames(model):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}/. Run extract_frames.py first.")

    print(f"[info] Found {len(frame_files)} frames")

    detections = []

    for fname in tqdm(frame_files, desc="Detecting"):
        frame_num = parse_frame_number(fname)
        # ffmpeg fps=1 outputs are 1-indexed: frame_0001.jpg = second 0
        timestamp_sec = frame_num - 1
        frame_path = str(FRAMES_DIR / fname)

        results = model(frame_path, verbose=False, conf=CONFIDENCE_THRESHOLD)

        for box in results[0].boxes:
            cls_id = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "video_id": VIDEO_ID,
                "frame_index": frame_num,
                "timestamp_sec": timestamp_sec,
                "class_label": model.names[cls_id],
                "x_min": round(x1, 1),
                "y_min": round(y1, 1),
                "x_max": round(x2, 1),
                "y_max": round(y2, 1),
                "confidence_score": round(float(box.conf), 4),
                "detector_name": MODEL_PATH.name if MODEL_PATH.exists() else FALLBACK_MODEL,
            })

    df = pd.DataFrame(detections)
    df = df.sort_values(["timestamp_sec", "class_label"]).reset_index(drop=True)
    df.to_parquet(str(OUTPUT_PATH), index=False)

    print(f"[done] {len(df)} detections from {len(frame_files)} frames → {OUTPUT_PATH}")
    print(f"\n[summary] Classes detected:")
    print(df["class_label"].value_counts().to_string())

    return df


if __name__ == "__main__":
    model = load_model()
    detect_all_frames(model)