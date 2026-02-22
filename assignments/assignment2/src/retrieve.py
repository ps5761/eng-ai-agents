"""
Step 4: Image-to-video semantic retrieval.

Given a query image, detect parts in it with the same model,
then find matching temporal segments in the video detection index.

Usage:
    python retrieve.py                  # run against full HF dataset
    python retrieve.py path/to/img.jpg  # single image query
"""
import os
import sys
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "outputs" / "detections.parquet"
OUTPUT_PATH = BASE_DIR / "outputs" / "retrieval_results.parquet"

MODEL_PATH = BASE_DIR / "models" / "best.pt"
FALLBACK_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25
GAP_THRESHOLD_SEC = 5  # max gap before splitting into separate clips


def load_model():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))
    return YOLO(FALLBACK_MODEL)


def load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Detection index not found at {INDEX_PATH}. Run detect.py first.")
    return pd.read_parquet(str(INDEX_PATH))


def detect_query_labels(model, image) -> list[dict]:
    """Run detector on a query image. Returns list of {class_label, confidence}."""
    results = model(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
    labels = []
    for box in results[0].boxes:
        labels.append({
            "class_label": model.names[int(box.cls)],
            "confidence": float(box.conf),
        })
    return labels


def merge_contiguous_segments(timestamps: list[float], gap: float) -> list[dict]:
    """
    Group sorted timestamps into contiguous segments.
    A new segment starts when the gap between consecutive timestamps
    exceeds `gap` seconds.
    """
    if not timestamps:
        return []

    segments = []
    start = timestamps[0]
    prev = timestamps[0]
    count = 1

    for t in timestamps[1:]:
        if t - prev <= gap:
            prev = t
            count += 1
        else:
            segments.append({
                "start_timestamp": start,
                "end_timestamp": prev,
                "num_supporting_detections": count,
            })
            start = t
            prev = t
            count = 1

    segments.append({
        "start_timestamp": start,
        "end_timestamp": prev,
        "num_supporting_detections": count,
    })
    return segments


def retrieve_clips(query_labels: list[dict], index_df: pd.DataFrame) -> list[dict]:
    """For each detected class in the query, find contiguous video segments."""
    results = []
    seen_labels = set()

    for det in query_labels:
        label = det["class_label"]
        if label in seen_labels:
            continue
        seen_labels.add(label)

        matches = index_df[index_df["class_label"] == label].sort_values("timestamp_sec")
        timestamps = matches["timestamp_sec"].tolist()
        segments = merge_contiguous_segments(timestamps, GAP_THRESHOLD_SEC)

        for seg in segments:
            seg["class_label"] = label
            seg["youtube_verify_url"] = (
                f"https://www.youtube.com/embed/YcvECxtXoxQ"
                f"?start={int(seg['start_timestamp'])}&end={int(seg['end_timestamp']) + 1}"
            )
            results.append(seg)

    return results


# ── Full HF dataset retrieval ───────────────────────────────────────────────

def run_retrieval_on_hf_dataset():
    """Run retrieval against every image in the HuggingFace query dataset."""
    model = load_model()
    index_df = load_index()

    print("[load] Loading query dataset from HuggingFace...")
    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")

    all_results = []

    for i, row in enumerate(ds):
        image = row["image"]
        query_labels = detect_query_labels(model, image)
        if not query_labels:
            continue

        clips = retrieve_clips(query_labels, index_df)
        for clip in clips:
            clip["query_index"] = i
            clip["query_timestamp"] = row.get("timestamp", "")
            all_results.append(clip)

    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(str(OUTPUT_PATH), index=False)
    print(f"[done] {len(results_df)} retrieval results → {OUTPUT_PATH}")
    return results_df


# ── Single image query ──────────────────────────────────────────────────────

def run_single_query(image_path: str):
    """Run retrieval for one query image (for testing / demo)."""
    model = load_model()
    index_df = load_index()

    image = Image.open(image_path)
    query_labels = detect_query_labels(model, image)

    print(f"[query] Detected in query image: {[d['class_label'] for d in query_labels]}")

    clips = retrieve_clips(query_labels, index_df)
    for clip in clips:
        print(
            f"  [{clip['class_label']}] "
            f"{clip['start_timestamp']:.1f}s – {clip['end_timestamp']:.1f}s "
            f"({clip['num_supporting_detections']} detections)  "
            f"{clip['youtube_verify_url']}"
        )
    return clips


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single_query(sys.argv[1])
    else:
        run_retrieval_on_hf_dataset()