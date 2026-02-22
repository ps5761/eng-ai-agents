# Assignment 2: Image-to-Video Semantic Retrieval

Given a car part image, retrieve video clips where that part appears.

## Pipeline

1. `src/download_video.py` — Download video via yt-dlp (H.264)
2. `src/extract_frames.py` — Sample frames at 1 FPS with ffmpeg
3. `src/detect.py` — Run YOLOv8n-seg car parts detector → Parquet
4. `src/retrieve.py` — Match query images against detection index
5. `src/upload_hf.py` — Push to HuggingFace

## Run

```bash
cd assignments/assignment2
python src/pipeline.py
```

## Model

YOLOv8n-seg fine-tuned on [carparts-seg](https://docs.ultralytics.com/datasets/segment/carparts-seg/) (5 epochs, Colab T4). Place weights at `models/best.pt`.

## Output

- **Detection index:** [huggingface.co/datasets/PrashanthNYU/car-parts-retrieval](https://huggingface.co/datasets/PrashanthNYU/car-parts-retrieval)
- 7,233 detections, 21 car part classes, 2,794 frames