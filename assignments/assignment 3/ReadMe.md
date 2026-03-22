# Assignment 3: UAV Drone Detection and Tracking

## Dataset

Roboflow Universe — [Drone Detection](https://universe.roboflow.com/drone-detection-i4yej/drone-detection-lzvig) (636 train, 77 val images). Classes: Bird, Drone, Plane. Only Drone class used during inference.

## Detector

YOLOv8n fine-tuned for 20 epochs (imgsz=640, batch=16) on Apple M2 Pro CPU. Best Drone mAP50 = 0.791, Recall = 0.782. Confidence threshold: 0.3.

## Kalman Filter

State vector: `[cx, cy, vx, vy]` — bounding box center + velocity. Constant velocity motion model. Observe position only.

Noise: Q = diag(10, 10, 5, 5), R = diag(20, 20). If drone is missing for 15+ consecutive frames, track is dropped.

## Failure Cases

Small/distant drones get missed. Constant velocity model lags on rapid maneuvers. Single-track design doesn't handle multiple drones.

## Output Videos

- [Video 1](https://youtu.be/6yRjuwiw_9M)
- [Video 2](https://youtu.be/lIyZT6gI1D4)

## HuggingFace Dataset

[PrashanthNYU/drone-detections-assignment3](https://huggingface.co/datasets/PrashanthNYU/drone-detections-assignment3)

## Run

```bash
conda activate drone
pip install -r requirements.txt
python train.py
python detect.py videos runs/detect/train2/weights/best.pt
python track.py videos detections/detections.json
```