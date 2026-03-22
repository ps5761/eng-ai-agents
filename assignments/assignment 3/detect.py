import os
import sys
import json
import cv2
from ultralytics import YOLO

def detect_drones(video_dir, model_path, output_dir="detections", conf=0.3, fps=5):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    all_results = {}

    for fname in sorted(os.listdir(video_dir)):
        if not fname.endswith(".mp4"):
            continue
        video_path = os.path.join(video_dir, fname)
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(orig_fps / fps))
        
        video_name = os.path.splitext(fname)[0]
        video_detections = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                results = model(frame, conf=conf, verbose=False)[0]
                boxes = results.boxes
                if len(boxes) > 0:
                    # save frame
                    out_name = f"{video_name}_frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, out_name), frame)
                    
                    # store detection info
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        video_detections.append({
                            "frame_idx": frame_idx,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(box.conf[0]),
                            "class": int(box.cls[0])
                        })
            frame_idx += 1
        cap.release()
        all_results[video_name] = video_detections
        print(f"{video_name}: {len(video_detections)} detections")

    # save metadata
    with open(os.path.join(output_dir, "detections.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "videos"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "best.pt"
    detect_drones(video_dir, model_path)
