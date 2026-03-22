import os
import sys
import json
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

def make_kalman_filter(initial_pos):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # state: [cx, cy, vx, vy]
    kf.x = np.array([initial_pos[0], initial_pos[1], 0., 0.])
    
    # state transition (constant velocity)
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    # measurement function (observe position only)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)
    
    # process noise
    kf.Q = np.diag([10., 10., 5., 5.])
    
    # measurement noise
    kf.R = np.diag([20., 20.])
    
    # initial covariance
    kf.P *= 100.
    
    return kf

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def track_and_render(video_dir, detections_file, output_dir="outputs", max_miss=15):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(detections_file) as f:
        all_detections = json.load(f)
    
    for fname in sorted(os.listdir(video_dir)):
        if not fname.endswith(".mp4"):
            continue
        video_name = os.path.splitext(fname)[0]
        if video_name not in all_detections:
            print(f"No detections for {video_name}, skipping")
            continue
        
        # group detections by frame
        dets_by_frame = {}
        for d in all_detections[video_name]:
            fidx = d["frame_idx"]
            if fidx not in dets_by_frame:
                dets_by_frame[fidx] = []
            dets_by_frame[fidx].append(d)
        
        video_path = os.path.join(video_dir, fname)
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # we'll use the same frame_interval as detection
        frame_interval = max(1, int(orig_fps / 5))
        
        out_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(out_path, fourcc, 5, (w, h))
        
        kf = None
        trajectory = []
        miss_count = 0
        frame_idx = 0
        drone_present_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            detection = None
            if frame_idx in dets_by_frame:
                # take highest confidence detection
                best = max(dets_by_frame[frame_idx], key=lambda d: d["confidence"])
                detection = best["bbox"]
            
            if kf is None and detection is not None:
                # initialize tracker
                center = bbox_center(detection)
                kf = make_kalman_filter(center)
                trajectory.append((int(center[0]), int(center[1])))
                miss_count = 0
            elif kf is not None:
                kf.predict()
                if detection is not None:
                    center = bbox_center(detection)
                    kf.update(np.array(center))
                    miss_count = 0
                else:
                    miss_count += 1
                
                est = kf.x[:2]
                trajectory.append((int(est[0]), int(est[1])))
                
                if miss_count > max_miss:
                    kf = None
                    trajectory = []
                    miss_count = 0
            
            # only write frames where drone is present (detected or tracked)
            if kf is not None or detection is not None:
                # draw bbox if we have a detection
                if detection is not None:
                    x1, y1, x2, y2 = [int(v) for v in detection]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "drone", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # draw trajectory polyline
                if len(trajectory) > 1:
                    pts = np.array(trajectory, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], False, (0, 0, 255), 2)
                
                # draw current estimated position
                if kf is not None:
                    cx, cy = int(kf.x[0]), int(kf.x[1])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
                out_video.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out_video.release()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "videos"
    det_file = sys.argv[2] if len(sys.argv) > 2 else "detections/detections.json"
    track_and_render(video_dir, det_file)