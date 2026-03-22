from roboflow import Roboflow
from ultralytics import YOLO

# 1. Download dataset — go to universe.roboflow.com, search "drone detection",
#    pick a dataset, click "Download Dataset" > YOLOv8 format, and it gives you this snippet:
rf = Roboflow(api_key="TflR4XJVum73dTtrjvdN")
project = rf.workspace("drone-detection-i4yej").project("drone-detection-lzvig")
version = project.version(4)
dataset = version.download("yolov8")

# 2. Fine-tune
model = YOLO("yolov8n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=20, imgsz=640, batch=16)