from ultralytics import YOLO
import os

os.makedirs("outputs/day15", exist_ok=True)

configs = [
    {"epochs": 30, "batch": 16, "imgsz": 640, "name": "yolov8_cfgA"},
    {"epochs": 50, "batch": 16, "imgsz": 640, "name": "yolov8_cfgB"},
    {"epochs": 30, "batch": 8,  "imgsz": 640, "name": "yolov8_cfgC"},
]

for cfg in configs:
    model = YOLO("yolov8n.pt")
    model.train(
        data="dataset/detect/data.yaml",
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        project="outputs/day15",
        name=cfg["name"],
        degrees=10,
        scale=0.2,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )