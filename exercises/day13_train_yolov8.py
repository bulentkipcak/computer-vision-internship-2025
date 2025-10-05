from ultralytics import YOLO
import os

os.makedirs("outputs/day13", exist_ok=True)

model = YOLO("yolov8n.pt")
results = model.train(
    data="dataset/detect/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    project="outputs/day13",
    name="yolov8n_shapes"
)

metrics = model.val(data="dataset/detect/data.yaml", imgsz=640)
print(metrics)