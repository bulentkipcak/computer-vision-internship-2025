from ultralytics import YOLO
import os

run_dir = "outputs/day13/yolov8n_shapes"
best = os.path.join(run_dir, "weights", "best.pt")
model = YOLO(best)

os.makedirs("outputs/day13/preds", exist_ok=True)
model.predict(
    source="dataset/detect/val/images",
    imgsz=640,
    save=True,
    project="outputs/day13",
    name="preds"
)
print("Predictions saved.")