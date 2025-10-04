from ultralytics import YOLO
import cv2
import os

os.makedirs("outputs/day12", exist_ok=True)

model = YOLO("yolov8n.pt")

img_path = "inputs/dog_and_cat.webp"
img = cv2.imread(img_path)

results = model.predict(source=img_path, save=True, project="outputs/day12", name="test")

print("Detection results:")
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"Class {cls}, Confidence {conf:.2f}, Box {xyxy}")
