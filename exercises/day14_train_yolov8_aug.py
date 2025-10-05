from ultralytics import YOLO
import os

os.makedirs("outputs/day14", exist_ok=True)

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/detect/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    project="outputs/day14",
    name="yolov8n_aug",
    degrees=10,       
    scale=0.2,        
    flipud=0.3,     
    fliplr=0.5,     
    hsv_h=0.015,   
    hsv_s=0.7,        
    hsv_v=0.4,      
)
