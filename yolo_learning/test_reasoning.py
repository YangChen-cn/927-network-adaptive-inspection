from ultralytics import YOLO
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO(
    project_root
    / "runs/detect/yolo_learning/runs/learn_10ep/weights/best.pt"
)

images = sorted(
    (project_root / "data/pcb_yolo/images/test").glob("*.jpg")
)

assert images, "test 目录里没有找到 jpg"

source = str(images[0])
print("Testing image:", source)

results = model.predict(
    source=source,
    imgsz=640,
    conf=0.25,
    device=device,
    save=True,
    project=project_root / "yolo_learning/runs",
    name="predict_640",
)

for r in results:
    print("speed(ms):", r.speed)
    print("detected boxes:", len(r.boxes))