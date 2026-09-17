from pathlib import Path

from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent
model = YOLO(project_root / "runs/detect/yolo_learning/runs/learn_10ep/weights/best.pt")

model.predict(
    source=project_root / "data/pcb_yolo/images/train/missing_hole02.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)