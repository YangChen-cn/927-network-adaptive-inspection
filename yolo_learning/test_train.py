# yolo_learning/test_train.py
from pathlib import Path

import torch
from ultralytics import YOLO

# 项目根目录 = 本文件所在目录的上一级（避免写死本机绝对路径）
project_root = Path(__file__).resolve().parent.parent

# 1. 选择设备：优先使用 Apple GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training device: {device}")

# 2. 加载通用预训练模型
model = YOLO("yolo11n.pt")

# 3. 在 PCB 数据上微调
model.train(
    data=project_root / "data/pcb_yolo/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=device,
    workers=0,              # 第一遍求稳，确认跑通后可调大
    seed=42,
    project=project_root / "yolo_learning/runs",
    name="learn_10ep"
)
