"""在 PCB 缺陷数据集上微调 YOLO。

用 COCO 预训练权重直接推理 PCB 图几乎检不出缺陷，所以必须微调，
否则后续「精度 vs 分辨率 vs 时延」的权衡实验没有意义（精度恒为 0）。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from . import paths
from .config import load_config

# 已验证可用的预训练权重地址（GitHub releases，无需登录）
_ASSET_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}",
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/{name}",
]


def resolve_device(pref: str | None = None) -> str:
    """选择推理设备：优先 Apple MPS，其次 CUDA，最后 CPU。"""
    if pref:
        return pref
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_weights(model_name: str = "yolo11n.pt") -> Path:
    """确保预训练权重在 models/ 下。

    显式指定路径，避免 ultralytics 把 .pt 下载到项目根目录污染仓库。
    """
    dest = paths.MODELS_DIR / model_name
    if dest.exists():
        return dest

    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for tmpl in _ASSET_URLS:
        url = tmpl.format(name=model_name)
        try:
            print(f"[weights] 下载 {url}")
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            print(f"[weights] 已保存到 {dest}")
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"[weights] 该地址失败 ({e})，尝试下一个")

    # 兜底：交给 ultralytics 自己解析（可能落到 CWD）
    print(f"[weights] 回退到 ultralytics 自动下载 {model_name}")
    from ultralytics import YOLO

    YOLO(model_name)
    fallback = Path(model_name)
    if fallback.exists():
        fallback.replace(dest)
    return dest


def run(
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    model_name: str = "yolo11n.pt",
    name: str = "pcb_yolo11n",
    workers: int = 4,
    patience: int = 30,
    config: Path | str | None = None,
) -> Path:
    """微调并返回 best.pt 路径。"""
    from ultralytics import YOLO

    cfg = load_config(config)
    paths.ensure_dirs()

    data_yaml = paths.DATA_YOLO / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"未找到 {data_yaml}，请先运行: python -m pcbvis prepare"
        )

    weights = ensure_weights(model_name)
    dev = resolve_device(device)

    print("=" * 62)
    print("开始微调")
    print("=" * 62)
    print(f"  预训练权重 : {weights}")
    print(f"  数据配置   : {data_yaml}")
    print(f"  设备       : {dev}")
    print(f"  epochs={epochs} imgsz={imgsz} batch={batch} patience={patience}")
    print()

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev,
        workers=workers,
        seed=cfg["dataset"]["seed"],
        patience=patience,
        project=str(paths.RESULTS_TRAIN),
        name=name,
        exist_ok=True,
        plots=True,
        val=True,
    )

    best = paths.RESULTS_TRAIN / name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"训练结束但未找到 {best}")

    # 记录本次训练的配置，便于论文里复现
    run_cfg = {
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": dev,
        "seed": cfg["dataset"]["seed"],
        "patience": patience,
        "data": str(data_yaml),
        "best_weights": str(best),
    }
    (paths.RESULTS_TRAIN / name / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print(f"训练完成，最优权重: {best}")
    print("=" * 62 + "\n")
    return best
