"""在验证/测试集上评估精度指标。

与 predict.py 一样，imgsz 是一级参数——跑不同 imgsz 就能直接得到
「精度 vs 分辨率」曲线，这是第二阶段自适应策略的标定依据。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import paths
from .predict import default_weights
from .train import resolve_device


def run(
    weights: Path | str | None = None,
    imgsz: int = 640,
    split: str = "val",
    device: str | None = None,
) -> dict:
    """评估并返回指标 dict。"""
    from ultralytics import YOLO

    paths.ensure_dirs()

    data_yaml = paths.DATA_YOLO / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到 {data_yaml}，请先运行: python -m pcbvis prepare")

    wpath = Path(weights) if weights else default_weights()
    if not wpath.exists():
        raise FileNotFoundError(f"权重不存在: {wpath}，请先运行: python -m pcbvis train")

    dev = resolve_device(device)
    print("=" * 62)
    print(f"评估: split={split} imgsz={imgsz} device={dev}")
    print(f"  权重: {wpath}")
    print("=" * 62)

    model = YOLO(str(wpath))
    metrics = model.val(
        data=str(data_yaml), imgsz=imgsz, split=split, device=dev, verbose=False
    )

    box = metrics.box
    names = getattr(metrics, "names", {}) or {}

    # 逐类 mAP50：ap50 与 ap_class_index 一一对应
    per_class = {}
    try:
        ap50 = box.ap50.tolist() if hasattr(box.ap50, "tolist") else list(box.ap50)
        idx = (
            box.ap_class_index.tolist()
            if hasattr(box.ap_class_index, "tolist")
            else list(box.ap_class_index)
        )
        for i, c in zip(idx, ap50):
            per_class[names.get(i, str(i))] = round(float(c), 4)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] 逐类指标解析失败（不影响总体指标）: {e}")

    out = {
        "split": split,
        "imgsz": imgsz,
        "weights": str(wpath),
        "device": dev,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_mAP50": per_class,
    }

    out_dir = paths.RESULTS_EVAL
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_imgsz{imgsz}_{split}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print(f"  mAP50     : {out['mAP50']:.4f}")
    print(f"  mAP50-95  : {out['mAP50-95']:.4f}")
    print(f"  precision : {out['precision']:.4f}")
    print(f"  recall    : {out['recall']:.4f}")
    if per_class:
        print("\n  逐类 mAP50:")
        for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
            print(f"    {k:18s} {v:.4f}")
    print(f"\n  已保存 -> {out_dir / f'metrics_imgsz{imgsz}_{split}.json'}")
    print("=" * 62 + "\n")

    if out["mAP50"] < 0.1:
        print("⚠ mAP50 极低，请检查标注转换或类别映射是否正确\n")

    return out
