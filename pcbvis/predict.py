"""在测试集上推理并输出带框图 + JSON。

imgsz 是本项目最关键的调节旋钮：分辨率降低 -> 时延下降、传输字节下降，
但小缺陷（约占图宽 2~3%）会先被牺牲，精度随之下降。
因此这里把 imgsz 做成一级参数，并逐图记录推理时延，
为第二阶段「分辨率 / 时延 / 带宽」权衡直接备好数据。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from . import paths
from .train import resolve_device


def default_weights() -> Path:
    """优先用微调后的权重，否则回退到预训练权重（效果会很差，仅用于冒烟测试）。"""
    best = paths.RESULTS_TRAIN / "pcb_yolo11n" / "weights" / "best.pt"
    if best.exists():
        return best
    return paths.MODELS_DIR / "yolo11n.pt"


def run(
    weights: Path | str | None = None,
    imgsz: int = 640,
    conf: float = 0.25,
    split: str = "test",
    device: str | None = None,
    limit: int | None = None,
) -> Path:
    """跑推理，返回输出目录。"""
    from ultralytics import YOLO

    paths.ensure_dirs()

    img_dir = paths.DATA_YOLO / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到 {img_dir}，请先运行: python -m pcbvis prepare")

    wpath = Path(weights) if weights else default_weights()
    if not wpath.exists():
        raise FileNotFoundError(f"权重不存在: {wpath}，请先运行: python -m pcbvis train")

    dev = resolve_device(device)
    out_dir = paths.RESULTS_PREDICT / f"imgsz{imgsz}"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if limit:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"{img_dir} 下没有图片")

    print("=" * 62)
    print(f"推理: imgsz={imgsz} conf={conf} split={split} device={dev}")
    print(f"  权重   : {wpath}")
    print(f"  图片数 : {len(images)}")
    print("=" * 62)

    model = YOLO(str(wpath))

    # 预热：首次推理包含 MPS/权重加载开销，先跑一次不计时
    model.predict(source=str(images[0]), imgsz=imgsz, conf=conf, device=dev, verbose=False)

    records = []
    latencies: list[float] = []

    for img_path in tqdm(images, desc=f"[predict] imgsz={imgsz}", unit="img"):
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_path), imgsz=imgsz, conf=conf, device=dev, verbose=False
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        r = results[0]
        names = r.names
        dets = []
        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                dets.append(
                    {
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "confidence": round(float(b.conf.item()), 4),
                        "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                    }
                )

        # 画框 + 叠加本次推理信息，便于肉眼对比不同 imgsz
        annotated = r.plot()
        h = annotated.shape[0]
        scale = max(0.8, h / 1200.0)
        cv2.putText(
            annotated,
            f"imgsz={imgsz}  conf>={conf}  {dt_ms:.0f} ms  dets={len(dets)}",
            (14, int(34 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            int(6 * scale),
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"imgsz={imgsz}  conf>={conf}  {dt_ms:.0f} ms  dets={len(dets)}",
            (14, int(34 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            int(2 * scale),
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        records.append(
            {
                "image": img_path.name,
                "imgsz": imgsz,
                "latency_ms": round(dt_ms, 2),
                "num_detections": len(dets),
                "detections": dets,
            }
        )

    latencies.sort()
    n = len(latencies)
    summary = {
        "imgsz": imgsz,
        "conf": conf,
        "split": split,
        "weights": str(wpath),
        "device": dev,
        "num_images": n,
        "latency_ms": {
            "mean": round(sum(latencies) / n, 2),
            "p50": round(latencies[n // 2], 2),
            "p95": round(latencies[min(int(n * 0.95), n - 1)], 2),
            "min": round(latencies[0], 2),
            "max": round(latencies[-1], 2),
        },
        "total_detections": sum(r["num_detections"] for r in records),
    }

    (out_dir / "predictions.json").write_text(
        json.dumps({"summary": summary, "predictions": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print(f"imgsz={imgsz} 平均时延 {summary['latency_ms']['mean']:.1f} ms "
          f"(p95 {summary['latency_ms']['p95']:.1f} ms)")
    print(f"共检出 {summary['total_detections']} 个目标")
    print(f"输出目录: {out_dir}")
    print("=" * 62 + "\n")
    return out_dir
