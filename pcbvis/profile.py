"""
分辨率 profiling：对每个 imgsz 一次性测出三件事——

  1. 精度  (mAP50 / mAP50-95 / P / R)，跑 val
  2. 单图推理时延，逐图计时（这是「本地推理耗时」的实测依据）
  3. JPEG 编码后的平均字节数（这是「卸载传输量」的实测依据）

三个量一起测，才能画出「精度 / 时延 / 带宽」三者的权衡曲线 —— 这是
第二阶段自适应策略的标定数据。

注意两个刻意的设计选择：
- 时延用**逐图**推理测，不用 val 的批量推理。产线上一次检查就是一张图，
  批量推理的 per-image 时间会偏乐观。
- JPEG 字节数按「长边缩放到 imgsz、保持宽高比」算，**不含 letterbox 补边**。
  补边是模型输入的内部细节，真实传输不会把黑边也发过去。
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import cv2

from . import paths
from .predict import default_weights
from .train import resolve_device

DEFAULT_IMGSZ = (320, 384, 416, 448, 480, 512, 544, 576, 608, 640)


# --------------------------------------------------------------------------- #
# JPEG 传输量
# --------------------------------------------------------------------------- #
def measure_jpeg_bytes(
    img_paths: list[Path], imgsz: int, quality: int = 85
) -> tuple[float, float, float]:
    """按长边缩放到 imgsz 后做 JPEG 编码，返回 (均值, 中位, 最大) 字节数。"""
    sizes: list[int] = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = imgsz / max(h, w)
        # 只在需要缩小时缩放；不放大（放大不会带来信息，只会浪费字节）
        if scale < 1.0:
            img = cv2.resize(
                img, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            sizes.append(int(buf.size))
    if not sizes:
        return 0.0, 0.0, 0.0
    sizes.sort()
    return (
        statistics.mean(sizes),
        float(sizes[len(sizes) // 2]),
        float(sizes[-1]),
    )


# --------------------------------------------------------------------------- #
# 单图时延
# --------------------------------------------------------------------------- #
def measure_latency(
    model, img_paths: list[Path], imgsz: int, device: str, warmup: int = 3
) -> tuple[float, float, float]:
    """逐图推理计时，返回 (均值, p50, p95) 毫秒。含 MPS 预热。

    ⚠ 单分辨率顺序测量容易被热漂移污染，跨分辨率比较请用
    measure_latency_interleaved()。
    """
    n = len(img_paths)
    if n == 0:
        return 0.0, 0.0, 0.0
    for p in img_paths[: min(warmup, n)]:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)

    times: list[float] = []
    for p in img_paths:
        t0 = time.perf_counter()
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return (
        statistics.mean(times),
        times[len(times) // 2],
        times[min(int(len(times) * 0.95), len(times) - 1)],
    )


def measure_latency_interleaved(
    model,
    img_paths: list[Path],
    imgsz_list: tuple[int, ...],
    device: str,
    rounds: int = 5,
    warmup: int = 3,
) -> dict[int, dict]:
    """交错（round-robin）测所有分辨率的时延，抵消热漂移。

    为什么必须交错：本机是 MacBook Air（无风扇），持续推理会降频。如果按
    「先测完 320、再测 384……」的顺序，后面的分辨率会被系统性拖慢，测出来的
    曲线不单调（实测 544 比 512 还快，物理上不可能）。

    做法：轮询 —— 每一轮把**所有**分辨率各测一遍，重复 rounds 轮，
    最后对每个分辨率取**跨轮中位数**。这样热漂移对所有分辨率的影响是均匀的，
    分辨率之间的相对关系才是可信的。
    """
    n = len(img_paths)
    if n == 0:
        return {}

    # 预热所有分辨率，避免首次调用（含 MPS 图编译）被算进去
    for imgsz in imgsz_list:
        for p in img_paths[: min(warmup, n)]:
            model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)

    per_round: dict[int, list[float]] = {s: [] for s in imgsz_list}
    # 同时记录 ultralytics 自己的耗时拆解（preprocess / inference / postprocess），
    # 用来回答「时延到底花在哪」——是模型算力，还是固定开销
    breakdown: dict[int, dict[str, list[float]]] = {
        s: {"preprocess": [], "inference": [], "postprocess": []} for s in imgsz_list
    }

    for r in range(rounds):
        for imgsz in imgsz_list:
            times = []
            for p in img_paths:
                t0 = time.perf_counter()
                res = model.predict(
                    source=str(p), imgsz=imgsz, device=device, verbose=False
                )
                times.append((time.perf_counter() - t0) * 1000.0)
                for k in breakdown[imgsz]:
                    v = getattr(res[0], "speed", {}).get(k)
                    if v is not None:
                        breakdown[imgsz][k].append(float(v))
            per_round[imgsz].append(statistics.mean(times))
        print(f"    [latency] 第 {r + 1}/{rounds} 轮完成")

    out: dict[int, dict] = {}
    for imgsz in imgsz_list:
        vals = sorted(per_round[imgsz])
        # 丢掉最快和最慢的一轮，再取均值——比单纯中位数更稳，且能压掉预热轮
        trimmed = vals[1:-1] if len(vals) >= 4 else vals
        bd = {
            k: round(statistics.median(v), 2) if v else None
            for k, v in breakdown[imgsz].items()
        }
        out[imgsz] = {
            "latency_mean_ms": round(statistics.mean(trimmed), 2),
            "latency_median_ms": round(vals[len(vals) // 2], 2),
            "latency_min_ms": round(vals[0], 2),
            "latency_max_ms": round(vals[-1], 2),
            "rounds": rounds,
            "per_round_ms": [round(v, 2) for v in per_round[imgsz]],
            "ultralytics_speed_ms": bd,
        }
    return out


# --------------------------------------------------------------------------- #
# 路径解析
# --------------------------------------------------------------------------- #
def _resolve_image_dir(data_yaml: Path, split: str) -> Path:
    """从 data.yaml 里解析出某个 split 的图片目录。

    data.yaml 的 `path` 是数据集根，`train/val/test` 是相对它的路径。
    这里以 yaml 为准而不是猜目录层级 —— 目录布局可能和 pcb_yolo 不同，
    也可能存在多个指向不同 test 子集的 yaml。
    """
    import yaml as _yaml

    cfg = _yaml.safe_load(data_yaml.read_text()) or {}
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    rel = cfg.get(split) or f"images/{split}"
    d = Path(rel)
    if not d.is_absolute():
        d = base / d
    return d


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(
    weights: Path | str | None = None,
    imgsz_list: tuple[int, ...] = DEFAULT_IMGSZ,
    splits: tuple[str, ...] = ("test",),
    data_yaml: Path | str | None = None,
    device: str | None = None,
    latency_sample: int = 30,
    jpeg_quality: int = 85,
    tag: str | None = None,
) -> list[dict]:
    """对每个 imgsz 跑精度 + 时延 + 字节数。返回结果列表。"""
    from ultralytics import YOLO

    paths.ensure_dirs()

    wpath = Path(weights) if weights else default_weights()
    if not wpath.exists():
        raise FileNotFoundError(f"权重不存在: {wpath}")

    dyaml = Path(data_yaml) if data_yaml else paths.DATA_YOLO / "data.yaml"
    if not dyaml.exists():
        raise FileNotFoundError(f"数据配置不存在: {dyaml}")

    dev = resolve_device(device)
    name = tag or wpath.parent.parent.name
    out_dir = paths.RESULTS_DIR / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"分辨率 profiling — {name}")
    print(f"  权重   : {wpath}")
    print(f"  数据   : {dyaml}")
    print(f"  设备   : {dev} | 分辨率 {list(imgsz_list)} | test split {list(splits)}")
    print("=" * 78)

    model = YOLO(str(wpath))
    results: list[dict] = []

    for imgsz in imgsz_list:
        for split in splits:
            # 该 split 的图片列表（用于时延采样与字节数统计）
            img_dir = _resolve_image_dir(dyaml, split)
            imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if not imgs:
                print(f"  ⚠ imgsz={imgsz} split={split}: 没找到图片，跳过")
                continue

            lat_paths = imgs[:latency_sample] if latency_sample else imgs

            # --- 1. 时延 ---
            lat_mean, lat_p50, lat_p95 = measure_latency(model, lat_paths, imgsz, dev)

            # --- 2. JPEG 字节数 ---
            jb_mean, jb_p50, jb_max = measure_jpeg_bytes(imgs, imgsz, jpeg_quality)

            # --- 3. 精度 ---
            m = model.val(
                data=str(dyaml),
                imgsz=imgsz,
                split=split,
                device=dev,
                verbose=False,
                project=str(out_dir),
                name=f"val_{name}_{split}_imgsz{imgsz}",
                exist_ok=True,
            )
            box = m.box
            per_class = {}
            try:
                ap50 = box.ap50.tolist() if hasattr(box.ap50, "tolist") else list(box.ap50)
                idx = (
                    box.ap_class_index.tolist()
                    if hasattr(box.ap_class_index, "tolist")
                    else list(box.ap_class_index)
                )
                names_map = getattr(m, "names", {}) or {}
                for i, c in zip(idx, ap50):
                    per_class[names_map.get(i, str(i))] = round(float(c), 4)
            except Exception:  # noqa: BLE001
                pass

            row = {
                "model": name,
                "split": split,
                "imgsz": imgsz,
                "mAP50": round(float(box.map50), 4),
                "mAP50-95": round(float(box.map), 4),
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
                "latency_mean_ms": round(lat_mean, 2),
                "latency_p50_ms": round(lat_p50, 2),
                "latency_p95_ms": round(lat_p95, 2),
                "jpeg_bytes_mean": round(jb_mean, 1),
                "jpeg_bytes_p50": round(jb_p50, 1),
                "jpeg_bytes_max": round(jb_max, 1),
                "jpeg_quality": jpeg_quality,
                "latency_sample": len(lat_paths),
                "num_images": len(imgs),
                "per_class_mAP50": per_class,
            }
            results.append(row)

            (out_dir / f"{name}_{split}_imgsz{imgsz}.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            print(
                f"  imgsz={imgsz:>4} [{split:>4}]  mAP50 {row['mAP50']:.4f} | "
                f"mAP50-95 {row['mAP50-95']:.4f} | P {row['precision']:.4f} | "
                f"R {row['recall']:.4f} | {lat_mean:6.1f} ms | "
                f"{jb_mean / 1024:6.1f} KB"
            )

    # 汇总 CSV
    if results:
        csv_path = out_dir / f"summary_{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in results[0] if k != "per_class_mAP50"])
            w.writeheader()
            for r in results:
                w.writerow({k: v for k, v in r.items() if k != "per_class_mAP50"})
        print(f"\n  汇总 CSV -> {csv_path}")

    _print_tables(results)
    return results


def run_latency(
    weights: Path | str | None = None,
    imgsz_list: tuple[int, ...] = DEFAULT_IMGSZ,
    data_yaml: Path | str | None = None,
    split: str = "test",
    device: str | None = None,
    sample: int = 20,
    rounds: int = 5,
    tag: str | None = None,
) -> dict:
    """只测时延（交错法），用于得到可信的「分辨率 vs 时延」曲线。"""
    from ultralytics import YOLO

    paths.ensure_dirs()
    wpath = Path(weights) if weights else default_weights()
    dyaml = Path(data_yaml) if data_yaml else paths.DATA_YOLO / "data.yaml"
    dev = resolve_device(device)
    name = tag or wpath.parent.parent.name

    img_dir = _resolve_image_dir(dyaml, split)
    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"{img_dir} 下没有图片")
    imgs = imgs[:sample]

    print("=" * 78)
    print(f"交错时延测量 — {name}")
    print(f"  {len(imgs)} 张图 × {len(imgsz_list)} 个分辨率 × {rounds} 轮 = "
          f"{len(imgs) * len(imgsz_list) * rounds} 次推理")
    print(f"  设备: {dev}（MacBook Air 无风扇，交错是为了抵消热漂移）")
    print("=" * 78)

    model = YOLO(str(wpath))
    res = measure_latency_interleaved(model, imgs, tuple(imgsz_list), dev, rounds=rounds)

    out_dir = paths.RESULTS_DIR / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"latency_{name}_{split}.json"
    p.write_text(
        json.dumps({"model": name, "split": split, "sample": len(imgs),
                    "latency": {str(k): v for k, v in res.items()}},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n  {'imgsz':>6} {'中位时延ms':>11} {'均值ms':>9} {'最小':>8} {'最大':>8}  逐轮")
    for imgsz in sorted(res):
        d = res[imgsz]
        print(f"  {imgsz:>6} {d['latency_median_ms']:>11.1f} {d['latency_mean_ms']:>9.1f} "
              f"{d['latency_min_ms']:>8.1f} {d['latency_max_ms']:>8.1f}  {d['per_round_ms']}")

    med = [res[s]["latency_median_ms"] for s in sorted(res)]
    print(f"\n  单调递增: {all(b >= a for a, b in zip(med, med[1:]))}")
    print(f"  中位时延范围: {min(med):.1f} ~ {max(med):.1f} ms")
    print(f"  已保存 -> {p}\n")
    return res


def _print_tables(results: list[dict]) -> None:
    """按 split 分别打印表格——绝不把两个 test set 混成一个 overall。"""
    if not results:
        return
    splits = sorted({r["split"] for r in results})
    for sp in splits:
        rows = sorted([r for r in results if r["split"] == sp], key=lambda r: r["imgsz"])
        print("\n" + "=" * 78)
        print(f"split = {sp}   ({rows[0]['num_images']} 张图)")
        print("=" * 78)
        print(f"  {'imgsz':>6} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7} "
              f"{'时延(ms)':>10} {'JPEG(KB)':>10}")
        for r in rows:
            print(f"  {r['imgsz']:>6} {r['mAP50']:>8.4f} {r['mAP50-95']:>9.4f} "
                  f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
                  f"{r['latency_mean_ms']:>10.1f} {r['jpeg_bytes_mean'] / 1024:>10.1f}")
        best = max(rows, key=lambda r: r["mAP50"])
        fast = min(rows, key=lambda r: r["latency_mean_ms"])
        print(f"\n  精度最优: imgsz={best['imgsz']} (mAP50 {best['mAP50']:.4f})")
        print(f"  时延最低: imgsz={fast['imgsz']} ({fast['latency_mean_ms']:.1f} ms, "
              f"mAP50 {fast['mAP50']:.4f})")


def plot(results: list[dict], tag: str) -> Path | None:
    """画三张曲线：mAP50 / 时延 / JPEG 字节 随 imgsz 的变化。按 split 分线。"""
    if not results:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = paths.RESULTS_DIR / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = sorted({r["split"] for r in results})

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    metrics = [
        ("mAP50", "mAP50", axes[0]),
        ("mAP50-95", "mAP50-95", axes[1]),
        ("latency_mean_ms", "Latency (ms/image)", axes[2]),
        ("jpeg_bytes_mean", "JPEG size (bytes)", axes[3]),
    ]
    for sp in splits:
        rows = sorted([r for r in results if r["split"] == sp], key=lambda r: r["imgsz"])
        xs = [r["imgsz"] for r in rows]
        for key, label, ax in metrics:
            ax.plot(xs, [r[key] for r in rows], marker="o", label=sp, linewidth=2)
            ax.set_xlabel("imgsz")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    for _, _, ax in metrics:
        ax.legend()
        ax.set_xticks(sorted({r["imgsz"] for r in results}))
    fig.suptitle(f"Resolution profile — {tag}", fontsize=13)
    fig.tight_layout()
    p = out_dir / f"curves_{tag}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  曲线图 -> {p}")
    return p
