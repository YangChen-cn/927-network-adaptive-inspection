"""
数据集下载与标注格式转换。

流程：HuggingFace 下载原始数据（jpg + Pascal VOC xml）
      -> 解析 XML 取缺陷框
      -> 转成 YOLO 归一化格式
      -> 按类别分层划分 train/val/test
      -> 生成 data.yaml

关于本数据集的几个实测坑（务必注意）：
1. XML 里的 <filename> 字段（如 "01_missing_hole_01.jpg"）与实际文件名
   （"missing_hole01.jpg"）**并不一致**，所以配对只能按文件名词干做，
   绝不能相信 <filename>。
2. 类别名以 XML 内的 <name> 为准，不要从文件夹名推断。
3. 图像分辨率为 3034x1586，缺陷框相对整图极小（约占宽度 2~3%），
   这直接决定了推理分辨率 imgsz 会显著影响召回率——这是本项目后续
   「分辨率 / 时延 / 带宽」权衡的核心变量。
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from . import paths


@dataclass
class Record:
    """一张图及其所有缺陷框。boxes 中每项为 (cls_id, cx, cy, w, h)，均已归一化。"""

    image: Path
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def download(cfg: dict, force: bool = False, retries: int = 8) -> Path:
    """从 HuggingFace 下载原始数据集到 data/raw/pcb_defect。

    带重试：HF 偶发 `httpx.RemoteProtocolError: Server disconnected`，
    单次失败不应让整轮下载白费。snapshot_download 本身会跳过已下好的文件，
    所以直接重入即可断点续传。
    """
    import time

    from huggingface_hub import snapshot_download

    repo_id = cfg["dataset"]["repo_id"]
    expected = int(cfg["dataset"].get("expected_images", 0))
    target = paths.DATA_RAW
    target.mkdir(parents=True, exist_ok=True)

    def have() -> int:
        return len(list(target.glob("*/*.jpg")))

    if not force and expected and have() >= expected:
        print(f"[download] 已有 {have()}/{expected} 张，跳过（--force 可强制重下）")
        return target

    print(f"[download] 从 HuggingFace 拉取 {repo_id}（目标 {expected} 张，约 1 GB）")
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
                allow_patterns=["*/*.jpg", "*/*.xml"],
                max_workers=4,  # 并发低一点，减少被服务端断连的概率
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            n = have()
            if expected and n >= expected:
                break
            wait = min(2**attempt, 20)
            print(
                f"[download] 第 {attempt}/{retries} 次中断（{type(e).__name__}），"
                f"已下 {n}/{expected}，{wait}s 后续传..."
            )
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"下载重试 {retries} 次仍失败，已获取 {have()}/{expected}。"
            f"可直接重跑本命令续传（已下载的文件不会重复下载）。最后错误: {last_err}"
        )

    n = have()
    print(f"[download] 完成，共 {n} 张图片")
    if expected and n < expected:
        print(f"[download] ⚠ 仍少于预期的 {expected} 张，建议重跑一次续传")
    return target


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def _parse_one(
    xml_path: Path,
    image_path: Path,
    class_to_id: dict[str, int],
    stats: Counter,
) -> Record | None:
    """解析单个 XML。尺寸以真实图片为准，XML 的 <size> 仅用于交叉校验。"""
    # 以图片实际尺寸为基准（这才是模型真正看到的尺寸）
    with Image.open(image_path) as im:
        W, H = im.size

    root = ET.parse(xml_path).getroot()

    # 交叉校验 XML 声明的尺寸；不一致说明标注与图片对不上，需要提醒
    size_el = root.find("size")
    if size_el is not None:
        try:
            xw = int(float(size_el.findtext("width")))
            xh = int(float(size_el.findtext("height")))
            if (xw, xh) != (W, H):
                stats["size_mismatch"] += 1
        except (TypeError, ValueError):
            pass

    boxes: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in class_to_id:
            if name:
                stats[f"unknown_class:{name}"] += 1
            continue

        bb = obj.find("bndbox")
        if bb is None:
            stats["missing_bndbox"] += 1
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            stats["bad_coords"] += 1
            continue

        # 夹紧到图像范围，避免越界框导致 YOLO 报错
        xmin = max(0.0, min(xmin, W))
        xmax = max(0.0, min(xmax, W))
        ymin = max(0.0, min(ymin, H))
        ymax = max(0.0, min(ymax, H))

        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0:
            stats["degenerate_box"] += 1
            continue

        boxes.append(
            (
                class_to_id[name],
                (xmin + xmax) / 2.0 / W,
                (ymin + ymax) / 2.0 / H,
                bw / W,
                bh / H,
            )
        )

    if not boxes:
        stats["image_without_box"] += 1
        return None

    return Record(image=image_path, width=W, height=H, boxes=boxes)


def collect_records(cfg: dict, limit: int | None = None) -> tuple[list[Record], Counter]:
    """遍历原始目录，按词干配对 jpg 与 xml，返回全部记录。"""
    classes: list[str] = cfg["dataset"]["classes"]
    suffix: str = cfg["dataset"]["annotation_suffix"]
    class_to_id = {c: i for i, c in enumerate(classes)}

    raw = paths.DATA_RAW
    if not raw.exists() or not list(raw.glob("*/*.jpg")):
        raise FileNotFoundError(f"原始数据不存在: {raw}，请先运行 prepare --download")

    image_paths = sorted(raw.glob("*/*.jpg"))
    if limit:
        # 冒烟测试用：每类取前 N 张
        per_class: dict[str, int] = defaultdict(int)
        picked = []
        for p in image_paths:
            cls_dir = p.parent.name
            if per_class[cls_dir] < limit:
                per_class[cls_dir] += 1
                picked.append(p)
        image_paths = picked

    stats: Counter = Counter()
    records: list[Record] = []
    seen_names: dict[str, str] = {}

    for img_path in tqdm(image_paths, desc="[convert] 解析标注", unit="img"):
        xml_path = img_path.with_name(img_path.stem + suffix)
        if not xml_path.exists():
            stats["missing_xml"] += 1
            continue

        # 检查重名：不同类别目录下若出现同名文件，复制到同一 split 目录会互相覆盖
        if img_path.name in seen_names:
            raise RuntimeError(
                f"文件名冲突: {img_path} 与 {seen_names[img_path.name]} 同名，"
                "需要在输出时加类别前缀"
            )
        seen_names[img_path.name] = str(img_path)

        rec = _parse_one(xml_path, img_path, class_to_id, stats)
        if rec is not None:
            stats["parsed"] += 1
            stats[f"cls:{img_path.parent.name}"] += 1
            records.append(rec)

    return records, stats


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def _block_index(stem: str, blocks: list[list[int]]) -> int:
    """图片序号落在第几块母板（从 1 开始）。"""
    idx = int("".join(ch for ch in stem if ch.isdigit()))
    for i, (lo, hi) in enumerate(blocks, 1):
        if lo <= idx <= hi:
            return i
    raise ValueError(f"图片序号 {idx}（{stem}）不在任何母板 block 范围内")


def split_records(
    records: list[Record],
    ratios: dict,
    seed: int,
    blocks: list[list[int]] | None = None,
    block_split: dict | None = None,
) -> dict[str, list[Record]]:
    """划分 train/val/test。

    【默认】按 PCB 母板 block 划分 —— 整块母板只进一个 split。
    这是防泄漏的关键：本数据集只有 10 块母板，同母板的图在非缺陷区域
    逐像素完全相同，若按图片随机划分会让同一块板同时进 train 和 test，
    模型靠"认板子"就能拿高分，指标虚高。

    未提供 blocks/block_split 时，回退到按类别目录分层随机划分
    （保留该路径仅作兼容，不推荐使用）。
    """
    if blocks and block_split:
        block2split: dict[int, str] = {}
        for sp, idxs in block_split.items():
            for i in idxs:
                block2split[int(i)] = sp

        missing = set(range(1, len(blocks) + 1)) - set(block2split)
        if missing:
            raise ValueError(f"以下母板 block 未被分配到任何 split: {sorted(missing)}")

        out: dict[str, list[Record]] = {s: [] for s in paths.SPLITS}
        for r in records:
            bi = _block_index(r.image.stem, blocks)
            out[block2split[bi]].append(r)
        return out

    # ---- 回退：按类别目录分层随机划分（有泄漏风险，仅在无母板信息时用）----
    by_dir: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_dir[r.image.parent.name].append(r)

    rng = random.Random(seed)
    out: dict[str, list[Record]] = {s: [] for s in paths.SPLITS}

    for cls_dir in sorted(by_dir):
        recs = sorted(by_dir[cls_dir], key=lambda r: r.image.name)  # 先排序保证确定性
        rng.shuffle(recs)
        n = len(recs)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        n_test = n - n_train - n_val  # 余数全部给 test，保证总数守恒
        if n_test < 0:
            raise ValueError(f"划分比例之和超过 1.0（类别 {cls_dir}, n={n}）")

        out["train"] += recs[:n_train]
        out["val"] += recs[n_train : n_train + n_val]
        out["test"] += recs[n_train + n_val :]

    return out


# --------------------------------------------------------------------------- #
# 写盘
# --------------------------------------------------------------------------- #
def write_yolo(
    cfg: dict, splits: dict[str, list[Record]], clean: bool = True
) -> Path:
    """写出 YOLO 目录结构 + data.yaml。"""
    out_root = paths.DATA_YOLO
    if clean:
        for sub in ("images", "labels"):
            d = out_root / sub
            if d.exists():
                shutil.rmtree(d)

    for split in paths.SPLITS:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, recs in splits.items():
        for rec in tqdm(recs, desc=f"[convert] 写出 {split}", unit="img"):
            shutil.copy2(rec.image, out_root / "images" / split / rec.image.name)
            lines = [
                f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for c, cx, cy, w, h in rec.boxes
            ]
            (out_root / "labels" / split / f"{rec.image.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    classes: list[str] = cfg["dataset"]["classes"]
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(classes)},
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    return yaml_path


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _report(
    records: list[Record],
    splits: dict[str, list[Record]],
    stats: Counter,
    blocks: list[list[int]] | None = None,
) -> None:
    classes_seen = Counter()
    box_sizes: list[float] = []
    for r in records:
        for c, _cx, _cy, w, _h in r.boxes:
            classes_seen[c] += 1
            box_sizes.append(w)  # 归一化宽度，用于说明缺陷有多小

    print("\n" + "=" * 62)
    print("数据准备完成")
    print("=" * 62)
    print(f"  有效图片      : {len(records)}")
    print(f"  缺陷框总数    : {sum(classes_seen.values())}")
    print(f"  平均框/图     : {sum(classes_seen.values()) / max(len(records), 1):.2f}")
    if box_sizes:
        box_sizes.sort()
        med = box_sizes[len(box_sizes) // 2]
        print(f"  缺陷框宽度    : 中位数 {med * 100:.2f}% 图宽 "
              f"(3034px 图上约 {med * 3034:.0f}px)")
    print(f"  训练/验证/测试: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")

    mapping = {i: c for i, c in enumerate(_CLASSES_REF)}
    print("\n  各类别框数:")
    for cid in sorted(classes_seen):
        print(f"    {cid} {mapping.get(cid, '?'):18s} {classes_seen[cid]:5d}")

    noise = {k: v for k, v in stats.items() if k.startswith(("unknown_class", "bad_", "missing_", "degenerate"))}
    if noise:
        print("\n  ⚠ 需要留意的问题:")
        for k, v in sorted(noise.items()):
            print(f"    {k}: {v}")
    if stats.get("size_mismatch"):
        print(f"\n  ⚠ {stats['size_mismatch']} 张图的 XML 声明尺寸与实际不一致（已按实际尺寸处理）")

    # 防泄漏自检：同一块母板是否跨 split
    if blocks:
        blk = lambda r: _block_index(r.image.stem, blocks)  # noqa: E731
        tr = {blk(r) for r in splits["train"]}
        va = {blk(r) for r in splits["val"]}
        te = {blk(r) for r in splits["test"]}
        leak = (tr & te) | (tr & va) | (va & te)
        print(f"\n  母板划分（防泄漏，共 {len(blocks)} 块母板）:")
        print(f"    train {sorted(tr)} | val {sorted(va)} | test {sorted(te)}")
        print(f"    跨 split 共用的母板: {sorted(leak) if leak else '无 ✅'}")
    print()


_CLASSES_REF: list[str] = []


def run(
    config: Path | str | None = None,
    do_download: bool = True,
    limit: int | None = None,
    force: bool = False,
) -> Path:
    """完整数据准备流程，返回 data.yaml 路径。"""
    from .config import load_config

    cfg = load_config(config)
    paths.ensure_dirs()

    global _CLASSES_REF
    _CLASSES_REF = cfg["dataset"]["classes"]

    if do_download:
        download(cfg, force=force)

    records, stats = collect_records(cfg, limit=limit)
    if not records:
        raise RuntimeError("没有解析到任何有效标注，请检查下载是否完整")

    dcfg = cfg["dataset"]
    splits = split_records(
        records,
        dcfg["split"],
        dcfg["seed"],
        blocks=dcfg.get("template_blocks"),
        block_split=dcfg.get("block_split"),
    )
    yaml_path = write_yolo(cfg, splits)
    _report(records, splits, stats, dcfg.get("template_blocks"))

    print(f"  data.yaml -> {yaml_path}\n")
    return yaml_path
