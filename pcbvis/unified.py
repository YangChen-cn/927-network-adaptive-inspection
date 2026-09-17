"""
统一数据集构建：PKU-Market-PCB + PCB-Defect 2025 -> 一份 YOLO detection 数据集。

三个关键设计（都基于实测，不是拍脑袋）：

1. 【类别统一】7 类。missing_hole 与 missing_pad 语义不同，不合并；
   其余 5 类命名完全一致，直接合并。详见 configs/unified.yaml 的注释。

2. 【防泄漏】PKU 只有 10 块 PCB 母板，每块母板合成 6 类缺陷各若干张，
   同母板的图在非缺陷区域【逐像素完全相同】。因此划分必须以「母板 block」
   为单位，否则同一块板会同时进 train 和 test。
   PCB-Defect 2025 的 230 张图尺寸全部不同（各自独立的板），无此问题。

3. 【两个独立 test set】合并 train/val，但 test 保持分开，
   这样才能判断 unified model 是否同时适应两个 domain。
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from . import paths

UNIFIED_DIR = paths.DATA_DIR / "unified"
UNIFIED_CONFIG = paths.CONFIG_DIR / "unified.yaml"


@dataclass
class Item:
    """一张图 + 其缺陷框 + 所属母板分组。boxes 为 (unified_cls, cx, cy, w, h) 归一化。"""

    image: Path
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    group: str = ""      # 母板分组标识，防泄漏划分用
    source: str = ""     # "pku" | "pcb2025"


def load_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path else UNIFIED_CONFIG
    return yaml.safe_load(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 下载 PCB-Defect 2025
# --------------------------------------------------------------------------- #
def download_pcb2025(cfg: dict, force: bool = False) -> Path:
    """下载并校验 PCB-Defect 2025（Mendeley，无需登录）。带 SHA256 校验。"""
    c = cfg["pcb2025"]
    raw = paths.ROOT / c["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "PCB_Defect.zip"
    coco = raw / c["coco_json"]

    if coco.exists() and not force:
        print(f"[pcb2025] 已解压，跳过（--force 可重下）")
        return raw

    need_download = force or not zip_path.exists()
    if not need_download:
        h = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if h != c["sha256"]:
            print(f"[pcb2025] zip 校验失败，重新下载")
            need_download = True

    if need_download:
        print(f"[pcb2025] 下载 {c['source']} (约 150 MB, CC BY 4.0)")
        urllib.request.urlretrieve(c["download_url"], zip_path)
        h = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if h != c["sha256"]:
            raise RuntimeError(f"SHA256 校验失败:\n  期望 {c['sha256']}\n  实际 {h}")
        print(f"[pcb2025] SHA256 校验通过 ✅")

    print(f"[pcb2025] 解压...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)
    return raw


# --------------------------------------------------------------------------- #
# 读取各数据集
# --------------------------------------------------------------------------- #
def _pku_block_index(idx: int, blocks: list[list[int]]) -> int:
    """序号落在第几个母板 block（从 1 开始）。"""
    for i, (lo, hi) in enumerate(blocks, 1):
        if lo <= idx <= hi:
            return i
    raise ValueError(f"序号 {idx} 不在任何母板 block 内")


def load_pku(cfg: dict) -> list[Item]:
    """读 PKU 原始 VOC XML，转统一 class id，并标出母板分组。"""
    c = cfg["pku"]
    cmap = {k: int(v) for k, v in c["category_map"].items()}
    suffix = c["annotation_suffix"]
    blocks = c["template_blocks"]
    raw = paths.ROOT / c["raw_dir"]

    items: list[Item] = []
    stats = {"unknown": 0, "no_box": 0, "missing_xml": 0}

    for img_path in tqdm(sorted(raw.glob(c["image_glob"])), desc="[pku] 解析", unit="img"):
        xml_path = img_path.with_name(img_path.stem + suffix)
        if not xml_path.exists():
            stats["missing_xml"] += 1
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        root = ET.parse(xml_path).getroot()

        boxes = []
        for o in root.findall("object"):
            name = (o.findtext("name") or "").strip().lower()
            if name not in cmap:
                stats["unknown"] += 1
                continue
            bb = o.find("bndbox")
            if bb is None:
                continue
            try:
                x0 = max(0.0, min(float(bb.findtext("xmin")), W))
                y0 = max(0.0, min(float(bb.findtext("ymin")), H))
                x1 = max(0.0, min(float(bb.findtext("xmax")), W))
                y1 = max(0.0, min(float(bb.findtext("ymax")), H))
            except (TypeError, ValueError):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append((cmap[name], (x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H))

        if not boxes:
            stats["no_box"] += 1
            continue

        idx = int("".join(ch for ch in img_path.stem if ch.isdigit()))
        blk = _pku_block_index(idx, blocks)
        items.append(Item(image=img_path, boxes=boxes,
                          group=f"pku_block{blk:02d}", source="pku"))

    if any(stats.values()):
        print(f"[pku] 统计: {stats}")
    return items


def load_pcb2025(cfg: dict) -> list[Item]:
    """读 PCB-Defect 2025 的 COCO JSON，转统一 class id。"""
    c = cfg["pcb2025"]
    raw = paths.ROOT / c["raw_dir"]
    coco = json.loads((raw / c["coco_json"]).read_text(encoding="utf-8"))
    imgs_dir = raw / c["images_dir"]

    # COCO categories 的 name -> 统一 id；原始 id 一律不沿用
    name2uid = {k: int(v) for k, v in c["category_map"].items()}
    catid2uid = {}
    for cat in coco["categories"]:
        nm = cat.get("name")
        if nm in name2uid:
            catid2uid[cat["id"]] = name2uid[nm]

    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    items: list[Item] = []
    skipped = 0
    for im in tqdm(coco["images"], desc="[pcb2025] 解析", unit="img"):
        p = imgs_dir / im["file_name"]
        if not p.exists():
            skipped += 1
            continue
        W, H = float(im["width"]), float(im["height"])
        boxes = []
        for a in by_img.get(im["id"], []):
            uid = catid2uid.get(a["category_id"])
            if uid is None:
                continue
            x, y, w, h = a["bbox"]                      # COCO: [x,y,w,h] 绝对像素
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append((uid, (x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H))
        if not boxes:
            continue
        # 实测 230 张尺寸各不相同，各自独立，按文件名自成一组
        items.append(Item(image=p, boxes=boxes,
                          group=f"pcb2025_{p.stem}", source="pcb2025"))

    if skipped:
        print(f"[pcb2025] ⚠ {skipped} 张图片文件缺失")
    return items


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def split_all(cfg: dict, pku: list[Item], p25: list[Item]) -> dict[str, list[Item]]:
    """PKU 按母板 block 划分；PCB2025 按图片随机划分。两个数据集各自独立划分后再合并。"""
    c = cfg["pku"]
    bmap = {k: [int(i) for i in v] for k, v in c["block_split"].items()}  # split -> [block]
    block2split = {}
    for sp, idxs in bmap.items():
        for i in idxs:
            block2split[f"pku_block{i:02d}"] = sp

    out: dict[str, list[Item]] = {"train": [], "val": [], "test_pku": [], "test_pcb2025": []}

    # --- PKU: 整块母板进同一个 split ---
    unknown_blocks = {it.group for it in pku} - set(block2split)
    if unknown_blocks:
        raise RuntimeError(f"有母板 block 未被分配: {sorted(unknown_blocks)}")
    for it in pku:
        sp = block2split[it.group]
        out["test_pku" if sp == "test" else sp].append(it)

    # --- PCB2025: 尺寸各异、无同源图，随机划分即可 ---
    rng = random.Random(cfg["split_seed"])
    items = sorted(p25, key=lambda it: it.image.name)
    rng.shuffle(items)
    n = len(items)
    s = cfg["pcb2025"]["split"]
    n_tr = int(n * s["train"])
    n_va = int(n * s["val"])
    out["train"] += items[:n_tr]
    out["val"] += items[n_tr:n_tr+n_va]
    out["test_pcb2025"] += items[n_tr+n_va:]

    return out


# --------------------------------------------------------------------------- #
# 写盘
# --------------------------------------------------------------------------- #
def write(cfg: dict, splits: dict[str, list[Item]], clean: bool = True) -> dict:
    root = UNIFIED_DIR
    if clean and root.exists():
        shutil.rmtree(root)
    for sub in ("images", "labels"):
        for sp in splits:
            (root / sub / sp).mkdir(parents=True, exist_ok=True)

    for sp, items in splits.items():
        for it in tqdm(items, desc=f"[write] {sp}", unit="img"):
            stem = f"{it.source}_{it.image.stem}"
            shutil.copy2(it.image, root / "images" / sp / f"{stem}.jpg")
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for c, cx, cy, w, h in it.boxes]
            (root / "labels" / sp / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

    classes = cfg["unified"]["classes"]
    names = {i: n for i, n in enumerate(classes)}
    base = {"path": str(root.resolve()), "names": names}

    def dump(fname, **kw):
        with open(root / fname, "w", encoding="utf-8") as f:
            yaml.safe_dump({**base, **kw}, f, sort_keys=False, allow_unicode=True)

    # 训练用：train + 合并后的 val
    dump("data.yaml", train="images/train", val="images/val", test="images/test_pku")
    # 两个独立测试集，各自一份 yaml
    dump("test_pku.yaml", train="images/train", val="images/val", test="images/test_pku")
    dump("test_pcb2025.yaml", train="images/train", val="images/val", test="images/test_pcb2025")

    mapping = {
        "unified_classes": names,
        "rationale": {
            "missing_hole_vs_missing_pad":
                "语义不同，未合并。missing_hole=该钻孔处无孔；missing_pad=铜焊盘缺失/破损。",
            "merged": "mouse_bite/open_circuit/short/spur/spurious_copper 两数据集命名一致，直接合并。",
        },
        "sources": {
            "pku": {"source": cfg["pku"]["source"], "original_to_unified": cfg["pku"]["category_map"],
                    "split_method": "按 10 块 PCB 母板分组划分（防泄漏）",
                    "template_blocks": cfg["pku"]["template_blocks"],
                    "block_split": cfg["pku"]["block_split"]},
            "pcb2025": {"source": cfg["pcb2025"]["source"], "license": cfg["pcb2025"]["license"],
                        "doi": "10.17632/vdj74sngvn.1",
                        "original_to_unified": cfg["pcb2025"]["category_map"],
                        "split_method": "按图片随机划分（230 张尺寸各异，无同源图）"},
        },
        "counts": {sp: len(v) for sp, v in splits.items()},
    }
    (root / "class_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return mapping


def run(config: Path | str | None = None, do_download: bool = True, force: bool = False) -> dict:
    cfg = load_config(config)
    paths.ensure_dirs()
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    if do_download:
        download_pcb2025(cfg, force=force)

    pku_raw = paths.ROOT / cfg["pku"]["raw_dir"]
    if not list(pku_raw.glob(cfg["pku"]["image_glob"])):
        raise FileNotFoundError(f"PKU 原始数据缺失: {pku_raw}，请先运行 python -m pcbvis prepare")

    pku = load_pku(cfg)
    p25 = load_pcb2025(cfg)
    splits = split_all(cfg, pku, p25)
    mapping = write(cfg, splits)
    report(cfg, splits, pku, p25)
    return mapping


def report(cfg: dict, splits: dict, pku: list[Item], p25: list[Item]) -> None:
    import collections
    names = cfg["unified"]["classes"]

    print("\n" + "=" * 74)
    print("统一数据集构建完成")
    print("=" * 74)
    print(f"  PKU      : {len(pku):>4} 张  ({len({i.group for i in pku})} 块母板)")
    print(f"  PCB2025  : {len(p25):>4} 张")
    print(f"  合计     : {len(pku)+len(p25):>4} 张")
    print()
    print(f"  {'split':<14} {'PKU':>6} {'PCB2025':>9} {'合计':>7}")
    for sp in ("train", "val", "test_pku", "test_pcb2025"):
        items = splits[sp]
        a = sum(1 for i in items if i.source == "pku")
        b = len(items) - a
        print(f"  {sp:<14} {a:>6} {b:>9} {len(items):>7}")

    print("\n  逐类框数:")
    print(f"  {'id':<3} {'类别':<18} {'train':>7} {'val':>7} {'test_pku':>9} {'test_pcb2025':>13}")
    for cid, nm in enumerate(names):
        row = []
        for sp in ("train", "val", "test_pku", "test_pcb2025"):
            row.append(sum(1 for it in splits[sp] for b in it.boxes if b[0] == cid))
        print(f"  {cid:<3} {nm:<18} {row[0]:>7} {row[1]:>7} {row[2]:>9} {row[3]:>13}")

    # 泄漏自检：同一母板是否跨 train/test
    tr_g = {it.group for it in splits["train"]}
    te_g = {it.group for it in splits["test_pku"]} | {it.group for it in splits["test_pcb2025"]}
    overlap = tr_g & te_g
    print(f"\n  泄漏自检 — train 与 test 共用的母板分组: {overlap if overlap else '无 ✅'}")
    print(f"\n  data.yaml / test_pku.yaml / test_pcb2025.yaml / class_mapping.json")
    print(f"  -> {UNIFIED_DIR}\n")
