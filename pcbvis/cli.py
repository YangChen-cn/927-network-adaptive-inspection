"""命令行入口。用法: python -m pcbvis <命令> [参数]"""

from __future__ import annotations

import argparse
import sys


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="配置文件路径（默认 configs/pcb.yaml）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcbvis",
        description="PCB 缺陷检测 — 数据准备 / 微调 / 推理 / 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型流程:\n"
            "  python -m pcbvis prepare\n"
            "  python -m pcbvis train\n"
            "  python -m pcbvis eval\n"
            "  python -m pcbvis predict --imgsz 640\n"
            "  python -m pcbvis sweep --imgsz 320,640,1280\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prepare ---
    p = sub.add_parser("prepare", help="下载数据集并转成 YOLO 格式")
    _add_common(p)
    p.add_argument("--no-download", action="store_true", help="跳过下载，只做转换")
    p.add_argument("--limit", type=int, default=None, help="每类只取 N 张（冒烟测试用）")
    p.add_argument("--force", action="store_true", help="强制重新下载")

    # --- train ---
    p = sub.add_parser("train", help="微调 YOLO")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="mps / cpu / cuda（默认自动探测）")
    p.add_argument("--model", default="yolo11n.pt", help="预训练权重名")
    p.add_argument("--name", default="pcb_yolo11n", help="训练输出子目录名")
    p.add_argument("--workers", type=int, default=4, help="DataLoader 进程数（macOS 卡住时设 0）")
    p.add_argument("--patience", type=int, default=30, help="早停耐心值")

    # --- predict ---
    p = sub.add_parser("predict", help="推理并输出带框图")
    p.add_argument("--weights", default=None, help="权重路径（默认用微调后的 best.pt）")
    p.add_argument("--imgsz", type=int, default=640, help="推理分辨率（核心旋钮）")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 张")

    # --- eval ---
    p = sub.add_parser("eval", help="评估精度指标")
    p.add_argument("--weights", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--device", default=None)

    # --- sweep ---
    p = sub.add_parser("sweep", help="扫多个 imgsz，得到「精度 vs 分辨率」曲线")
    p.add_argument("--weights", default=None)
    p.add_argument("--imgsz", default="320,416,640,1280", help="逗号分隔")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--device", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.command

    if cmd == "prepare":
        from . import data_prep

        data_prep.run(
            config=args.config,
            do_download=not args.no_download,
            limit=args.limit,
            force=args.force,
        )

    elif cmd == "train":
        from . import train as train_mod

        train_mod.run(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            model_name=args.model,
            name=args.name,
            workers=args.workers,
            patience=args.patience,
            config=args.config,
        )

    elif cmd == "predict":
        from . import predict as predict_mod

        predict_mod.run(
            weights=args.weights,
            imgsz=args.imgsz,
            conf=args.conf,
            split=args.split,
            device=args.device,
            limit=args.limit,
        )

    elif cmd == "eval":
        from . import evaluate

        evaluate.run(
            weights=args.weights, imgsz=args.imgsz, split=args.split, device=args.device
        )

    elif cmd == "sweep":
        from . import evaluate

        sizes = [int(s) for s in str(args.imgsz).split(",") if s.strip()]
        rows = []
        for s in sizes:
            print(f"\n########## imgsz={s} ##########\n")
            m = evaluate.run(
                weights=args.weights, imgsz=s, split=args.split, device=args.device
            )
            rows.append(m)

        print("\n" + "=" * 62)
        print("精度 vs 分辨率")
        print("=" * 62)
        print(f"  {'imgsz':>7} {'mAP50':>9} {'mAP50-95':>10} {'precision':>10} {'recall':>8}")
        for m in rows:
            print(
                f"  {m['imgsz']:>7} {m['mAP50']:>9.4f} {m['mAP50-95']:>10.4f} "
                f"{m['precision']:>10.4f} {m['recall']:>8.4f}"
            )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
