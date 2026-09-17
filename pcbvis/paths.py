"""集中管理项目内所有路径。其他模块只从这里取路径，避免到处硬编码。"""

from pathlib import Path

# 项目根目录 = 本包的上一级
ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "configs"

DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / "pcb_defect"   # 原始下载（jpg + xml）
DATA_YOLO = DATA_DIR / "pcb_yolo"            # 转换后的 YOLO 数据集

MODELS_DIR = ROOT / "models"                 # 预训练权重 + 微调权重

RESULTS_DIR = ROOT / "results"
RESULTS_TRAIN = RESULTS_DIR / "train"
RESULTS_PREDICT = RESULTS_DIR / "predict"
RESULTS_EVAL = RESULTS_DIR / "eval"

# 各子集名称，与 data.yaml 中的 key 保持一致
SPLITS = ("train", "val", "test")


def ensure_dirs() -> None:
    """创建所有需要写入的目录（幂等）。"""
    for d in (
        CONFIG_DIR,
        DATA_RAW,
        DATA_YOLO,
        MODELS_DIR,
        RESULTS_TRAIN,
        RESULTS_PREDICT,
        RESULTS_EVAL,
    ):
        d.mkdir(parents=True, exist_ok=True)
