"""配置读取。"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import paths

DEFAULT_CONFIG = paths.CONFIG_DIR / "pcb.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """读取 YAML 配置，默认 configs/pcb.yaml。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
