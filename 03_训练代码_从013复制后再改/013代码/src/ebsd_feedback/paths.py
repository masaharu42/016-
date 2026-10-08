from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _image_mode() -> str:
    """013 has one independent four-channel IPF+GB experiment."""
    raw = os.environ.get("EBSD_013_MODE", "IPF_GB").strip().upper().replace("-", "_")
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if index + 1 < len(sys.argv):
            candidate = sys.argv[index + 1].strip().upper().replace("-", "_")
            if candidate in {"IPF_GB", "IPFGB"}:
                raw = candidate
    if raw not in {"IPF_GB", "IPFGB"}:
        raise ValueError(f"013 只支持 IPF_GB 四通道模式，收到 {raw}")
    return "IPF_GB"


IMAGE_MODE = _image_mode()
IMAGE_CHANNELS = 4
CODE_ROOT = PROJECT_ROOT / "05_代码"
CONFIG_ROOT = PROJECT_ROOT / "06_配置文件"
MODELING_DATA_ROOT = PROJECT_ROOT / "02_整理后建模数据"
IMAGE_ROOT = PROJECT_ROOT / "03_图像数据" / IMAGE_MODE
REAL_IMAGE_ROOT = IMAGE_ROOT
PRETRAINED_ROOT = PROJECT_ROOT / "07_预训练模型_服务器生成" / IMAGE_MODE
FOLD_MODEL_ROOT = PROJECT_ROOT / "08_严格留一模型_服务器生成" / IMAGE_MODE
LOG_ROOT = PROJECT_ROOT / "09_训练日志_服务器生成" / IMAGE_MODE
PREDICTION_ROOT = PROJECT_ROOT / "10_预测结果_服务器生成" / IMAGE_MODE


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
