from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from .data import AlloyRepository
from .paths import LOG_ROOT, PROJECT_ROOT
from .utils import environment_snapshot


def run_checks(require_cuda: bool = True) -> dict:
    environment = environment_snapshot()
    errors, warnings = [], []
    if require_cuda and not torch.cuda.is_available():
        errors.append("未检测到CUDA GPU")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        environment["compute_capability"] = f"{major}.{minor}"
        environment["bf16_supported"] = torch.cuda.is_bf16_supported()
        environment["compiled_architectures"] = torch.cuda.get_arch_list()
        if not torch.cuda.is_bf16_supported():
            errors.append("GPU或当前PyTorch构建不支持BF16")
    version = torch.__version__.split("+")[0]
    if version != "2.12.1":
        warnings.append(f"推荐PyTorch 2.12.1，当前为{torch.__version__}")
    if torch.version.cuda != "13.2":
        warnings.append(f"推荐cu132构建，当前PyTorch CUDA为{torch.version.cuda}")
    repository = AlloyRepository()
    image_sizes = {}
    for alloy_id in repository.alloy_ids:
        path = Path(repository.row(alloy_id)["image_path"])
        if not path.exists():
            errors.append(f"缺少真实EBSD: {path}")
            continue
        with Image.open(path) as image:
            image_sizes[alloy_id] = [image.width, image.height]
    result = {
        "project_root": str(PROJECT_ROOT),
        "alloy_count": len(repository.alloy_ids),
        "image_count": len(image_sizes),
        "image_sizes": image_sizes,
        "environment": environment,
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    output = LOG_ROOT / "环境与数据检查.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
