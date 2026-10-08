from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_torch_save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temp_path)
    os.replace(temp_path, path)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    result: dict[str, Any] = {
        "available": True,
        "name": props.name,
        "memory_allocated_gb": round(torch.cuda.memory_allocated(index) / 2**30, 3),
        "memory_reserved_gb": round(torch.cuda.memory_reserved(index) / 2**30, 3),
        "peak_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated(index) / 2**30, 3
        ),
        "memory_total_gb": round(props.total_memory / 2**30, 3),
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip().split(",")
        result.update(
            utilization_pct=float(query[0]),
            temperature_c=float(query[1]),
            power_w=float(query[2]),
            memory_used_mib=float(query[3]),
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return result


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_snapshot(),
    }


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total
