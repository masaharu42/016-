from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def configure_torch_performance(config: Any, device: torch.device) -> None:
    """Enable fixed-shape CUDA optimizations without changing the objective."""
    performance = getattr(config, "performance", None)
    matmul_precision = str(getattr(performance, "matmul_precision", "high"))
    if matmul_precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(getattr(performance, "cudnn_benchmark", True))
    allow_tf32 = bool(getattr(performance, "allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def use_channels_last(module: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled:
        module.to(memory_format=torch.channels_last)
    return module


def format_image_batch(image: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and image.ndim == 4:
        return image.contiguous(memory_format=torch.channels_last)
    return image


def maybe_compile(module: torch.nn.Module, config: Any, name: str) -> torch.nn.Module:
    """Compile a fixed-shape module, falling back to eager execution if unsupported."""
    performance = getattr(config, "performance", None)
    if not bool(getattr(performance, "compile", False)) or not hasattr(torch, "compile"):
        return module
    mode = str(getattr(performance, "compile_mode", "max-autotune-no-cudagraphs"))
    try:
        import importlib

        dynamo = importlib.import_module("torch._dynamo")
        dynamo.config.suppress_errors = True
        compiled = torch.compile(module, mode=mode, dynamic=False)
        print(f"已启用torch.compile: {name} ({mode})", flush=True)
        return compiled
    except Exception as exc:  # pragma: no cover - depends on server compiler stack
        print(f"torch.compile不可用，{name}回退普通执行: {exc}", flush=True)
        return module


def make_adamw(
    parameters: Any,
    device: torch.device,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    options: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": betas,
    }
    if device.type == "cuda":
        options["fused"] = True
    return torch.optim.AdamW(parameters, **options)


def infinite_batches(loader: Any) -> Iterator[Any]:
    while True:
        yield from loader


def make_cosine_scheduler(
    optimizer: Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return LambdaLR(optimizer, schedule)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler | None:
    if device.type == "cuda" and precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def backward(loss: torch.Tensor, scaler: torch.amp.GradScaler | None) -> None:
    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def optimizer_step(
    optimizer: Optimizer,
    model: torch.nn.Module,
    gradient_clip: float,
    scaler: torch.amp.GradScaler | None,
) -> float:
    if scaler:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(norm.detach().cpu())


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        loaded_shadow = state["shadow"]
        # Checkpoints are intentionally loaded on CPU. Keep the freshly
        # initialized EMA tensors' device/dtype so resumed training and
        # inference can update/copy them against a CUDA model immediately.
        self.shadow = {
            name: value.detach().to(
                device=self.shadow[name].device,
                dtype=self.shadow[name].dtype,
            )
            if name in self.shadow
            else value.detach().clone()
            for name, value in loaded_shadow.items()
        }

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value)
