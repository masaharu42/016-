from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .utils import atomic_json_dump, environment_snapshot, gpu_snapshot

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "计算中"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class SmoothedMetrics:
    def __init__(self, window: int = 50) -> None:
        self.values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.values[key].append(float(value))

    def means(self) -> dict[str, float]:
        return {key: sum(values) / len(values) for key, values in self.values.items() if values}


class TrainingMonitor:
    """Writes console, CSV, JSONL, TensorBoard and machine-readable live status."""

    def __init__(
        self,
        run_dir: Path,
        stage: str,
        total_steps: int,
        metric_names: Iterable[str],
        log_every: int = 10,
        smoothing_window: int = 50,
        enable_tensorboard: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.stage = stage
        self.total_steps = total_steps
        self.metric_names = list(dict.fromkeys(metric_names))
        self.log_every = log_every
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.step_durations: deque[float] = deque(maxlen=100)
        self.metrics = SmoothedMetrics(smoothing_window)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:  # pragma: no cover
            pass
        run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = run_dir / "训练指标.csv"
        self.jsonl_path = run_dir / "训练指标.jsonl"
        self.text_path = run_dir / "训练过程.log"
        self.status_path = run_dir / "实时状态.json"
        self.csv_handle = self.csv_path.open("a", newline="", encoding="utf-8-sig")
        fields = [
            "step",
            "total_steps",
            "elapsed_seconds",
            "eta_seconds",
            "seconds_per_step",
            "steps_per_second",
            *self.metric_names,
        ]
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=fields, extrasaction="ignore")
        if self.csv_path.stat().st_size == 0:
            self.csv_writer.writeheader()
        self.jsonl_handle = self.jsonl_path.open("a", encoding="utf-8")
        self.text_handle = self.text_path.open("a", encoding="utf-8")
        self.tensorboard = (
            SummaryWriter(str(run_dir / "tensorboard"))
            if SummaryWriter and enable_tensorboard
            else None
        )
        atomic_json_dump(environment_snapshot(), run_dir / "运行环境.json")

    def log(self, step: int, raw_metrics: dict[str, float], force: bool = False) -> None:
        now = time.time()
        self.step_durations.append(max(now - self.last_step_time, 1e-9))
        self.last_step_time = now
        self.metrics.update(raw_metrics)
        if not force and step != 1 and step % self.log_every != 0 and step != self.total_steps:
            return
        smoothed = self.metrics.means()
        seconds_per_step = sum(self.step_durations) / len(self.step_durations)
        eta = max(self.total_steps - step, 0) * seconds_per_step
        elapsed = now - self.start_time
        gpu = gpu_snapshot()
        row: dict[str, Any] = {
            "step": step,
            "total_steps": self.total_steps,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(eta, 3),
            "seconds_per_step": round(seconds_per_step, 6),
            "steps_per_second": round(1.0 / max(seconds_per_step, 1e-9), 6),
            **{key: smoothed.get(key, float("nan")) for key in self.metric_names},
        }
        self.csv_writer.writerow(row)
        self.csv_handle.flush()
        payload = {**row, "stage": self.stage, "timestamp": now, "gpu": gpu}
        self.jsonl_handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self.jsonl_handle.flush()
        atomic_json_dump(payload, self.status_path)
        if self.tensorboard:
            for key, value in smoothed.items():
                self.tensorboard.add_scalar(f"train/{key}", value, step)
            for key in ("utilization_pct", "temperature_c", "power_w", "memory_used_mib"):
                if key in gpu:
                    self.tensorboard.add_scalar(f"gpu/{key}", gpu[key], step)
        loss_parts = [
            f"{name}={smoothed[name]:.4g}"
            for name in self.metric_names
            if name in smoothed and ("loss" in name or name in {"lr", "grad_norm"})
        ]
        vram = (
            f"显存 当前{gpu.get('memory_allocated_gb', 0):.1f}GB "
            f"峰值{gpu.get('peak_memory_allocated_gb', 0):.1f}/"
            f"{gpu.get('memory_total_gb', 0):.1f}GB"
        )
        util = f"GPU {gpu['utilization_pct']:.0f}%" if "utilization_pct" in gpu else ""
        line = (
            f"[{self.stage}] {step:06d}/{self.total_steps:06d} "
            f"({100 * step / max(self.total_steps, 1):5.1f}%) | "
            f"耗时 {format_duration(elapsed)} | 预计剩余 {format_duration(eta)} | "
            f"{seconds_per_step:.3f}秒/步 | "
            + " | ".join(loss_parts[:10])
            + f" | {vram} {util}"
        )
        print(line, flush=True)
        self.text_handle.write(line + "\n")
        self.text_handle.flush()

    def add_images(self, name: str, images: Any, step: int) -> None:
        if self.tensorboard:
            self.tensorboard.add_images(name, images, step)

    def close(self, status: str = "completed") -> None:
        current: dict[str, Any] = {}
        if self.status_path.exists():
            current = json.loads(self.status_path.read_text(encoding="utf-8"))
        current["status"] = status
        current["finished_at"] = time.time()
        atomic_json_dump(current, self.status_path)
        self.csv_handle.close()
        self.jsonl_handle.close()
        self.text_handle.close()
        if self.tensorboard:
            self.tensorboard.close()


def watch_status(run_dir: Path, interval: float = 2.0) -> None:
    status_path = run_dir / "实时状态.json"
    last_text = ""
    print(f"监控目录: {run_dir}")
    try:
        while True:
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                metrics = [
                    f"{key}={value:.5g}"
                    for key, value in status.items()
                    if isinstance(value, float) and ("loss" in key or key in {"lr", "grad_norm"})
                ]
                text = (
                    f"{status.get('stage')} {status.get('step')}/{status.get('total_steps')} | "
                    f"ETA {format_duration(float(status.get('eta_seconds', 0)))} | "
                    + " | ".join(metrics[:10])
                )
                if text != last_text:
                    print(text, flush=True)
                    last_text = text
                if status.get("status") in {"completed", "failed", "interrupted"}:
                    return
            else:
                print("等待训练创建实时状态文件...", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("已停止监控；训练进程不会被停止。", file=sys.stderr)
