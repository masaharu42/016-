from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checks import run_checks
from .config import load_config
from .folds import prepare_strict_fold
from .inference import generate_three_images
from .monitor import watch_status
from .training.diffusion import train_diffusion
from .training.mechanics import train_mechanics
from .training.vae import train_vae
from .training.image_descriptor import train_image_descriptor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="013 IPF-Z+GB 四通道条件扩散EBSD生成项目")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_mode(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--mode",
            choices=("IPF_GB",),
            default=None,
            help="013 独立 IPF_GB 四通道分支",
        )

    check = subparsers.add_parser("check", help="检查服务器环境和22个合金数据")
    check.add_argument("--allow-cpu", action="store_true")
    add_mode(check)
    fold = subparsers.add_parser("prepare-fold", help="准备一个严格留一折")
    fold.add_argument("--holdout", required=True, help="例如 ID01")
    add_mode(fold)
    vae = subparsers.add_parser("train-vae", help="训练边界感知VAE")
    vae.add_argument("--holdout", required=True)
    vae.add_argument("--config", default="01_边界感知VAE.yaml")
    add_mode(vae)
    mechanics = subparsers.add_parser("train-mechanics", help="训练冻结用力学代理")
    mechanics.add_argument("--holdout", required=True)
    mechanics.add_argument("--config", default="02_力学代理.yaml")
    add_mode(mechanics)
    descriptor = subparsers.add_parser("train-descriptor", help="预训练图像描述符代理，内层选步后21合金重拟合")
    descriptor.add_argument("--holdout", required=True)
    descriptor.add_argument("--config", default="02b_图像描述符代理.yaml")
    add_mode(descriptor)
    diffusion = subparsers.add_parser("train-diffusion", help="训练基础条件扩散")
    diffusion.add_argument("--holdout", required=True)
    diffusion.add_argument("--config", default="03_基础条件扩散.yaml")
    add_mode(diffusion)
    feedback = subparsers.add_parser("train-feedback", help="用真实曲线反馈微调扩散权重")
    feedback.add_argument("--holdout", required=True)
    feedback.add_argument("--config", default="04_力学反馈微调.yaml")
    add_mode(feedback)
    sample = subparsers.add_parser("sample", help="固定种子生成3张EBSD及曲线")
    sample.add_argument("--holdout", required=True)
    sample.add_argument("--composition", help="可选：一行9元素的新成分CSV")
    sample.add_argument("--config", default="05_三图推理.yaml")
    add_mode(sample)
    monitor = subparsers.add_parser("monitor", help="在另一终端查看实时状态和ETA")
    monitor.add_argument("--run-dir", required=True)
    monitor.add_argument("--interval", type=float, default=2.0)
    add_mode(monitor)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check":
        result = run_checks(require_cuda=not args.allow_cpu)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit(2)
    elif args.command == "prepare-fold":
        print(f"折清单已生成: {prepare_strict_fold(args.holdout)}")
    elif args.command == "train-vae":
        print(f"VAE训练完成: {train_vae(load_config(args.config), args.holdout)}")
    elif args.command == "train-mechanics":
        print(f"力学代理训练完成: {train_mechanics(load_config(args.config), args.holdout)}")
    elif args.command == "train-descriptor":
        print(f"图像描述符代理完成: {train_image_descriptor(load_config(args.config), args.holdout)}")
    elif args.command == "train-diffusion":
        print(f"基础扩散训练完成: {train_diffusion(load_config(args.config), args.holdout)}")
    elif args.command == "train-feedback":
        print(f"力学反馈微调完成: {train_diffusion(load_config(args.config), args.holdout)}")
    elif args.command == "sample":
        print(
            f"预测结果目录: {generate_three_images(load_config(args.config), args.holdout, args.composition)}"
        )
    elif args.command == "monitor":
        watch_status(Path(args.run_dir), args.interval)


if __name__ == "__main__":
    main()
