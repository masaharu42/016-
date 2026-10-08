"""Sequential fold runner with provenance guard, lock and observable subprocesses."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "05_代码" / "src"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0); handle.write("0"); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit("另一个013入口正在运行，请勿同时启动。系统在进程退出后自动释放锁。")
    try:
        yield
    finally:
        handle.close()


def main():
    parser = argparse.ArgumentParser(description="013 一键连续前五折，自动跳过已完成阶段、续训滚动断点")
    parser.add_argument("--count", type=int, default=5, help="质量顺序前N折，默认5")
    parser.add_argument("--all", action="store_true", help="全部22折，沿既定顺序")
    parser.add_argument("--ids", nargs="+", help="指定若干折，自动依既定顺序执行")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--plan", action="store_true", help="只打印顺序，不加载GPU、不创建模型")
    args = parser.parse_args()
    order = json.loads((ROOT / "00_项目说明" / "04_质量顺序.json").read_text(encoding="utf-8"))["full_order"]
    if not 1 <= args.count <= len(order):
        parser.error("--count 必须为1到22")
    if args.ids and (len(args.ids) != len(set(args.ids)) or set(args.ids)-set(order)):
        parser.error("--ids 包含未知或重复ID")
    selected = [x for x in order if x in args.ids] if args.ids else order if args.all else order[:args.count]
    print("013 执行顺序: " + " → ".join(selected), flush=True)
    if args.plan:
        return
    os.environ["PYTHONPATH"] = str(ROOT / "05_代码" / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["EBSD_013_MODE"] = "IPF_GB"
    from ebsd_feedback.paths import FOLD_MODEL_ROOT, PREDICTION_ROOT, LOG_ROOT
    from ebsd_feedback.utils import atomic_json_dump
    from ebsd_feedback.data import AlloyRepository
    stages = [
        ("prepare-fold", None, "00_折准备/成分到描述符与曲线_GPR.joblib"),
        ("train-vae", "01_边界感知VAE.yaml", "01_边界感知VAE/VAE_最终模型.pt"),
        ("train-mechanics", "02_力学代理.yaml", "02_力学代理/力学代理_最终模型.pt"),
        ("train-descriptor", "02b_图像描述符代理.yaml", "02b_图像描述符代理/图像描述符_最终模型.pt"),
        ("train-diffusion", "03_基础条件扩散.yaml", "03_基础条件扩散/扩散_最终模型.pt"),
        ("sample-base", "05b_基础扩散三图推理.yaml", "base/预测说明.json"),
        ("train-feedback", "04_力学反馈微调.yaml", "04_力学反馈扩散/扩散_最终模型.pt"),
        ("sample-feedback", "05_三图推理.yaml", "feedback/预测说明.json"),
    ]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with lock(ROOT / "09_训练日志_服务器生成" / ".run013.lock"):
        subprocess.run([sys.executable, "-u", "-m", "ebsd_feedback.cli", "check"], cwd=ROOT, check=True)
        if args.check_only:
            print("检查通过，未开始训练。", flush=True)
            return
        repo = AlloyRepository()
        tracked = list((ROOT / "05_代码" / "src").rglob("*.py")) + list((ROOT / "06_配置文件").glob("*.yaml"))
        tracked += [Path(repo.row(i)["image_path"]) for i in repo.alloy_ids]
        for directory in ("01_成分表", "02_组织描述符", "03_力学目标", "04_应力应变曲线", "05_GB_CSL统计"):
            tracked += list((ROOT / "02_整理后建模数据" / directory).glob("*.csv"))
        fingerprint = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(tracked))}
        for position, holdout in enumerate(selected, 1):
            fold = FOLD_MODEL_ROOT / holdout
            signature = fold / "013输入指纹.json"
            if signature.exists():
                previous = json.loads(signature.read_text(encoding="utf-8"))
                if previous != fingerprint:
                    raise RuntimeError(f"{holdout} 代码/配置/数据已变化，停止混用旧断点。请保留现有结果，使用独立项目目录开展新实验。")
            else:
                if fold.exists() and any(fold.rglob("*.pt")):
                    raise RuntimeError("发现无013指纹的模型，不自动复用其他项目权重")
                atomic_json_dump(fingerprint, signature)
            for stage_index, (stage, config, relative) in enumerate(stages, 1):
                sample = stage.startswith("sample-")
                target = (PREDICTION_ROOT / holdout if sample else fold) / relative
                if target.exists():
                    print(f"[{position}/{len(selected)} {holdout}] 已完成，跳过 {stage}", flush=True)
                    continue
                command = [sys.executable, "-u", "-m", "ebsd_feedback.cli", "sample" if sample else stage, "--holdout", holdout]
                if config:
                    command += ["--config", config]
                print(f"\n===== 第{position}/{len(selected)}折 {holdout} 阶段{stage_index}/{len(stages)} {stage} =====", flush=True)
                process = subprocess.Popen(command, cwd=ROOT)
                payload = {"fold": holdout, "fold_position": position, "selected_ids": selected,
                    "stage": stage, "pid": process.pid, "status": "running", "started_at": time.time()}
                status = LOG_ROOT / "一键运行状态.json"
                atomic_json_dump(payload, status)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=45)
                            break
                        except subprocess.TimeoutExpired:
                            print(f"[运行中] {holdout} / {stage}，PID={process.pid}（编译/GPR阶段也会显示此提示）", flush=True)
                except KeyboardInterrupt:
                    import signal
                    process.send_signal(signal.SIGINT)
                    print("已请求子进程保存断点，请等待退出。", flush=True)
                    process.wait()
                    payload.update(status="interrupted", finished_at=time.time())
                    atomic_json_dump(payload, status)
                    raise
                payload.update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=time.time())
                atomic_json_dump(payload, status)
                if code != 0:
                    raise SystemExit(f"{holdout}/{stage} 退出码={code}；修复后重跑相同命令会从保存断点继续。")
                if not target.exists():
                    raise RuntimeError(f"阶段退出成功但缺少完成文件: {target}")
        print("\n指定折数全部完成。请查看10_预测结果_服务器生成/IPF_GB。", flush=True)


if __name__ == "__main__":
    main()
