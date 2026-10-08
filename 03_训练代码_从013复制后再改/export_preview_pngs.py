# -*- coding: utf-8 -*-
"""Export 256x256 RGB previews that match crops.csv. Does not touch latents or the csv."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

FULL_H = 448
FULL_W = 768
WIN = 256
VIEWS = {
    "v0_rot0": (),
    "v2_rot180": ("rot180",),
    "v4_flipLR": ("flipLR",),
    "v6_flipLR_rot180": ("flipLR", "rot180"),
}
TRANS = {
    "rot90": Image.Transpose.ROTATE_90,
    "rot180": Image.Transpose.ROTATE_180,
    "rot270": Image.Transpose.ROTATE_270,
    "flipLR": Image.Transpose.FLIP_LEFT_RIGHT,
}
IMG_ROOT = Path(r"D:\013_IPF_GB混合方案项目\03_图像数据\IPF_GB")
CSV_PATH = Path(r"D:\016_切块扩散第一轮\01_窗口清单_每个窗口的位置和描述符\crops.csv")
OUT = Path(r"D:\016_切块扩散第一轮\06_可打开的窗口小图")
SUMMARY = Path(r"C:\Users\ASUS\AppData\Local\Temp\016_png_summary.txt")


def apply_ops(image: Image.Image, ops) -> Image.Image:
    for op in ops:
        image = image.transpose(TRANS[op])
    return image


def load_resized_rgb(path: Path) -> Image.Image:
    """Same RGB path as round1_common.load_resized: convert RGB, Lanczos to 768x448."""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((FULL_W, FULL_H), Image.Resampling.LANCZOS)


def main() -> None:
    rows = []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if len(rows) != 5376:
        raise RuntimeError(f"crops.csv rows {len(rows)} != 5376")
    by_alloy = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["alloy_id"], row["view"], row["pixel_y"], row["pixel_x"])
        if key in seen:
            raise RuntimeError(f"duplicate {key}")
        seen.add(key)
        by_alloy[row["alloy_id"]].append(row)
    if "ID03" in by_alloy:
        raise RuntimeError("ID03 must not be exported")
    if len(by_alloy) != 21:
        raise RuntimeError(f"alloys {len(by_alloy)}")
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    example = ""
    for alloy in sorted(by_alloy):
        src = IMG_ROOT / alloy / f"{alloy}_真实EBSD_IPF加晶界.bmp"
        if not src.is_file():
            raise FileNotFoundError(src)
        base = load_resized_rgb(src)
        cache = {}
        dest = OUT / alloy
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for row in by_alloy[alloy]:
            view = row["view"]
            if view not in VIEWS:
                raise RuntimeError(f"unexpected view {alloy} {view}")
            py = int(row["pixel_y"])
            px = int(row["pixel_x"])
            if py != int(row["latent_y"]) * 4 or px != int(row["latent_x"]) * 4:
                raise RuntimeError(f"pixel != latent*4 {alloy} {view} {py} {px}")
            if view not in cache:
                turned = apply_ops(base, VIEWS[view])
                if turned.size != (FULL_W, FULL_H):
                    raise RuntimeError(f"{alloy} {view} size {turned.size}")
                cache[view] = turned
            image = cache[view]
            if py < 0 or px < 0 or py + WIN > image.size[1] or px + WIN > image.size[0]:
                raise RuntimeError(f"oob {alloy} {view} y={py} x={px}")
            crop = image.crop((px, py, px + WIN, py + WIN))
            if crop.size != (WIN, WIN) or crop.mode != "RGB":
                raise RuntimeError(f"crop {crop.size} {crop.mode}")
            name = f"{alloy}_{view}_y{py}_x{px}.png"
            crop.save(dest / name, format="PNG")
            n += 1
            if not example:
                example = str(dest / name)
        base.close()
        for image in cache.values():
            image.close()
        pngs = list(dest.glob("*.png"))
        if len(pngs) != 256 or n != 256:
            raise RuntimeError(f"{alloy} wrote {n} files, dir has {len(pngs)}")
        counts[alloy] = len(pngs)
        print(f"{alloy} {len(pngs)}", flush=True)
    note = (
        "这些是训练窗口的图片，每张 256×256，可以在资源管理器里直接打开。\r\n"
        "训练仍然按清单从整图潜变量上现切，不会读取这个文件夹里的图片。\r\n"
        "90 度朝向没有小图，原因和 crops.csv 里没有它们一样："
        "转 90 度后画布变成 448×768，右侧再留出 256 像素当验证，就放不下一个 256 的训练窗口。\r\n"
    )
    (OUT / "这里是什么.txt").write_text(note, encoding="utf-8")
    lines = [f"{alloy}\t{count}" for alloy, count in sorted(counts.items())]
    lines.append(f"TOTAL\t{sum(counts.values())}")
    lines.append(f"EXAMPLE\t{example}")
    lines.append(f"OUT\t{OUT}")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DONE {sum(counts.values())} {example}", flush=True)


if __name__ == "__main__":
    main()
