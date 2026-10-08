"""Read-only source audit; writes independent visual/provenance outputs."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "05_代码" / "src"))
from ebsd_feedback.data import AlloyRepository, resize_full_frame, image_to_tensor, pack_image
from ebsd_feedback.constants import DESCRIPTOR_COLUMNS, COMPOSITION_COLUMNS, CURVE_TARGET_COLUMNS


def main():
    repo = AlloyRepository()
    out = ROOT / "04_数据审计"
    out.mkdir(parents=True, exist_ok=True)
    records, thumbnails = [], []
    for aid in repo.alloy_ids:
        source_path = Path(repo.row(aid)["image_path"])
        with Image.open(source_path) as im:
            source_rgb = np.asarray(im.convert("RGB"))
            black = source_rgb.max(axis=2) < 51
            packed = resize_full_frame(im, 448, 768)
            tensor = image_to_tensor(packed)
            rgb = packed.convert("RGB")
            gb = packed.getchannel("A")
            rgb.save(out / f"{aid}_IPF.png")
            gb.save(out / f"{aid}_GB_overlay.png")
            records.append({"alloy_id": aid, "source": source_path.relative_to(ROOT).as_posix(),
                "source_mode": im.mode, "source_width": im.width, "source_height": im.height,
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "boundary_fraction_native": float(black.mean()), "four_channel_shape": str(tuple(tensor.shape)),
                "boundary_fraction_resized": float((tensor[3] < 0).float().mean()),
                "threshold_rule": "max(R,G,B)<51 on original uint8 RGB; mask nearest resize; white interior",
                "note": "black overlay includes any black invalid pixels; not misorientation-based grain reconstruction"})
            tile = Image.new("RGB", (512, 182), "white")
            tile.paste(rgb.resize((256,150)), (0,30))
            tile.paste(gb.convert("RGB").resize((256,150)), (256,30))
            ImageDraw.Draw(tile).text((6,8), f"{aid}: IPF-Z+GB RGB | derived GB", fill="black")
            thumbnails.append(tile)
    sheet = Image.new("RGB", (1024, 182*11), "white")
    for i,tile in enumerate(thumbnails):
        sheet.paste(tile, ((i%2)*512, (i//2)*182))
    sheet.save(out / "22合金输入通道核对.jpg", quality=95)
    pd.DataFrame(records).to_csv(out / "22合金输入审计.csv", index=False, encoding="utf-8-sig")
    repo.table[["alloy_id", *COMPOSITION_COLUMNS, *DESCRIPTOR_COLUMNS, *CURVE_TARGET_COLUMNS]].to_csv(
        out / "22合金建模主表.csv", index=False, encoding="utf-8-sig")
    if any(not 0 < r["boundary_fraction_native"] < .5 for r in records):
        raise ValueError("黑色叠加比例异常，请检查审计图")
    (out / "数据说明.json").write_text(json.dumps({"count": len(records), "ipf_axis": "Z (user confirmed)",
        "rgb_is_not_xyz": True, "source_is_not_rgba": True, "output_channels": ["IPF_R", "IPF_G", "IPF_B", "GB_white_interior"],
        "limitation": "IPF colors do not uniquely encode full orientation; image proxy is not physical measurement",
        "modeling_tables": "Active 01 composition,02 GB descriptors,03 targets,04 dense curves,05 mapping copied from 012; 010 ancillary tables/raw CTF retained"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"22合金输入检查完成: {out}", flush=True)


if __name__ == "__main__":
    main()
