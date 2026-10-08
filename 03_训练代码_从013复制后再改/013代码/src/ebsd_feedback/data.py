from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import (
    COMPOSITION_COLUMNS,
    CURVE_TARGET_COLUMNS,
    DENSE_CURVE_MAX_STRAIN_PCT,
    DENSE_CURVE_STRAIN_STEP_PCT,
    DESCRIPTOR_COLUMNS,
    canonical_alloy_id,
)
from .paths import IMAGE_CHANNELS, IMAGE_MODE, IMAGE_ROOT, MODELING_DATA_ROOT, PROJECT_ROOT, REAL_IMAGE_ROOT


def pack_image(image: Image.Image) -> Image.Image:
    """Internal RGBA container: A is derived GB white-interior, NOT source alpha."""
    if image.mode == "RGBA":
        return image.copy()
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gb = (rgb.max(axis=2) >= 51).astype(np.uint8) * 255
    return Image.fromarray(np.concatenate([rgb, gb[..., None]], axis=2), mode="RGBA")


def resize_packed(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    # Never resize RGBA as transparency: it would premultiply the IPF colors.
    packed = pack_image(image)
    rgb = packed.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    gb = packed.getchannel("A").resize(size, Image.Resampling.NEAREST)
    return Image.merge("RGBA", (*rgb.split(), gb))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert the RGB IPF+GB composite to four normalized channels.

    The source BMPs are RGB, not RGBA. Black pixels are the exported GB overlay;
    they are retained in RGB and also exposed as an explicit fourth boundary mask.
    The threshold is recorded in the dataset manifest by the training scripts.
    """
    # Extract native-resolution overlay BEFORE resampling (when not packed yet).
    array = np.asarray(pack_image(image), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resize_full_frame(image: Image.Image, height: int, width: int) -> Image.Image:
    return resize_packed(image, (width, height))


def random_square_crop(image: Image.Image, size: int, min_scale: float = 0.55) -> Image.Image:
    width, height = image.size
    side = max(16, int(min(width, height) * random.uniform(min_scale, 1.0)))
    left = random.randint(0, max(width - side, 0))
    top = random.randint(0, max(height - side, 0))
    crop = image.crop((left, top, left + side, top + side))
    return resize_packed(crop, (size, size))


class AlloyRepository:
    def __init__(self) -> None:
        composition_path = MODELING_DATA_ROOT / "01_成分表" / "22个合金_9元素成分.csv"
        descriptor_filename = (
            "22个合金组织描述符_GB_CSL3.csv"
            if IMAGE_MODE == "GB_CSL3"
            else "22个合金组织描述符_GB.csv"
        )
        descriptor_path = MODELING_DATA_ROOT / "02_组织描述符" / descriptor_filename
        curve_path = MODELING_DATA_ROOT / "03_力学目标" / "22个合金_9个力学目标.csv"
        mapping_path = MODELING_DATA_ROOT / "05_GB_CSL统计" / "22个合金文件对应表.csv"
        composition = pd.read_csv(composition_path)
        descriptors = pd.read_csv(descriptor_path)
        curves = pd.read_csv(curve_path)
        mapping = pd.read_csv(mapping_path)
        for frame in (composition, descriptors, curves, mapping):
            frame["alloy_id"] = frame["alloy_id"].map(canonical_alloy_id)
        self.table = composition[["alloy_id", *COMPOSITION_COLUMNS]].merge(
            descriptors[["alloy_id", *DESCRIPTOR_COLUMNS]], on="alloy_id", validate="one_to_one"
        )
        self.table = self.table.merge(
            curves[["alloy_id", *CURVE_TARGET_COLUMNS]], on="alloy_id", validate="one_to_one"
        )
        self.table = self.table.merge(
            mapping[["alloy_id", "real_image_relative_path"]],
            on="alloy_id",
            validate="one_to_one",
        ).sort_values("alloy_id", ignore_index=True)
        if len(self.table) != 22:
            raise ValueError(f"期望22个合金，实际得到{len(self.table)}个")
        if self.table[[*COMPOSITION_COLUMNS, *DESCRIPTOR_COLUMNS, *CURVE_TARGET_COLUMNS]].isna().any().any():
            raise ValueError("建模主表包含缺失数值")
        # Original IPF-Z+GB RGB exports, independent of 012 output folders.
        self.table["image_mode"] = IMAGE_MODE
        self.table["image_path"] = self.table["alloy_id"].map(
            lambda alloy_id: str(IMAGE_ROOT / str(alloy_id) /
                                  f"{alloy_id}_真实EBSD_IPF加晶界.bmp")
        )
        missing = [path for path in self.table["image_path"] if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                f"{IMAGE_MODE} 图像缺失 {len(missing)} 个，例如: {missing[0]}"
            )

    @property
    def alloy_ids(self) -> list[str]:
        return self.table["alloy_id"].tolist()

    def split_ids(self, holdout_id: str) -> tuple[list[str], list[str]]:
        holdout_id = canonical_alloy_id(holdout_id)
        if holdout_id not in self.alloy_ids:
            raise ValueError(f"未知留出合金: {holdout_id}")
        return [value for value in self.alloy_ids if value != holdout_id], [holdout_id]

    def row(self, alloy_id: str) -> pd.Series:
        alloy_id = canonical_alloy_id(alloy_id)
        rows = self.table[self.table["alloy_id"] == alloy_id]
        if len(rows) != 1:
            raise KeyError(alloy_id)
        return rows.iloc[0]


def load_fold_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["alloy_id"] = frame["alloy_id"].map(canonical_alloy_id)
    return frame


class EbsdDataset(Dataset[dict[str, Any]]):
    """Alloy-balanced image dataset; repetitions add crops, not independent alloy labels."""

    def __init__(
        self,
        records: pd.DataFrame,
        alloy_ids: Sequence[str],
        samples_per_alloy: int,
        view: str,
        image_height: int,
        image_width: int,
        patch_size: int,
        random_augment: bool,
        cache_images: bool = False,
        include_metadata: bool = True,
    ) -> None:
        self.records = records.copy()
        self.records["alloy_id"] = self.records["alloy_id"].map(canonical_alloy_id)
        wanted = {canonical_alloy_id(value) for value in alloy_ids}
        self.records = self.records[self.records["alloy_id"].isin(wanted)].reset_index(drop=True)
        if set(self.records["alloy_id"]) != wanted:
            missing = wanted - set(self.records["alloy_id"])
            raise ValueError(f"折清单缺少合金: {sorted(missing)}")
        self.samples_per_alloy = samples_per_alloy
        self.view = view
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        # Full images carry whole-scan descriptor labels. Cropping changes their
        # physical scale/region statistics; use full frames for supervised stages.
        self.random_augment = random_augment if view == "patch" else False
        self.include_metadata = include_metadata
        self.image_cache: dict[str, np.ndarray] = {}
        if cache_images:
            for image_path in self.records["image_path"].astype(str).unique():
                with Image.open(image_path) as source:
                    self.image_cache[image_path] = np.asarray(
                        pack_image(source), dtype=np.uint8
                    ).copy()
        self.full_tensor_cache = {}
        if cache_images and view == "full":
            for path, pixels in self.image_cache.items():
                self.full_tensor_cache[path] = image_to_tensor(resize_full_frame(
                    Image.fromarray(pixels), self.image_height, self.image_width))
        dense_path = (
            MODELING_DATA_ROOT
            / "04_应力应变曲线"
            / "22个合金中位数曲线长表.csv"
        )
        dense_table = pd.read_csv(dense_path)
        dense_table["alloy_id"] = dense_table["alloy_id"].map(canonical_alloy_id)
        self.dense_strain = np.arange(
            0.0,
            DENSE_CURVE_MAX_STRAIN_PCT + DENSE_CURVE_STRAIN_STEP_PCT * 0.5,
            DENSE_CURVE_STRAIN_STEP_PCT,
            dtype=np.float32,
        )
        self.dense_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for alloy_id in wanted:
            rows = dense_table[dense_table["alloy_id"] == alloy_id].sort_values(
                "engineering_strain_pct"
            )
            if rows.empty:
                raise ValueError(f"缺少合金级中位数逐点曲线: {alloy_id}")
            strain = rows["engineering_strain_pct"].to_numpy(dtype=np.float32)
            stress = rows["median_stress_MPa"].to_numpy(dtype=np.float32)
            valid = self.dense_strain <= float(strain[-1]) + 1e-6
            interpolated = np.zeros_like(self.dense_strain)
            interpolated[valid] = np.interp(self.dense_strain[valid], strain, stress)
            self.dense_curves[alloy_id] = (interpolated, valid)
        self._dense_strain_tensor = torch.from_numpy(self.dense_strain)
        self._metadata_cache: list[dict[str, torch.Tensor]] = []
        if self.include_metadata:
            numeric_groups = {
                "composition": [f"comp_{name}" for name in COMPOSITION_COLUMNS],
                "descriptor_true": [f"desc_true_{name}" for name in DESCRIPTOR_COLUMNS],
                "descriptor_condition": [f"desc_cond_{name}" for name in DESCRIPTOR_COLUMNS],
                "descriptor_std": [f"desc_std_{name}" for name in DESCRIPTOR_COLUMNS],
                "curve_true": [f"curve_true_{name}" for name in CURVE_TARGET_COLUMNS],
                "curve_baseline": [f"curve_baseline_{name}" for name in CURVE_TARGET_COLUMNS],
                "curve_residual": [f"curve_residual_{name}" for name in CURVE_TARGET_COLUMNS],
            }
            if "ori_ipf_bin_0_0" in self.records:
                from .orientation import ORIENTATION_COLUMNS
                numeric_groups["orientation_true"] = [f"ori_{name}" for name in ORIENTATION_COLUMNS]
            for _, row in self.records.iterrows():
                dense_stress, dense_mask = self.dense_curves[row["alloy_id"]]
                cached = {
                    "curve_dense_strain": self._dense_strain_tensor,
                    "curve_dense_stress": torch.from_numpy(dense_stress),
                    "curve_dense_mask": torch.from_numpy(dense_mask),
                }
                for key, columns in numeric_groups.items():
                    if all(column in row.index for column in columns):
                        cached[key] = torch.from_numpy(
                            row[columns].to_numpy(dtype=np.float32, copy=True)
                        )
                self._metadata_cache.append(cached)

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_alloy

    def _image(self, row: pd.Series) -> torch.Tensor:
        image_path = str(row["image_path"])
        if image_path in self.full_tensor_cache:
            return self.full_tensor_cache[image_path]
        if image_path in self.image_cache:
            image = Image.fromarray(self.image_cache[image_path])
        else:
            with Image.open(image_path) as source:
                image = pack_image(source)
        try:
            if self.view == "patch":
                if self.random_augment:
                    image = random_square_crop(image, self.patch_size)
                else:
                    side = min(image.size)
                    left = (image.width - side) // 2
                    top = (image.height - side) // 2
                    image = resize_packed(image.crop((left, top, left + side, top + side)),
                                          (self.patch_size, self.patch_size))
            elif self.view == "full":
                if self.random_augment and random.random() < 0.5:
                    scale = random.uniform(0.88, 1.0)
                    crop_w, crop_h = int(image.width * scale), int(image.height * scale)
                    left = random.randint(0, max(image.width - crop_w, 0))
                    top = random.randint(0, max(image.height - crop_h, 0))
                    image = image.crop((left, top, left + crop_w, top + crop_h))
                image = resize_full_frame(image, self.image_height, self.image_width)
            else:
                raise ValueError(f"未知图像视图: {self.view}")
            return image_to_tensor(image)
        finally:
            image.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = (index // self.samples_per_alloy) % len(self.records)
        row = self.records.iloc[row_index]
        item: dict[str, Any] = {"image": self._image(row), "alloy_id": row["alloy_id"]}
        if self.include_metadata:
            item.update(self._metadata_cache[row_index])
        return item


def repository_records() -> pd.DataFrame:
    repository = AlloyRepository()
    result = repository.table.copy()
    result = result.rename(
        columns={
            **{name: f"comp_{name}" for name in COMPOSITION_COLUMNS},
            **{name: f"desc_true_{name}" for name in DESCRIPTOR_COLUMNS},
            **{name: f"curve_true_{name}" for name in CURVE_TARGET_COLUMNS},
        }
    )
    return result
