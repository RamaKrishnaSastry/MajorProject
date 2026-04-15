"""
Prepare IDRiD data with domain-shift robustness safeguards.

Key differences vs simple split scripts:
1) Never mixes official test labels/images into training by default.
2) Uses joint DR+DME stratification for train/val split stability.
3) Adds train-only style perturbation variants to improve robustness.
4) Exports explicit train/val/test CSVs and image folders for auditability.

Usage (Kaggle notebook cell):
    python tools/prepare_idrid_domain_robust.py

Optional arguments:
    --output-dir /kaggle/working
    --val-size 0.2
    --augment-per-image 1
    --seed 42
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Paths:
    raw_root: str
    output_dir: str

    @property
    def train_img_src(self) -> str:
        return os.path.join(
            self.raw_root,
            "B.%20Disease%20Grading",
            "B. Disease Grading",
            "1. Original Images",
            "a. Training Set",
        )

    @property
    def test_img_src(self) -> str:
        return os.path.join(
            self.raw_root,
            "B.%20Disease%20Grading",
            "B. Disease Grading",
            "1. Original Images",
            "b. Testing Set",
        )

    @property
    def train_csv_src(self) -> str:
        return os.path.join(
            self.raw_root,
            "B.%20Disease%20Grading",
            "B. Disease Grading",
            "2. Groundtruths",
            "a. IDRiD_Disease Grading_Training Labels.csv",
        )

    @property
    def test_csv_src(self) -> str:
        return os.path.join(
            self.raw_root,
            "B.%20Disease%20Grading",
            "B. Disease Grading",
            "2. Groundtruths",
            "b. IDRiD_Disease Grading_Testing Labels.csv",
        )

    @property
    def train_dst(self) -> str:
        return os.path.join(self.output_dir, "train_images")

    @property
    def val_dst(self) -> str:
        return os.path.join(self.output_dir, "val_images")

    @property
    def test_dst(self) -> str:
        return os.path.join(self.output_dir, "test_images")


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    image_col: Optional[str] = None
    dr_col: Optional[str] = None
    dme_col: Optional[str] = None

    for col in cols:
        low = col.lower()
        if image_col is None and "image" in low:
            image_col = col
        elif dr_col is None and "retinopathy" in low:
            dr_col = col
        elif dme_col is None and "macular" in low:
            dme_col = col

    if not image_col or not dr_col or not dme_col:
        raise ValueError(
            "Could not infer required columns from CSV. "
            f"Detected image={image_col}, dr={dr_col}, dme={dme_col}, cols={cols}"
        )

    return image_col, dr_col, dme_col


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    image_col, dr_col, dme_col = _detect_columns(df)

    cleaned = pd.DataFrame(
        {
            "Image name": df[image_col].astype(str).str.strip() + ".jpg",
            "Retinopathy grade": df[dr_col].astype(int),
            "Risk of macular edema": df[dme_col].astype(int),
        }
    )
    return cleaned


def resolve_image_path(image_name: str, train_src: str, test_src: str) -> Optional[str]:
    cands = [
        os.path.join(train_src, image_name),
        os.path.join(test_src, image_name),
    ]
    for cand in cands:
        if os.path.exists(cand):
            return cand

    stem, _ = os.path.splitext(image_name)
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".bmp", ".JPG", ".PNG", ".TIF"]:
        for root in [train_src, test_src]:
            cand = os.path.join(root, stem + ext)
            if os.path.exists(cand):
                return cand
    return None


def _joint_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["Retinopathy grade"].astype(str)
        + "_"
        + df["Risk of macular edema"].astype(str)
    )


def stratified_train_val_split(
    train_df: pd.DataFrame,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    joint = _joint_key(train_df)
    counts = joint.value_counts()

    # Joint stratification is preferred for multi-task stability.
    if (counts >= 2).all() and counts.size > 1:
        strat = joint
        strategy = "joint_dme_dr"
    else:
        # Fallback to DR stratification if joint bins are too sparse.
        dr_counts = train_df["Retinopathy grade"].value_counts()
        if (dr_counts >= 2).all() and dr_counts.size > 1:
            strat = train_df["Retinopathy grade"]
            strategy = "dr_only"
        else:
            strat = None
            strategy = "random"

    tr, va = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=strat,
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True), strategy


def _apply_style_perturbation(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train-only style perturbation to reduce domain sensitivity."""
    img = img_bgr.astype(np.float32) / 255.0

    # Brightness/contrast jitter.
    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-0.08, 0.08))
    img = np.clip(img * alpha + beta, 0.0, 1.0)

    # Gamma perturbation.
    gamma = float(rng.uniform(0.85, 1.15))
    img = np.power(np.clip(img, 1e-6, 1.0), gamma)

    # HSV jitter for camera color variability.
    hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h_shift = int(rng.integers(-6, 7))
    s_scale = float(rng.uniform(0.85, 1.2))
    v_scale = float(rng.uniform(0.85, 1.2))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + h_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * s_scale, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * v_scale, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Mild blur/noise to simulate acquisition differences.
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), sigmaX=0)
    noise_std = float(rng.uniform(0.0, 0.02))
    noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)

    # JPEG artifact simulation.
    jpg_q = int(rng.integers(65, 101))
    enc = cv2.imencode(
        ".jpg",
        (img * 255.0).astype(np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q],
    )[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec


def _copy_or_augment_row(
    row: pd.Series,
    dst_dir: str,
    src_train: str,
    src_test: str,
    augment_index: Optional[int],
    rng: np.random.Generator,
) -> Optional[Dict]:
    image_name = str(row["Image name"])
    src = resolve_image_path(image_name, src_train, src_test)
    if src is None:
        print(f"Warning: missing image for {image_name}")
        return None

    stem, ext = os.path.splitext(image_name)
    ext = ext if ext else ".jpg"

    if augment_index is None:
        out_name = f"{stem}{ext}"
        out_path = os.path.join(dst_dir, out_name)
        shutil.copy2(src, out_path)
    else:
        out_name = f"{stem}__aug{augment_index}{ext}"
        out_path = os.path.join(dst_dir, out_name)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to decode {src}")
            return None
        img_aug = _apply_style_perturbation(img, rng)
        cv2.imwrite(out_path, img_aug)

    return {
        "Image name": out_name,
        "Retinopathy grade": int(row["Retinopathy grade"]),
        "Risk of macular edema": int(row["Risk of macular edema"]),
    }


def export_split(
    df: pd.DataFrame,
    dst_dir: str,
    src_train: str,
    src_test: str,
    augment_per_image: int,
    seed: int,
    train_mode: bool,
) -> pd.DataFrame:
    os.makedirs(dst_dir, exist_ok=True)
    for p in Path(dst_dir).glob("*"):
        if p.is_file():
            p.unlink()

    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    for _, row in df.iterrows():
        base_row = _copy_or_augment_row(
            row=row,
            dst_dir=dst_dir,
            src_train=src_train,
            src_test=src_test,
            augment_index=None,
            rng=rng,
        )
        if base_row is not None:
            rows.append(base_row)

        if train_mode:
            for idx in range(1, augment_per_image + 1):
                aug_row = _copy_or_augment_row(
                    row=row,
                    dst_dir=dst_dir,
                    src_train=src_train,
                    src_test=src_test,
                    augment_index=idx,
                    rng=rng,
                )
                if aug_row is not None:
                    rows.append(aug_row)

    return pd.DataFrame(rows)


def _dist(df: pd.DataFrame, col: str) -> Dict[str, int]:
    s = df[col].value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in s.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IDRiD with domain-shift robust train split")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/kaggle/input/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset",
    )
    parser.add_argument("--output-dir", type=str, default="/kaggle/working")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--augment-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = Paths(raw_root=args.raw_root, output_dir=args.output_dir)

    os.makedirs(paths.output_dir, exist_ok=True)
    os.makedirs(paths.train_dst, exist_ok=True)
    os.makedirs(paths.val_dst, exist_ok=True)
    os.makedirs(paths.test_dst, exist_ok=True)

    train_df = load_and_clean(paths.train_csv_src)
    test_df = load_and_clean(paths.test_csv_src)

    train_split, val_split, split_strategy = stratified_train_val_split(
        train_df,
        val_size=float(args.val_size),
        seed=int(args.seed),
    )

    # Keep official test untouched to avoid label leakage.
    train_export = export_split(
        train_split,
        dst_dir=paths.train_dst,
        src_train=paths.train_img_src,
        src_test=paths.test_img_src,
        augment_per_image=max(0, int(args.augment_per_image)),
        seed=int(args.seed),
        train_mode=True,
    )
    val_export = export_split(
        val_split,
        dst_dir=paths.val_dst,
        src_train=paths.train_img_src,
        src_test=paths.test_img_src,
        augment_per_image=0,
        seed=int(args.seed) + 11,
        train_mode=False,
    )
    test_export = export_split(
        test_df,
        dst_dir=paths.test_dst,
        src_train=paths.train_img_src,
        src_test=paths.test_img_src,
        augment_per_image=0,
        seed=int(args.seed) + 23,
        train_mode=False,
    )

    train_csv_path = os.path.join(paths.output_dir, "train_labels.csv")
    val_csv_path = os.path.join(paths.output_dir, "val_labels.csv")
    test_csv_path = os.path.join(paths.output_dir, "test_labels.csv")

    train_export.to_csv(train_csv_path, index=False)
    val_export.to_csv(val_csv_path, index=False)
    test_export.to_csv(test_csv_path, index=False)

    model_dataset = train_export.copy()
    model_dataset["image_path"] = model_dataset["Image name"].apply(
        lambda x: os.path.join(paths.train_dst, x)
    )
    model_dataset_path = os.path.join(paths.output_dir, "model_training_dataset.csv")
    model_dataset.to_csv(model_dataset_path, index=False)

    split_info = {
        "split_strategy": split_strategy,
        "seed": int(args.seed),
        "val_size": float(args.val_size),
        "augment_per_image": int(args.augment_per_image),
        "source_counts": {
            "train_original": int(len(train_split)),
            "val_original": int(len(val_split)),
            "test_original": int(len(test_df)),
            "train_exported": int(len(train_export)),
            "val_exported": int(len(val_export)),
            "test_exported": int(len(test_export)),
        },
        "dr_distribution": {
            "train": _dist(train_export, "Retinopathy grade"),
            "val": _dist(val_export, "Retinopathy grade"),
            "test": _dist(test_export, "Retinopathy grade"),
        },
        "dme_distribution": {
            "train": _dist(train_export, "Risk of macular edema"),
            "val": _dist(val_export, "Risk of macular edema"),
            "test": _dist(test_export, "Risk of macular edema"),
        },
        "notes": [
            "Official IDRiD test labels are not used for training.",
            "Train split includes style-perturbed replicas to improve domain robustness.",
            "Use val_labels.csv for threshold tuning; keep test_labels.csv for final reporting.",
        ],
    }

    split_info_path = os.path.join(paths.output_dir, "split_info_domain_robust.json")
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("\nDataset preparation complete (domain-robust mode).")
    print(f"Training Images Folder: {paths.train_dst}")
    print(f"Validation Images Folder: {paths.val_dst}")
    print(f"Testing Images Folder: {paths.test_dst}")
    print(f"Training CSV: {train_csv_path}")
    print(f"Validation CSV: {val_csv_path}")
    print(f"Testing CSV: {test_csv_path}")
    print(f"Model Training Dataset CSV: {model_dataset_path}")
    print(f"Split Diagnostics: {split_info_path}")


if __name__ == "__main__":
    main()
