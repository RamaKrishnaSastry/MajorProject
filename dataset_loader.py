"""
IRDID / IDRiD Dataset Loader

Expects the following on-disk layout::

    dataset_path/
    ├── images/
    │   ├── IDRiD_001.jpg
    │   └── ...
    └── DME_Grades.csv          # columns: image_id, dme_grade  (+ optional dr_grade)

If the real dataset is unavailable, a synthetic fallback is generated
automatically so that the full pipeline can be validated offline.
"""

import os
import math
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import preprocess_fundus_image, batch_preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DME_CLASSES = 3      # 0 = No DME, 1 = Possible DME, 2 = Present DME
DR_CLASSES = 5       # 0-4
BATCH_SIZE = 8
TARGET_SIZE = (512, 512)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helper – synthetic dataset generation
# ---------------------------------------------------------------------------

def _generate_synthetic_dataset(
    dataset_path: str,
    n_samples: int = 100,
    seed: int = 42,
) -> None:
    """Create a minimal synthetic dataset for offline testing."""
    rng = np.random.default_rng(seed)
    images_dir = Path(dataset_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(1, n_samples + 1):
        name = f"IDRiD_{i:03d}.jpg"
        img = rng.integers(30, 200, size=(256, 256, 3), dtype=np.uint8)
        # Add a bright circular region to mimic a fundus
        cv2.circle(img, (128, 128), 100, (180, 180, 180), -1)
        cv2.imwrite(str(images_dir / name), img)
        records.append(
            {
                "image_id": name,
                "dr_grade": int(rng.integers(0, 5)),
                "dme_grade": int(rng.integers(0, 3)),
            }
        )

    pd.DataFrame(records).to_csv(
        Path(dataset_path) / "DME_Grades.csv", index=False
    )
    logger.info("Synthetic dataset created at %s (%d samples)", dataset_path, n_samples)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class IDRiDDataset:
    """TensorFlow data pipeline for the IDRiD / IRDID dataset.

    Parameters
    ----------
    dataset_path:
        Root directory containing ``images/`` and ``DME_Grades.csv``.
    split:
        ``'train'`` or ``'val'``.
    augment:
        Whether to apply random augmentation (train split only).
    val_fraction:
        Fraction of data to use for validation (default 0.2).
    target_size:
        Spatial size fed to the model.
    batch_size:
        Mini-batch size.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        augment: bool = True,
        val_fraction: float = 0.2,
        target_size: tuple = TARGET_SIZE,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.augment = augment and split == "train"
        self.val_fraction = val_fraction
        self.target_size = target_size
        self.batch_size = batch_size
        self.seed = seed

        self._prepare_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_data(self) -> None:
        csv_path = self.dataset_path / "DME_Grades.csv"
        if not csv_path.exists():
            # Try DR_Grades.csv as fallback
            csv_path = self.dataset_path / "DR_Grades.csv"
        if not csv_path.exists():
            logger.warning(
                "No CSV file found in %s – generating synthetic dataset.",
                self.dataset_path,
            )
            _generate_synthetic_dataset(str(self.dataset_path))
            csv_path = self.dataset_path / "DME_Grades.csv"

        df = pd.read_csv(csv_path)

        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]
        if "image_id" not in df.columns:
            raise ValueError("CSV must contain an 'image_id' column.")
        if "dme_grade" not in df.columns:
            raise ValueError("CSV must contain a 'dme_grade' column.")

        images_dir = self.dataset_path / "images"

        # Resolve full image paths and drop rows with missing files
        def _resolve(row):
            name = str(row["image_id"])
            p = images_dir / name
            if p.exists():
                return str(p)
            # Try common extensions
            for ext in IMG_EXTENSIONS:
                stem = Path(name).stem
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    return str(candidate)
            return None

        df["image_path"] = df.apply(_resolve, axis=1)
        missing = df["image_path"].isna().sum()
        if missing:
            logger.warning("Dropping %d rows with missing image files.", missing)
        df = df.dropna(subset=["image_path"]).reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError("No valid images found. Check dataset_path.")

        # Train / val split
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(df))
        val_n = max(1, int(len(df) * self.val_fraction))
        val_idx = idx[:val_n]
        train_idx = idx[val_n:]

        chosen = train_idx if self.split == "train" else val_idx
        df = df.iloc[chosen].reset_index(drop=True)

        self.df = df
        self.image_paths = df["image_path"].tolist()
        self.dme_labels = df["dme_grade"].astype(int).tolist()
        self.dr_labels = (
            df["dr_grade"].astype(int).tolist()
            if "dr_grade" in df.columns
            else [0] * len(df)
        )

        # Compute class weights for imbalanced datasets
        counts = np.bincount(self.dme_labels, minlength=DME_CLASSES).astype(float)
        zero_classes = np.where(counts == 0)[0].tolist()
        if zero_classes:
            logger.warning(
                "Zero samples for DME class(es) %s in %s split – check data quality.",
                zero_classes, self.split,
            )
        counts = np.where(counts == 0, 1, counts)  # avoid division by zero
        total = counts.sum()
        self._class_weights = {
            i: total / (DME_CLASSES * counts[i]) for i in range(DME_CLASSES)
        }

    # ------------------------------------------------------------------
    # tf.data pipeline helpers
    # ------------------------------------------------------------------

    def _load_and_preprocess(self, image_path: tf.Tensor, dme_label: tf.Tensor, dr_label: tf.Tensor):
        """tf.py_function wrapper around our NumPy preprocessing."""
        img_np = preprocess_fundus_image(
            image_path.numpy().decode("utf-8"), self.target_size
        )
        img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)
        dme_oh = tf.one_hot(dme_label, DME_CLASSES)
        dr_val = tf.cast(dr_label, tf.float32)
        return img_tensor, dme_oh, dr_val

    def _tf_load(self, image_path, dme_label, dr_label):
        img, dme_oh, dr_val = tf.py_function(
            func=self._load_and_preprocess,
            inp=[image_path, dme_label, dr_label],
            Tout=[tf.float32, tf.float32, tf.float32],
        )
        h, w = self.target_size
        img.set_shape([h, w, 3])
        dme_oh.set_shape([DME_CLASSES])
        dr_val.set_shape([])
        return img, {"dme_risk": dme_oh, "dr_output": dr_val}

    @staticmethod
    def _augment(img, labels):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(self) -> tf.data.Dataset:
        """Build and return a batched, prefetched ``tf.data.Dataset``.

        Each element is ``(image, {"dme_risk": one_hot, "dr_output": scalar})``.
        """
        paths_ds = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.dme_labels, self.dr_labels)
        )
        ds = paths_ds.map(self._tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        if self.augment:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        if self.split == "train":
            ds = ds.shuffle(buffer_size=min(500, len(self.image_paths)),
                            seed=self.seed)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_class_weights(self) -> dict:
        """Return per-class weights for DME labels (for ``class_weight`` arg)."""
        return self._class_weights

    def __len__(self) -> int:
        return len(self.image_paths)

    def steps_per_epoch(self) -> int:
        return math.ceil(len(self) / self.batch_size)

    def visualize_batch(self, n_samples: int = 4) -> None:
        """Display *n_samples* images from the first batch with their labels."""
        ds = self.get_dataset().unbatch().take(n_samples)
        dme_names = ["No DME", "Possible DME", "Present DME"]

        fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
        if n_samples == 1:
            axes = [axes]

        for ax, (img, labels) in zip(axes, ds):
            ax.imshow(img.numpy())
            dme_idx = int(np.argmax(labels["dme_risk"].numpy()))
            dr_val = float(labels["dr_output"].numpy())
            ax.set_title(f"DME: {dme_names[dme_idx]}\nDR: {dr_val:.0f}", fontsize=9)
            ax.axis("off")

        plt.suptitle(f"Sample batch – {self.split} split", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"sample_batch_{self.split}.png", dpi=100)
        plt.show()
        logger.info("Saved sample_batch_%s.png", self.split)
