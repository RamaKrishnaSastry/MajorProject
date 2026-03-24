"""
dataset_loader.py
IDRiD dataset loader with QWK-aware sampling.

Expected dataset layout:
    dataset/
    ├── images/
    │   ├── IDRiD_001.jpg
    │   └── ...
    ├── DME_Grades.csv          (columns: image_id, dme_grade [0-2])
    └── IDRiD_metadata.csv      (optional; columns: image_id, dr_grade [0-4])

Class distribution (typical IDRiD):
    Grade 0 (No DME):      ~49 %
    Grade 1 (Mild DME):    ~33 %
    Grade 2 (Moderate DME):~18 %
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from preprocess import preprocess_fundus_image


# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_DME_CLASSES = 3
NUM_DR_CLASSES = 5
AUTOTUNE = tf.data.AUTOTUNE


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_and_preprocess_tf(image_path: str, label: int):
    """Load and preprocess a single image inside a tf.py_function wrapper."""
    image = tf.py_function(
        func=lambda p: preprocess_fundus_image(
            p.numpy().decode("utf-8"), target_size=IMAGE_SIZE
        ),
        inp=[image_path],
        Tout=tf.float32,
    )
    image.set_shape((*IMAGE_SIZE, 3))
    return image, label


def _augment(image: tf.Tensor, label: int):
    """
    Light augmentation valid for fundus images:
    - Horizontal flip  (medically valid: left/right eye symmetry)
    - ±15 % brightness
    - ±15° rotation
    Vertical flip is NOT applied (anatomically invalid).
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)

    angle = tf.random.uniform((), minval=-15.0, maxval=15.0) * (np.pi / 180.0)
    image = _rotate_image(image, angle)
    return image, label


def _rotate_image(image: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    """Rotate image by *angle* radians using bilinear interpolation."""
    import tensorflow_addons as tfa  # optional dependency
    return tfa.image.rotate(image, angle, interpolation="BILINEAR")


def _rotate_image_fallback(image: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    """Rotation via tf.raw_ops when tensorflow_addons is unavailable."""
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    cx = tf.cast(w, tf.float32) / 2.0
    cy = tf.cast(h, tf.float32) / 2.0
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    transform = [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a,
                 sin_a, cos_a, cy - cx * sin_a - cy * cos_a,
                 0.0, 0.0]
    transform = tf.stack(transform)[tf.newaxis]
    image_batch = image[tf.newaxis]
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_batch,
        transforms=tf.cast(transform, tf.float32),
        output_shape=tf.stack([h, w]),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )
    return rotated[0]


def _safe_augment(image: tf.Tensor, label: int):
    """Augmentation that gracefully falls back if tfa is not installed."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)

    angle = tf.random.uniform((), minval=-15.0, maxval=15.0) * (np.pi / 180.0)
    try:
        import tensorflow_addons as tfa  # noqa: F401
        image = _rotate_image(image, angle)
    except ImportError:
        image = _rotate_image_fallback(image, angle)
    return image, label


# ── Dataset class ─────────────────────────────────────────────────────────────
class IDRiDDataset:
    """
    tf.data pipeline for the IDRiD DME grading sub-challenge.

    Args:
        dataset_path: Root directory of the dataset.
        split: One of ``'train'``, ``'val'``, or ``'test'``.
        augment: Apply random augmentation (training only).
        val_fraction: Fraction of training data reserved for validation.
        seed: Random seed for reproducibility.
        batch_size: Mini-batch size.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        augment: bool = True,
        val_fraction: float = 0.20,
        seed: int = 42,
        batch_size: int = BATCH_SIZE,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.augment = augment and split == "train"
        self.val_fraction = val_fraction
        self.seed = seed
        self.batch_size = batch_size

        self._image_paths, self._labels = self._load_metadata()
        self._class_weights = self._compute_class_weights()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load_metadata(self):
        """Read CSV, resolve image paths, and return (paths, labels)."""
        csv_path = self._find_csv()
        df = pd.read_csv(csv_path)

        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        image_col = self._detect_column(df, ["image_id", "image", "filename", "id"])
        label_col = self._detect_column(df, ["dme_grade", "dme", "grade", "label"])

        image_ids = df[image_col].astype(str).str.strip().values
        labels = df[label_col].astype(int).values

        image_dir = self.dataset_path / "images"
        paths = []
        valid_labels = []
        for img_id, lbl in zip(image_ids, labels):
            p = self._resolve_image_path(image_dir, img_id)
            if p is not None:
                paths.append(str(p))
                valid_labels.append(lbl)
            else:
                print(f"  [WARN] Image not found, skipping: {img_id}")

        if len(paths) == 0:
            raise RuntimeError(
                f"No images found in {image_dir}. "
                "Check dataset_path and image directory structure."
            )

        paths = np.array(paths)
        valid_labels = np.array(valid_labels, dtype=np.int32)

        # Stratified split
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_fraction, random_state=self.seed
        )
        train_idx, val_idx = next(splitter.split(paths, valid_labels))

        if self.split == "train":
            return paths[train_idx], valid_labels[train_idx]
        else:  # val / test share the same held-out portion
            return paths[val_idx], valid_labels[val_idx]

    def _find_csv(self) -> Path:
        candidates = [
            "DME_Grades.csv",
            "dme_grades.csv",
            "IDRiD_metadata.csv",
            "labels.csv",
        ]
        for name in candidates:
            p = self.dataset_path / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No grade CSV found in {self.dataset_path}. "
            f"Expected one of: {candidates}"
        )

    @staticmethod
    def _detect_column(df: pd.DataFrame, candidates: list) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(
            f"Could not find a suitable column. "
            f"Tried: {candidates}. Available: {list(df.columns)}"
        )

    @staticmethod
    def _resolve_image_path(image_dir: Path, image_id: str):
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ""):
            p = image_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        return None

    def _compute_class_weights(self) -> dict:
        """
        Compute per-class weights to handle class imbalance.

        Formula: weight_c = N / (C * count_c)
        where N = total samples, C = number of classes.
        This keeps the expected total gradient magnitude constant.
        """
        labels = self._labels
        n = len(labels)
        weights = {}
        for c in range(NUM_DME_CLASSES):
            count = np.sum(labels == c)
            weights[c] = n / (NUM_DME_CLASSES * (count + 1))
        # Normalise so that the mean weight == 1
        mean_w = np.mean(list(weights.values()))
        weights = {c: w / mean_w for c, w in weights.items()}
        return weights

    # ── Public API ──────────────────────────────────────────────────────────

    def get_dataset(self) -> tf.data.Dataset:
        """
        Build and return a ready-to-use ``tf.data.Dataset``.

        Returns batches of ``(image, one_hot_label)`` where image has shape
        ``(batch_size, 512, 512, 3)`` and label has shape
        ``(batch_size, NUM_DME_CLASSES)``.
        """
        paths_ds = tf.data.Dataset.from_tensor_slices(self._image_paths)
        labels_ds = tf.data.Dataset.from_tensor_slices(self._labels)
        ds = tf.data.Dataset.zip((paths_ds, labels_ds))

        if self.split == "train":
            ds = ds.shuffle(buffer_size=len(self._image_paths), seed=self.seed)

        ds = ds.map(_load_and_preprocess_tf, num_parallel_calls=AUTOTUNE)

        if self.augment:
            ds = ds.map(_safe_augment, num_parallel_calls=AUTOTUNE)

        # One-hot encode labels
        ds = ds.map(
            lambda img, lbl: (
                img,
                tf.one_hot(lbl, depth=NUM_DME_CLASSES),
            ),
            num_parallel_calls=AUTOTUNE,
        )

        ds = ds.batch(self.batch_size).prefetch(AUTOTUNE)
        return ds

    def get_raw_dataset(self) -> tf.data.Dataset:
        """
        Return a dataset that yields ``(image, integer_label)`` pairs
        without batching — useful for evaluation / QWK computation.
        """
        paths_ds = tf.data.Dataset.from_tensor_slices(self._image_paths)
        labels_ds = tf.data.Dataset.from_tensor_slices(self._labels)
        ds = tf.data.Dataset.zip((paths_ds, labels_ds))
        ds = ds.map(_load_and_preprocess_tf, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(AUTOTUNE)
        return ds

    def get_class_weights(self) -> dict:
        """Return ``{class_index: weight}`` dict for ``model.fit``."""
        return self._class_weights

    @property
    def num_samples(self) -> int:
        return len(self._image_paths)

    @property
    def steps_per_epoch(self) -> int:
        return max(1, self.num_samples // self.batch_size)

    def class_distribution(self) -> dict:
        """Return ``{class_index: count}`` for diagnostic purposes."""
        return {c: int(np.sum(self._labels == c)) for c in range(NUM_DME_CLASSES)}

    def print_summary(self):
        dist = self.class_distribution()
        total = sum(dist.values())
        print(f"\nIDRiD Dataset — split='{self.split}'  total={total}")
        print(f"{'Grade':<10}{'Count':>8}{'%':>8}{'Weight':>10}")
        print("-" * 38)
        for c, cnt in dist.items():
            pct = 100.0 * cnt / total if total > 0 else 0
            w = self._class_weights.get(c, 1.0)
            print(f"  {c:<8}{cnt:>8}{pct:>7.1f}%{w:>10.3f}")
        print()
