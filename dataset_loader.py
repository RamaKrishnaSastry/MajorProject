"""
dataset_loader.py - IRDID dataset pipeline for multi-output DME+DR classification.

Handles:
- Loading DME_Grades.csv
- Mapping image paths
- Creating tf.data pipeline with preprocessing
- Computing class weights for imbalanced data
- Train / validation split
- Multi-output target formatting (DR regression + DME classification)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocess import make_preprocess_fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DME_CLASSES = {0: "No DME", 1: "Mild", 2: "Moderate"}
NUM_DME_CLASSES = len(DME_CLASSES)

DR_CLASSES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe NPDR", 4: "Proliferative DR"}
NUM_DR_CLASSES = len(DR_CLASSES)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# CSV / path helpers
# ---------------------------------------------------------------------------

def load_dme_csv(csv_path: str, image_dir: str) -> pd.DataFrame:
    """Load *DME_Grades.csv* and resolve absolute image paths.

    The CSV is expected to have at least two columns:
    - ``Image name`` (or the first column) – image filename without extension
    - ``Retinopathy grade`` – DR label 0-4 (optional; defaults to 0 if absent)
    - ``Risk of macular edema `` (trailing space allowed) – DME label 0-2

    Parameters
    ----------
    csv_path : str
        Path to the grades CSV file.
    image_dir : str
        Root directory where fundus images are stored.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``image_path``, ``dme_label``, and ``dr_label``.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # remove accidental whitespace

    # Flexible column detection
    image_col = _find_column(df, ["Image name", "image_name", "filename", "Image"])
    dme_col = _find_column(
        df,
        [
            "Risk of macular edema",
            "DME_grade",
            "dme_label",
            "DME",
        ],
    )
    dr_col = _find_column(
        df,
        [
            "Retinopathy grade",
            "DR_grade",
            "dr_label",
            "DR",
        ],
    )

    if image_col is None:
        raise ValueError(
            f"Could not detect image name column in CSV. Available: {list(df.columns)}"
        )
    if dme_col is None:
        raise ValueError(
            f"Could not detect DME label column in CSV. Available: {list(df.columns)}"
        )

    if dr_col is None:
        logger.warning(
            "No DR label column found in CSV; DR labels will default to 0."
        )

    logger.info(
        "Using image column '%s', DME column '%s', DR column '%s'",
        image_col, dme_col, dr_col or "(none – defaulting to 0)",
    )

    records = []
    image_dir = Path(image_dir)
    for _, row in df.iterrows():
        name = str(row[image_col]).strip()
        dme_label = int(row[dme_col])
        dr_label = int(row[dr_col]) if dr_col is not None else 0
        # Try common extensions
        path = _resolve_image_path(image_dir, name)
        if path is not None:
            records.append({"image_path": str(path), "dme_label": dme_label, "dr_label": dr_label})
        else:
            logger.warning("Image not found for '%s' – skipping.", name)

    result = pd.DataFrame(records)
    logger.info("Loaded %d valid samples.", len(result))
    return result


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *df*."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _resolve_image_path(image_dir: Path, name: str) -> Optional[Path]:
    """Try to locate an image file with several common extensions."""
    for ext in ["", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        candidate = image_dir / (name + ext)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Mock dataset generator (for testing without real data)
# ---------------------------------------------------------------------------

def create_mock_dataset(
    output_dir: str,
    num_samples: int = 150,
    image_size: Tuple[int, int] = (512, 512),
    seed: int = 42,
    balanced: bool = True,
) -> Tuple[str, str]:
    """Generate a small synthetic dataset for pipeline testing.

    Creates random fundus-like images and a corresponding CSV file with both
    DR (Retinopathy grade, 0-4) and DME (Risk of macular edema, 0-2) labels.

    Parameters
    ----------
    output_dir : str
        Directory where mock images and CSV will be saved.
    num_samples : int
        Total number of synthetic images to create. Defaults to 150 to provide
        50 samples per DME class for stable train/val splits.
    image_size : tuple
        (H, W) pixel dimensions of each image.
    seed : int
        Random seed for reproducibility.
    balanced : bool
        If True (default), generate equal DME samples per class so that all
        three DME grades are well-represented. If False, use a realistic
        imbalanced distribution that mimics real IDRiD proportions.

    Returns
    -------
    tuple
        ``(csv_path, image_dir)`` paths to use with :func:`load_dme_csv`.
    """
    import cv2

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if balanced:
        # Equal DME samples per class: ensures all classes present in train AND val
        samples_per_class = num_samples // NUM_DME_CLASSES
        dme_labels = np.repeat(np.arange(NUM_DME_CLASSES), samples_per_class)
        # Handle remainder
        remainder = num_samples - len(dme_labels)
        if remainder > 0:
            extra = rng.integers(0, NUM_DME_CLASSES, size=remainder)
            dme_labels = np.concatenate([dme_labels, extra])
        rng.shuffle(dme_labels)
    else:
        # Class distribution mimicking real IDRiD imbalance (No DME is most common)
        class_probs = [0.65, 0.20, 0.15]
        dme_labels = rng.choice(NUM_DME_CLASSES, size=num_samples, p=class_probs)

    # Generate independent DR labels (0-4, 5 classes)
    dr_labels = rng.integers(0, NUM_DR_CLASSES, size=num_samples)

    records = []
    for i, (dme_label, dr_label) in enumerate(zip(dme_labels, dr_labels)):
        name = f"mock_{i:04d}"
        img = rng.integers(30, 220, size=(*image_size, 3), dtype=np.uint8)
        # Add a circular vignette to simulate fundus style
        h, w = image_size
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - w / 2) ** 2 + (Y - h / 2) ** 2)
        mask = dist > (min(h, w) / 2 * 0.95)
        img[mask] = 0
        cv2.imwrite(str(image_dir / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        records.append({
            "Image name": name,
            "Retinopathy grade": int(dr_label),
            "Risk of macular edema": int(dme_label),
        })

    csv_path = output_dir / "DME_Grades.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    logger.info("Mock dataset created: %d images in '%s'.", num_samples, output_dir)
    return str(csv_path), str(image_dir)


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_dme_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute per-class weights to handle class imbalance.

    Uses scikit-learn's *balanced* strategy: weight ∝ 1 / class_frequency.

    Parameters
    ----------
    labels : np.ndarray
        Integer class labels for all training samples.

    Returns
    -------
    dict
        Mapping ``{class_index: weight}``.
    """
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    logger.info("Ordinal class weights: %s", class_weight_dict)
    return class_weight_dict


# ---------------------------------------------------------------------------
# tf.data pipeline builders
# ---------------------------------------------------------------------------

def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply random augmentations to image only (not labels).
    
    Suitable for fundus images.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _build_tf_dataset(
    image_paths: np.ndarray,
    dme_labels: np.ndarray,
    dr_labels: np.ndarray,
    preprocess_fn,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    cache: bool,
    seed: int,
) -> tf.data.Dataset:
    """Internal helper – build a *tf.data.Dataset* from arrays.
    
    Returns targets as dict {'dr_output': ..., 'dme_risk': ...} to match the
    multi-output model structure.

    DR head  : regression (MSE).  Target = dr_label ∈ [0, 4], shape (1,).
               Compatible with EyePACS-pretrained ``Dense(1, relu)`` head.
    DME head : 3-class softmax (categorical crossentropy).
               Target = one-hot, shape (3,).
    """
    # Convert DME labels to one-hot for 3-class classification
    labels_dme_oh = tf.keras.utils.to_categorical(dme_labels, num_classes=NUM_DME_CLASSES)

    # DR labels: keep native DR grades in [0, 4] (5 integer classes) for ReLU regression head
    labels_dr = dr_labels.astype(np.float32).reshape(-1, 1)

    # Dataset with both DR and DME labels
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_dr, labels_dme_oh))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)
    
    # Load image and create multi-output target dict
    def load_and_format(path, dr_label, dme_label):
        image = preprocess_fn(path)
        targets = {
            'dr_output': dr_label,      # Shape: (1,) normalized float
            'dme_risk': dme_label,      # Shape: (3,) one-hot
        }
        return image, targets
    
    ds = ds.map(
        load_and_format,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    if augment:
        # Augment only the image, not the labels
        ds = ds.map(
            lambda image, targets: (_augment_image(image), targets),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    
    if cache:
        ds = ds.cache()
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(
    csv_path: str,
    image_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    batch_size: int = 8,
    val_split: float = 0.2,
    augment_train: bool = True,
    cache: bool = False,
    seed: int = 42,
    border_fraction: float = 0.10,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
    """Build train and validation *tf.data* datasets for multi-output DR+DME model.

    Parameters
    ----------
    csv_path : str
        Path to ``DME_Grades.csv``.
    image_dir : str
        Root directory containing fundus images.
    target_size : tuple
        Image resize target (H, W).
    batch_size : int
        Number of samples per batch.
    val_split : float
        Fraction of data to use for validation.
    augment_train : bool
        Whether to apply random augmentation to training images.
    cache : bool
        Cache preprocessed images in memory (needs sufficient RAM).
    seed : int
        Random seed for reproducible splits.
    border_fraction : float
        CLAHE / crop preprocessing parameter.
    clip_limit : float
        CLAHE clip limit.
    grid_size : int
        CLAHE grid size.

    Returns
    -------
    tuple
        ``(train_ds, val_ds, class_weights)``
        
    Notes
    -----
    Each batch returns:
    - images: (batch_size, H, W, 3)
    - targets: dict with keys 'dr_output' and 'dme_risk'
    """
    df = load_dme_csv(csv_path, image_dir)

    if len(df) == 0:
        raise ValueError("Dataset is empty – check csv_path and image_dir.")

    paths = df["image_path"].values
    dme_labels = df["dme_label"].values.astype(int)
    dr_labels = df["dr_label"].values.astype(int)

    train_paths, val_paths, train_dme, val_dme, train_dr, val_dr = train_test_split(
        paths, dme_labels, dr_labels, test_size=val_split, random_state=seed, stratify=dme_labels
    )

    logger.info(
        "Split: %d train / %d val samples.", len(train_paths), len(val_paths)
    )
    
    # Log class distribution
    train_dist = {c: int(np.sum(train_dme == c)) for c in range(NUM_DME_CLASSES)}
    val_dist = {c: int(np.sum(val_dme == c)) for c in range(NUM_DME_CLASSES)}
    logger.info("Train distribution: %s | Val distribution: %s", train_dist, val_dist)

    class_weights = compute_dme_class_weights(train_dme)

    preprocess_fn = make_preprocess_fn(
        target_size=target_size,
        border_fraction=border_fraction,
        clip_limit=clip_limit,
        grid_size=grid_size,
    )

    train_ds = _build_tf_dataset(
        train_paths,
        train_dme,
        train_dr,
        preprocess_fn,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        cache=cache,
        seed=seed,
    )
    val_ds = _build_tf_dataset(
        val_paths,
        val_dme,
        val_dr,
        preprocess_fn,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        cache=cache,
        seed=seed,
    )

    return train_ds, val_ds, class_weights


def save_split_info(
    csv_path: str,
    image_dir: str,
    output_path: str = "split_info.json",
    val_split: float = 0.2,
    seed: int = 42,
) -> None:
    """Persist train/val split metadata to a JSON file.

    Useful for reproducibility and auditing.

    Parameters
    ----------
    csv_path : str
        Path to ``DME_Grades.csv``.
    image_dir : str
        Root image directory.
    output_path : str
        Destination JSON file.
    val_split : float
        Validation fraction.
    seed : int
        Random seed used for splitting.
    """
    df = load_dme_csv(csv_path, image_dir)
    paths = df["image_path"].values
    dme_labels = df["dme_label"].values.astype(int)

    train_paths, val_paths, train_dme, val_dme = train_test_split(
        paths, dme_labels, test_size=val_split, random_state=seed, stratify=dme_labels
    )

    info = {
        "total_samples": len(df),
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "class_distribution": {
            "train": {str(c): int(np.sum(train_dme == c)) for c in range(NUM_DME_CLASSES)},
            "val": {str(c): int(np.sum(val_dme == c)) for c in range(NUM_DME_CLASSES)},
        },
        "val_split": val_split,
        "seed": seed,
    }

    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Split info saved to '%s'.", output_path)
