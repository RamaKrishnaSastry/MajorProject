"""
dataset_loader_eyepacs.py - EyePACS dataset loader for mixed DR training.

Implements:
- Loading EyePACS CSV with DR grades (5-class scale: 0-4)
- Ignoring EyePACS DME labels (binary, unreliable for mixed training)
- Combining IDRiD + EyePACS data for Stage 1 DR backbone training
- Computing class weights for combined dataset
- Creating tf.data pipelines with shared preprocessing
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from dataset_loader import (
    NUM_DR_CLASSES,
    _find_column,
    _resolve_image_path,
    _build_tf_dataset,
)
from preprocess import make_preprocess_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# EyePACS Constants
# ---------------------------------------------------------------------------

EYEPACS_DR_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


# ---------------------------------------------------------------------------
# EyePACS CSV Loading
# ---------------------------------------------------------------------------

def load_eyepacs_dr_csv(
    csv_path: str,
    image_dir: str,
    include_dme: bool = False,
) -> pd.DataFrame:
    """Load EyePACS CSV with DR grades (5-class scale).

    EyePACS DR labels are on a 0-4 scale matching IDRiD, making them
    compatible for mixed training.

    Note: EyePACS DME labels are binary and less reliable than IDRiD.
    This function ignores them (include_dme=False) by default to avoid
    corrupting the mixed training set with noisy labels.

    Parameters
    ----------
    csv_path : str
        Path to EyePACS trainLabels.csv or similar.
    image_dir : str
        Root directory where EyePACS images are stored.
    include_dme : bool, optional
        If True, preserve binary DME labels (not recommended for mixed training).
        Default: False (DME labels omitted).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'image_path' (str): full path to image file
        - 'dr_label' (int): DR grade 0-4
        - 'dme_label' (int, optional): DME label 0 (no) or 1 (yes) if include_dme=True

    Raises
    ------
    ValueError
        If required columns cannot be found in CSV.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # EyePACS typically uses 'image' column for filename and 'level' for DR grade
    image_col = _find_column(
        df,
        ["image", "Image", "image_name", "Image name", "filename"],
    )
    dr_col = _find_column(
        df,
        ["level", "Level", "dr_grade", "DR_grade", "retinopathy_grade"],
    )

    if image_col is None:
        raise ValueError(
            f"Could not detect image name column in CSV. Available: {list(df.columns)}"
        )
    if dr_col is None:
        raise ValueError(
            f"Could not detect DR label column in CSV. Available: {list(df.columns)}"
        )

    logger.info(
        "Using image column '%s', DR column '%s'",
        image_col,
        dr_col,
    )

    records = []
    image_dir = Path(image_dir)
    for _, row in df.iterrows():
        name = str(row[image_col]).strip()
        dr_label = int(row[dr_col])

        # Validate DR label is in [0, 4]
        if not 0 <= dr_label <= 4:
            logger.warning(
                "Invalid DR label %d for image '%s' – skipping.",
                dr_label,
                name,
            )
            continue

        # Resolve image path
        path = _resolve_image_path(image_dir, name)
        if path is not None:
            record = {
                "image_path": str(path),
                "dr_label": dr_label,
            }
            if include_dme:
                dme_col = _find_column(
                    df,
                    ["macular_edema", "Macular_Edema", "dme_label", "DME", "dme"],
                )
                if dme_col is not None:
                    dme_label = int(row[dme_col])
                    record["dme_label"] = dme_label
            records.append(record)
        else:
            logger.warning("Image not found for '%s' – skipping.", name)

    result = pd.DataFrame(records)
    logger.info("Loaded %d valid EyePACS DR samples.", len(result))
    return result


# ---------------------------------------------------------------------------
# Mixed Dataset Combining (IDRiD + EyePACS)
# ---------------------------------------------------------------------------

def combine_idrid_eyepacs(
    idrid_df: pd.DataFrame,
    eyepacs_df: pd.DataFrame,
    dr_loss_weight_eyepacs: float = 1.0,
    dr_loss_weight_idrid: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Combine IDRiD and EyePACS DataFrames for mixed training.

    For mixed training on DR (Stage 1), we combine:
    - IDRiD: 516 images with both DR and DME labels
    - EyePACS: ~88,000 images with DR labels only

    During Stage 1, the model is trained on DR task from both datasets.
    DME head is not used in Stage 1, so lack of EyePACS DME labels is acceptable.

    Note: Only uses DR labels. DME is ignored (EyePACS DME too unreliable).

    Parameters
    ----------
    idrid_df : pd.DataFrame
        IDRiD DataFrame from dataset_loader.load_dme_csv() with columns:
        ['image_path', 'dme_label', 'dr_label']
    eyepacs_df : pd.DataFrame
        EyePACS DataFrame from load_eyepacs_dr_csv() with columns:
        ['image_path', 'dr_label']
    dr_loss_weight_eyepacs : float, optional
        Weight multiplier for EyePACS DR loss. Default: 1.0 (equal weight).
        Can be < 1.0 to down-weight EyePACS if it dominates training.
    dr_loss_weight_idrid : float, optional
        Weight multiplier for IDRiD DR loss. Default: 1.0 (equal weight).

    Returns
    -------
    tuple
        - combined_df (pd.DataFrame): Merged DataFrame with columns:
          ['image_path', 'dr_label', 'source']
          where source is 'idrid' or 'eyepacs'
        - metadata (dict): Metadata about combined dataset:
          {
            'total_samples': int,
            'idrid_samples': int,
            'eyepacs_samples': int,
            'idrid_dr_distribution': dict,
            'eyepacs_dr_distribution': dict,
          }
    """
    # Add source column
    idrid_df = idrid_df.copy()
    eyepacs_df = eyepacs_df.copy()

    idrid_df["source"] = "idrid"
    eyepacs_df["source"] = "eyepacs"

    # Use only DR labels
    idrid_subset = idrid_df[["image_path", "dr_label", "source"]].copy()
    eyepacs_subset = eyepacs_df[["image_path", "dr_label", "source"]].copy()

    # Combine
    combined_df = pd.concat(
        [idrid_subset, eyepacs_subset],
        ignore_index=True,
    )

    # Metadata
    idrid_distribution = idrid_subset["dr_label"].value_counts().to_dict()
    eyepacs_distribution = eyepacs_subset["dr_label"].value_counts().to_dict()

    metadata = {
        "total_samples": len(combined_df),
        "idrid_samples": len(idrid_subset),
        "eyepacs_samples": len(eyepacs_subset),
        "idrid_dr_distribution": {int(k): int(v) for k, v in idrid_distribution.items()},
        "eyepacs_dr_distribution": {int(k): int(v) for k, v in eyepacs_distribution.items()},
    }

    logger.info("Combined dataset:")
    logger.info("  Total samples: %d", len(combined_df))
    logger.info("  IDRiD samples: %d", len(idrid_subset))
    logger.info("  EyePACS samples: %d", len(eyepacs_subset))
    logger.info("  IDRiD DR distribution: %s", metadata["idrid_dr_distribution"])
    logger.info("  EyePACS DR distribution: %s", metadata["eyepacs_dr_distribution"])

    return combined_df, metadata


# ---------------------------------------------------------------------------
# Class Weight Computation for Mixed Dataset
# ---------------------------------------------------------------------------

def compute_dr_class_weights_mixed(
    dr_labels: np.ndarray,
    clip_ratio: float = 7.0,
) -> Dict[int, float]:
    """Compute balanced DR class weights for mixed IDRiD+EyePACS dataset.

    This function computes inverse-frequency weights for the DR classification
    task across the combined dataset. The weighting ensures that less common
    grades (e.g., Proliferative DR, which is rare) still contribute meaningfully
    to the loss gradient.

    Parameters
    ----------
    dr_labels : np.ndarray
        Array of DR labels [0, 1, 2, 3, 4] from combined IDRiD+EyePACS.
    clip_ratio : float, optional
        Maximum ratio between highest and lowest weight. If exceeded, weights
        are clipped at this ratio. Default: 7.0.

    Returns
    -------
    dict
        Class weights {0: w0, 1: w1, 2: w2, 3: w3, 4: w4}
    """
    weights = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_DR_CLASSES),
        y=dr_labels,
    )

    # Normalize to mean 1.0
    weights = weights / weights.mean()

    # Apply clipping ratio
    max_weight = weights.max()
    min_weight = weights.min()
    if min_weight > 0:
        ratio = max_weight / min_weight
        if ratio > clip_ratio:
            logger.warning(
                "DR class weight ratio %.2f exceeds clip_ratio %.2f – clipping.",
                ratio,
                clip_ratio,
            )
            # Clip to ratio, then re-normalize
            target_min = max_weight / clip_ratio
            weights = np.clip(weights, target_min, max_weight)
            weights = weights / weights.mean()

    result = {int(i): float(w) for i, w in enumerate(weights)}
    logger.info("DR class weights (mixed dataset): %s", result)
    return result


# ---------------------------------------------------------------------------
# Pipeline Creation
# ---------------------------------------------------------------------------

def create_mixed_dr_train_val_datasets(
    idrid_csv: str,
    idrid_image_dir: str,
    eyepacs_csv: str,
    eyepacs_image_dir: str,
    batch_size: int = 8,
    val_split: float = 0.20,
    augment_train: bool = True,
    seed: int = 42,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    Dict[int, float],
    Dict,
]:
    """Create train/val datasets for mixed IDRiD+EyePACS Stage 1 training.

    This function:
    1. Loads IDRiD and EyePACS data
    2. Combines them for DR-only training (Stage 1)
    3. Splits combined data into train/val (preserving dataset distribution)
    4. Creates tf.data pipelines with preprocessing
    5. Returns class weights for the combined DR task

    For Stage 2 (DME fine-tuning), call create_idrid_only_dataset() instead.

    Parameters
    ----------
    idrid_csv : str
        Path to IDRiD DME_Grades.csv
    idrid_image_dir : str
        Path to IDRiD images directory
    eyepacs_csv : str
        Path to EyePACS trainLabels.csv
    eyepacs_image_dir : str
        Path to EyePACS images directory
    batch_size : int, optional
        Batch size for tf.data.Dataset. Default: 8.
    val_split : float, optional
        Fraction of data for validation [0, 1]. Default: 0.20.
    augment_train : bool, optional
        If True, apply augmentation to training set. Default: True.
    seed : int, optional
        Random seed for reproducibility. Default: 42.

    Returns
    -------
    tuple
        - train_ds (tf.data.Dataset): Training dataset
        - val_ds (tf.data.Dataset): Validation dataset
        - dr_class_weights (dict): DR class weights {0: w0, ..., 4: w4}
        - metadata (dict): Metadata about the combined dataset
    """
    logger.info("\n" + "=" * 60)
    logger.info("Loading mixed IDRiD + EyePACS dataset for Stage 1 DR training")
    logger.info("=" * 60)

    # Load individual datasets
    logger.info("Loading IDRiD...")
    idrid_df = load_dme_csv(idrid_csv, idrid_image_dir)
    logger.info("IDRiD loaded: %d samples", len(idrid_df))

    logger.info("Loading EyePACS...")
    eyepacs_df = load_eyepacs_dr_csv(eyepacs_csv, eyepacs_image_dir, include_dme=False)
    logger.info("EyePACS loaded: %d samples", len(eyepacs_df))

    # Combine datasets
    combined_df, metadata = combine_idrid_eyepacs(idrid_df, eyepacs_df)

    # Split into train/val while preserving source distribution
    np.random.seed(seed)
    train_df, val_df = train_test_split(
        combined_df,
        test_size=val_split,
        random_state=seed,
        stratify=combined_df["source"],  # keep IDRiD/EyePACS split proportions
    )

    logger.info(
        "Train/val split: %d train, %d val (%.1f%% val)",
        len(train_df),
        len(val_df),
        100 * val_split,
    )

    # Compute DR class weights on training set
    dr_class_weights = compute_dr_class_weights_mixed(train_df["dr_label"].values)

    # Create tf.data pipelines for DR-only task
    preprocess_fn = make_preprocess_fn(augment=augment_train)
    train_ds = _build_tf_dataset(
        train_df[["image_path", "dr_label"]].values,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        preprocess_fn=preprocess_fn,
        include_dme=False,  # Stage 1 is DR-only
    )

    preprocess_fn_val = make_preprocess_fn(augment=False)
    val_ds = _build_tf_dataset(
        val_df[["image_path", "dr_label"]].values,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        preprocess_fn=preprocess_fn_val,
        include_dme=False,  # Stage 1 is DR-only
    )

    logger.info("Mixed dataset pipelines created.")
    logger.info("Train dataset: %d batches of size %d", len(train_ds), batch_size)
    logger.info("Val dataset: %d batches of size %d", len(val_ds), batch_size)

    return train_ds, val_ds, dr_class_weights, metadata



