"""
dataset_loader_advanced.py - QWK-aware advanced dataset loader for DR+DME classification.

Extends dataset_loader.py with:
- QWK-aware stratification for balanced ordinal splits
- Ordinal penalty-weighted class weights
- Dataset balance visualisation (before/after class weighting)
- Cross-validation aware splitting with reproducibility
- Ordinal consistency checking in train/val splits
- Medical domain knowledge integration (class importance weighting)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from dataset_loader import (
    DME_CLASSES,
    NUM_DME_CLASSES,
    load_dme_csv,
    create_mock_dataset,
    _build_tf_dataset,
)
from preprocess import make_preprocess_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Ordinal class weight computation
# ---------------------------------------------------------------------------

# Clinical importance weights – rarer/more severe classes get extra emphasis
MEDICAL_IMPORTANCE_WEIGHTS = {
    0: 1.0,   # No DME   – baseline
    1: 1.2,   # Mild     – early detection important
    2: 1.5,   # Moderate – treatment decision boundary
    3: 2.0,   # Severe   – urgent referral needed
}


def compute_ordinal_class_weights(
    labels: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
    medical_importance: bool = True,
    ordinal_penalty: bool = True,
) -> Dict[int, float]:
    """Compute class weights combining frequency balancing, ordinal penalty,
    and medical domain importance.

    Strategy:
    1. Frequency-based balancing (inverse class frequency)
    2. Optional medical importance multiplier (rarer/severe = higher weight)
    3. Optional ordinal penalty: classes at distribution extremes weighted more

    Parameters
    ----------
    labels : np.ndarray
        Integer class labels for training samples.
    num_classes : int
        Number of ordinal classes.
    medical_importance : bool
        Apply medical domain importance multipliers.
    ordinal_penalty : bool
        Add ordinal-awareness penalty to extreme class weights.

    Returns
    -------
    dict
        Mapping ``{class_index: weight}``.
    """
    classes = np.arange(num_classes)
    present_classes = np.unique(labels.astype(int))

    # Base: sklearn balanced weights for present classes
    base_weights_arr = compute_class_weight(
        "balanced", classes=present_classes, y=labels.astype(int)
    )
    base_weights = dict(zip(present_classes.tolist(), base_weights_arr.tolist()))

    # Fill missing classes with max weight
    max_w = max(base_weights.values()) if base_weights else 1.0
    class_weights = {i: base_weights.get(i, max_w) for i in range(num_classes)}

    if medical_importance:
        for i in range(num_classes):
            class_weights[i] *= MEDICAL_IMPORTANCE_WEIGHTS.get(i, 1.0)

    if ordinal_penalty:
        # Amplify boundary classes to reduce edge confusion:
        # +10% for class 0 (No DME) to distinguish it from Mild
        # +30% for the severe class (highest ordinal) to emphasise rare severe cases
        class_weights[0] *= 1.1
        class_weights[num_classes - 1] *= 1.3

    # Normalise so mean weight = 1.0
    mean_w = np.mean(list(class_weights.values()))
    if mean_w > 0:
        class_weights = {k: v / mean_w for k, v in class_weights.items()}

    logger.info("Ordinal class weights: %s", {k: round(v, 3) for k, v in class_weights.items()})
    return class_weights


# ---------------------------------------------------------------------------
# QWK-aware stratified split
# ---------------------------------------------------------------------------

def ordinal_stratified_split(
    paths: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.2,
    seed: int = 42,
    num_classes: int = NUM_DME_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split that preserves ordinal class distribution.

    Ensures every ordinal class is represented proportionally in both splits.
    Falls back to standard stratify if any class has too few samples.

    Parameters
    ----------
    paths : np.ndarray
        Image path array.
    labels : np.ndarray
        Integer ordinal labels.
    val_split : float
        Validation fraction.
    seed : int
        Random seed.
    num_classes : int
        Number of ordinal classes.

    Returns
    -------
    tuple
        ``(train_paths, val_paths, train_labels, val_labels)``
    """
    labels = labels.astype(int)
    class_counts = {i: int(np.sum(labels == i)) for i in range(num_classes)}
    logger.info("Class counts before split: %s", class_counts)

    # Check if any class has fewer than 2 samples (required for stratify)
    min_count = min(class_counts.get(i, 0) for i in range(num_classes))
    if min_count < 2:
        logger.warning(
            "Some classes have < 2 samples; falling back to non-stratified split."
        )
        stratify = None
    else:
        stratify = labels

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=val_split, random_state=seed, stratify=stratify
    )

    # Verify ordinal consistency
    _check_ordinal_consistency(train_labels, val_labels, num_classes)

    logger.info(
        "Split: %d train / %d val | Train distribution: %s | Val distribution: %s",
        len(train_paths),
        len(val_paths),
        {i: int(np.sum(train_labels == i)) for i in range(num_classes)},
        {i: int(np.sum(val_labels == i)) for i in range(num_classes)},
    )
    return train_paths, val_paths, train_labels, val_labels


def _check_ordinal_consistency(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
) -> None:
    """Log a warning if a class is absent from train or val set."""
    train_classes = set(np.unique(train_labels))
    val_classes = set(np.unique(val_labels))
    all_classes = set(range(num_classes))

    missing_in_train = all_classes - train_classes
    missing_in_val = all_classes - val_classes

    if missing_in_train:
        logger.warning(
            "Classes missing from training set: %s. Consider oversampling.",
            missing_in_train,
        )
    if missing_in_val:
        logger.warning(
            "Classes missing from validation set: %s. QWK may be unreliable.",
            missing_in_val,
        )


# ---------------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------------

def build_kfold_splits(
    paths: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate stratified K-fold cross-validation splits.

    Parameters
    ----------
    paths : np.ndarray
        Image path array.
    labels : np.ndarray
        Integer ordinal labels.
    n_splits : int
        Number of folds.
    seed : int
        Random seed.

    Returns
    -------
    list of tuple
        Each element is ``(train_paths, val_paths, train_labels, val_labels)``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        t_paths = paths[train_idx]
        v_paths = paths[val_idx]
        t_labels = labels[train_idx]
        v_labels = labels[val_idx]
        logger.info(
            "Fold %d/%d: %d train / %d val",
            fold_idx + 1, n_splits, len(t_paths), len(v_paths),
        )
        splits.append((t_paths, v_paths, t_labels, v_labels))
    return splits


# ---------------------------------------------------------------------------
# Dataset balance visualisation
# ---------------------------------------------------------------------------

def plot_dataset_balance(
    labels: np.ndarray,
    class_weights: Dict[int, float],
    class_names: Optional[List[str]] = None,
    output_path: str = "dataset_balance.png",
    num_classes: int = NUM_DME_CLASSES,
    title: str = "Dataset Class Balance",
) -> None:
    """Plot class distribution before and after class weighting.

    Parameters
    ----------
    labels : np.ndarray
        Full label array.
    class_weights : dict
        Computed class weights.
    class_names : list, optional
        Human-readable class names.
    output_path : str
        Output PNG path.
    num_classes : int
        Number of classes.
    title : str
        Plot title.
    """
    if not _PLOTTING_AVAILABLE:
        logger.warning("Plotting not available – skipping dataset balance plot.")
        return

    names = class_names or [DME_CLASSES.get(i, f"Class {i}") for i in range(num_classes)]
    labels = labels.astype(int)

    counts = np.array([int(np.sum(labels == i)) for i in range(num_classes)])
    weights = np.array([class_weights.get(i, 1.0) for i in range(num_classes)])
    effective_counts = counts * weights

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Raw counts
    axes[0].bar(names, counts, color="steelblue", edgecolor="black", alpha=0.85)
    axes[0].set_title("Raw Class Counts", fontsize=12)
    axes[0].set_xlabel("DME Class")
    axes[0].set_ylabel("Sample Count")
    for i, c in enumerate(counts):
        axes[0].text(i, c + 0.5, str(c), ha="center", va="bottom", fontsize=10)

    # Class weights
    axes[1].bar(names, weights, color="darkorange", edgecolor="black", alpha=0.85)
    axes[1].set_title("Class Weights", fontsize=12)
    axes[1].set_xlabel("DME Class")
    axes[1].set_ylabel("Weight")
    for i, w in enumerate(weights):
        axes[1].text(i, w + 0.02, f"{w:.2f}", ha="center", va="bottom", fontsize=10)

    # Effective sample contribution
    axes[2].bar(names, effective_counts, color="seagreen", edgecolor="black", alpha=0.85)
    axes[2].set_title("Effective Weighted Counts", fontsize=12)
    axes[2].set_xlabel("DME Class")
    axes[2].set_ylabel("Effective Count")
    for i, ec in enumerate(effective_counts):
        axes[2].text(i, ec + 0.5, f"{ec:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Dataset balance plot saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# Advanced dataset builder
# ---------------------------------------------------------------------------

def build_datasets_advanced(
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
    medical_importance: bool = True,
    ordinal_penalty: bool = True,
    output_dir: str = ".",
    save_split_info: bool = True,
    save_balance_plot: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, float], Dict]:
    """Build QWK-aware train/val tf.data datasets.

    Parameters
    ----------
    csv_path : str
        Path to ``DME_Grades.csv``.
    image_dir : str
        Root directory containing fundus images.
    target_size : tuple
        Image resize target (H, W).
    batch_size : int
        Batch size.
    val_split : float
        Validation fraction.
    augment_train : bool
        Apply augmentation to training images.
    cache : bool
        Cache dataset in memory.
    seed : int
        Random seed.
    border_fraction : float
        CLAHE crop fraction.
    clip_limit : float
        CLAHE clip limit.
    grid_size : int
        CLAHE grid size.
    medical_importance : bool
        Use medical domain importance in class weight computation.
    ordinal_penalty : bool
        Use ordinal penalty in class weight computation.
    output_dir : str
        Directory to save artefacts.
    save_split_info : bool
        Save split statistics to JSON.
    save_balance_plot : bool
        Save dataset balance visualisation.

    Returns
    -------
    tuple
        ``(train_ds, val_ds, class_weights, split_info)``
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_dme_csv(csv_path, image_dir)
    if len(df) == 0:
        raise ValueError("Dataset is empty – check csv_path and image_dir.")

    paths = df["image_path"].values
    labels = df["dme_label"].values.astype(int)

    # QWK-aware stratified split
    train_paths, val_paths, train_labels, val_labels = ordinal_stratified_split(
        paths, labels, val_split=val_split, seed=seed, num_classes=NUM_DME_CLASSES,
    )

    # Ordinal class weights
    class_weights = compute_ordinal_class_weights(
        train_labels,
        num_classes=NUM_DME_CLASSES,
        medical_importance=medical_importance,
        ordinal_penalty=ordinal_penalty,
    )

    preprocess_fn = make_preprocess_fn(
        target_size=target_size,
        border_fraction=border_fraction,
        clip_limit=clip_limit,
        grid_size=grid_size,
    )

    train_ds = _build_tf_dataset(
        train_paths, train_labels, preprocess_fn,
        batch_size=batch_size, shuffle=True, augment=augment_train,
        cache=cache, seed=seed,
    )
    val_ds = _build_tf_dataset(
        val_paths, val_labels, preprocess_fn,
        batch_size=batch_size, shuffle=False, augment=False,
        cache=cache, seed=seed,
    )

    split_info = {
        "total_samples": int(len(paths)),
        "train_samples": int(len(train_paths)),
        "val_samples": int(len(val_paths)),
        "class_distribution": {
            "train": {
                str(i): int(np.sum(train_labels == i)) for i in range(NUM_DME_CLASSES)
            },
            "val": {
                str(i): int(np.sum(val_labels == i)) for i in range(NUM_DME_CLASSES)
            },
        },
        "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
        "val_split": val_split,
        "seed": seed,
        "medical_importance": medical_importance,
        "ordinal_penalty": ordinal_penalty,
    }

    if save_split_info:
        split_path = os.path.join(output_dir, "split_info_advanced.json")
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)
        logger.info("Advanced split info saved to '%s'.", split_path)

    if save_balance_plot:
        balance_path = os.path.join(output_dir, "dataset_balance.png")
        plot_dataset_balance(
            labels, class_weights, output_path=balance_path,
        )

    return train_ds, val_ds, class_weights, split_info


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for advanced dataset loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced QWK-aware dataset loader")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--create-mock", action="store_true")
    parser.add_argument("--kfold", type=int, default=0,
                        help="Number of CV folds (0 = no cross-validation)")
    args = parser.parse_args()

    if args.mock or args.create_mock or args.csv is None:
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_adv", num_samples=80)
        logger.info("Mock dataset created at /tmp/mock_irdid_adv")
    else:
        csv_path, image_dir = args.csv, args.image_dir

    os.makedirs(args.output_dir, exist_ok=True)

    if args.kfold > 0:
        df = load_dme_csv(csv_path, image_dir)
        paths = df["image_path"].values
        labels = df["dme_label"].values.astype(int)
        splits = build_kfold_splits(paths, labels, n_splits=args.kfold, seed=args.seed)
        logger.info("Generated %d CV splits.", len(splits))
        fold_info = [
            {"fold": i + 1, "train": len(s[0]), "val": len(s[1])}
            for i, s in enumerate(splits)
        ]
        with open(os.path.join(args.output_dir, "kfold_info.json"), "w") as f:
            json.dump(fold_info, f, indent=2)
        print(json.dumps(fold_info, indent=2))
    else:
        train_ds, val_ds, class_weights, split_info = build_datasets_advanced(
            csv_path=csv_path,
            image_dir=image_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()
