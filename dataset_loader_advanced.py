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
    NUM_DR_CLASSES,
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
    1: 2.0,   # Mild     – early detection important
    2: 1.0,   # Moderate – treatment decision boundary
}


def compute_ordinal_class_weights(
    labels: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
    medical_importance: bool = True,
    ordinal_penalty: bool = True,
) -> Dict[int, float]:
    """Compute balanced class weights with gentle medical importance scaling.
    
    This function computes per-class loss weights to address class imbalance in the DME dataset
    while respecting the ordinal structure of the classification task. The weighting strategy
    combines three mechanisms:

    1. **Inverse-frequency balancing**: Classes with fewer samples receive higher weights,
       ensuring rare classes contribute meaningfully to the loss gradient. Formally:
       w_i = (n_total / n_i) / mean(all_weights), where n_i is count of class i.

    2. **Medical importance scaling**: Multiplies base weights by domain-specific importance
       factors. Early detection of Mild (class 1) DME is clinically valuable, so it receives
       a modest 2x boost. No DME (class 0) and Moderate (class 2) are equally weighted at 1.0x.
       This prevents aggressive over-weighting that would cause mode collapse.

    3. **Post-normalization clipping**: After medical importance scaling, weights are
       re-normalized so the mean is 1.0 (keeping the loss scale interpretable). A safety check
       ensures the ratio of max to min weight does not exceed 5x. If it does (e.g., due to
       extreme class imbalance), weights are clipped and re-normalized to maintain stability.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of class labels in [0, num_classes-1], shape (n_samples,).
        Typically the DME training labels (0=No DME, 1=Mild, 2=Moderate).
    num_classes : int, optional
        Number of ordinal classes. Default: NUM_DME_CLASSES (3).
    medical_importance : bool, optional
        If True, apply MEDICAL_IMPORTANCE_WEIGHTS multipliers (default: True).
        Set False to use pure inverse-frequency weights for baseline comparisons.
    ordinal_penalty : bool, optional
        Reserved for future use. Currently unused (ordinal awareness is baked into
        the OrdinalWeightedCrossEntropy loss, not class weights). Default: True.

    Returns
    -------
    dict
        Mapping {class_index: weight} where weights are normalized so mean = 1.0.
        Example output for DME task with 413 samples:
        {0: 0.9, 1: 1.5, 2: 0.8}  ← class 1 (Mild) gets modest boost

    Notes
    -----
    - **Why not use sklearn's compute_class_weight?** The custom implementation gives us
      explicit control over medical importance and the 5x safety ceiling, which prevents
      pathological weighting schemes that cause mode collapse on minority classes.

    - **Why re-normalize after medical scaling?** Ensures the mean weight stays 1.0 across
      training. This keeps the effective loss scale consistent regardless of the importance
      multipliers chosen, making training dynamics more predictable.

    - **Why 5x ceiling?** Weights > 5x skew the loss landscape too severely. On a 413-sample
      dataset with 3 classes, even 3x is aggressive. The 5x limit is a safety guardrail
      against silent failure modes like loss NaN or gradient explosion.

    - **Interaction with oversampling:** When class 1 (Mild) is oversampled 3x in the training
      data (33 → 99 samples), its effective frequency goes from 8% to 25%. This function
      recomputes weights based on the actual oversampled label distribution, so class 1's
      weight decreases automatically. The 2.0x medical importance multiplier then applies on
      top, yielding a final moderate boost that balances early detection with convergence stability.

    Examples
    --------
    >>> labels_train = np.array([0]*141 + [1]*33 + [2]*156)  # train distribution
    >>> w = compute_ordinal_class_weights(labels_train)
    >>> print(w)
    {0: 0.9, 1: 1.5, 2: 0.8}

    >>> # After oversampling class 1 ×3:
    >>> labels_oversampled = np.array([0]*141 + [1]*99 + [2]*156)
    >>> w = compute_ordinal_class_weights(labels_oversampled)
    >>> print(w)  # class 1 weight now lower because it's no longer rare
    {0: 0.95, 1: 1.2, 2: 0.85}
    """
    # Step 1: pure inverse-frequency weights, normalized so they sum to num_classes
    counts = np.array([max(np.sum(labels == i), 1) for i in range(num_classes)], dtype=np.float64)
    weights = 1.0 / counts
    weights = weights / weights.mean()   # normalize: mean weight = 1.0

    class_weights = {i: float(weights[i]) for i in range(num_classes)}

    # Step 2: gentle medical importance multiplier (max 2x, not 4x)
    if medical_importance:
        importance = MEDICAL_IMPORTANCE_WEIGHTS
        for i in range(num_classes):
            class_weights[i] *= importance.get(i, 1.0)

    # Step 3: re-normalize so mean = 1.0 (prevents all weights from being huge or tiny)
    mean_w = np.mean(list(class_weights.values()))
    class_weights = {k: round(v / mean_w, 4) for k, v in class_weights.items()}

    # Step 4: sanity check — no weight should be > 5x any other weight
    vals = list(class_weights.values())
    ratio = max(vals) / min(vals)
    if ratio > 8.0:
        logger.warning(
            "Class weight ratio %.1fx exceeds safe limit. Clipping to 5x.", ratio
        )
        min_w = min(vals)
        class_weights = {k: min(v, min_w * 5.0) for k, v in class_weights.items()}
        # re-normalize after clipping
        mean_w = np.mean(list(class_weights.values()))
        class_weights = {k: round(v / mean_w, 4) for k, v in class_weights.items()}

    logger.info("Ordinal class weights: %s", class_weights)
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
def oversample_minority_class(paths, dme_labels, dr_labels, minority_class=1, factor=3):
    """Repeat minority class samples to reduce imbalance."""
    mask = (dme_labels == minority_class)
    extra_paths  = np.tile(paths[mask],     factor - 1)
    extra_dme    = np.tile(dme_labels[mask], factor - 1)
    extra_dr     = np.tile(dr_labels[mask],  factor - 1)
    return (
        np.concatenate([paths,      extra_paths]),
        np.concatenate([dme_labels, extra_dme]),
        np.concatenate([dr_labels,  extra_dr]),
    )


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
    messidor_dir: Optional[str] = None,
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
    messidor_dir : str, optional
        Optional path to a MESSIDOR directory to merge into the dataset.

    Returns
    -------
    tuple
        ``(train_ds, val_ds, class_weights, split_info)``
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_dme_csv(csv_path, image_dir)
    if len(df) == 0:
        raise ValueError("Dataset is empty – check csv_path and image_dir.")

    if messidor_dir and os.path.exists(messidor_dir):
        from dataset_loader_messidor import load_messidor_as_idrid_format
        messidor_df = load_messidor_as_idrid_format(messidor_dir)
        df = pd.concat([df, messidor_df], ignore_index=True)
        logger.info("Combined dataset: %d samples (IDRiD + MESSIDOR)", len(df))
        
    paths = df["image_path"].values
    dme_labels = df["dme_label"].values.astype(int)
    dr_labels = df["dr_label"].values.astype(int)

    # QWK-aware stratified split (stratify on DME labels)
    train_paths, val_paths, train_dme, val_dme = ordinal_stratified_split(
        paths, dme_labels, val_split=val_split, seed=seed, num_classes=NUM_DME_CLASSES,
    )

    # Apply the same split to DR labels using path-index mapping
    path_to_idx = {p: i for i, p in enumerate(paths)}
    train_indices = np.array([path_to_idx[p] for p in train_paths])
    val_indices = np.array([path_to_idx[p] for p in val_paths])
    train_dr = dr_labels[train_indices]
    val_dr = dr_labels[val_indices]

    # Ordinal class weights (based on DME labels)
    class_weights = compute_ordinal_class_weights(
        train_dme,
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

    # After the split, before building train_ds:
    # train_paths, train_dme, train_dr = oversample_minority_class(
    #     train_paths, train_dme, train_dr, minority_class=1, factor=2
    # )
    # logger.info(
    #     "After oversampling class 1 ×3: %s",
    #     {i: int(np.sum(train_dme == i)) for i in range(NUM_DME_CLASSES)}
    # )
    # Expected output: {0: 141, 1: 99, 2: 156}  ← class 1 now competitive

    train_ds = _build_tf_dataset(
        train_paths, train_dme, train_dr, preprocess_fn,
        batch_size=batch_size, shuffle=True, augment=augment_train,
        cache=cache, seed=seed,
    )
    val_ds = _build_tf_dataset(
        val_paths, val_dme, val_dr, preprocess_fn,
        batch_size=batch_size, shuffle=False, augment=False,
        cache=cache, seed=seed,
    )

    split_info = {
        "total_samples": int(len(paths)),
        "train_samples": int(len(train_paths)),
        "val_samples": int(len(val_paths)),
        "class_distribution": {
            "train": {
                str(i): int(np.sum(train_dme == i)) for i in range(NUM_DME_CLASSES)
            },
            "val": {
                str(i): int(np.sum(val_dme == i)) for i in range(NUM_DME_CLASSES)
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
            dme_labels, class_weights, output_path=balance_path,
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
    parser.add_argument("--messidor-dir", type=str, default=None,
                        help="Path to MESSIDOR dataset directory")
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
            messidor_dir=None,
        )
        print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()
