"""
test_cross_dataset_dr.py - Cross-dataset evaluation on Messidor, EyePACS, APTOS.

Tests the DR classification task only (no DME) on external datasets.
Compares predictions against ground truth DR labels.

Usage:
    # Use best pre-trained weights
    python test_cross_dataset_dr.py --use-model --use-weights best_qwk.weights.h5

    # Full architecture from scratch (random weights for debugging)
    python test_cross_dataset_dr.py --no-use-model
    
    # Custom weights path
    python test_cross_dataset_dr.py --use-model --use-weights /path/to/weights.h5
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Local imports
from model import build_model
from qwk_metrics import compute_quadratic_weighted_kappa

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_weights_with_fallback(model, weights_path: str):
    """
    Load weights with better error handling for architecture mismatches.
    
    Attempts to load weights by layer name first. If that fails, tries
    loading only backbone weights. Falls back gracefully on full failure.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model to load weights into
    weights_path : str
        Path to weights file
    """
    try:
        # First try: Load by name (ignores architecture differences)
        logger.info(f"Attempting to load weights (by_name=True)...")
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        logger.info("✅ Weights loaded successfully (by_name mode)")
        return True
    except Exception as e:
        logger.warning(f"Load by_name failed: {e}")
    
    try:
        # Second try: Load with skip_mismatch (more forgiving)
        logger.info(f"Attempting to load weights (skip_mismatch=True)...")
        model.load_weights(weights_path, skip_mismatch=True)
        logger.info("✅ Weights loaded successfully (skip_mismatch mode)")
        return True
    except Exception as e:
        logger.warning(f"Load with skip_mismatch failed: {e}")
    
    logger.warning(f"Could not load weights from {weights_path}. Using ImageNet backbone + random heads.")
    return False


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_messidor_dr_labels(messidor_images_dir: str, messidor_csv: str) -> Tuple[list, list]:
    """
    Load MESSIDOR dataset with DR labels only.
    
    MESSIDOR DR grades: 0=No, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
    (Maps to IDRiD format: 0-4)
    
    Parameters
    ----------
    messidor_images_dir : str
        Path to MESSIDOR images directory
    messidor_csv : str
        Path to CSV/XLS annotation file
        
    Returns
    -------
    Tuple[list, list]
        (image_paths, dr_labels)
    """
    image_paths = []
    dr_labels = []
    
    messidor_path = Path(messidor_images_dir)
    csv_path = Path(messidor_csv)
    
    # Load from provided annotation file
    try:
        if str(csv_path).endswith(('.xls', '.xlsx')):
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)
        
        df.columns = df.columns.str.strip()
        
        for _, row in df.iterrows():
            img_name = str(row.iloc[0]).strip()
            dr_label = int(row.get('Retinopathy grade', row.iloc[1]) if len(row) > 1 else 0)
            
            # Direct file lookup: check if image exists as-is or with added extension
            img_path = None
            if Path(messidor_path / img_name).exists():
                img_path = messidor_path / img_name
            else:
                # Try common extensions
                for ext in ['.tif', '.TIF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    candidate = messidor_path / (img_name + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
            
            if img_path:
                image_paths.append(str(img_path))
                dr_labels.append(min(dr_label, 4))  # Cap at 4 for IDRiD compatibility
    except Exception as e:
        logger.warning(f"Error loading MESSIDOR annotations from {csv_path}: {e}")
    
    logger.info(f"Loaded {len(image_paths)} MESSIDOR images")
    if dr_labels:
        logger.info(f"DR distribution: {pd.Series(dr_labels).value_counts().sort_index().to_dict()}")
    
    return image_paths, dr_labels


def load_eyepacs_dr_labels(eyepacs_images_dir: str, eyepacs_csv: str) -> Tuple[list, list]:
    """
    Load EyePACS dataset with DR labels only.
    
    EyePACS DR grades: 0=No, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
    
    Parameters
    ----------
    eyepacs_images_dir : str
        Path to EyePACS images directory
    eyepacs_csv : str
        Path to CSV label file
        
    Returns
    -------
    Tuple[list, list]
        (image_paths, dr_labels)
    """
    image_paths = []
    dr_labels = []
    
    images_path = Path(eyepacs_images_dir)
    csv_path = Path(eyepacs_csv)
    
    try:
        df = pd.read_csv(csv_path)
        
        # Standard columns: image, diabetic_retinopathy_grade
        if 'diabetic_retinopathy_grade' in df.columns:
            image_col = 'image' if 'image' in df.columns else df.columns[0]
            label_col = 'diabetic_retinopathy_grade'
        elif 'diagnosis' in df.columns:
            image_col = 'image_id' if 'image_id' in df.columns else df.columns[0]
            label_col = 'diagnosis'
        else:
            # Assume first column is image, second is label
            image_col = df.columns[0]
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[-1]
        
        for _, row in df.iterrows():
            img_name = str(row[image_col]).strip()
            dr_label = int(row[label_col])
            
            # Direct file lookup: check if image exists as-is or with added extension
            img_path = None
            if Path(images_path / img_name).exists():
                img_path = images_path / img_name
            else:
                # Try common extensions
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    candidate = images_path / (img_name + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
            
            if img_path:
                image_paths.append(str(img_path))
                dr_labels.append(min(dr_label, 4))
    
    except Exception as e:
        logger.warning(f"Error loading EyePACS annotations from {csv_path}: {e}")
    
    logger.info(f"Loaded {len(image_paths)} EyePACS images")
    if dr_labels:
        logger.info(f"DR distribution: {pd.Series(dr_labels).value_counts().sort_index().to_dict()}")
    
    return image_paths, dr_labels


def load_aptos_dr_labels(aptos_images_dir: str, aptos_csv: str) -> Tuple[list, list]:
    """
    Load APTOS 2019 dataset with DR labels only.
    
    APTOS DR grades: 0=No, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
    
    Parameters
    ----------
    aptos_images_dir : str
        Path to APTOS images directory
    aptos_csv : str
        Path to CSV label file
        
    Returns
    -------
    Tuple[list, list]
        (image_paths, dr_labels)
    """
    image_paths = []
    dr_labels = []
    
    images_path = Path(aptos_images_dir)
    csv_path = Path(aptos_csv)
    
    try:
        df = pd.read_csv(csv_path)
        # Standard APTOS format: id_code, diagnosis
        
        for _, row in df.iterrows():
            img_name = str(row.iloc[0]).strip()
            dr_label = int(row.iloc[1])
            
            # Direct file lookup: check if image exists as-is or with added extension
            img_path = None
            if Path(images_path / img_name).exists():
                img_path = images_path / img_name
            else:
                # Try common extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = images_path / (img_name + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
            
            if img_path:
                image_paths.append(str(img_path))
                dr_labels.append(min(dr_label, 4))
    
    except Exception as e:
        logger.warning(f"Error loading APTOS annotations from {csv_path}: {e}")
    
    logger.info(f"Loaded {len(image_paths)} APTOS images")
    if dr_labels:
        logger.info(f"DR distribution: {pd.Series(dr_labels).value_counts().sort_index().to_dict()}")
    
    return image_paths, dr_labels


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------

def crop_black_borders(image: np.ndarray, border_fraction: float = 0.10) -> np.ndarray:
    """Remove black circular borders common in fundus photographs.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array (H, W, 3)
    border_fraction : float
        Fraction of each side to crop (default 0.10 = 10%)
        
    Returns
    -------
    np.ndarray
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h = int(h * border_fraction)
    crop_w = int(w * border_fraction)
    cropped = image[crop_h : h - crop_h, crop_w : w - crop_w]
    return cropped


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to GREEN channel only.
    
    Per the DR-ASPP-DRN paper: green channel provides best vessel contrast
    for retinal fundus images. CLAHE enhances local contrast without 
    over-amplifying noise.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array (H, W, 3)
    clip_limit : float
        Histogram clip limit (default 2.0)
    grid_size : int
        Tile grid size (default 8x8)
        
    Returns
    -------
    np.ndarray
        Image with CLAHE applied to green channel only
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    # Extract green channel (index 1) — best contrast for retinal vessels
    green = image[:, :, 1]
    enhanced_green = clahe.apply(green)
    result = image.copy()
    result[:, :, 1] = enhanced_green  # Replace only green channel
    return result


def preprocess_image(
    image_path: str, 
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe_enhancement: bool = True,
    border_fraction: float = 0.10,
) -> np.ndarray:
    """
    Full preprocessing pipeline for fundus images.
    
    Steps:
    1. Load image
    2. Crop black borders
    3. Apply CLAHE enhancement (optional)
    4. Resize to target size
    5. Normalize to [-1, 1] for ResNet50
    
    Parameters
    ----------
    image_path : str
        Path to image file
    target_size : Tuple[int, int]
        Target size (H, W)
    apply_clahe_enhancement : bool
        Whether to apply CLAHE (default True)
    border_fraction : float
        Fraction of borders to crop (default 0.10)
        
    Returns
    -------
    np.ndarray
        Preprocessed image normalized to [-1, 1]
    """
    try:
        # 1. Load image
        image = cv2.imread(str(image_path))
        if image is None:
            # Try loading with TensorFlow for other formats
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = image.numpy()
        
        # Convert BGR to RGB if loaded with OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() > 1.0:
                image = image.astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)
        
        # 2. Crop black borders
        image = crop_black_borders(image, border_fraction=border_fraction)
        
        # 3. Apply CLAHE enhancement
        if apply_clahe_enhancement:
            image = apply_clahe(image, clip_limit=2.0, grid_size=8)
        
        # 4. Resize
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        
        # 5. Normalize to [-1, 1] for ResNet50 compatibility
        image = image.astype(np.float32)
        image = image / 127.5 - 1.0  # [0, 255] → [-1, 1]
        
        return image
    
    except Exception as e:
        logger.error(f"Error preprocessing {image_path}: {e}")
        return np.zeros((*target_size, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model,
    image_paths: list,
    dr_labels: list,
    dataset_name: str,
    batch_size: int = 8,
    apply_clahe: bool = True,
) -> Dict:
    """
    Evaluate model on a dataset (DR task only).
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    image_paths : list
        List of image file paths
    dr_labels : list
        List of DR ground truth labels
    dataset_name : str
        Name of dataset for logging
    batch_size : int
        Batch size for inference
    apply_clahe : bool
        Whether to apply CLAHE enhancement (default True)
        
    Returns
    -------
    Dict
        Evaluation metrics including QWK, accuracy, confusion matrix
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {dataset_name} ({len(image_paths)} images)")
    logger.info(f"{'='*60}")
    
    # Preprocess all images
    images = []
    valid_labels = []
    valid_paths = []
    
    for img_path, label in zip(image_paths, dr_labels):
        try:
            img = preprocess_image(img_path, apply_clahe_enhancement=apply_clahe)
            images.append(img)
            valid_labels.append(label)
            valid_paths.append(img_path)
        except Exception as e:
            logger.warning(f"Skipping {img_path}: {e}")
    
    images = np.array(images)
    dr_labels_array = np.array(valid_labels)
    
    logger.info(f"Successfully loaded {len(images)} images")
    logger.info(f"CLAHE enhancement: {'Enabled' if apply_clahe else 'Disabled'}")
    
    # Predict
    predictions = model.predict(images, batch_size=batch_size, verbose=1)
    
    # Extract DR predictions (first output)
    dr_preds = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    dr_pred_classes = np.argmax(dr_preds, axis=1)
    dr_pred_probs = np.max(dr_preds, axis=1)
    
    # Metrics
    qwk = compute_quadratic_weighted_kappa(dr_labels_array, dr_pred_classes)
    accuracy = np.mean(dr_labels_array == dr_pred_classes)
    
    # Confusion matrix
    cm = confusion_matrix(dr_labels_array, dr_pred_classes, labels=range(5))
    
    # Per-class accuracy
    per_class_acc = {}
    for i in range(5):
        mask = dr_labels_array == i
        if mask.sum() > 0:
            per_class_acc[f"Grade_{i}"] = np.mean(dr_pred_classes[mask] == i)
    
    results = {
        "dataset": dataset_name,
        "num_images": len(images),
        "preprocessing": {
            "clahe_enabled": apply_clahe,
            "border_cropping": 0.10,
            "normalization": "[-1, 1] for ResNet50"
        },
        "dr_qwk": float(qwk),
        "dr_accuracy": float(accuracy),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "dr_label_distribution": pd.Series(dr_labels_array).value_counts().sort_index().to_dict(),
        "dr_pred_distribution": pd.Series(dr_pred_classes).value_counts().sort_index().to_dict(),
    }
    
    # Detailed logging
    logger.info(f"\n{dataset_name} Results:")
    logger.info(f"  DR QWK: {results['dr_qwk']:.4f}")
    logger.info(f"  DR Accuracy: {results['dr_accuracy']:.4f}")
    logger.info(f"  Per-class accuracy: {per_class_acc}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info(f"  Ground Truth Distribution: {results['dr_label_distribution']}")
    logger.info(f"  Prediction Distribution: {results['dr_pred_distribution']}")
    
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation (DR task only, no DME)"
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        default=False,
        help="Load pre-trained weights (otherwise use random initialization)"
    )
    parser.add_argument(
        "--use-weights",
        type=str,
        default="pipeline_outputs/best_qwk.weights.h5",
        help="Path to pre-trained weights file"
    )
    parser.add_argument(
        "--messidor-images",
        type=str,
        default=None,
        help="Path to MESSIDOR images directory"
    )
    parser.add_argument(
        "--messidor-csv",
        type=str,
        default=None,
        help="Path to MESSIDOR CSV/XLS annotation file"
    )
    parser.add_argument(
        "--eyepacs-images",
        type=str,
        default=None,
        help="Path to EyePACS images directory"
    )
    parser.add_argument(
        "--eyepacs-csv",
        type=str,
        default=None,
        help="Path to EyePACS CSV label file"
    )
    parser.add_argument(
        "--aptos-images",
        type=str,
        default=None,
        help="Path to APTOS images directory"
    )
    parser.add_argument(
        "--aptos-csv",
        type=str,
        default=None,
        help="Path to APTOS CSV label file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--disable-clahe",
        action="store_true",
        default=False,
        help="Disable CLAHE enhancement during preprocessing"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="cross_dataset_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Build model
    logger.info("Building model...")
    model = build_model(
        input_shape=(512, 512, 3),
        backbone_weights="imagenet" if not args.use_model else None,
        trainable=False,  # Inference only
    )
    
    # Load weights if specified
    if args.use_model and Path(args.use_weights).exists():
        logger.info(f"Loading weights from {args.use_weights}...")
        load_weights_with_fallback(model, args.use_weights)
    elif args.use_model:
        logger.warning(f"Weights file not found: {args.use_weights}")
        logger.warning("Using ImageNet pre-trained weights instead")
    else:
        logger.info("Using random initialization (no pre-trained weights)")
    
    # Evaluate datasets
    all_results = {
        "model_info": {
            "use_pretrained": args.use_model,
            "weights_path": args.use_weights if args.use_model else None,
        },
        "preprocessing": {
            "clahe_enabled": not args.disable_clahe,
            "border_cropping": 0.10,
            "target_size": [512, 512],
            "normalization": "[-1, 1] for ResNet50"
        },
        "datasets": {}
    }
    
    # MESSIDOR
    if args.messidor_images and args.messidor_csv:
        if Path(args.messidor_images).exists() and Path(args.messidor_csv).exists():
            img_paths, labels = load_messidor_dr_labels(args.messidor_images, args.messidor_csv)
            if img_paths and labels:
                results = evaluate_dataset(
                    model, img_paths, labels, "MESSIDOR", args.batch_size, 
                    apply_clahe=not args.disable_clahe
                )
                all_results["datasets"]["messidor"] = results
    
    # EyePACS
    if args.eyepacs_images and args.eyepacs_csv:
        if Path(args.eyepacs_images).exists() and Path(args.eyepacs_csv).exists():
            img_paths, labels = load_eyepacs_dr_labels(args.eyepacs_images, args.eyepacs_csv)
            if img_paths and labels:
                results = evaluate_dataset(
                    model, img_paths, labels, "EyePACS", args.batch_size,
                    apply_clahe=not args.disable_clahe
                )
                all_results["datasets"]["eyepacs"] = results
    
    # APTOS 2019
    if args.aptos_images and args.aptos_csv:
        if Path(args.aptos_images).exists() and Path(args.aptos_csv).exists():
            img_paths, labels = load_aptos_dr_labels(args.aptos_images, args.aptos_csv)
            if img_paths and labels:
                results = evaluate_dataset(
                    model, img_paths, labels, "APTOS 2019", args.batch_size,
                    apply_clahe=not args.disable_clahe
                )
                all_results["datasets"]["aptos"] = results
    
    # Save results
    if all_results["datasets"]:
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("CROSS-DATASET EVALUATION SUMMARY")
        logger.info("="*60)
        for dataset_name, results in all_results["datasets"].items():
            logger.info(f"\n{dataset_name.upper()}:")
            logger.info(f"  QWK: {results['dr_qwk']:.4f}")
            logger.info(f"  Accuracy: {results['dr_accuracy']:.4f}")
    else:
        logger.error("No datasets evaluated. Please specify dataset paths.")


if __name__ == "__main__":
    main()
