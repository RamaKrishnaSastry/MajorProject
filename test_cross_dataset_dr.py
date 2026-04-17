"""
test_cross_dataset_dr.py - Dataset evaluation for DR grading (IDRiD and external sets).

Tests the DR classification task only (no DME).
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
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tensorflow import keras
from tensorflow.keras import layers

# Local imports
from qwk_metrics import compute_quadratic_weighted_kappa

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Model Architecture (copied from model.py to ensure exact compatibility)
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package="MajorProject")
class ResizeToMatch(layers.Layer):
    """Resize the first tensor to match spatial size of a reference tensor."""

    def call(self, inputs):
        source, reference = inputs
        target_hw = tf.shape(reference)[1:3]
        return tf.image.resize(source, target_hw)

    def get_config(self):
        return super().get_config()


def build_backbone(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    weights: str = "imagenet",
    trainable: bool = True,
    weights_path: Optional[str] = None,
) -> keras.Model:
    """Build a ResNet50 backbone."""
    try:
        base = keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    except Exception as exc:
        if weights == "imagenet":
            logger.warning(
                "Could not load ImageNet weights (%s). Falling back to random init.",
                exc,
            )
            base = keras.applications.ResNet50(
                include_top=False,
                weights=None,
                input_shape=input_shape,
            )
        else:
            raise
    base.trainable = trainable
    output = base.get_layer("conv4_block6_out").output
    backbone = keras.Model(inputs=base.input, outputs=output, name="resnet50_conv4_backbone")
    backbone.trainable = trainable

    if weights_path is not None:
        backbone.load_weights(weights_path, skip_mismatch=True)
        logger.info("✅ Loaded custom backbone weights from '%s'.", weights_path)

    logger.info(
        "✅ Backbone: ResNet50@conv4_block6_out, trainable=%s, output shape=%s",
        trainable,
        backbone.output_shape,
    )
    return backbone


def build_aspp(
    x: tf.Tensor,
    filters: int = 256,
    name_prefix: str = "aspp",
) -> tf.Tensor:
    """Atrous Spatial Pyramid Pooling (ASPP) block."""
    b0 = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                       name=f"{name_prefix}_b0")(x)
    b0 = layers.BatchNormalization(name=f"{name_prefix}_b0_bn")(b0)
    b0 = layers.Activation("relu", name=f"{name_prefix}_b0_relu")(b0)

    b1 = layers.Conv2D(filters, 3, padding="same", dilation_rate=6, use_bias=False,
                       name=f"{name_prefix}_b1")(x)
    b1 = layers.BatchNormalization(name=f"{name_prefix}_b1_bn")(b1)
    b1 = layers.Activation("relu", name=f"{name_prefix}_b1_relu")(b1)

    b2 = layers.Conv2D(filters, 3, padding="same", dilation_rate=12, use_bias=False,
                       name=f"{name_prefix}_b2")(x)
    b2 = layers.BatchNormalization(name=f"{name_prefix}_b2_bn")(b2)
    b2 = layers.Activation("relu", name=f"{name_prefix}_b2_relu")(b2)

    b3 = layers.Conv2D(filters, 3, padding="same", dilation_rate=18, use_bias=False,
                       name=f"{name_prefix}_b3")(x)
    b3 = layers.BatchNormalization(name=f"{name_prefix}_b3_bn")(b3)
    b3 = layers.Activation("relu", name=f"{name_prefix}_b3_relu")(b3)

    b4 = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    b4 = layers.Reshape((1, 1, -1), name=f"{name_prefix}_gap_reshape")(b4)
    b4 = layers.Conv2D(filters, 1, use_bias=False, name=f"{name_prefix}_b4_conv")(b4)
    b4 = layers.BatchNormalization(name=f"{name_prefix}_b4_bn")(b4)
    b4 = layers.Activation("relu", name=f"{name_prefix}_b4_relu")(b4)
    
    b4 = ResizeToMatch(name=f"{name_prefix}_b4_resize")([b4, x])

    out = layers.Concatenate(name=f"{name_prefix}_concat")([b0, b1, b2, b3, b4])
    out = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                        name=f"{name_prefix}_proj")(out)
    out = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(out)
    out = layers.Activation("relu", name=f"{name_prefix}_proj_relu")(out)
    
    logger.debug("✅ ASPP output shape: %s", out.shape)
    return out


def build_dr_head(
    x: tf.Tensor,
    num_classes: int = 5,
    dropout_rate: float = 0.5,
    hidden_units: int = 256,
) -> tf.Tensor:
    """Build the Diabetic Retinopathy (DR) classification head."""
    x = layers.GlobalAveragePooling2D(name="dr_gap")(x)
    x = layers.LayerNormalization(name="dr_ln0")(x)

    shortcut = layers.Dense(hidden_units, use_bias=True, name="dr_res_proj")(x)
    shortcut = layers.BatchNormalization(name="dr_res_proj_bn")(shortcut)

    h = layers.Dense(hidden_units, use_bias=False, name="dr_fc1")(x)
    h = layers.BatchNormalization(name="dr_fc1_bn")(h)
    h = layers.Activation("swish", name="dr_fc1_act")(h)
    h = layers.Dropout(dropout_rate * 0.5, name="dr_fc1_dropout")(h)
    h = layers.Dense(hidden_units, use_bias=False, name="dr_fc2")(h)
    h = layers.BatchNormalization(name="dr_fc2_bn")(h)

    x = layers.Add(name="dr_residual_add")([shortcut, h])
    x = layers.Activation("swish", name="dr_residual_act")(x)
    x = layers.Dropout(dropout_rate, name="dr_dropout")(x)
    x = layers.Dense(num_classes, use_bias=False, activation="softmax", name="dr_output")(x)
    return x


def build_dme_head(
    x: tf.Tensor,
    num_classes: int = 3,
    dropout_rate: float = 0.5,
    hidden_units: int = 256,
    residual_mlp: bool = False,
) -> tf.Tensor:
    """Build the DME (Diabetic Macular Edema) classification head."""
    x = layers.GlobalAveragePooling2D(name="dme_gap")(x)
    if residual_mlp:
        x = layers.LayerNormalization(name="dme_ln0")(x)
        shortcut = layers.Dense(hidden_units, use_bias=False, name="dme_res_proj")(x)

        h = layers.Dense(hidden_units, use_bias=False, name="dme_fc1")(x)
        h = layers.Activation("swish", name="dme_fc1_act")(h)
        h = layers.Dropout(dropout_rate * 0.5, name="dme_fc1_dropout")(h)
        h = layers.Dense(hidden_units, use_bias=False, name="dme_fc2")(h)

        x = layers.Add(name="dme_residual_add")([shortcut, h])
        x = layers.Activation("swish", name="dme_residual_act")(x)
        x = layers.Dropout(dropout_rate, name="dme_dropout")(x)
    else:
        x = layers.Dense(hidden_units, use_bias=False, activation="relu", name="dme_fc1")(x)
        x = layers.Dropout(dropout_rate, name="dme_dropout")(x)
    x = layers.Dense(num_classes, use_bias=False, activation="softmax", name="dme_risk")(x)
    return x


def build_model(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    backbone_weights: str = "imagenet",
    num_dme_classes: int = 3,
    num_dr_classes: int = 5,
    aspp_filters: int = 256,
    dr_head_units: int = 256,
    dme_head_units: int = 256,
    dme_head_residual: bool = False,
    dropout_rate: float = 0.5,
    trainable: bool = True,
    backbone_weights_path: Optional[str] = None,
) -> keras.Model:
    """Build the complete multi-task DR + DME model."""
    inputs = keras.Input(shape=input_shape, name="input_image")

    backbone = build_backbone(
        input_shape=input_shape,
        weights=backbone_weights,
        trainable=trainable,
        weights_path=backbone_weights_path,
    )
    features = backbone(inputs)
    logger.info("✅ Backbone: %d parameters", backbone.count_params())

    aspp_out = build_aspp(features, filters=aspp_filters)
    logger.info("✅ ASPP module added")
    
    dr_out = build_dr_head(
        aspp_out,
        num_classes=num_dr_classes,
        dropout_rate=dropout_rate,
        hidden_units=dr_head_units,
    )

    dme_out = build_dme_head(
        aspp_out,
        num_classes=num_dme_classes,
        dropout_rate=dropout_rate,
        hidden_units=dme_head_units,
        residual_mlp=dme_head_residual,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={"dr_output": dr_out, "dme_risk": dme_out},
        name="multitask_dr_dme",
    )

    model.trainable = trainable

    logger.info("✅ Built multi-task model:")
    logger.info("   Total parameters: %d", model.count_params())
    logger.info("   Trainable: %s", trainable)
    logger.info("   Input shape: %s", input_shape)

    return model


def _snapshot_layer_weights(model, layer_names):
    snapshots = {}
    for layer_name in layer_names:
        try:
            layer = model.get_layer(layer_name)
            snapshots[layer_name] = [np.copy(w) for w in layer.get_weights()]
        except Exception:
            snapshots[layer_name] = None
    return snapshots


def _did_layer_change(model, layer_name, before_weights, tol: float = 1e-8) -> bool:
    if before_weights is None:
        return False
    try:
        after_weights = model.get_layer(layer_name).get_weights()
    except Exception:
        return False
    if len(before_weights) != len(after_weights):
        return False
    max_delta = 0.0
    for w_before, w_after in zip(before_weights, after_weights):
        if w_before.shape != w_after.shape:
            return False
        max_delta = max(max_delta, float(np.max(np.abs(w_before - w_after))))
    return max_delta > tol


def load_weights_with_fallback(model, weights_path: str, tracked_layer_names=("dr_output",)):
    """
    Load weights with better error handling for architecture mismatches.
    
    Loads compatible layers with ``skip_mismatch=True`` and verifies whether
    tracked layers (e.g., ``dr_output``) actually changed after loading.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model to load weights into
    weights_path : str
        Path to weights file
    """
    snapshots = _snapshot_layer_weights(model, tracked_layer_names)
    layer_loaded = {name: False for name in tracked_layer_names}

    try:
        logger.info("Attempting to load weights (skip_mismatch=True)...")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.load_weights(weights_path, skip_mismatch=True)

        for name in tracked_layer_names:
            layer_loaded[name] = _did_layer_change(model, name, snapshots.get(name))

        for warning_obj in caught:
            msg = str(warning_obj.message)
            if "could not be loaded" in msg or "shape" in msg:
                logger.warning("Weight load warning: %s", msg.splitlines()[0])

        logger.info("✅ Weights load call completed")
        logger.info("Tracked layer load status: %s", layer_loaded)
        return True, layer_loaded
    except Exception as e:
        logger.warning("Load failed: %s", e)
        return False, layer_loaded


def build_legacy_dr_regression_model(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    backbone_weights: Optional[str] = None,
    trainable: bool = False,
):
    """Build legacy DR regression model for old Stage-2 checkpoints (fallback only).

    Legacy DR head: GAP -> Dense(256,relu) -> Dropout(0.4) -> Dense(1,relu)
    with layer names preserved for checkpoint compatibility.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    backbone = build_backbone(
        input_shape=input_shape,
        weights=backbone_weights,
        trainable=trainable,
    )
    features = backbone(inputs)
    aspp_features = build_aspp(features, filters=256)

    x = tf.keras.layers.GlobalAveragePooling2D(name="dr_gap")(aspp_features)
    x = tf.keras.layers.Dense(256, activation="relu", name="dr_fc1")(x)
    x = tf.keras.layers.Dropout(0.4, name="dr_dropout")(x)
    dr_output = tf.keras.layers.Dense(1, activation="relu", name="dr_output")(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs={"dr_output": dr_output},
        name="dr_only_legacy_regression",
    )


def build_best_matching_model(weights_path: str):
    """Build a model variant that best matches the provided checkpoint.
    
    First tries to load a pre-built model file (.model.h5). If not available,
    tries loading weights into current production pipeline architecture.
    Finally falls back to legacy architecture for backward compatibility.
    """
    # Check if pre-built model exists (prefer this over weights-only loading)
    model_file = weights_path.replace(".weights.h5", ".model.h5")
    if Path(model_file).exists():
        try:
            logger.info(f"Loading pre-built model from {model_file}...")
            model = keras.models.load_model(model_file)
            logger.info("✅ Pre-built model loaded successfully.")
            return model, "classification", "prebuilt-production-model"
        except Exception as e:
            logger.warning(f"Could not load pre-built model from {model_file}: {e}")
    
    logger.info("Building pipeline architecture from weights_path...")
    logger.info(f"Loading weights from {weights_path}...")
    
    # Try current architecture first
    logger.info("Trying current multi-task classification architecture...")
    current_model = build_model(
        input_shape=(512, 512, 3),
        backbone_weights=None,
        trainable=False,
    )
    ok, status = load_weights_with_fallback(current_model, weights_path, tracked_layer_names=("dr_output",))
    if ok and status.get("dr_output", False):
        logger.info("✅ Current architecture matched DR head weights.")
        return current_model, "classification", "current-multitask"

    # Fallback to legacy if current doesn't work
    logger.warning(
        "Current architecture did not load DR head weights. "
        "Trying legacy DR regression architecture..."
    )
    legacy_model = build_legacy_dr_regression_model(
        input_shape=(512, 512, 3),
        backbone_weights=None,
        trainable=False,
    )
    ok_legacy, status_legacy = load_weights_with_fallback(
        legacy_model,
        weights_path,
        tracked_layer_names=("dr_output",),
    )
    if ok_legacy and status_legacy.get("dr_output", False):
        logger.info("✅ Legacy architecture matched DR head weights.")
        return legacy_model, "regression", "legacy-dr-regression"

    raise RuntimeError(
        "Could not match DR head layer from checkpoint in either supported architecture "
        "(current classification or legacy regression). "
        "Use a checkpoint produced by this codebase/version, or export a compatible DR head."
    )


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_idrid_dr_labels(idrid_images_dir: str, idrid_csv: str) -> Tuple[list, list]:
    """
    Load IDRiD dataset with DR labels only.

    IDRiD DR grades: 0=No, 1=Mild, 2=Moderate, 3=Severe NPDR, 4=PDR

    Parameters
    ----------
    idrid_images_dir : str
        Path to IDRiD images directory
    idrid_csv : str
        Path to IDRiD CSV annotation file

    Returns
    -------
    Tuple[list, list]
        (image_paths, dr_labels)
    """
    image_paths = []
    dr_labels = []

    images_path = Path(idrid_images_dir)
    csv_path = Path(idrid_csv)

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        image_col = None
        for candidate in ["Image name", "image_name", "image", "Image", df.columns[0]]:
            if candidate in df.columns:
                image_col = candidate
                break

        dr_col = None
        for candidate in ["Retinopathy grade", "dr_grade", "DR_grade", "diagnosis"]:
            if candidate in df.columns:
                dr_col = candidate
                break

        if image_col is None:
            raise ValueError(f"Could not detect image column. Columns={list(df.columns)}")
        if dr_col is None:
            raise ValueError(f"Could not detect DR label column. Columns={list(df.columns)}")

        for _, row in df.iterrows():
            img_name = str(row[image_col]).strip()
            dr_label = int(row[dr_col])

            img_path = None
            if Path(images_path / img_name).exists():
                img_path = images_path / img_name
            else:
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"]:
                    candidate = images_path / (img_name + ext)
                    if candidate.exists():
                        img_path = candidate
                        break

            if img_path:
                image_paths.append(str(img_path))
                dr_labels.append(min(dr_label, 4))

    except Exception as e:
        logger.warning(f"Error loading IDRiD annotations from {csv_path}: {e}")

    logger.info(f"Loaded {len(image_paths)} IDRiD images")
    if dr_labels:
        logger.info(f"DR distribution: {pd.Series(dr_labels).value_counts().sort_index().to_dict()}")

    return image_paths, dr_labels

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

def crop_black_borders(image: np.ndarray, border_fraction: float = 0.05) -> np.ndarray:
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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 0 and h > 0:
            return image[y : y + h, x : x + w]

    h, w = image.shape[:2]
    crop_h = int(h * border_fraction)
    crop_w = int(w * border_fraction)
    return image[crop_h : h - crop_h, crop_w : w - crop_w]


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
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_image(
    image_path: str, 
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe_enhancement: bool = True,
    border_fraction: float = 0.05,
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
    preprocess_ensemble: bool = False,
    tta_mode: str = "none",
    dr_prediction_mode: str = "classification",
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
    preprocess_ensemble : bool
        If True, run inference with CLAHE on and off and average logits/probabilities.
    tta_mode : str
        Test-time augmentation mode: none, hflip, rot4, dihedral8.
        
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
    logger.info(f"Preprocessing ensemble: {'Enabled' if preprocess_ensemble else 'Disabled'}")
    logger.info(f"TTA mode: {tta_mode}")
    
    def _extract_dr_output(predictions_obj):
        if isinstance(predictions_obj, dict):
            dr_vals = predictions_obj.get("dr_output")
            if dr_vals is None:
                dr_vals = next(iter(predictions_obj.values()))
        elif isinstance(predictions_obj, (tuple, list)):
            dr_vals = predictions_obj[0]
        else:
            dr_vals = predictions_obj
        return np.asarray(dr_vals)

    def _tta_transforms(mode: str):
        key = (mode or "none").lower()
        if key == "hflip":
            return [
                ("identity", lambda x: x),
                ("hflip", lambda x: np.flip(x, axis=2)),
            ]
        if key == "rot4":
            return [
                ("r0", lambda x: x),
                ("r90", lambda x: np.rot90(x, 1, axes=(1, 2))),
                ("r180", lambda x: np.rot90(x, 2, axes=(1, 2))),
                ("r270", lambda x: np.rot90(x, 3, axes=(1, 2))),
            ]
        if key == "dihedral8":
            return [
                ("r0", lambda x: x),
                ("r90", lambda x: np.rot90(x, 1, axes=(1, 2))),
                ("r180", lambda x: np.rot90(x, 2, axes=(1, 2))),
                ("r270", lambda x: np.rot90(x, 3, axes=(1, 2))),
                ("h", lambda x: np.flip(x, axis=2)),
                ("v", lambda x: np.flip(x, axis=1)),
                ("d1", lambda x: np.flip(np.rot90(x, 1, axes=(1, 2)), axis=2)),
                ("d2", lambda x: np.flip(np.rot90(x, 1, axes=(1, 2)), axis=1)),
            ]
        return [("none", lambda x: x)]

    def _predict_dr_with_tta(input_images: np.ndarray) -> np.ndarray:
        transforms = _tta_transforms(tta_mode)
        dr_outputs = []
        for _, transform in transforms:
            aug_images = transform(input_images)
            preds_obj = model.predict(aug_images, batch_size=batch_size, verbose=0)
            dr_outputs.append(_extract_dr_output(preds_obj))
        return np.mean(np.stack(dr_outputs, axis=0), axis=0)

    # Predict with optional TTA and optional preprocessing ensemble.
    dr_preds = _predict_dr_with_tta(images)
    if preprocess_ensemble:
        alt_images = []
        for img_path in valid_paths:
            # Use opposite CLAHE setting for robust domain-shift averaging.
            alt_images.append(
                preprocess_image(img_path, apply_clahe_enhancement=not apply_clahe)
            )
        alt_images = np.array(alt_images)
        dr_preds_alt = _predict_dr_with_tta(alt_images)
        dr_preds = 0.5 * (np.asarray(dr_preds) + np.asarray(dr_preds_alt))

    dr_preds = np.asarray(dr_preds)

    if dr_prediction_mode == "regression" or (dr_preds.ndim == 2 and dr_preds.shape[-1] == 1):
        dr_scores = dr_preds.reshape(-1)
        # Legacy models may output [0,1] (sigmoid) or [0,4] (relu).
        if float(np.nanmax(dr_scores)) <= 1.5:
            dr_scores = np.clip(dr_scores, 0.0, 1.0) * 4.0
        else:
            dr_scores = np.clip(dr_scores, 0.0, 4.0)
        dr_pred_classes = np.rint(dr_scores).astype(np.int32)
        dr_pred_classes = np.clip(dr_pred_classes, 0, 4)
        dr_pred_probs = np.ones_like(dr_scores, dtype=np.float32)
    else:
        if dr_preds.ndim == 1:
            logger.warning("1D DR output detected; interpreting as regression scores.")
            dr_scores = np.clip(dr_preds, 0.0, 4.0)
            dr_pred_classes = np.rint(dr_scores).astype(np.int32)
            dr_pred_classes = np.clip(dr_pred_classes, 0, 4)
            dr_pred_probs = np.ones_like(dr_scores, dtype=np.float32)
        else:
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
            "preprocess_ensemble": preprocess_ensemble,
            "border_cropping": 0.05,
            "normalization": "[-1, 1] for ResNet50"
        },
        "tta_mode": str(tta_mode).lower(),
        "dr_prediction_mode": dr_prediction_mode,
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
        description="Dataset evaluation for DR task only (no DME)"
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
        "--idrid-images",
        type=str,
        default=None,
        help="Path to IDRiD images directory"
    )
    parser.add_argument(
        "--idrid-csv",
        type=str,
        default=None,
        help="Path to IDRiD CSV annotation file"
    )
    parser.add_argument(
        "--only-idrid",
        action="store_true",
        default=False,
        help="Evaluate only IDRiD and ignore MESSIDOR/EyePACS/APTOS even if provided"
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
        "--preprocess-ensemble",
        action="store_true",
        default=False,
        help="Average predictions from CLAHE on/off preprocessing (domain-shift robust)"
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        default="none",
        choices=["none", "hflip", "rot4", "dihedral8"],
        help="Test-time augmentation mode"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="cross_dataset_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()

    # Build model with checkpoint-aware architecture selection
    dr_prediction_mode = "classification"
    model_variant = "current-multitask"

    if args.use_model and Path(args.use_weights).exists():
        logger.info("Building model for checkpoint compatibility...")
        logger.info(f"Loading weights from {args.use_weights}...")
        model, dr_prediction_mode, model_variant = build_best_matching_model(args.use_weights)
    elif args.use_model:
        logger.warning(f"Weights file not found: {args.use_weights}")
        logger.warning("Using ImageNet pre-trained backbone with current architecture")
        model = build_model(
            input_shape=(512, 512, 3),
            backbone_weights="imagenet",
            trainable=False,
        )
    else:
        logger.info("Using random initialization (no pre-trained weights)")
        model = build_model(
            input_shape=(512, 512, 3),
            backbone_weights=None,
            trainable=False,
        )
    
    # Evaluate datasets
    all_results = {
        "model_info": {
            "use_pretrained": args.use_model,
            "weights_path": args.use_weights if args.use_model else None,
            "model_variant": model_variant,
            "dr_prediction_mode": dr_prediction_mode,
        },
        "preprocessing": {
            "clahe_enabled": not args.disable_clahe,
            "preprocess_ensemble": bool(args.preprocess_ensemble),
            "tta_mode": str(args.tta_mode).lower(),
            "border_cropping": 0.05,
            "target_size": [512, 512],
            "normalization": "[-1, 1] for ResNet50"
        },
        "datasets": {}
    }
    
    # IDRiD (primary evaluation target)
    if args.idrid_images and args.idrid_csv:
        if Path(args.idrid_images).exists() and Path(args.idrid_csv).exists():
            img_paths, labels = load_idrid_dr_labels(args.idrid_images, args.idrid_csv)
            if img_paths and labels:
                results = evaluate_dataset(
                    model, img_paths, labels, "IDRiD", args.batch_size,
                    apply_clahe=not args.disable_clahe,
                    preprocess_ensemble=bool(args.preprocess_ensemble),
                    tta_mode=str(args.tta_mode).lower(),
                    dr_prediction_mode=dr_prediction_mode,
                )
                all_results["datasets"]["idrid"] = results

    if not args.only_idrid:
        # MESSIDOR
        if args.messidor_images and args.messidor_csv:
            if Path(args.messidor_images).exists() and Path(args.messidor_csv).exists():
                img_paths, labels = load_messidor_dr_labels(args.messidor_images, args.messidor_csv)
                if img_paths and labels:
                    results = evaluate_dataset(
                        model, img_paths, labels, "MESSIDOR", args.batch_size,
                        apply_clahe=not args.disable_clahe,
                        preprocess_ensemble=bool(args.preprocess_ensemble),
                        tta_mode=str(args.tta_mode).lower(),
                        dr_prediction_mode=dr_prediction_mode,
                    )
                    all_results["datasets"]["messidor"] = results

        # EyePACS
        if args.eyepacs_images and args.eyepacs_csv:
            if Path(args.eyepacs_images).exists() and Path(args.eyepacs_csv).exists():
                img_paths, labels = load_eyepacs_dr_labels(args.eyepacs_images, args.eyepacs_csv)
                if img_paths and labels:
                    results = evaluate_dataset(
                        model, img_paths, labels, "EyePACS", args.batch_size,
                        apply_clahe=not args.disable_clahe,
                        preprocess_ensemble=bool(args.preprocess_ensemble),
                        tta_mode=str(args.tta_mode).lower(),
                        dr_prediction_mode=dr_prediction_mode,
                    )
                    all_results["datasets"]["eyepacs"] = results

        # APTOS 2019
        if args.aptos_images and args.aptos_csv:
            if Path(args.aptos_images).exists() and Path(args.aptos_csv).exists():
                img_paths, labels = load_aptos_dr_labels(args.aptos_images, args.aptos_csv)
                if img_paths and labels:
                    results = evaluate_dataset(
                        model, img_paths, labels, "APTOS 2019", args.batch_size,
                        apply_clahe=not args.disable_clahe,
                        preprocess_ensemble=bool(args.preprocess_ensemble),
                        tta_mode=str(args.tta_mode).lower(),
                        dr_prediction_mode=dr_prediction_mode,
                    )
                    all_results["datasets"]["aptos"] = results
    else:
        logger.info("--only-idrid enabled: skipping MESSIDOR, EyePACS, and APTOS evaluation.")
    
    # Save results
    if all_results["datasets"]:
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        for dataset_name, results in all_results["datasets"].items():
            logger.info(f"\n{dataset_name.upper()}:")
            logger.info(f"  QWK: {results['dr_qwk']:.4f}")
            logger.info(f"  Accuracy: {results['dr_accuracy']:.4f}")
    else:
        logger.error("No datasets evaluated. Please specify dataset paths.")


if __name__ == "__main__":
    main()
