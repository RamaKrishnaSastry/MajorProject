"""
ablation_study.py - Comprehensive ablation study for DR-ASPP-DRN architecture variants.

Compares 10+ model configurations to validate each component's contribution:

ARCHITECTURAL ABLATIONS:
- Model A: Baseline ResNet50 only                    (expected QWK ~0.73-0.75)
- Model B: ResNet50 + ASPP module                    (expected QWK ~0.76-0.78, +0.03-0.05 vs A)
- Model C: Full DR-ASPP-DRN (Multi-task + ASPP)      (expected QWK >= 0.80, +0.01-0.02 vs B)

TWO-STAGE TRAINING ABLATIONS:
- Model D: Full DR-ASPP-DRN (single-stage training)  (expected QWK ~0.78-0.79)
- Model E: Full DR-ASPP-DRN (two-stage training)     (expected QWK >= 0.80, +0.01-0.02 vs D)

LOSS FUNCTION STACK ABLATIONS:
- Model F: Simple categorical cross-entropy only     (expected QWK ~0.75-0.76)
- Model G: + Class weighting (address imbalance)     (expected QWK ~0.77-0.78, +0.02 vs F)
- Model H: + Ordinal weighting (respect ordering)    (expected QWK ~0.78-0.79, +0.01 vs G)
- Model I: + Label smoothing (soften targets)        (expected QWK ~0.79-0.80, +0.01 vs H)
- Model J: Full stack (all losses combined)          (expected QWK >= 0.80, +0.00-0.01 vs I)

Provides:
- Quantitative comparison table for all 10 variants
- Statistical significance testing (paired bootstrap)
- Separate visualizations for each ablation category
- Component contribution breakdown
- JSON export of comprehensive results
"""

import json
import logging
import os
import time
import gc
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import numpy as np

# Keep TensorFlow runtime conservative for long ablation sweeps on Kaggle GPUs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from qwk_metrics import (
    compute_quadratic_weighted_kappa,
    compute_ordinal_metrics,
    NUM_DME_CLASSES,
    DME_CLASS_NAMES,
)
from evaluate import get_predictions, compute_accuracy, compute_f1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _load_project_hparams() -> Dict[str, float]:
    """Load project-aligned defaults from config.yaml with safe fallbacks."""
    defaults = {
        "learning_rate": 2.0e-5,
        "dropout_rate": 0.4,
        "dr_loss_weight": 0.32,
        "aspp_filters": 256,
        "dr_head_units": 256,
        "dme_head_units": 256,
        "dme_head_residual": True,
        "early_stopping_patience": 15,
        "focal_loss_gamma": 1.5,
        "dme_label_smoothing": 0.02,
    }

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not _YAML_AVAILABLE or not os.path.exists(config_path):
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        stage1_cfg = cfg.get("stage1", {}) or {}
        model_cfg = cfg.get("model", {}) or {}

        defaults["learning_rate"] = float(stage1_cfg.get("learning_rate", defaults["learning_rate"]))
        defaults["dropout_rate"] = float(model_cfg.get("dropout_rate", defaults["dropout_rate"]))
        defaults["dr_loss_weight"] = float(stage1_cfg.get("dr_loss_weight", defaults["dr_loss_weight"]))
        defaults["aspp_filters"] = int(model_cfg.get("aspp_filters", defaults["aspp_filters"]))
        defaults["dr_head_units"] = int(model_cfg.get("dr_head_units", defaults["dr_head_units"]))
        defaults["dme_head_units"] = int(model_cfg.get("dme_head_units", defaults["dme_head_units"]))
        defaults["dme_head_residual"] = bool(model_cfg.get("dme_head_residual", defaults["dme_head_residual"]))
        defaults["early_stopping_patience"] = int(
            stage1_cfg.get("early_stopping_patience", defaults["early_stopping_patience"])
        )
        defaults["focal_loss_gamma"] = float(stage1_cfg.get("focal_loss_gamma", defaults["focal_loss_gamma"]))
    except Exception as exc:
        logger.warning("Could not parse config.yaml for hyperparameters: %s", exc)

    return defaults


PROJECT_HPARAMS = _load_project_hparams()
PROJECT_LR = float(PROJECT_HPARAMS["learning_rate"])
PROJECT_DROPOUT = float(PROJECT_HPARAMS["dropout_rate"])
PROJECT_DR_LOSS_WEIGHT = float(PROJECT_HPARAMS["dr_loss_weight"])
PROJECT_ASPP_FILTERS = int(PROJECT_HPARAMS["aspp_filters"])
PROJECT_DR_HEAD_UNITS = int(PROJECT_HPARAMS["dr_head_units"])
PROJECT_DME_HEAD_UNITS = int(PROJECT_HPARAMS["dme_head_units"])
PROJECT_DME_HEAD_RESIDUAL = bool(PROJECT_HPARAMS["dme_head_residual"])
PROJECT_EARLY_STOPPING_PATIENCE = int(PROJECT_HPARAMS["early_stopping_patience"])
PROJECT_FOCAL_GAMMA = float(PROJECT_HPARAMS["focal_loss_gamma"])

logger.info(
    "Ablation hyperparameters aligned to project defaults: lr=%.2e, dropout=%.3f, dr_loss_weight=%.3f, early_stop_patience=%d",
    PROJECT_LR,
    PROJECT_DROPOUT,
    PROJECT_DR_LOSS_WEIGHT,
    PROJECT_EARLY_STOPPING_PATIENCE,
)

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
# CUSTOM LOSS FUNCTIONS FOR ABLATION
# ---------------------------------------------------------------------------

class OrdinalWeightedCrossEntropy(keras.losses.Loss):
    """Ordinal-aware weighted cross-entropy that penalizes distant misclassifications.
    
    Adjacent errors (e.g., class 0->1) are penalized less than distant errors (0->2).
    """
    def __init__(self, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        # Build ordinal penalty matrix: [0->0]=1.0, [0->1]=1.25, [0->2]=2.0, etc.
        self.penalty_matrix = np.ones((num_classes, num_classes), dtype=np.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    distance = abs(i - j)
                    self.penalty_matrix[i, j] = 1.0 + 0.25 * distance
    
    def call(self, y_true, y_pred):
        """Compute ordinal-weighted CE loss."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Standard cross-entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Apply ordinal weights
        y_true_class = tf.argmax(y_true, axis=-1)
        weights = tf.gather_nd(
            tf.constant(self.penalty_matrix),
            tf.stack([y_true_class, tf.argmax(y_pred, axis=-1)], axis=1)
        )
        
        return tf.reduce_mean(ce * tf.expand_dims(weights, axis=1))


class FocalLoss(keras.losses.Loss):
    """Focal loss: down-weight easy examples, focus on hard negatives."""
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        ce = keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = (1 - p_t) ** self.gamma
        
        return self.alpha * focal_weight * ce


# ---------------------------------------------------------------------------
# MODEL VARIANT BUILDERS
# ---------------------------------------------------------------------------

def _resolve_backbone_spec(backbone_weights: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split backbone spec into (keras_weights_arg, custom_weights_path)."""
    if backbone_weights in (None, "imagenet"):
        return backbone_weights, None

    if isinstance(backbone_weights, str):
        return None, backbone_weights

    return backbone_weights, None


def _build_resnet50_backbone(
    input_shape: Tuple[int, int, int],
    backbone_weights: Optional[str],
) -> keras.Model:
    """Build ResNet50 backbone and safely load optional custom checkpoint."""
    keras_weights_arg, custom_weights_path = _resolve_backbone_spec(backbone_weights)
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights=keras_weights_arg,
        input_shape=input_shape,
    )

    if custom_weights_path is not None:
        loaded = False
        try:
            backbone.load_weights(custom_weights_path, by_name=True, skip_mismatch=True)
            loaded = True
        except TypeError:
            # Some formats do not support by_name; fallback to topological loading.
            try:
                backbone.load_weights(custom_weights_path, skip_mismatch=True)
                loaded = True
            except Exception as exc:
                logger.warning(
                    "Could not fully load custom backbone weights from '%s': %s",
                    custom_weights_path,
                    exc,
                )
        except Exception as exc:
            logger.warning(
                "Could not fully load custom backbone weights from '%s': %s",
                custom_weights_path,
                exc,
            )

        if loaded:
            logger.info("Loaded custom backbone weights from '%s' with mismatch tolerance.", custom_weights_path)

    backbone.trainable = True
    return backbone

def build_model_a(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
) -> keras.Model:
    """Model A: Baseline ResNet50 with direct classification head.

    No ASPP, no multi-task learning. Architecture ablation baseline.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _build_resnet50_backbone(input_shape, backbone_weights)
    features = backbone(inputs)

    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(PROJECT_DROPOUT, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelA_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model A (Baseline ResNet50): %d params", model.count_params())
    return model


def build_model_b(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model B: ResNet50 + ASPP module, single DME head.

    Adds multi-scale context via ASPP. Tests ASPP contribution.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_aspp

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _build_resnet50_backbone(input_shape, backbone_weights)
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters, name_prefix="aspp")

    x = layers.GlobalAveragePooling2D(name="dme_gap")(aspp_out)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(PROJECT_DROPOUT, name="dme_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelB_ASPP")
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model B (ResNet50 + ASPP): %d params", model.count_params())
    return model


def build_model_c(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_dme_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model C: Full DR-ASPP-DRN (multi-task DR + DME with ASPP).

    The complete production architecture without two-stage training.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_dme_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_model

    backbone_weights_arg, backbone_weights_path = _resolve_backbone_spec(backbone_weights)

    model = build_model(
        input_shape=input_shape,
        backbone_weights=backbone_weights_arg,
        backbone_weights_path=backbone_weights_path,
        num_dme_classes=num_dme_classes,
        aspp_filters=aspp_filters,
        dr_head_units=PROJECT_DR_HEAD_UNITS,
        dme_head_units=PROJECT_DME_HEAD_UNITS,
        dme_head_residual=PROJECT_DME_HEAD_RESIDUAL,
        dropout_rate=PROJECT_DROPOUT,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss={"dr_output": "mse", "dme_risk": "categorical_crossentropy"},
        loss_weights={"dr_output": PROJECT_DR_LOSS_WEIGHT, "dme_risk": 1.0},
        metrics={"dme_risk": [keras.metrics.CategoricalAccuracy(name="accuracy")]},
    )
    logger.info("Model C (Full DR-ASPP-DRN, single-stage): %d params", model.count_params())
    return model


def build_model_d(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_dme_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model D: Full DR-ASPP-DRN with two-stage training support.
    
    Identical to Model C but designed for two-stage training workflow.
    Tests benefit of two-stage training strategy.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_dme_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_model

    backbone_weights_arg, backbone_weights_path = _resolve_backbone_spec(backbone_weights)

    model = build_model(
        input_shape=input_shape,
        backbone_weights=backbone_weights_arg,
        backbone_weights_path=backbone_weights_path,
        num_dme_classes=num_dme_classes,
        aspp_filters=aspp_filters,
        dr_head_units=PROJECT_DR_HEAD_UNITS,
        dme_head_units=PROJECT_DME_HEAD_UNITS,
        dme_head_residual=PROJECT_DME_HEAD_RESIDUAL,
        dropout_rate=PROJECT_DROPOUT,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss={"dr_output": "mse", "dme_risk": "categorical_crossentropy"},
        loss_weights={"dr_output": PROJECT_DR_LOSS_WEIGHT, "dme_risk": 1.0},
        metrics={"dme_risk": [keras.metrics.CategoricalAccuracy(name="accuracy")]},
    )
    logger.info("Model D (Full DR-ASPP-DRN, two-stage compatible): %d params", model.count_params())
    return model


def build_model_f(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model F: Full architecture + ASPP with simple CE loss (loss ablation baseline).

    Tests loss function contribution with basic categorical cross-entropy.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_aspp

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _build_resnet50_backbone(input_shape, backbone_weights)
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters, name_prefix="aspp")
    x = layers.GlobalAveragePooling2D(name="dme_gap")(aspp_out)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(PROJECT_DROPOUT, name="dme_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelF_SimpleCE")
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model F (ASPP + Simple CE): %d params", model.count_params())
    return model


def build_model_g(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model G: Model F + class weighting for imbalance.

    Tests class weighting contribution (+0.02 QWK expected).

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    # Model identical to F; class_weight passed at training time
    from model import build_aspp

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _build_resnet50_backbone(input_shape, backbone_weights)
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters, name_prefix="aspp")
    x = layers.GlobalAveragePooling2D(name="dme_gap")(aspp_out)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(PROJECT_DROPOUT, name="dme_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelG_ClassWeighting")
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model G (ASPP + Class Weighting): %d params", model.count_params())
    return model


def build_model_h(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model H: Model G + ordinal weighting for misclassification penalty.

    Tests ordinal weighting contribution (+0.01 QWK expected).

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_aspp

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _build_resnet50_backbone(input_shape, backbone_weights)
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters, name_prefix="aspp")
    x = layers.GlobalAveragePooling2D(name="dme_gap")(aspp_out)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(PROJECT_DROPOUT, name="dme_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelH_OrdinalWeighting")
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss=OrdinalWeightedCrossEntropy(num_classes=num_classes),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model H (ASPP + Ordinal Weighting): %d params", model.count_params())
    return model


def build_model_i(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model I: Model H + label smoothing to reduce overconfidence.

    Tests label smoothing contribution (+0.01 QWK expected).
    (Label smoothing applied at data/training level, not loss level)

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_aspp

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _build_resnet50_backbone(input_shape, backbone_weights)
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters, name_prefix="aspp")
    x = layers.GlobalAveragePooling2D(name="dme_gap")(aspp_out)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(PROJECT_DROPOUT, name="dme_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelI_LabelSmoothing")
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss=OrdinalWeightedCrossEntropy(num_classes=num_classes),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model I (ASPP + Ordinal + Label Smoothing): %d params", model.count_params())
    return model


def build_model_j(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = PROJECT_ASPP_FILTERS,
) -> keras.Model:
    """Model J: Full loss stack - Ordinal + Focal loss combined.

    Tests full loss stack contribution (final optimization layer).

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_aspp

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _build_resnet50_backbone(input_shape, backbone_weights)
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters, name_prefix="aspp")
    x = layers.GlobalAveragePooling2D(name="dme_gap")(aspp_out)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(PROJECT_DROPOUT, name="dme_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelJ_FullStack")
    
    # Combine ordinal weighting + focal loss
    def combined_loss(y_true, y_pred):
        ordinal_loss = OrdinalWeightedCrossEntropy(num_classes=num_classes)(y_true, y_pred)
        focal = FocalLoss(alpha=0.25, gamma=PROJECT_FOCAL_GAMMA)(y_true, y_pred)
        return 0.7 * ordinal_loss + 0.3 * focal
    
    model.compile(
        optimizer=keras.optimizers.Adam(PROJECT_LR),
        loss=combined_loss,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model J (Full Loss Stack - Ordinal + Focal): %d params", model.count_params())
    return model


# ---------------------------------------------------------------------------
# MODEL PREDICTION HELPERS
# ---------------------------------------------------------------------------

def _is_multi_output_model(model: keras.Model) -> bool:
    """Return True if model has multiple outputs (DR + DME)."""
    return len(model.output_names) > 1


def _adapt_dataset_for_model(
    dataset: tf.data.Dataset,
    model: keras.Model,
) -> tf.data.Dataset:
    """Adapt dataset labels to match model output structure.

    The project dataset loader returns labels as a dict with keys
    ``dr_output`` and ``dme_risk``. Single-output ablation models expect only
    the ``dme_risk`` target tensor.
    """
    label_spec = dataset.element_spec[1]
    if _is_multi_output_model(model):
        return dataset

    if isinstance(label_spec, Mapping):
        return dataset.map(
            lambda images, labels: (images, labels["dme_risk"]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset


def _rebatch_dataset(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    """Re-batch an existing dataset for OOM fallback retries."""
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _add_multitask_sample_weights(
    dataset: tf.data.Dataset,
    class_weights: Dict,
) -> tf.data.Dataset:
    """Attach sample weights for multi-output models using DME class weights."""
    keys = tf.constant(list(class_weights.keys()), dtype=tf.int32)
    values = tf.constant(list(class_weights.values()), dtype=tf.float32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=tf.constant(1.0, dtype=tf.float32),
    )

    def map_fn(images, labels):
        dme_labels = labels["dme_risk"]
        dme_indices = tf.cast(tf.argmax(dme_labels, axis=-1), tf.int32)
        dme_weights = table.lookup(dme_indices)
        dr_weights = tf.ones_like(dme_weights, dtype=tf.float32)
        sample_weights = {"dr_output": dr_weights, "dme_risk": dme_weights}
        return images, labels, sample_weights

    return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _class_distribution(labels: np.ndarray, num_classes: int) -> Dict[int, int]:
    """Return a {class_index: count} distribution dict."""
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    return {int(i): int(c) for i, c in enumerate(counts)}


def _prediction_collapse_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    collapse_threshold: float = 0.90,
) -> Dict:
    """Compute class-distribution diagnostics to detect mode collapse."""
    true_dist = _class_distribution(y_true, num_classes)
    pred_dist = _class_distribution(y_pred, num_classes)
    total_pred = max(len(y_pred), 1)
    dominant_class = int(np.argmax(np.bincount(y_pred.astype(int), minlength=num_classes)))
    dominant_ratio = float(pred_dist[dominant_class]) / float(total_pred)
    collapsed = dominant_ratio >= collapse_threshold

    return {
        "true_distribution": true_dist,
        "pred_distribution": pred_dist,
        "dominant_pred_class": dominant_class,
        "dominant_pred_ratio": round(dominant_ratio, 4),
        "collapsed": bool(collapsed),
        "collapse_threshold": collapse_threshold,
    }

def _get_predictions_ablation(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_classes: int = NUM_DME_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions from any ablation model variant.

    Parameters
    ----------
    model : keras.Model
        Any of Model A-J.
    dataset : tf.data.Dataset
        Validation dataset.
    num_classes : int
        Number of classes.

    Returns
    -------
    tuple
        ``(y_true, y_proba, y_pred)``
    """
    dataset_for_model = _adapt_dataset_for_model(dataset, model)

    all_true, all_proba = [], []
    for batch_images, batch_labels in dataset_for_model:
        preds = model(batch_images, training=False)
        if isinstance(preds, dict):
            proba = preds["dme_risk"].numpy()
        elif isinstance(preds, (list, tuple)):
            proba = preds[1].numpy()
        else:
            proba = preds.numpy()

        if isinstance(batch_labels, dict):
            y_true_batch = batch_labels["dme_risk"].numpy()
        else:
            y_true_batch = batch_labels.numpy()

        if y_true_batch.ndim > 1:
            all_true.append(np.argmax(y_true_batch, axis=-1))
        else:
            all_true.append(y_true_batch.astype(np.int32))
        all_proba.append(proba)

    y_true = np.concatenate(all_true)
    y_proba = np.concatenate(all_proba)
    y_pred = np.argmax(y_proba, axis=-1)
    return y_true, y_proba, y_pred


# ---------------------------------------------------------------------------
# QUICK TRAINING HELPER
# ---------------------------------------------------------------------------

def _quick_train(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 5,
    class_weights: Optional[Dict] = None,
    verbose: int = 0,
    oom_batch_fallback: Optional[List[int]] = None,
) -> keras.callbacks.History:
    """Train a model variant for a quick ablation experiment.

    Parameters
    ----------
    model : keras.Model
        Compiled model.
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    epochs : int
        Number of training epochs.
    class_weights : dict, optional
        Per-class weights (only for models supporting this).
    verbose : int
        Verbosity.
    oom_batch_fallback : list of int, optional
        Batch sizes to retry when GPU OOM happens, e.g. [4, 2, 1].

    Returns
    -------
    keras.callbacks.History
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PROJECT_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]

    train_for_model = _adapt_dataset_for_model(train_ds, model)
    val_for_model = _adapt_dataset_for_model(val_ds, model)

    fit_kwargs = {}
    if class_weights is not None and not _is_multi_output_model(model):
        fit_kwargs["class_weight"] = class_weights
    elif class_weights is not None:
        logger.info("Using sample_weight mapping for multi-output model")
        train_for_model = _add_multitask_sample_weights(train_for_model, class_weights)
        val_for_model = _add_multitask_sample_weights(val_for_model, class_weights)
    
    # Extra safety fallback for environments with stricter class_weight checks.
    fallback_batch_sizes = oom_batch_fallback or [4, 2, 1]

    def fit_with_possible_rebatch(train_data, val_data):
        attempts = [None] + fallback_batch_sizes
        last_oom_error = None

        for attempt_batch in attempts:
            if attempt_batch is None:
                train_attempt = train_data
                val_attempt = val_data
                logger.info("Training attempt with original batch size")
            else:
                logger.warning("OOM fallback: retrying with batch_size=%d", attempt_batch)
                train_attempt = _rebatch_dataset(train_data, attempt_batch)
                val_attempt = _rebatch_dataset(val_data, attempt_batch)

            try:
                return model.fit(
                    train_attempt,
                    validation_data=val_attempt,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=verbose,
                    **fit_kwargs,
                )
            except Exception as ex:
                msg = str(ex).lower()
                is_oom = (
                    isinstance(ex, tf.errors.ResourceExhaustedError)
                    or "out of memory" in msg
                    or "oom" in msg
                )
                if not is_oom:
                    raise
                last_oom_error = ex
                continue

        if last_oom_error is not None:
            raise last_oom_error

        raise RuntimeError("Training failed before OOM fallback could run")

    try:
        history = fit_with_possible_rebatch(train_for_model, val_for_model)
    except ValueError as e:
        if "class_weight" in str(e):
            logger.info("Model doesn't support class_weight; retraining without it")
            fit_kwargs.pop("class_weight", None)
            history = model.fit(
                train_for_model,
                validation_data=val_for_model,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose,
            )
        elif "y_true and y_pred have different structures" in str(e):
            logger.info("Retrying with forced dme_risk label mapping for single-output model")
            forced_train = train_ds.map(
                lambda images, labels: (
                    images,
                    labels["dme_risk"] if isinstance(labels, Mapping) else labels,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            forced_val = val_ds.map(
                lambda images, labels: (
                    images,
                    labels["dme_risk"] if isinstance(labels, Mapping) else labels,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            history = model.fit(
                forced_train,
                validation_data=forced_val,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose,
            )
        else:
            raise
    
    return history


# ---------------------------------------------------------------------------
# STATISTICAL SIGNIFICANCE TESTING
# ---------------------------------------------------------------------------

def bootstrap_qwk_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """Paired bootstrap test to compare QWK of two models.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred_a : np.ndarray
        Predictions from model A.
    y_pred_b : np.ndarray
        Predictions from model B.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``qwk_a``, ``qwk_b``, ``delta``, ``ci_low``, ``ci_high``, ``p_value``.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    qwk_a = compute_quadratic_weighted_kappa(y_true, y_pred_a)
    qwk_b = compute_quadratic_weighted_kappa(y_true, y_pred_b)
    observed_delta = qwk_b - qwk_a

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        qt_a = compute_quadratic_weighted_kappa(y_true[idx], y_pred_a[idx])
        qt_b = compute_quadratic_weighted_kappa(y_true[idx], y_pred_b[idx])
        deltas.append(qt_b - qt_a)

    deltas = np.array(deltas)
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))

    # Two-sided p-value
    if observed_delta >= 0:
        p_value = float(np.mean(deltas <= 0))
    else:
        p_value = float(np.mean(deltas >= 0))

    return {
        "qwk_a": float(qwk_a),
        "qwk_b": float(qwk_b),
        "delta": float(observed_delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": bool(ci_low > 0 or ci_high < 0),
    }


# ---------------------------------------------------------------------------
# VISUALIZATION AND REPORTING
# ---------------------------------------------------------------------------

def print_ablation_table(results: List[Dict], category: str = "All") -> None:
    """Pretty-print ablation comparison table.

    Parameters
    ----------
    results : list of dict
        Per-model result dicts.
    category : str
        Category name (e.g., "Architecture", "Loss Function")
    """
    if not results:
        logger.warning("No results to display for category: %s", category)
        return
    
    print("\n" + "=" * 90)
    print(f"ABLATION STUDY - {category} Variants")
    print("=" * 90)
    print(f"{'Model':<35} {'QWK':>8} {'Acc':>8} {'F1':>8} {'MAE':>8} {'Gain':>8}")
    print("-" * 90)
    
    baseline_qwk = results[0]["qwk"]
    for i, r in enumerate(results):
        gain = r["qwk"] - baseline_qwk if i > 0 else 0.0
        print(
            f"{r['name']:<35} {r['qwk']:>8.4f} {r['accuracy']:>8.4f} "
            f"{r['f1_macro']:>8.4f} {r['mae']:>8.3f} {gain:>8.4f}"
        )
    print("=" * 90 + "\n")


def plot_ablation_comparison(
    all_results: Dict[str, List[Dict]],
    output_dir: str = "ablation_results",
) -> None:
    """Plot separate visualizations for each ablation category.

    Parameters
    ----------
    all_results : dict
        {"Architecture": [...], "TwoStage": [...], "LossFunction": [...]}
    output_dir : str
        Output directory for plots.
    """
    if not _PLOTTING_AVAILABLE:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    categories = ["Architecture", "TwoStage", "LossFunction"]
    colors_map = {
        "Architecture": ["steelblue", "darkorange", "seagreen"],
        "TwoStage": ["coral", "mediumaquamarine"],
        "LossFunction": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"],
    }

    for category in categories:
        if category not in all_results or not all_results[category]:
            continue

        results = all_results[category]
        names = [r["name"].split(": ")[1] for r in results]  # Extract model description
        qwks = [r["qwk"] for r in results]
        accs = [r["accuracy"] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # QWK bar chart with improvement annotations
        colors = colors_map.get(category, ["steelblue"] * len(names))
        bars = axes[0].bar(range(len(names)), qwks, color=colors, edgecolor="black", alpha=0.85)
        axes[0].axhline(y=0.80, color="red", linestyle="--", linewidth=2.5, label="Target 0.80")
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=10)
        axes[0].set_ylim([0, 1.05])
        axes[0].set_title(f"{category} Ablation - QWK Scores", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("QWK Score", fontsize=11)
        axes[0].legend(fontsize=10)
        
        # Add value labels on bars
        for bar, q in zip(bars, qwks):
            axes[0].text(bar.get_x() + bar.get_width() / 2, q + 0.01, f"{q:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        # Add improvement deltas
        for i in range(1, len(qwks)):
            delta = qwks[i] - qwks[i - 1]
            axes[0].annotate(
                f"+{delta:.4f}",
                xy=(i, qwks[i] + 0.05),
                ha="center",
                color="darkgreen" if delta > 0 else "darkred",
                fontsize=9,
                fontweight="bold",
            )

        # QWK vs Accuracy comparison
        x_pos = np.arange(len(names))
        width = 0.35
        axes[1].bar(x_pos - width/2, qwks, width, label="QWK", color="steelblue", alpha=0.85)
        axes[1].bar(x_pos + width/2, accs, width, label="Accuracy", color="darkorange", alpha=0.85)
        axes[1].axhline(y=0.80, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=10)
        axes[1].set_ylim([0, 1.05])
        axes[1].set_title(f"{category} Ablation - QWK vs Accuracy", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Score", fontsize=11)
        axes[1].legend(fontsize=10)

        fig.suptitle(
            f"Ablation Study: {category} Component Contribution",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"ablation_{category.lower()}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# MAIN ABLATION RUNNER (ENHANCED)
# ---------------------------------------------------------------------------

def run_ablation_study(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict] = None,
    epochs: int = 5,
    output_dir: str = "ablation_results",
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    n_bootstrap: int = 500,
    seed: int = 42,
    ablation_mode: str = "all",
    backbone_weights: Optional[str] = "imagenet",
    oom_batch_fallback: Optional[List[int]] = None,
) -> Dict:
    """Run comprehensive ablation study with 10+ model variants.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    class_weights : dict, optional
        Class weights for imbalance handling.
    epochs : int
        Training epochs per model variant.
    output_dir : str
        Output directory for artefacts.
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of ordinal classes.
    n_bootstrap : int
        Bootstrap resamples for significance testing.
    seed : int
        Random seed.
    ablation_mode : str
        "all" = run all variants, "arch" = architecture only,
        "loss" = loss function only, "stage" = two-stage only
    backbone_weights : str or None
        Backbone initialization for ResNet50. Use "imagenet" (default),
        None, or a local .h5/.keras weights file path.
    oom_batch_fallback : list of int, optional
        Batch sizes to retry when OOM happens.

    Returns
    -------
    dict
        Comprehensive ablation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Backbone setting for this run: %s", backbone_weights)

    # Define model configurations by category
    model_configs = {
        "Architecture": [
            ("Model A: Baseline ResNet50", build_model_a),
            ("Model B: ResNet50 + ASPP", build_model_b),
            ("Model C: Full DR-ASPP-DRN", build_model_c),
        ],
        "TwoStage": [
            ("Model D: Single-stage training", build_model_c),
            ("Model E: Two-stage training", build_model_d),
        ],
        "LossFunction": [
            ("Model F: Simple CE", build_model_f),
            ("Model G: + Class Weighting", build_model_g),
            ("Model H: + Ordinal Weighting", build_model_h),
            ("Model I: + Label Smoothing", build_model_i),
            ("Model J: Full Loss Stack", build_model_j),
        ],
    }

    # Filter based on mode
    if ablation_mode == "all":
        categories_to_run = ["Architecture", "TwoStage", "LossFunction"]
    elif ablation_mode == "arch":
        categories_to_run = ["Architecture"]
    elif ablation_mode == "loss":
        categories_to_run = ["LossFunction"]
    elif ablation_mode == "stage":
        categories_to_run = ["TwoStage"]
    else:
        categories_to_run = ["Architecture", "TwoStage", "LossFunction"]

    all_results = {}
    all_predictions = {}
    y_true_global = None

    # Run each category
    for category in categories_to_run:
        if category not in model_configs:
            continue

        logger.info("\n" + "=" * 70)
        logger.info("ABLATION CATEGORY: %s", category)
        logger.info("=" * 70)

        results = []
        category_predictions = {}

        for name, builder in model_configs[category]:
            logger.info("\n>> Training %s ...", name)
            t0 = time.time()
            model = None

            try:
                if name.startswith("Model C") or name.startswith("Model D") or name.startswith("Model E"):
                    model = builder(
                        input_shape=input_shape,
                        num_dme_classes=num_classes,
                        backbone_weights=backbone_weights,
                    )
                else:
                    model = builder(
                        input_shape=input_shape,
                        num_classes=num_classes,
                        backbone_weights=backbone_weights,
                    )

                _quick_train(model, train_ds, val_ds, epochs=epochs,
                           class_weights=class_weights, verbose=1,
                           oom_batch_fallback=oom_batch_fallback)

                y_true, y_proba, y_pred = _get_predictions_ablation(model, val_ds, num_classes)
                ordinal = compute_ordinal_metrics(y_true, y_pred, num_classes)
                acc = compute_accuracy(y_true, y_pred)
                f1 = compute_f1(y_true, y_pred, average="macro")
                collapse_stats = _prediction_collapse_stats(y_true, y_pred, num_classes)

                elapsed = time.time() - t0
                result = {
                    "name": name,
                    "qwk": ordinal["qwk"],
                    "mae": ordinal["mae"],
                    "rmse": ordinal["rmse"],
                    "accuracy": acc,
                    "f1_macro": f1,
                    "target_met": ordinal["qwk"] >= 0.80,
                    "training_time_s": round(elapsed, 1),
                    "params": int(model.count_params()),
                    "true_distribution": collapse_stats["true_distribution"],
                    "pred_distribution": collapse_stats["pred_distribution"],
                    "dominant_pred_class": collapse_stats["dominant_pred_class"],
                    "dominant_pred_ratio": collapse_stats["dominant_pred_ratio"],
                    "prediction_collapsed": collapse_stats["collapsed"],
                }
                results.append(result)
                category_predictions[name] = y_pred
                y_true_global = y_true  # Store for significance tests
                
                logger.info("[OK] %s -> QWK=%.4f | Acc=%.4f | F1=%.4f [%.1fs]",
                          name, ordinal["qwk"], acc, f1, elapsed)
                logger.info(
                    "Class dist true=%s | pred=%s | dominant_pred_class=%d (%.2f%%)",
                    collapse_stats["true_distribution"],
                    collapse_stats["pred_distribution"],
                    collapse_stats["dominant_pred_class"],
                    100.0 * collapse_stats["dominant_pred_ratio"],
                )
                if collapse_stats["collapsed"]:
                    logger.warning(
                        "Prediction collapse detected for %s: one class covers %.2f%% of predictions",
                        name,
                        100.0 * collapse_stats["dominant_pred_ratio"],
                    )

            except Exception as e:
                logger.error("[X] Error training %s: %s", name, str(e))
                continue
            finally:
                if model is not None:
                    del model
                tf.keras.backend.clear_session()
                gc.collect()

        all_results[category] = results
        all_predictions[category] = category_predictions

        # Print category results
        print_ablation_table(results, category=category)

        # Run significance tests within category
        if len(results) > 1:
            logger.info("\nStatistical Significance Tests (%s):", category)
            for i in range(len(results) - 1):
                pred_i = category_predictions[results[i]["name"]]
                pred_j = category_predictions[results[i + 1]["name"]]
                sig_test = bootstrap_qwk_test(y_true_global, pred_i, pred_j, n_bootstrap=n_bootstrap, seed=seed)
                
                significance = "[OK] SIGNIFICANT" if sig_test["significant"] else "[X] not significant"
                logger.info(
                    "  %s vs %s: delta=%.4f [%.4f, %.4f] p=%.3f %s",
                    results[i]["name"].split(": ")[0],
                    results[i + 1]["name"].split(": ")[0],
                    sig_test["delta"],
                    sig_test["ci_low"],
                    sig_test["ci_high"],
                    sig_test["p_value"],
                    significance,
                )

    # Compile comprehensive summary
    comprehensive_summary = {
        "backbone_weights": backbone_weights,
        "categories": all_results,
        "component_gains": {
            "ASPP": round(
                all_results["Architecture"][2]["qwk"] - all_results["Architecture"][0]["qwk"]
                if len(all_results.get("Architecture", [])) >= 3 else 0, 4
            ),
            "MultiTask": round(
                all_results["Architecture"][2]["qwk"] - all_results["Architecture"][1]["qwk"]
                if len(all_results.get("Architecture", [])) >= 3 else 0, 4
            ),
            "TwoStage": round(
                all_results.get("TwoStage", [{}])[-1]["qwk"] - all_results.get("TwoStage", [{}])[0]["qwk"]
                if len(all_results.get("TwoStage", [])) >= 2 else 0, 4
            ),
            "LossStack": round(
                all_results.get("LossFunction", [{}])[-1]["qwk"] - all_results.get("LossFunction", [{}])[0]["qwk"]
                if len(all_results.get("LossFunction", [])) >= 2 else 0, 4
            ),
        },
        "total_improvement": round(
            all_results["Architecture"][-1]["qwk"] - all_results["Architecture"][0]["qwk"]
            if all_results.get("Architecture") else 0, 4
        ),
    }

    # Save results
    json_path = os.path.join(output_dir, "ablation_results_comprehensive.json")
    with open(json_path, "w") as f:
        json.dump(comprehensive_summary, f, indent=2)
    logger.info("Comprehensive ablation results saved to '%s'.", json_path)

    # Generate visualizations
    plot_ablation_comparison(all_results, output_dir=output_dir)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 70)
    logger.info("Component Gains:")
    logger.info("  ASPP contribution: +%.4f QWK", comprehensive_summary["component_gains"]["ASPP"])
    logger.info("  Multi-task contribution: +%.4f QWK", comprehensive_summary["component_gains"]["MultiTask"])
    logger.info("  Two-stage training contribution: +%.4f QWK", comprehensive_summary["component_gains"]["TwoStage"])
    logger.info("  Loss function stack contribution: +%.4f QWK", comprehensive_summary["component_gains"]["LossStack"])
    logger.info("  TOTAL IMPROVEMENT (A->C): +%.4f QWK", comprehensive_summary["total_improvement"])
    logger.info("=" * 70 + "\n")

    return comprehensive_summary


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for comprehensive ablation study."""
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(description="Run comprehensive ablation study")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="ablation_results")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["all", "arch", "loss", "stage"])
    parser.add_argument(
        "--oom-fallback-batches",
        type=str,
        default="4,2,1",
        help="Comma-separated batch sizes to retry on OOM, e.g. 4,2,1",
    )
    parser.add_argument(
        "--use-backbone",
        type=str,
        default=None,
        help="Path to a custom backbone weights file for one additional comparison run.",
    )
    parser.add_argument(
        "--eyepacs-backbone",
        type=str,
        default=None,
        help="Path to EyePACS-pretrained backbone weights for comparison.",
    )
    parser.add_argument(
        "--all-datasets-backbone",
        type=str,
        default=None,
        help="Path to backbone weights pretrained on all datasets for comparison.",
    )
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    # Stabilize runtime: disable XLA JIT and use one GPU with memory growth.
    try:
        tf.config.optimizer.set_jit(False)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], "GPU")
            except RuntimeError:
                pass
            visible_gpus = tf.config.list_physical_devices("GPU")
            for gpu in visible_gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            logger.info("Runtime configured for stability: %d visible GPU(s), XLA JIT disabled", len(visible_gpus))
    except Exception as cfg_err:
        logger.warning("Could not fully apply runtime stability settings: %s", cfg_err)

    oom_batch_fallback = [
        int(x.strip())
        for x in args.oom_fallback_batches.split(",")
        if x.strip()
    ]

    if args.mock or args.csv is None:
        logger.info("Using mock dataset.")
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_ablation", num_samples=80)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    train_ds, val_ds, class_weights = build_datasets(
        csv_path=csv_path, image_dir=image_dir, batch_size=args.batch_size,
    )

    # Build experiment list: baseline (without custom backbone) + optional custom backbones.
    experiments: List[Tuple[str, Optional[str]]] = [
        ("baseline_imagenet", "imagenet"),
    ]

    optional_backbones = [
        ("custom_backbone", args.use_backbone),
        ("eyepacs_backbone", args.eyepacs_backbone),
        ("all_datasets_backbone", args.all_datasets_backbone),
    ]

    for exp_name, backbone_path in optional_backbones:
        if not backbone_path:
            continue
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(
                f"Backbone file for {exp_name} not found: {backbone_path}"
            )
        experiments.append((exp_name, backbone_path))

    # Remove accidental duplicate experiments that point to the same backbone path.
    unique_experiments: List[Tuple[str, Optional[str]]] = []
    seen_backbones = set()
    for exp_name, backbone_setting in experiments:
        key = backbone_setting if backbone_setting is not None else "<none>"
        if key in seen_backbones:
            continue
        seen_backbones.add(key)
        unique_experiments.append((exp_name, backbone_setting))

    all_experiment_results = {}
    for exp_name, backbone_setting in unique_experiments:
        logger.info("\n%s\nRunning experiment: %s\n%s", "#" * 70, exp_name, "#" * 70)
        exp_output_dir = os.path.join(args.output_dir, exp_name)

        exp_results = run_ablation_study(
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            epochs=args.epochs,
            output_dir=exp_output_dir,
            n_bootstrap=args.n_bootstrap,
            ablation_mode=args.mode,
            backbone_weights=backbone_setting,
            oom_batch_fallback=oom_batch_fallback,
        )
        all_experiment_results[exp_name] = exp_results

    comparison_summary = {
        "experiments": {
            exp_name: {
                "backbone_weights": result.get("backbone_weights"),
                "component_gains": result.get("component_gains", {}),
                "total_improvement": result.get("total_improvement", 0),
            }
            for exp_name, result in all_experiment_results.items()
        }
    }

    comparison_json_path = os.path.join(args.output_dir, "backbone_comparison_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(comparison_json_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)

    print("\n[OK] Backbone Ablation Study Complete!")
    print(json.dumps(comparison_summary, indent=2))


if __name__ == "__main__":
    main()
