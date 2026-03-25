"""
model.py - Model architecture components for the multi-task DR + DME system.

Provides:
- Backbone building (ResNet50)
- ASPP (Atrous Spatial Pyramid Pooling) module
- Multi-task model (DR + DME heads)
- DME fine-tuning model (only DME head trainable)
"""

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def build_backbone(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    weights: str = "imagenet",
    trainable: bool = True,
) -> keras.Model:
    """Build a ResNet50 backbone.

    Parameters
    ----------
    input_shape : tuple
        ``(H, W, C)`` input dimensions.
    weights : str
        Pre-trained weights source.  Use ``"imagenet"`` or ``None``.
    trainable : bool
        Whether backbone weights should be updated during training.

    Returns
    -------
    keras.Model
        ResNet50 model (without top classification layer).
    """
    base = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base.trainable = trainable
    logger.info(
        "Backbone: ResNet50, trainable=%s, output shape=%s",
        trainable,
        base.output_shape,
    )
    return base


# ---------------------------------------------------------------------------
# ASPP module
# ---------------------------------------------------------------------------

def build_aspp(
    x: tf.Tensor,
    filters: int = 256,
    name_prefix: str = "aspp",
) -> tf.Tensor:
    """Atrous Spatial Pyramid Pooling (ASPP) block.

    Captures multi-scale contextual information using dilated convolutions at
    several dilation rates, plus a global average pooling branch.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map tensor.
    filters : int
        Number of output filters for each branch.
    name_prefix : str
        Prefix applied to all layer names within this block.

    Returns
    -------
    tf.Tensor
        Concatenated multi-scale features.
    """
    # 1x1 conv
    b0 = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                       name=f"{name_prefix}_b0")(x)
    b0 = layers.BatchNormalization(name=f"{name_prefix}_b0_bn")(b0)
    b0 = layers.Activation("relu", name=f"{name_prefix}_b0_relu")(b0)

    # 3x3 dilation 6
    b1 = layers.Conv2D(filters, 3, padding="same", dilation_rate=6, use_bias=False,
                       name=f"{name_prefix}_b1")(x)
    b1 = layers.BatchNormalization(name=f"{name_prefix}_b1_bn")(b1)
    b1 = layers.Activation("relu", name=f"{name_prefix}_b1_relu")(b1)

    # 3x3 dilation 12
    b2 = layers.Conv2D(filters, 3, padding="same", dilation_rate=12, use_bias=False,
                       name=f"{name_prefix}_b2")(x)
    b2 = layers.BatchNormalization(name=f"{name_prefix}_b2_bn")(b2)
    b2 = layers.Activation("relu", name=f"{name_prefix}_b2_relu")(b2)

    # 3x3 dilation 18
    b3 = layers.Conv2D(filters, 3, padding="same", dilation_rate=18, use_bias=False,
                       name=f"{name_prefix}_b3")(x)
    b3 = layers.BatchNormalization(name=f"{name_prefix}_b3_bn")(b3)
    b3 = layers.Activation("relu", name=f"{name_prefix}_b3_relu")(b3)

    # Global average pooling branch
    gap_shape = tf.keras.backend.int_shape(x)[1:3]
    b4 = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    b4 = layers.Reshape((1, 1, tf.keras.backend.int_shape(x)[-1]),
                        name=f"{name_prefix}_gap_reshape")(b4)
    b4 = layers.Conv2D(filters, 1, use_bias=False, name=f"{name_prefix}_b4_conv")(b4)
    b4 = layers.BatchNormalization(name=f"{name_prefix}_b4_bn")(b4)
    b4 = layers.Activation("relu", name=f"{name_prefix}_b4_relu")(b4)
    b4 = layers.UpSampling2D(
        size=gap_shape if gap_shape[0] is not None else (16, 16),
        interpolation="bilinear",
        name=f"{name_prefix}_b4_upsample",
    )(b4)

    out = layers.Concatenate(name=f"{name_prefix}_concat")([b0, b1, b2, b3, b4])
    out = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                        name=f"{name_prefix}_proj")(out)
    out = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(out)
    out = layers.Activation("relu", name=f"{name_prefix}_proj_relu")(out)
    return out


# ---------------------------------------------------------------------------
# DR head
# ---------------------------------------------------------------------------

def build_dr_head(x: tf.Tensor) -> tf.Tensor:
    """Build the Diabetic Retinopathy (DR) regression head.

    Outputs a single scalar value representing DR severity (0–4).

    Parameters
    ----------
    x : tf.Tensor
        Input feature map from ASPP.

    Returns
    -------
    tf.Tensor
        DR severity output tensor.
    """
    x = layers.GlobalAveragePooling2D(name="dr_gap")(x)
    x = layers.Dense(256, activation="relu", name="dr_fc1")(x)
    x = layers.Dropout(0.4, name="dr_dropout")(x)
    x = layers.Dense(1, activation="sigmoid", name="dr_output")(x)
    return x


# ---------------------------------------------------------------------------
# DME head
# ---------------------------------------------------------------------------

def build_dme_head(x: tf.Tensor, num_classes: int = 4) -> tf.Tensor:
    """Build the DME (Diabetic Macular Edema) classification head.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map from ASPP.
    num_classes : int
        Number of DME severity classes (default: 4 – No/Mild/Moderate/Severe).

    Returns
    -------
    tf.Tensor
        Softmax probability distribution over DME classes.
    """
    x = layers.GlobalAveragePooling2D(name="dme_gap")(x)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(0.4, name="dme_dropout")(x)
    x = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)
    return x


# ---------------------------------------------------------------------------
# Full multi-task model
# ---------------------------------------------------------------------------

def build_model(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    backbone_weights: str = "imagenet",
    num_dme_classes: int = 4,
    aspp_filters: int = 256,
) -> keras.Model:
    """Build the complete multi-task DR + DME model.

    Architecture:
    ``Input → Backbone (ResNet50) → ASPP → DR head + DME head``

    Parameters
    ----------
    input_shape : tuple
        Model input shape ``(H, W, C)``.
    backbone_weights : str
        Initial backbone weights (``"imagenet"`` or ``None``).
    num_dme_classes : int
        Number of output classes for the DME head.
    aspp_filters : int
        Number of filters in each ASPP branch.

    Returns
    -------
    keras.Model
        Compiled Keras functional model with two outputs:
        ``dr_output`` and ``dme_risk``.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    backbone = build_backbone(input_shape, weights=backbone_weights, trainable=True)
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters)

    dr_out = build_dr_head(aspp_out)
    dme_out = build_dme_head(aspp_out, num_classes=num_dme_classes)

    model = keras.Model(
        inputs=inputs,
        outputs={"dr_output": dr_out, "dme_risk": dme_out},
        name="multitask_dr_dme",
    )
    logger.info("Built multi-task model. Parameters: %d", model.count_params())
    return model


# ---------------------------------------------------------------------------
# DME fine-tuning model (backbone / ASPP / DR head all frozen)
# ---------------------------------------------------------------------------

def build_model_dme_tuning(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    pretrained_weights: Optional[str] = None,
    num_dme_classes: int = 4,
    aspp_filters: int = 256,
) -> keras.Model:
    """Build the DME fine-tuning model with only the DME head trainable."""
    model = build_model(
        input_shape=input_shape,
        backbone_weights="imagenet",
        num_dme_classes=num_dme_classes,
        aspp_filters=aspp_filters,
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights, skip_mismatch=True)
        logger.info("Loaded pretrained weights from '%s'.", pretrained_weights)

    # Freeze everything first
    model.trainable = False

    # Unfreeze only DME head layers
    dme_layer_names = {"dme_gap", "dme_fc1", "dme_dropout", "dme_risk"}
    for layer in model.layers:
        if layer.name in dme_layer_names:
            layer.trainable = True

    trainable_layers = [l.name for l in model.layers if l.trainable]
    logger.info("Trainable layers: %s", trainable_layers)
    
    # CRITICAL: Force Keras to recognize the trainable layers
    # by triggering a model.build() with a dummy batch
    try:
        dummy_input = tf.zeros((1, *input_shape))
        _ = model(dummy_input, training=False)
        logger.info("Model built successfully. Trainable weights: %d", len(model.trainable_weights))
    except Exception as e:
        logger.warning("Could not build model with dummy input: %s", e)

    return model


def print_model_summary(model: keras.Model) -> None:
    """Print a concise summary of trainable vs frozen layers."""
    print(f"\n{'='*60}")
    print(f"Model: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"{'='*60}")
    print(f"{'Layer':<30} {'Trainable':<10} {'Output Shape'}")
    print("-" * 60)
    for layer in model.layers:
        shape = str(getattr(layer, "output_shape", "?"))
        print(f"{layer.name:<30} {str(layer.trainable):<10} {shape}")
    print("=" * 60)
