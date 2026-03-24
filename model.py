"""
DR-ASPP-DRN Model Architecture

Components
----------
- ``build_backbone``         : ResNet50-based DRN backbone
- ``ASPP``                   : Atrous Spatial Pyramid Pooling module
- ``dme_head``               : 3-class DME classification head
- ``build_model``            : Full multi-task model (DR regression + DME)
- ``build_model_dme_tuning`` : Fine-tuning variant – backbone+ASPP frozen,
                               only DME head is trainable
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def build_backbone(pretrained_weights: str = None) -> Model:
    """Build a DRN backbone based on ResNet50 with dilated convolutions.

    The last ResNet50 striding is removed so that the spatial resolution
    is maintained at H/16 × W/16 instead of H/32 × W/32, giving richer
    spatial features for the ASPP module.

    Parameters
    ----------
    pretrained_weights:
        Path to a ``.weights.h5`` file saved by ``model.save_weights()``.
        When *None* the backbone is initialised with ImageNet weights.

    Returns
    -------
    tf.keras.Model
        Model with output shape ``(batch, H/16, W/16, 1024)`` for the
        ``conv4_block6_out`` layer.
    """
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, 3),
    )

    # Use Block-3 output (conv4_block6_out) → 1024 channels, H/16 × W/16
    output = base.get_layer("conv4_block6_out").output
    backbone = Model(inputs=base.input, outputs=output, name="backbone")

    if pretrained_weights:
        try:
            backbone.load_weights(pretrained_weights, skip_mismatch=True)
            logger.info("Loaded backbone weights from %s", pretrained_weights)
        except Exception as exc:
            logger.warning("Could not load backbone weights: %s", exc)

    return backbone


# ---------------------------------------------------------------------------
# ASPP
# ---------------------------------------------------------------------------

def ASPP(
    inputs: tf.Tensor,
    out_channels: int = 256,
    rates: tuple = (6, 12, 18),
) -> tf.Tensor:
    """Atrous Spatial Pyramid Pooling (ASPP).

    Captures multi-scale context by combining:
    - 1×1 convolution
    - 3×3 dilated convolutions at *rates*
    - Global average pooling branch

    Parameters
    ----------
    inputs:
        Feature map tensor of shape ``(B, H, W, C)``.
    out_channels:
        Number of output channels for every branch and the final merge.
    rates:
        Dilation rates for the 3×3 atrous convolutions.

    Returns
    -------
    tf.Tensor
        Fused feature map of shape ``(B, H, W, out_channels)``.
    """
    branches = []

    # 1×1 branch
    x1 = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                       name="aspp_1x1")(inputs)
    x1 = layers.BatchNormalization(name="aspp_bn_1x1")(x1)
    x1 = layers.Activation("relu", name="aspp_relu_1x1")(x1)
    branches.append(x1)

    # Dilated 3×3 branches
    for i, r in enumerate(rates):
        xr = layers.Conv2D(
            out_channels, 3,
            dilation_rate=r,
            padding="same",
            use_bias=False,
            name=f"aspp_d{r}",
        )(inputs)
        xr = layers.BatchNormalization(name=f"aspp_bn_d{r}")(xr)
        xr = layers.Activation("relu", name=f"aspp_relu_d{r}")(xr)
        branches.append(xr)

    # Global average pooling branch
    x5 = layers.GlobalAveragePooling2D(name="aspp_gap")(inputs)
    x5 = layers.Dense(out_channels, use_bias=False, name="aspp_dense")(x5)
    x5 = layers.BatchNormalization(name="aspp_bn_gap")(x5)
    x5 = layers.Activation("relu", name="aspp_relu_gap")(x5)
    x5 = layers.Reshape((1, 1, out_channels), name="aspp_reshape")(x5)
    # Resize to spatial size of inputs (Keras-compatible)
    x5 = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear"),
        name="aspp_upsample",
    )([x5, inputs])
    branches.append(x5)

    # Fuse all branches
    x = layers.Concatenate(name="aspp_concat")(branches)
    x = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                      name="aspp_merge")(x)
    x = layers.BatchNormalization(name="aspp_bn_merge")(x)
    x = layers.Activation("relu", name="aspp_relu_merge")(x)
    return x


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

def dr_head(features: tf.Tensor) -> tf.Tensor:
    """DR regression head (predicts severity 0-4)."""
    x = layers.GlobalAveragePooling2D(name="dr_gap")(features)
    x = layers.Dense(256, activation="relu", name="dr_fc1")(x)
    x = layers.Dropout(0.4, name="dr_dropout")(x)
    return layers.Dense(1, activation="relu", name="dr_output")(x)


def dme_head(features: tf.Tensor) -> tf.Tensor:
    """DME classification head (3-class softmax)."""
    x = layers.GlobalAveragePooling2D(name="dme_gap")(features)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(0.4, name="dme_dropout")(x)
    return layers.Dense(3, activation="softmax", name="dme_risk")(x)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(backbone_weights_path: str = None) -> Model:
    """Build the full multi-task DR + DME model.

    Parameters
    ----------
    backbone_weights_path:
        Optional path to pre-trained backbone weights.

    Returns
    -------
    tf.keras.Model
        Model with two outputs: ``dr_output`` (scalar) and ``dme_risk``
        (3-class probability vector).
    """
    inp = layers.Input(shape=(None, None, 3), name="input_image")

    backbone = build_backbone(backbone_weights_path)
    features = backbone(inp)

    aspp_out = ASPP(features)

    dr_out = dr_head(aspp_out)
    dme_out = dme_head(aspp_out)

    model = Model(inputs=inp, outputs=[dr_out, dme_out], name="dr_aspp_drn")
    return model


def build_model_dme_tuning(backbone_weights_path: str = None) -> Model:
    """Build the DME fine-tuning variant.

    Strategy
    --------
    - Backbone: **frozen** (pre-trained features preserved)
    - ASPP:     **frozen**
    - DR head:  **frozen** (or absent – dummy output with weight 0)
    - DME head: **trainable**

    Parameters
    ----------
    backbone_weights_path:
        Path to ``.weights.h5`` produced by the pre-training stage.
        Weights are loaded with ``skip_mismatch=True`` so that the new
        DME head layers are safely ignored.

    Returns
    -------
    tf.keras.Model
        Model ready to be compiled and trained for DME fine-tuning.
    """
    inp = layers.Input(shape=(512, 512, 3), name="input_image")

    # -- Backbone (frozen) -----------------------------------------------
    backbone = build_backbone()
    features = backbone(inp, training=False)

    # -- ASPP (frozen) ---------------------------------------------------
    aspp_out = ASPP(features)

    # -- DR dummy output (frozen, loss weight = 0) ----------------------
    dr_out = dr_head(aspp_out)

    # -- DME head (trainable) -------------------------------------------
    dme_out = dme_head(aspp_out)

    model = Model(
        inputs=inp,
        outputs=[dr_out, dme_out],
        name="dr_aspp_drn_dme_tuning",
    )

    # Load pre-trained weights (backbone + ASPP layers only)
    if backbone_weights_path:
        try:
            model.load_weights(backbone_weights_path, skip_mismatch=True)
            logger.info("Loaded weights from %s (skip_mismatch=True)", backbone_weights_path)
        except Exception as exc:
            logger.warning("Could not load weights: %s", exc)

    # Freeze everything except the DME head
    trainable_prefixes = ("dme_gap", "dme_fc1", "dme_dropout", "dme_risk")
    for layer in model.layers:
        if layer.name.startswith(trainable_prefixes):
            layer.trainable = True
        else:
            layer.trainable = False

    _log_trainable_summary(model)
    return model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _log_trainable_summary(model: Model) -> None:
    trainable = [l.name for l in model.layers if l.trainable and l.weights]
    frozen = [l.name for l in model.layers if not l.trainable and l.weights]
    logger.info("Trainable layers  (%d): %s", len(trainable), trainable)
    logger.info("Frozen layers     (%d): %s", len(frozen), frozen)
