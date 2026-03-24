"""
model.py
DR-ASPP-DRN model architecture.

Components
----------
- DRN Backbone  : ResNet50 with dilated convolutions (RFA).
- ASPP Module   : Atrous Spatial Pyramid Pooling for multi-scale fusion.
- DR Head       : 5-class ordinal regression head.
- DME Head      : 3-class softmax classification head.
- build_model() : Full multi-task model.
- build_model_dme_tuning() : Backbone/ASPP/DR frozen; only DME head trains.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


# ── DRN Backbone ──────────────────────────────────────────────────────────────

def build_backbone(
    input_shape: tuple = (512, 512, 3),
    pretrained_weights: str = None,
) -> Model:
    """
    Build the Dilated Residual Network (DRN) backbone.

    Based on ResNet50 with Receptive Field Augmentation (RFA):
    - Removes stride-2 from ``conv5_block1_1_conv`` (layer 4 entry).
    - Sets ``dilation_rate=2`` on all ``conv4_*`` 3x3 convolutions.
    - Sets ``dilation_rate=4`` on all ``conv5_*`` 3x3 convolutions.

    This maintains 4× higher spatial resolution compared to the standard
    ResNet50 while preserving the same receptive field coverage, which
    is critical for detecting small retinal lesions.

    Args:
        input_shape: Spatial dimensions of the input image (H, W, C).
        pretrained_weights: Optional path to ``.weights.h5`` file
            previously saved from a pretraining run.

    Returns:
        Keras Model with output shape ``(batch, H/16, W/16, 2048)``.
    """
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # ── Apply dilated convolutions to conv4 (dilation=2) ──────────────────
    for layer in base.layers:
        if "conv4" in layer.name and isinstance(layer, layers.Conv2D):
            cfg = layer.get_config()
            if cfg["kernel_size"] == (3, 3) or cfg["kernel_size"] == 3:
                layer.dilation_rate = (2, 2)
                layer.strides = (1, 1)
                layer.padding = "same"

    # ── Apply dilated convolutions to conv5 (dilation=4) ──────────────────
    for layer in base.layers:
        if "conv5" in layer.name and isinstance(layer, layers.Conv2D):
            cfg = layer.get_config()
            if cfg["kernel_size"] == (3, 3) or cfg["kernel_size"] == 3:
                layer.dilation_rate = (4, 4)
                layer.strides = (1, 1)
                layer.padding = "same"
            # Remove stride-2 from the first 1x1 projection in conv5_block1
            if "conv5_block1_0_conv" in layer.name or (
                "conv5_block1_1_conv" in layer.name
                and (cfg["kernel_size"] == (1, 1) or cfg["kernel_size"] == 1)
            ):
                layer.strides = (1, 1)

    inp = layers.Input(shape=input_shape, name="backbone_input")
    features = base(inp, training=False)
    backbone_model = Model(inp, features, name="drn_backbone")

    if pretrained_weights is not None:
        backbone_model.load_weights(pretrained_weights, skip_mismatch=True)
        print(f"[backbone] Loaded weights from: {pretrained_weights}")

    return backbone_model


# ── ASPP Module ───────────────────────────────────────────────────────────────

def aspp_module(
    inputs: tf.Tensor,
    out_channels: int = 256,
    rates: tuple = (6, 12, 18),
    name_prefix: str = "aspp",
) -> tf.Tensor:
    """
    Atrous Spatial Pyramid Pooling (ASPP).

    Fuses features extracted at multiple dilation rates to capture
    both fine-grained (microaneurysms) and coarse (vascular distribution)
    retinal structures simultaneously.

    Branches:
        1. 1×1 Conv  — local point features.
        2-4. 3×3 Dilated Convs at ``rates`` — medium/large context.
        5. Global Average Pooling → Dense → Reshape — image-level context.

    Args:
        inputs: Feature tensor from the backbone, shape (B, H, W, C).
        out_channels: Number of output filters per branch.
        rates: Dilation rates for atrous convolutions.
        name_prefix: Prefix for Keras layer names (ensures uniqueness).

    Returns:
        Fused feature map, shape (B, H, W, out_channels).
    """
    conv_kwargs = dict(use_bias=False, padding="same")

    # Branch 1 — 1×1 conv
    x1 = layers.Conv2D(out_channels, 1, name=f"{name_prefix}_1x1", **conv_kwargs)(inputs)
    x1 = layers.BatchNormalization(name=f"{name_prefix}_bn_1x1")(x1)
    x1 = layers.Activation("relu", name=f"{name_prefix}_relu_1x1")(x1)

    branches = [x1]

    # Branches 2-4 — dilated 3×3 convs
    for i, r in enumerate(rates):
        xr = layers.Conv2D(
            out_channels, 3,
            dilation_rate=r,
            name=f"{name_prefix}_dil{r}",
            **conv_kwargs,
        )(inputs)
        xr = layers.BatchNormalization(name=f"{name_prefix}_bn_dil{r}")(xr)
        xr = layers.Activation("relu", name=f"{name_prefix}_relu_dil{r}")(xr)
        branches.append(xr)

    # Branch 5 — global average pooling context
    x5 = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(inputs)
    x5 = layers.Dense(out_channels, use_bias=False, name=f"{name_prefix}_gap_fc")(x5)
    x5 = layers.BatchNormalization(name=f"{name_prefix}_gap_bn")(x5)
    x5 = layers.Activation("relu", name=f"{name_prefix}_gap_relu")(x5)
    x5 = layers.Reshape((1, 1, out_channels), name=f"{name_prefix}_gap_reshape")(x5)
    # Resize to match spatial dimensions of input features
    x5 = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
        name=f"{name_prefix}_gap_upsample",
    )([x5, inputs])

    branches.append(x5)

    # Fuse
    x = layers.Concatenate(name=f"{name_prefix}_concat")(branches)
    x = layers.Conv2D(out_channels, 1, name=f"{name_prefix}_proj", **conv_kwargs)(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_proj_relu")(x)
    return x


# ── Classification Heads ──────────────────────────────────────────────────────

def dr_head(features: tf.Tensor, num_classes: int = 5) -> tf.Tensor:
    """
    DR severity regression/classification head.

    Args:
        features: Spatial feature tensor (B, H, W, C).
        num_classes: Number of DR grades (0–4).

    Returns:
        Scalar logit tensor (B, 1) for ordinal regression.
    """
    x = layers.GlobalAveragePooling2D(name="dr_gap")(features)
    x = layers.Dense(256, activation="relu", name="dr_fc1")(x)
    x = layers.Dropout(0.4, name="dr_dropout")(x)
    out = layers.Dense(1, activation="relu", name="dr_output")(x)
    return out


def dme_head(
    features: tf.Tensor,
    num_classes: int = 3,
    dropout_rate: float = 0.4,
) -> tf.Tensor:
    """
    DME risk classification head (trainable during fine-tuning).

    Args:
        features: Spatial feature tensor (B, H, W, C).
        num_classes: Number of DME grades (0–2 for IDRiD).
        dropout_rate: Dropout probability for regularisation.

    Returns:
        Softmax probability tensor (B, num_classes).
    """
    x = layers.GlobalAveragePooling2D(name="dme_gap")(features)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(dropout_rate, name="dme_dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)
    return out


# ── Full Multi-Task Model ─────────────────────────────────────────────────────

def build_model(
    input_shape: tuple = (512, 512, 3),
    pretrained_weights: str = None,
) -> Model:
    """
    Build the complete DR-ASPP-DRN multi-task model.

    Architecture:
        Input (512×512×3)
        → DRN Backbone
        → ASPP Module
        → DR Head  (output: dr_output, shape (B, 1))
        → DME Head (output: dme_risk,  shape (B, 3))

    Args:
        input_shape: Spatial dimensions of the input image.
        pretrained_weights: Optional path to pretrained backbone weights.

    Returns:
        Compiled Keras Model.
    """
    inp = layers.Input(shape=input_shape, name="input_image")
    backbone = build_backbone(input_shape, pretrained_weights)
    features = backbone(inp)
    aspp_out = aspp_module(features)
    dr_out = dr_head(aspp_out)
    dme_out = dme_head(aspp_out)
    model = Model(inputs=inp, outputs=[dr_out, dme_out], name="DR_ASPP_DRN")
    return model


def build_model_dme_tuning(
    backbone_weights_path: str = None,
    freeze_backbone: bool = True,
    input_shape: tuple = (512, 512, 3),
) -> Model:
    """
    Build DR-ASPP-DRN configured for DME fine-tuning.

    Frozen components (transfer learning):
        - DRN Backbone
        - ASPP Module
        - DR Head

    Trainable components:
        - DME Head (dme_gap, dme_fc1, dme_dropout, dme_risk)

    This two-stage strategy prevents catastrophic forgetting of
    pretrained retinal features while adapting the DME head to
    the IDRiD domain distribution.

    Args:
        backbone_weights_path: Path to ``pretrain_final.weights.h5``.
        freeze_backbone: Freeze backbone + ASPP layers (default True).
        input_shape: Spatial dimensions of the input image.

    Returns:
        Keras Model ready for DME fine-tuning.
    """
    inp = layers.Input(shape=input_shape, name="input_image")

    # ── Backbone ────────────────────────────────────────────────────────────
    backbone = build_backbone(input_shape, backbone_weights_path)
    if freeze_backbone:
        backbone.trainable = False
    features = backbone(inp, training=False)

    # ── ASPP ────────────────────────────────────────────────────────────────
    aspp_out = aspp_module(features, name_prefix="aspp")
    if freeze_backbone:
        # Mark all ASPP layers non-trainable by name convention
        # (they are created inline, so freezing is done via the Model later)
        pass

    # ── DR Head (frozen) ────────────────────────────────────────────────────
    dr_out = dr_head(aspp_out)

    # ── DME Head (trainable) ────────────────────────────────────────────────
    dme_out = dme_head(aspp_out)

    model = Model(
        inputs=inp,
        outputs=[dr_out, dme_out],
        name="DR_ASPP_DRN_DME_Tuning",
    )

    if freeze_backbone:
        # Freeze everything except DME head layers
        dme_layer_names = {"dme_gap", "dme_fc1", "dme_dropout", "dme_risk"}
        for layer in model.layers:
            if layer.name in dme_layer_names:
                layer.trainable = True
            else:
                layer.trainable = False

    # Print trainable status
    print("\n[Model] Trainable layers:")
    for layer in model.layers:
        if layer.trainable:
            print(f"  ✔ {layer.name}")
    print()

    return model
