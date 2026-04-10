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


@keras.utils.register_keras_serializable(package="MajorProject")
class ResizeToMatch(layers.Layer):
    """Resize the first tensor to match spatial size of a reference tensor."""

    def call(self, inputs):
        source, reference = inputs
        target_hw = tf.shape(reference)[1:3]
        return tf.image.resize(source, target_hw)

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def build_backbone(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    weights: str = "imagenet",
    trainable: bool = True,
    weights_path: Optional[str] = None,
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
    weights_path : str, optional
        Path to a custom ``.h5`` weights file.  When provided the backbone is
        initialised with ``weights`` (e.g. ``None``) and the custom weights are
        loaded afterwards, overriding any ImageNet initialisation.

    Returns
    -------
        keras.Model
        ResNet50 model truncated at ``conv4_block6_out`` to match the
        EyePACS preprocessing/training architecture.
    """
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
    # EyePACS pretraining used the Block-3 endpoint (conv4_block6_out)
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
    # 1x1 conv branch
    b0 = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                       name=f"{name_prefix}_b0")(x)
    b0 = layers.BatchNormalization(name=f"{name_prefix}_b0_bn")(b0)
    b0 = layers.Activation("relu", name=f"{name_prefix}_b0_relu")(b0)

    # 3x3 dilation rate 6
    b1 = layers.Conv2D(filters, 3, padding="same", dilation_rate=6, use_bias=False,
                       name=f"{name_prefix}_b1")(x)
    b1 = layers.BatchNormalization(name=f"{name_prefix}_b1_bn")(b1)
    b1 = layers.Activation("relu", name=f"{name_prefix}_b1_relu")(b1)

    # 3x3 dilation rate 12
    b2 = layers.Conv2D(filters, 3, padding="same", dilation_rate=12, use_bias=False,
                       name=f"{name_prefix}_b2")(x)
    b2 = layers.BatchNormalization(name=f"{name_prefix}_b2_bn")(b2)
    b2 = layers.Activation("relu", name=f"{name_prefix}_b2_relu")(b2)

    # 3x3 dilation rate 18
    b3 = layers.Conv2D(filters, 3, padding="same", dilation_rate=18, use_bias=False,
                       name=f"{name_prefix}_b3")(x)
    b3 = layers.BatchNormalization(name=f"{name_prefix}_b3_bn")(b3)
    b3 = layers.Activation("relu", name=f"{name_prefix}_b3_relu")(b3)

    # Global average pooling branch
    b4 = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    b4 = layers.Reshape((1, 1, -1), name=f"{name_prefix}_gap_reshape")(b4)
    b4 = layers.Conv2D(filters, 1, use_bias=False, name=f"{name_prefix}_b4_conv")(b4)
    b4 = layers.BatchNormalization(name=f"{name_prefix}_b4_bn")(b4)
    b4 = layers.Activation("relu", name=f"{name_prefix}_b4_relu")(b4)
    
    # Dynamically resize GAP branch to match current feature-map spatial size
    b4 = ResizeToMatch(name=f"{name_prefix}_b4_resize")([b4, x])

    # Concatenate all branches
    out = layers.Concatenate(name=f"{name_prefix}_concat")([b0, b1, b2, b3, b4])
    out = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                        name=f"{name_prefix}_proj")(out)
    out = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(out)
    out = layers.Activation("relu", name=f"{name_prefix}_proj_relu")(out)
    
    logger.debug("✅ ASPP output shape: %s", out.shape)
    return out


# ---------------------------------------------------------------------------
# DR head (Classification)
# ---------------------------------------------------------------------------

def build_dr_head(
    x: tf.Tensor,
    num_classes: int = 5,
    dropout_rate: float = 0.5,
    hidden_units: int = 256,
) -> tf.Tensor:
    """Build the Diabetic Retinopathy (DR) classification head.

    Outputs a softmax distribution over DR grades 0..4.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map from ASPP.
    num_classes : int
        Number of DR classes.
    dropout_rate : float
        Dropout rate applied before output projection.
    hidden_units : int
        Width of the residual MLP classifier block.

    Returns
    -------
    tf.Tensor
        DR grade probability distribution.
    """
    x = layers.GlobalAveragePooling2D(name="dr_gap")(x)
    x = layers.LayerNormalization(name="dr_ln0")(x)

    shortcut = layers.Dense(hidden_units, use_bias=False, name="dr_res_proj")(x)
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
    x = layers.Dense(num_classes, activation="softmax", name="dr_output")(x)
    return x


# ---------------------------------------------------------------------------
# DME head (Classification)
# ---------------------------------------------------------------------------

def build_dme_head(
    x: tf.Tensor,
    num_classes: int = 3,
    dropout_rate: float = 0.5,
    hidden_units: int = 256,
) -> tf.Tensor:
    """Build the DME (Diabetic Macular Edema) classification head.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map from ASPP.
    num_classes : int
        Number of DME severity classes (default: 3 – No DME/Mild/Moderate).
    dropout_rate : float
        Dropout rate used in the DME head.
    hidden_units : int
        Width of the DME head hidden projection.

    Returns
    -------
    tf.Tensor
        Softmax probability distribution over DME classes.
    """
    x = layers.GlobalAveragePooling2D(name="dme_gap")(x)
    x = layers.Dense(hidden_units, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(dropout_rate, name="dme_dropout")(x)
    x = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)
    return x


# ---------------------------------------------------------------------------
# Full multi-task model (STANDARD TRAINING)
# ---------------------------------------------------------------------------

def build_model(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    backbone_weights: str = "imagenet",
    num_dme_classes: int = 3,
    num_dr_classes: int = 5,
    aspp_filters: int = 256,
    dr_head_units: int = 256,
    dme_head_units: int = 256,
    dropout_rate: float = 0.5,
    trainable: bool = True,
    backbone_weights_path: Optional[str] = None,
) -> keras.Model:
    """Build the complete multi-task DR + DME model.

    Architecture:
    ``Input → Backbone (ResNet50) → ASPP → DR head + DME head``

    DR head: 5-class softmax classification (0..4).
    DME head: 3-class softmax (0=No DME, 1=Mild, 2=Moderate).

    Parameters
    ----------
    input_shape : tuple
        Model input shape ``(H, W, C)``.
    backbone_weights : str
        Initial backbone weights (``"imagenet"`` or ``None``).
    num_dme_classes : int
        Number of output classes for the DME head (default: 3).
    num_dr_classes : int
        Number of output classes for the DR head (default: 5).
    aspp_filters : int
        Number of filters in each ASPP branch (default: 256).
    dr_head_units : int
        Width of the DR head residual MLP.
    dme_head_units : int
        Width of the DME head hidden projection.
    trainable : bool
        Whether all model weights should be trainable (default: True).
    backbone_weights_path : str, optional
        Path to a custom ``.h5`` backbone weights file.  When provided, these
        weights are loaded into the backbone after construction, taking
        priority over ``backbone_weights``.

    Returns
    -------
    keras.Model
        Functional model with two outputs: ``dr_output`` and ``dme_risk``.
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Backbone (ResNet50 with optional pre-training)
    backbone = build_backbone(
        input_shape=input_shape,
        weights=backbone_weights,
        trainable=trainable,  # ✅ PROPAGATE trainable flag
        weights_path=backbone_weights_path,
    )
    features = backbone(inputs)
    logger.info("✅ Backbone: %d parameters", backbone.count_params())

    # ASPP module
    aspp_out = build_aspp(features, filters=aspp_filters)
    logger.info("✅ ASPP module added")
    # DR head (regression – compatible with EyePACS pre-trained weights)
    dr_out = build_dr_head(
        aspp_out,
        num_classes=num_dr_classes,
        dropout_rate=dropout_rate,
        hidden_units=dr_head_units,
    )

    # DME head (3-class classification)
    dme_out = build_dme_head(
        aspp_out,
        num_classes=num_dme_classes,
        dropout_rate=dropout_rate,
        hidden_units=dme_head_units,
    )

    # Build model
    model = keras.Model(
        inputs=inputs,
        outputs={"dr_output": dr_out, "dme_risk": dme_out},
        name="multitask_dr_dme",
    )

    # Ensure all layers are marked properly
    model.trainable = trainable

    logger.info("✅ Built multi-task model:")
    logger.info("   Total parameters: %d", model.count_params())
    logger.info("   Trainable: %s", trainable)
    logger.info("   Input shape: %s", input_shape)

    return model


# ---------------------------------------------------------------------------
# DME fine-tuning model (ONLY DME head trainable)
# ---------------------------------------------------------------------------

def build_model_dme_tuning(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    pretrained_weights: Optional[str] = None,
    num_dme_classes: int = 3,
    num_dr_classes: int = 5,
    aspp_filters: int = 256,
    dr_head_units: int = 256,
    dme_head_units: int = 256,
    dropout_rate: float = 0.5,
    backbone_weights_path: Optional[str] = None,
) -> keras.Model:
    """Build the DME fine-tuning model with only DME head trainable.
    
    Backbone, ASPP, and DR head are frozen. Only DME head gets trained.

    Parameters
    ----------
    input_shape : tuple
        Model input shape ``(H, W, C)``.
    pretrained_weights : str, optional
        Path to pre-trained model weights to load.
    num_dme_classes : int
        Number of DME severity classes (default: 3).
    num_dr_classes : int
        Number of DR classes.
    aspp_filters : int
        Number of filters in ASPP.
    dr_head_units : int
        Width of the DR head residual MLP.
    dme_head_units : int
        Width of the DME head hidden projection.
    backbone_weights_path : str, optional
        Path to a custom ``.h5`` backbone weights file.  When provided, these
        weights are loaded into the backbone before the full model weights are
        applied.

    Returns
    -------
    keras.Model
        Model with only DME head trainable.
    """
    # Build full model with trainable backbone first
    model = build_model(
        input_shape=input_shape,
        backbone_weights="imagenet",
        num_dme_classes=num_dme_classes,
        num_dr_classes=num_dr_classes,
        aspp_filters=aspp_filters,
        dr_head_units=dr_head_units,
        dme_head_units=dme_head_units,
        dropout_rate=dropout_rate,
        trainable=True,  # Start with all trainable
        backbone_weights_path=backbone_weights_path,
    )

    # Load pre-trained weights if provided
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights, skip_mismatch=True)
        logger.info("✅ Loaded pre-trained weights from '%s'", pretrained_weights)

    # ✅ FREEZE everything
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False

    # ✅ UNFREEZE only DME head layers
    dme_layer_patterns = {"dme_gap", "dme_fc1", "dme_dropout", "dme_risk"}
    for layer in model.layers:
        if any(pattern in layer.name for pattern in dme_layer_patterns):
            layer.trainable = True
            logger.info("✅ Unfroze layer: %s", layer.name)

    # Count trainable vs frozen
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = model.count_params()
    frozen_params = total_params - trainable_params

    logger.info("✅ DME Fine-tuning Model:")
    logger.info("   Total parameters: %d", total_params)
    logger.info("   Trainable (DME head): %d", trainable_params)
    logger.info("   Frozen (backbone+ASPP+DR): %d", frozen_params)

    return model


# ---------------------------------------------------------------------------
# Utility: Print trainability summary
# ---------------------------------------------------------------------------

def print_model_summary(model: keras.Model) -> None:
    """Print a detailed summary of model trainability and layer info."""
    print("\n" + "=" * 80)
    print(f"MODEL: {model.name}")
    print("=" * 80)
    print(f"{'Layer':<35} {'Trainable':<12} {'Parameters':>15}")
    print("-" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for layer in model.layers:
        params = layer.count_params()
        total_params += params
        if layer.trainable:
            trainable_params += params
            status = "✅ YES"
        else:
            status = "❌ NO"
        
        print(f"{layer.name:<35} {status:<12} {params:>15,}")
    
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Frozen: {total_params - trainable_params:,}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Utility: Quick model check
# ---------------------------------------------------------------------------

def verify_model_trainability(model: keras.Model) -> bool:
    """Verify that model has trainable weights."""
    if not model.trainable_weights:
        logger.error("❌ ERROR: Model has NO trainable weights!")
        return False
    
    trainable_count = len(model.trainable_weights)
    total_count = len(model.weights)
    logger.info("✅ Model verification passed:")
    logger.info("   Trainable weights: %d", trainable_count)
    logger.info("   Total weights: %d", total_count)
    return True
