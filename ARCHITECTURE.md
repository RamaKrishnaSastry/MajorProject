# DR-ASPP Multi-Task Architecture (Current Main Branch)

This document describes the implemented model used by the production pipeline.

## Task Definitions

- DME task: 3-class ordinal classification
  - 0: No DME
  - 1: Mild
  - 2: Moderate
- DR task: 5-class classification
  - 0: No DR
  - 1: Mild
  - 2: Moderate
  - 3: Severe NPDR
  - 4: Proliferative DR

## High-Level Topology

```text
Input (512x512x3)
  -> ResNet50 backbone (truncated at conv4_block6_out)
  -> ASPP multi-scale context block
  -> Shared feature tensor
       |-> DR classification head (softmax 5)
       `-> DME classification head (softmax 3)
```

## Backbone

Implementation details:
- Base network: keras.applications.ResNet50
- Endpoint: conv4_block6_out
- Typical output shape at 512 input: (None, 32, 32, 1024)
- Supports ImageNet initialization and optional custom backbone weight loading

Design intent:
- Keep stronger mid/high-resolution feature maps than very deep terminal blocks
- Improve lesion localization sensitivity for small retinal findings

## ASPP Block

The ASPP block aggregates context at multiple receptive fields:
- 1x1 conv branch
- 3x3 dilated conv branches with dilation rates 6, 12, 18
- Global pooling branch projected and resized back to feature resolution
- Branch concatenation followed by projection

Why ASPP matters here:
- Retinal findings vary significantly in scale
- Multi-scale context helps both DME and DR grading heads

## DR Head

Current DR head is classification-based (not regression):
- GlobalAveragePooling
- LayerNormalization
- Residual MLP block with BatchNorm + swish activations
- Dropout regularization
- Dense(num_dr_classes, activation="softmax") named dr_output

Training behavior:
- Optimized with categorical cross-entropy
- Optional class weighting from training distribution
- Monitored by val_dr_qwk and DR-specific callbacks/checkpoints

## DME Head

Current DME head is 3-class softmax classification:
- GlobalAveragePooling
- Configurable head style:
  - baseline dense projection, or
  - residual MLP variant (config flag dme_head_residual)
- Dense(num_dme_classes, activation="softmax") named dme_risk

Training behavior:
- Optimized using ordinal-aware weighted cross-entropy when enabled
- Supports label smoothing and optional soft ordinal weighting mode
- Primary optimization metric remains val_qwk

## Multi-Task Loss Setup

Model compiles with task-specific losses and weights:
- DME loss weight: fixed at 1.0
- DR loss weight: configurable (dr_loss_weight)

This allows DR influence to be tuned without destabilizing DME optimization.

## Stage-Wise Training Semantics

Stage 1:
- Backbone frozen
- Train ASPP + both heads
- Bias initialization for heads from observed train class counts

Stage 2:
- Restore from Stage 1 checkpoint
- Unfreeze backbone
- Freeze selected BN layers for stability at small batch sizes
- Use collapse guard and revert-if-worse safeguards

## Checkpointing and Selection

The training stack tracks:
- best_qwk.weights.h5 (DME-oriented)
- best_dr.weights.h5 (DR-oriented)
- best_joint.weights.h5 (tiered DME+DR policy)

Joint policy uses a threshold ladder with a DME floor to enforce balanced behavior.

## Evaluation-Time Enhancements

Implemented in evaluate_comprehensive.py:
- Optional geometric TTA:
  - none, hflip, rot4, dihedral8
- Optional checkpoint probability ensembling
- Optional threshold calibration for both DME and DR

These are evaluation-only controls and do not modify training weights.

## Practical Notes

- The codebase now consistently treats DR as classification, not scalar regression.
- Historical docs mentioning DR sigmoid regression or 4-class DME are outdated.
- For exact runtime behavior, use effective_config.json emitted per run.
