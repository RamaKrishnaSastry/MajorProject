# MajorProject - Multi-Task DR + DME Detection System

Production-focused deep learning pipeline for joint grading of:

- Diabetic Macular Edema (DME): 3-class ordinal classification
- Diabetic Retinopathy (DR): 5-class classification

## Main Branch Benchmark Status

This main branch currently tracks the best validated run where both tasks crossed the project bar:

- DME QWK > 0.80
- DR QWK > 0.80

Use this as the reference milestone when comparing future experiments, ablations, or branch-level changes.

## Why This Project Exists

The pipeline is designed for real training workflows, not just notebook demos:

- Two-stage training with safe stage2 fine-tuning controls
- QWK-first monitoring and checkpointing
- Joint DME+DR model selection policy
- Robust evaluation with calibration, geometric TTA, and checkpoint ensembling
- Reproducible output artifacts for experiment comparison

## Repository Layout

```text
MajorProject/
|-- main_pipeline.py
|-- train_enhanced.py
|-- evaluate_comprehensive.py
|-- qwk_metrics.py
|-- model.py
|-- dataset_loader.py
|-- dataset_loader_advanced.py
|-- preprocess.py
|-- config.yaml
|-- QUICKSTART.md
|-- USAGE_GUIDE.md
|-- ARCHITECTURE.md
|-- RESULTS.md
|-- CHANGELOG.md
`-- docs/
```

## Model Summary

Backbone and shared trunk:

- ResNet50 backbone truncated at conv4_block6_out
- ASPP multi-scale context block

Task heads:

- DR head: residual MLP + softmax over 5 DR grades
- DME head: softmax over 3 ordinal DME grades

## Training Strategy

Stage 1 (initial training):

- Backbone frozen
- Train ASPP + heads
- Warmup + QWK callbacks
- Saves best_qwk, best_dr, and best_joint (when criteria are met)

Stage 2 (fine-tuning):

- Starts from Stage 1 checkpoint
- Backbone unfrozen with BN freeze safeguards
- Collapse guard and revert-if-worse protections
- Joint checkpointing remains active

## Selection Policy (Joint Focus)

The pipeline uses a strict-to-relaxed tier ladder for checkpoint and final-stage selection.
Primary target tier:

- DME >= 0.70 and DR >= 0.80

With DME floor enforcement and fallback tiers if primary constraints are not met.

## Evaluation Stack

Comprehensive evaluation includes:

- DME: QWK, MAE, RMSE, accuracy, F1, boundary confusion
- DR: QWK, accuracy, MAE, RMSE, confusion matrix
- Optional threshold calibration (DME + DR)
- Optional geometric TTA:
  - none
  - hflip
  - rot4
  - dihedral8
- Optional checkpoint probability ensembling

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Smoke test:

```bash
python main_pipeline.py --mock --epochs 5 --single-stage
```

Full run:

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml
```

## Common Run Modes

```bash
# Stage 1 only
python main_pipeline.py --csv /path/to/DME_Grades.csv --image-dir /path/to/images --single-stage

# Stage 2 only (requires stage1 artifacts)
python main_pipeline.py --csv /path/to/DME_Grades.csv --image-dir /path/to/images --stage2-only

# Select stage2 schedule
python main_pipeline.py --csv /path/to/DME_Grades.csv --image-dir /path/to/images --long-stage2
python main_pipeline.py --csv /path/to/DME_Grades.csv --image-dir /path/to/images --short-stage2
```

## Dataset Format

Required columns (aliases supported by loaders):

- Image name
- Risk of macular edema (0..2)
- Retinopathy grade (0..4)

Example:

```csv
Image name,Risk of macular edema,Retinopathy grade
img001,0,0
img002,1,2
img003,2,4
```

Label spaces:

- DME: 0 No DME, 1 Mild, 2 Moderate
- DR: 0 No DR, 1 Mild, 2 Moderate, 3 Severe NPDR, 4 Proliferative DR

## Most Important Config Knobs

Data and split:

- batch_size
- val_split
- use_advanced_loader
- oversample_minority_enabled

Training:

- stage1.epochs, stage1.learning_rate
- stage2.epochs, stage2.learning_rate
- stage2.dr_loss_weight
- stage2.collapse*guard*\* settings

Joint selection:

- joint_selection.thresholds
- joint_selection.dme_floor

Evaluation:

- evaluation.tta_mode
- evaluation.checkpoint_ensemble_enabled
- evaluation.calibrate_dme_thresholds
- evaluation.calibrate_dr_thresholds
- evaluation.dr_calibration_max_accuracy_drop

## Output Artifacts

Typical outputs:

```text
pipeline_outputs/
|-- effective_config.json
|-- model_stage1.weights.h5
|-- model_stage2.weights.h5
|-- history_stage1.json
|-- history_stage2.json
|-- qwk_epoch_table_stage1.csv
|-- qwk_epoch_table_stage2.csv
|-- pipeline_report.json
|-- eval_stage1/
|   `-- comprehensive_metrics.json
|-- eval_stage2/
|   `-- comprehensive_metrics.json
`-- checkpoints/
    |-- stage1/
    `-- stage2/
```

Start comparisons from:

- pipeline_outputs/pipeline_report.json
- pipeline_outputs/eval_stage1/comprehensive_metrics.json
- pipeline_outputs/eval_stage2/comprehensive_metrics.json

## Documentation Map

- QUICKSTART.md: command-first examples
- USAGE_GUIDE.md: practical runbook and troubleshooting
- ARCHITECTURE.md: detailed model and callback internals
- RESULTS.md: benchmark interpretation and reporting template
- CHANGELOG.md: change history
- docs/INSTALL.md: setup instructions
- docs/FIXES.md: targeted fix notes

## Reproducibility Notes

The pipeline saves:

- effective_config.json for each run
- split information and class balance artifacts
- per-epoch histories and QWK traces
- stage-wise evaluation reports

Use a fixed seed and the same split strategy when making run-to-run comparisons.
