# Pipeline Guide - DR + DME Main Branch Workflow

This document explains the production training/evaluation flow implemented in this repository.

It is intentionally aligned with the current main branch behavior, where the best validated run crossed 0.80 QWK on both DME and DR.

## Pipeline Stages

```text
Stage 0: Data preparation and split
Stage 1: Backbone-frozen training (ASPP + heads)
Stage 1 Evaluation: comprehensive metrics, optional TTA + ensemble + calibration
Stage 2: Full-model fine-tuning with safeguards
Stage 2 Evaluation: same comprehensive stack
Final Aggregation: joint-tier stage selection + pipeline report
```

## Stage 0 - Data Preparation

Implemented by dataset_loader_advanced.py (default path when use_advanced_loader=true).

Key behaviors:
- Flexible CSV column resolution for image, DME, and DR labels
- Joint-aware stratification logic to reduce split instability
- Optional minority oversampling (commonly Mild DME)
- Ordinal-aware DME class weighting
- Saved split metadata and class-balance visualizations

Primary artifacts:
- split_info_advanced.json
- dataset_balance.png

## Stage 1 - Initial Training

Intent:
- Stabilize task heads on strong shared features while keeping the backbone frozen

Behavior:
- ResNet50 conv4 feature extractor frozen
- ASPP + DR head + DME head trained
- QWK-first callbacks track DME and DR every epoch
- Checkpoints:
  - best_qwk.weights.h5
  - best_dr.weights.h5
  - best_joint.weights.h5 (when tier and floor constraints are met)

## Stage 2 - Fine-Tuning

Intent:
- Improve beyond Stage 1 while avoiding catastrophic collapse

Behavior:
- Restores from Stage 1 checkpoint
- Unfreezes backbone with BN freeze controls
- Lower learning rate and shorter warmup by default
- Safety controls:
  - collapse guard (hard drop and patience logic)
  - revert-if-worse against Stage 1 baseline QWK

Stage 2 saves to its own checkpoint directory and is evaluated independently.

## Joint Checkpoint and Final Stage Selection

Main branch uses a strict-to-relaxed threshold ladder.

Primary tier:
- DME >= 0.70 and DR >= 0.80

Additional tiers relax both thresholds when primary is missed.

DME floor enforcement:
- joint checkpoint saving is blocked when DME is below dme_floor

Final stage selection:
- aggregate_results ranks stages by joint tier first
- falls back to DME-best bookkeeping fields for traceability

## Comprehensive Evaluation Stack

evaluate_comprehensive.py runs both task evaluations in one pass where possible.

DME reporting:
- QWK
- MAE, RMSE
- accuracy, F1
- boundary confusion summaries
- confusion matrix + dashboard figures

DR reporting:
- QWK
- accuracy
- MAE, RMSE
- confusion matrix

Optional enhancements:
- DME threshold calibration
- DR threshold calibration with configurable max accuracy drop
- geometric TTA (none/hflip/rot4/dihedral8)
- checkpoint probability ensembling

## Configuration Guide

config.yaml controls all major behavior.

Most tuned blocks:
- stage1
- stage2
- joint_selection
- evaluation

Important examples:
- stage2.dr_loss_weight
- stage2.collapse_guard_*
- joint_selection.thresholds and dme_floor
- evaluation.tta_mode
- evaluation.checkpoint_ensemble_enabled
- evaluation.calibrate_dr_thresholds
- evaluation.dr_calibration_max_accuracy_drop

## Commands

Full pipeline:

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml
```

Stage 1 only:

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml \
  --single-stage
```

Stage 2 only:

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml \
  --stage2-only
```

Mock smoke test:

```bash
python main_pipeline.py --mock --epochs 5 --single-stage
```

## Output Reference

```text
pipeline_outputs/
|-- effective_config.json
|-- history_stage1.json
|-- history_stage2.json
|-- qwk_stage1.json
|-- qwk_stage2.json
|-- qwk_epoch_table_stage1.csv
|-- qwk_epoch_table_stage2.csv
|-- model_stage1.weights.h5
|-- model_stage2.weights.h5
|-- eval_stage1/
|   |-- comprehensive_metrics.json
|   |-- dr_metrics.json
|   |-- ordinal_confusion_matrix.png
|   `-- comprehensive_dashboard.png
|-- eval_stage2/
|   |-- comprehensive_metrics.json
|   |-- dr_metrics.json
|   |-- ordinal_confusion_matrix.png
|   `-- comprehensive_dashboard.png
|-- checkpoints/
|   |-- stage1/
|   `-- stage2/
`-- pipeline_report.json
```

## Recommended Reporting Practice

When sharing results from this branch:
- Report both raw and calibrated QWK for both tasks
- Report selected stage plus best_joint_stage
- Include stage-wise QWK epoch tables
- Keep seed and split strategy fixed for fair comparison

## Related Documents

- README.md for project-level summary
- QUICKSTART.md for copy-paste workflows
- USAGE_GUIDE.md for troubleshooting and operational guidance
- RESULTS.md for benchmark interpretation and reporting format
