# Usage Guide - Operational Runbook

This guide is for day-to-day training, evaluation, debugging, and reproducible reporting on the current main branch.

## 1) Environment Setup

```bash
pip install -r requirements.txt
```

Optional sanity checks:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

## 2) Dataset Contract

Expected CSV columns (flexible aliases are supported):
- Image name
- Risk of macular edema (0..2)
- Retinopathy grade (0..4)

Recommended checks before long runs:
- No missing image files
- Label values inside expected ranges
- Sufficient representation of minority DR grades

## 3) Core Run Modes

Full 2-stage pipeline:

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

## 4) Config Tuning Workflow

Recommended order:
1. Keep split and seed fixed.
2. Tune stage1 for stable dual-task signal.
3. Tune stage2 dr_loss_weight and guard settings.
4. Adjust evaluation calibration/TTA/ensemble only after training behavior stabilizes.

High-impact config fields:
- stage1.learning_rate, stage1.focal_loss_gamma
- stage2.learning_rate, stage2.dr_loss_weight
- stage2.collapse_guard_ratio
- stage2.collapse_guard_hard_drop
- stage2.stage2_revert_if_worse
- joint_selection.thresholds and dme_floor
- evaluation.tta_mode
- evaluation.checkpoint_ensemble_enabled
- evaluation.dr_calibration_max_accuracy_drop

## 5) Reading Pipeline Outputs Correctly

Primary output files:
- pipeline_outputs/pipeline_report.json
- pipeline_outputs/eval_stage1/comprehensive_metrics.json
- pipeline_outputs/eval_stage2/comprehensive_metrics.json

Interpretation checklist:
- Compare stage1 vs stage2 raw QWK for both tasks
- Check whether calibration was applied or rejected for each task
- Confirm selected stage and selection_mode
- Inspect best_joint_tier_index in pipeline report

## 6) Best-Run Reproduction Guidance

Main branch reference currently corresponds to a best run where both DME and DR crossed 0.80 QWK.

For reproducibility:
- Lock seed and split
- Keep long/short stage2 mode fixed
- Keep TTA mode and ensemble toggle fixed
- Keep calibration gates unchanged
- Archive effective_config.json and stage metrics for each rerun

## 7) Troubleshooting by Symptom

Symptom: DME improves but DR degrades
- Increase stage2.dr_loss_weight moderately
- Verify DR-best checkpoint is being saved and considered in evaluation
- Review DR calibration gate (dr_calibration_max_accuracy_drop)

Symptom: No best_joint checkpoint saved
- DME floor may be blocking saves
- Inspect joint threshold ladder and dme_floor
- Confirm both task QWK curves are not diverging

Symptom: Stage2 sudden collapse
- Check collapse guard trigger logs
- Reduce stage2 LR or increase warmup
- Confirm stage2 init checkpoint quality

Symptom: DR calibration never applied
- DR gain may be positive but blocked by accuracy-drop constraint
- Adjust dr_calibration_max_accuracy_drop carefully

Symptom: Validation too slow
- Lower TTA mode from dihedral8 to hflip or none
- Disable checkpoint ensembling for quick iteration runs

## 8) Evaluation Cost vs Quality Tradeoff

Fast iteration mode:
- evaluation.tta_mode: none
- evaluation.checkpoint_ensemble_enabled: false
- calibration optional

High-fidelity reporting mode:
- evaluation.tta_mode: dihedral8
- evaluation.checkpoint_ensemble_enabled: true
- calibration enabled with explicit acceptance gates

## 9) Maintenance Commands

Run targeted tests:

```bash
python test_qwk_calculation.py
python test_qwk_multioutput.py
python test_ordinal_weighted_loss.py
python test_model_collapse_detection.py
```

Check working tree before/after changes:

```bash
git status --short
```

## 10) Recommended Experiment Log Fields

Track these fields for every run:
- run tag
- seed
- data split settings
- stage1 and stage2 key hyperparameters
- stage1 and stage2 DME raw/calibrated QWK
- stage1 and stage2 DR raw/calibrated QWK
- selected stage and selection mode
- evaluation options (TTA, ensemble, calibration gates)

Using a consistent log schema makes branch comparisons reliable.
