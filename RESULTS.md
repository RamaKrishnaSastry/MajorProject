# Results and Benchmark Reporting

This document defines how to interpret and report performance for the current main branch.

## Main Branch Milestone

Current main branch reference milestone:
- Best validated run crossed 0.80 QWK on both tasks
  - DME QWK > 0.80
  - DR QWK > 0.80

Treat this as the baseline to beat when introducing architectural, loss, or data strategy changes.

## Primary Metrics

Task-level primary metrics:
- DME: QWK (primary), MAE, RMSE, accuracy, macro F1
- DR: QWK (primary), MAE, RMSE, accuracy

Operational quality checks:
- Per-class behavior (especially minority or boundary classes)
- Boundary confusion rates for DME ordinal boundaries
- DR confusion concentration among adjacent grades

## Target Bands

DME QWK interpretation:
- >= 0.80: excellent (target band)
- 0.75-0.79: strong but below target
- 0.60-0.74: moderate
- < 0.60: weak

DR QWK interpretation:
- >= 0.80: excellent (target band)
- 0.75-0.79: strong but below target
- 0.60-0.74: moderate
- < 0.60: weak

## Stage-Wise Result Reading

Always read stage outputs separately before the final aggregate decision:
- Stage 1 can be more stable and may dominate joint selection in hard splits
- Stage 2 can improve one task while hurting the other without safeguards

Use these files:
- pipeline_outputs/eval_stage1/comprehensive_metrics.json
- pipeline_outputs/eval_stage2/comprehensive_metrics.json
- pipeline_outputs/pipeline_report.json

## Joint Selection Context

The final selected stage follows the joint tier ladder and DME floor policy.

Implications:
- A stage with better raw DME may still not be selected if DR drops too much in joint tiers
- Calibrated metrics are informative but selection currently uses raw task metrics unless explicitly changed

## Recommended Report Template

For each experiment, report:
- Seed and split strategy
- Stage 1: DME raw/calibrated QWK, DR raw/calibrated QWK
- Stage 2: DME raw/calibrated QWK, DR raw/calibrated QWK
- Selected stage and selection_mode
- best_joint_tier_index and threshold reached
- TTA and ensemble settings used at evaluation
- Whether DR/DME calibration was applied or rejected

Minimal JSON snippet to include in experiment notes:

```json
{
  "run_id": "<timestamp_or_tag>",
  "seed": 42,
  "selected_stage": "stage1",
  "selection_mode": "joint_tiers",
  "stage1": {
    "dme_qwk_raw": 0.0,
    "dr_qwk_raw": 0.0,
    "dme_qwk_calibrated": 0.0,
    "dr_qwk_calibrated": 0.0
  },
  "stage2": {
    "dme_qwk_raw": 0.0,
    "dr_qwk_raw": 0.0,
    "dme_qwk_calibrated": 0.0,
    "dr_qwk_calibrated": 0.0
  },
  "best_joint_tier_index": 0,
  "tta_mode": "dihedral8",
  "checkpoint_ensemble_enabled": true
}
```

## Common Failure Patterns

1. DME up, DR down:
- Usually indicates weak DR coupling, unstable grade calibration, or checkpoint-selection mismatch.

2. Early-stage class collapse:
- Often seen as single-class prediction spikes in first epochs.
- Mitigate with warmup, class weighting sanity checks, and LR control.

3. Stage 2 regression after strong Stage 1:
- Use collapse guard and revert-if-worse behavior.
- Confirm stage2 init weights path and checkpoint policy.

4. Calibration gains rejected:
- Happens when qwk gain is real but accuracy-drop gate is too strict.
- Tune dr_calibration_max_accuracy_drop carefully.

## Artifact Checklist for Best-Run Claims

When claiming a new best run, archive all of:
- effective_config.json
- split_info_advanced.json
- qwk_epoch_table_stage1.csv
- qwk_epoch_table_stage2.csv
- eval_stage1/comprehensive_metrics.json
- eval_stage2/comprehensive_metrics.json
- pipeline_report.json
- selected checkpoint files under checkpoints/

## Reproducibility and Fair Comparison

For fair benchmark comparisons:
- Keep seed fixed
- Keep split strategy fixed
- Keep evaluation controls fixed (TTA mode, ensemble on/off, calibration flags)
- Compare both stage-level and selected-stage metrics

If any of the above changes, mark the run as a separate benchmark regime.
