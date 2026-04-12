# Quickstart Guide (Main Branch)

Copy-paste workflows for the current DR+DME multi-task pipeline.

## 0) Install

```bash
pip install -r requirements.txt
```

## 1) Smoke Test (No Real Dataset Needed)

```bash
python main_pipeline.py --mock --epochs 5 --single-stage --output-dir pipeline_outputs_smoke
```

Expected outcome:
- Stage 1 trains and evaluates end-to-end
- pipeline_report.json is generated

## 2) Full Real-Data Pipeline

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml \
  --output-dir pipeline_outputs
```

## 3) Stage-Selective Runs

Stage 1 only:

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml \
  --single-stage
```

Stage 2 only (requires existing Stage 1 artifacts):

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml \
  --stage2-only
```

## 4) Stage 2 Schedule Choice

Long schedule (default):

```bash
python main_pipeline.py --csv /path/to/DME_Grades.csv --image-dir /path/to/images --long-stage2
```

Legacy short schedule:

```bash
python main_pipeline.py --csv /path/to/DME_Grades.csv --image-dir /path/to/images --short-stage2
```

## 5) Custom Backbone Initialization

EyePACS backbone:

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --use-eyepacs /path/to/eyepacs_backbone.weights.h5
```

Generic custom backbone:

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --backbone-weights-path /path/to/custom_backbone.h5
```

## 6) Evaluation-Only Utility Commands

Comprehensive evaluator (single model weights):

```bash
python evaluate_comprehensive.py \
  --weights /path/to/model_stage1.weights.h5 \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --output-dir eval_manual
```

Basic evaluator:

```bash
python evaluate.py \
  --weights /path/to/model_stage1.weights.h5 \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --output-dir eval_basic
```

## 7) Common Testing Commands

```bash
python test_qwk_calculation.py
python test_qwk_multioutput.py
python test_ordinal_weighted_loss.py
python test_model_collapse_detection.py
python test_mock_dataset_balance.py
```

## 8) Inspecting Results Quickly

```python
import json

with open("pipeline_outputs/pipeline_report.json", "r") as f:
    report = json.load(f)

print("Selected stage:", report.get("best_stage"))
print("Selection mode:", report.get("selection_mode"))
print("Best DME raw QWK:", report.get("best_qwk"))
print("Best joint stage:", report.get("best_joint_stage"))
print("Best joint DR raw QWK:", report.get("best_joint_dr_qwk"))
```

```python
import json

with open("pipeline_outputs/eval_stage1/comprehensive_metrics.json", "r") as f:
    s1 = json.load(f)

with open("pipeline_outputs/eval_stage2/comprehensive_metrics.json", "r") as f:
    s2 = json.load(f)

print("Stage1 DME raw QWK:", s1["calibration"]["dme"].get("baseline_qwk", s1.get("qwk")))
print("Stage1 DR raw QWK:", s1["dr"].get("calibration", {}).get("baseline_qwk", s1["dr"].get("dr_qwk")))
print("Stage2 DME raw QWK:", s2["calibration"]["dme"].get("baseline_qwk", s2.get("qwk")))
print("Stage2 DR raw QWK:", s2["dr"].get("calibration", {}).get("baseline_qwk", s2["dr"].get("dr_qwk")))
```

## 9) Recommended Main-Branch Reproduction Recipe

To compare against the current best main-branch milestone (both QWK > 0.80), keep these fixed:
- seed
- split strategy and val_split
- stage schedule (long vs short)
- evaluation options (TTA mode, checkpoint ensemble, calibration gates)

Then archive:
- effective_config.json
- qwk epoch tables
- stage-wise comprehensive metrics
- pipeline_report.json

## 10) Expected Output Tree

```text
pipeline_outputs/
|-- effective_config.json
|-- model_stage1.weights.h5
|-- model_stage2.weights.h5
|-- history_stage1.json
|-- history_stage2.json
|-- qwk_epoch_table_stage1.csv
|-- qwk_epoch_table_stage2.csv
|-- eval_stage1/
|   |-- comprehensive_metrics.json
|   `-- dr_metrics.json
|-- eval_stage2/
|   |-- comprehensive_metrics.json
|   `-- dr_metrics.json
`-- pipeline_report.json
```
