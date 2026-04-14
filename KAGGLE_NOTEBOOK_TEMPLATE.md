# Complete Kaggle Notebook Template for Mixed IDRiD+EyePACS Training

This is a ready-to-use template for running mixed training on Kaggle. Copy-paste each cell sequentially.

---

## Cell 1: Setup and Dependencies

```python
# Install required packages
!pip install -q tensorflow keras scikit-learn pandas opencv-python pyyaml

# Setup paths
import os
import sys
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Output directory (auto-saves to Kaggle output)
OUTPUT_DIR = "/kaggle/working/pipeline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Dependencies installed")
print(f"✅ Output directory: {OUTPUT_DIR}")
```

---

## Cell 2: Find and Verify Dataset Paths

```python
import os
from pathlib import Path

# Kaggle mounts datasets in /kaggle/input/
input_dir = Path("/kaggle/input")
print("Available datasets:")
for dataset in input_dir.iterdir():
    print(f"  - {dataset.name}")

# Find IDRiD
idrid_path = None
eyepacs_path = None

for dataset in input_dir.iterdir():
    dataset_name = dataset.name.lower()
    
    # Look for IDRiD
    if 'idrid' in dataset_name or 'indian' in dataset_name:
        idrid_path = str(dataset)
        print(f"\n✅ Found IDRiD: {idrid_path}")
        
        # List contents
        for item in os.listdir(idrid_path)[:10]:
            print(f"   - {item}")
    
    # Look for EyePACS
    if 'eyepacs' in dataset_name or 'diabetic' in dataset_name:
        eyepacs_path = str(dataset)
        print(f"\n✅ Found EyePACS: {eyepacs_path}")
        
        # List contents
        for item in os.listdir(eyepacs_path)[:10]:
            print(f"   - {item}")

if not idrid_path:
    print("\n⚠️ IDRiD dataset not found - add 'idrid' dataset to this notebook")
if not eyepacs_path:
    print("\n⚠️ EyePACS dataset not found - add 'eyepacs' dataset to this notebook")
```

---

## Cell 3: Locate CSV and Image Files

```python
from pathlib import Path

def find_file(root_path, filename_patterns):
    """Find file matching any pattern."""
    for root, dirs, files in os.walk(root_path):
        for file in files:
            for pattern in filename_patterns:
                if pattern.lower() in file.lower():
                    return os.path.join(root, file)
    return None

def find_dir(root_path, dirname_patterns):
    """Find directory matching any pattern."""
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            for pattern in dirname_patterns:
                if pattern.lower() in dir_name.lower():
                    return os.path.join(root, dir_name)
    return None

# Find IDRiD files
print("Looking for IDRiD files...")
idrid_csv = find_file(idrid_path, ["dme_grades", "dme.csv", "grades.csv"])
idrid_img_dir = find_dir(idrid_path, ["training set", "train", "images"])

print(f"IDRiD CSV: {idrid_csv}")
print(f"IDRiD Images: {idrid_img_dir}")

if idrid_img_dir:
    img_count = len([f for f in os.listdir(idrid_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  → {img_count} images found")

# Find EyePACS files
print("\nLooking for EyePACS files...")
eyepacs_csv = find_file(eyepacs_path, ["trainlabels", "labels.csv"])
eyepacs_img_dir = find_dir(eyepacs_path, ["train"])

print(f"EyePACS CSV: {eyepacs_csv}")
print(f"EyePACS Images: {eyepacs_img_dir}")

if eyepacs_img_dir:
    img_count = len([f for f in os.listdir(eyepacs_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  → {img_count} images found")

# Verify all paths exist
assert idrid_csv and os.path.exists(idrid_csv), "IDRiD CSV not found"
assert idrid_img_dir and os.path.exists(idrid_img_dir), "IDRiD images not found"
assert eyepacs_csv and os.path.exists(eyepacs_csv), "EyePACS CSV not found"
assert eyepacs_img_dir and os.path.exists(eyepacs_img_dir), "EyePACS images not found"

print("\n✅ All files found and verified!")
```

---

## Cell 4: Create Config File for Kaggle

```python
# Create optimized config for Kaggle (GPU constraints)
config_yaml = """
# config_mixed_eyepacs_kaggle.yaml - Kaggle GPU optimized

seed: 42
long_stage2_mode: true
input_shape: [512, 512, 3]
num_dme_classes: 3
num_dr_classes: 5
output_dir: /kaggle/working/pipeline_outputs
checkpoint_dir: /kaggle/working/pipeline_outputs/checkpoints

# DATA LOADING
batch_size: 12          # Reduce from 16 for Kaggle GPU memory
val_split: 0.20
augment_train: true
use_advanced_loader: true

dme_class_weight_clip_ratio: 7.0
oversample_minority_enabled: true
oversample_minority_class_id: 1
oversample_factor: 2.5

medical_importance: true
ordinal_penalty: true

border_fraction: 0.10
clip_limit: 2.0
grid_size: 8

# STAGE 1: DR BACKBONE (IDRiD + EyePACS)
stage1:
  epochs: 40              # Reduced from 50 for faster Kaggle execution
  learning_rate: 3.0e-5
  early_stopping_patience: 12
  lr_reduce_patience: 6
  lr_reduce_factor: 0.5
  min_lr: 1.0e-7
  ordinal_loss_weighting: true
  focal_loss_gamma: 1.5
  dme_label_smoothing: 0.02
  dme_soft_ordinal_weights: false
  dr_loss_weight: 0.32
  dr_class_weighting: true
  dr_class_weight_clip_ratio: 7.5
  warmup_epochs: 5

# STAGE 2: DME FINE-TUNING (IDRiD Only)
stage2:
  epochs: 35
  learning_rate: 2.0e-7
  early_stopping_patience: 20
  lr_reduce_patience: 8
  lr_reduce_factor: 0.5
  min_lr: 1.0e-8
  ordinal_loss_weighting: true
  focal_loss_gamma: 1.5
  dme_label_smoothing: 0.005
  dme_soft_ordinal_weights: true
  dr_loss_weight: 0.05
  dr_class_weighting: true
  dr_class_weight_clip_ratio: 7.5
  warmup_epochs: 3
  stage2_freeze_aspp_bn: true
  stage2_checkpoint_use_stage1_baseline: true
  collapse_guard_enabled: true
  collapse_guard_ratio: 0.95
  collapse_guard_min_abs_qwk: 0.68
  collapse_guard_patience: 3
  collapse_guard_hard_drop: 0.20
  collapse_guard_start_epoch: 1
  stage2_revert_if_worse: true
  stage2_min_improvement: 0.003

joint_selection:
  enabled: true
  thresholds:
    - [0.70, 0.80]
    - [0.72, 0.78]
    - [0.75, 0.75]
    - [0.70, 0.75]
    - [0.70, 0.72]
    - [0.70, 0.70]
    - [0.68, 0.70]
  fallback_step: 0.02
  min_threshold: 0.60
  dme_floor: 0.70
"""

# Save config to current directory
config_path = "/kaggle/working/config_mixed_kaggle.yaml"
with open(config_path, 'w') as f:
    f.write(config_yaml)

print(f"✅ Config saved to {config_path}")
```

---

## Cell 5: Clone Your Repository Code

```python
# Option A: Clone from GitHub
!git clone https://github.com/yourusername/your-repo.git /tmp/repo 2>/dev/null || echo "Git clone skipped"

# Add repo to path
import sys
sys.path.insert(0, "/tmp/repo")

# Option B: Copy files directly if already in Kaggle dataset
# (You can upload your code files as a Kaggle dataset)

# Verify we can import
try:
    from main_pipeline import run_pipeline
    print("✅ main_pipeline imported successfully")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Make sure your code files are in the path")
```

---

## Cell 6: Verify Imports

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Try importing main components
try:
    from main_pipeline import run_pipeline, load_config
    from dataset_loader_eyepacs import create_mixed_dr_train_val_datasets
    print("\n✅ All imports successful!")
except ImportError as e:
    print(f"\n⚠️ Import error: {e}")
    raise
```

---

## Cell 7: Run Mixed Training (Main!)

```python
import time
import datetime

print("="*70)
print("MIXED IDRiD + EyePACS TRAINING")
print("="*70)
print(f"Start time: {datetime.datetime.now()}")
print(f"\nPaths:")
print(f"  IDRiD CSV: {idrid_csv}")
print(f"  IDRiD Images: {idrid_img_dir}")
print(f"  EyePACS CSV: {eyepacs_csv}")
print(f"  EyePACS Images: {eyepacs_img_dir}")
print(f"\nConfig: {config_path}")
print("="*70)

# Run the full pipeline
t_start = time.time()

try:
    result = run_pipeline(
        csv_path=idrid_csv,
        image_dir=idrid_img_dir,
        config_path=config_path,
        eyepacs_csv=eyepacs_csv,
        eyepacs_image_dir=eyepacs_img_dir,
        two_stage=True,
    )
    
    t_elapsed = (time.time() - t_start) / 3600  # hours
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {t_elapsed:.2f} hours")
    print(f"\nResults:")
    print(f"  Best stage: {result['best_stage']}")
    print(f"  Best DME QWK: {result['best_qwk']:.4f}")
    print(f"  Best DR QWK: {result.get('best_dr_qwk', 'N/A')}")
    print(f"  Target met (≥0.80): {result['target_met']}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    
    # Save results summary
    summary = {
        'training_time_hours': t_elapsed,
        'best_stage': result['best_stage'],
        'best_dme_qwk': float(result['best_qwk']),
        'best_dr_qwk': float(result.get('best_dr_qwk', 0)),
        'target_met': result['target_met'],
        'completion_time': str(datetime.datetime.now()),
    }
    
    with open(f"{OUTPUT_DIR}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to training_summary.json")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    raise
```

---

## Cell 8: List Generated Models

```python
import os
from pathlib import Path

print("Generated models and outputs:\n")

models_dir = Path(OUTPUT_DIR) / "model_final_best.model.h5"
if models_dir.exists():
    size_mb = models_dir.stat().st_size / (1024*1024)
    print(f"✅ model_final_best.model.h5 ({size_mb:.1f} MB)")

checkpoints_dir = Path(OUTPUT_DIR) / "checkpoints"
if checkpoints_dir.exists():
    for stage_dir in checkpoints_dir.iterdir():
        print(f"\n  {stage_dir.name}:")
        for file in stage_dir.glob("*.h5"):
            size_mb = file.stat().st_size / (1024*1024)
            print(f"    - {file.name} ({size_mb:.1f} MB)")

# List all files
print(f"\nAll files in {OUTPUT_DIR}:")
for file in Path(OUTPUT_DIR).rglob("*"):
    if file.is_file():
        rel_path = file.relative_to(OUTPUT_DIR)
        print(f"  {rel_path}")
```

---

## Cell 9: Load and Test Model

```python
import tensorflow as tf
from tensorflow import keras

# Load the best model
model_path = f"{OUTPUT_DIR}/model_final_best.model.h5"

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"\nModel info:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output names: {list(model.output_names)}")
    
    # Model summary
    print(f"\nModel architecture:")
    model.summary()
else:
    print(f"⚠️ Model not found at {model_path}")
```

---

## Cell 10: Download Results

```python
# All files in /kaggle/working/ are automatically downloaded
# when the notebook finishes

print("✅ All outputs saved to /kaggle/working/")
print("\nFiles ready for download:")
print("  - pipeline_outputs/model_final_best.model.h5  (main model)")
print("  - pipeline_outputs/checkpoints/               (all checkpoints)")
print("  - pipeline_outputs/pipeline_report.json       (metrics)")
print("  - pipeline_outputs/effective_config.json      (config used)")
print("  - training_summary.json                       (summary)")
print("\nThese will download automatically when notebook completes.")
```

---

## How to Use This Template

### Step 1: Prepare Kaggle
- Create a **new Notebook** in Kaggle
- Add **IDRiD dataset** as input
- Add **EyePACS dataset** as input

### Step 2: Paste Code
Copy each cell above into your Kaggle notebook cells (in order)

### Step 3: Run Cells
- Cell 1: Install packages
- Cell 2: Find datasets
- Cell 3: Locate files
- Cell 4: Create config
- Cell 5: Clone code (or upload manually)
- Cell 6: Verify imports
- **Cell 7: RUN TRAINING** ← Main execution
- Cell 8: List outputs
- Cell 9: Load model (optional)
- Cell 10: Download results

### Step 4: Monitor
Watch Cell 7 progress (takes 3-5 hours with 2× T4 GPU)

### Step 5: Download
All files in `/kaggle/working/` auto-download when complete

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| ImportError | Cell 5: Make sure GitHub clone worked or upload code as dataset |
| Dataset not found | Cell 2: Add IDRiD/EyePACS datasets to notebook inputs |
| Out of memory | Cell 4: Reduce `batch_size` from 12 to 8 |
| "CSV not found" | Cell 3: Check dataset structure, file names might differ |
| Timeout | This is normal - training takes 3-5 hours. Kaggle sessions support long runs. |

---

## Expected Output

```
======================================================================
MIXED IDRiD + EyePACS TRAINING
======================================================================

Stage 0: Data preparation…
  Combined dataset: 88,516 samples (516 IDRiD + 88,000 EyePACS)
  
Stage 1: Initial Training (DR backbone)
  Epochs: 40
  Final DME QWK: 0.7234

Stage 2: Fine-Tuning (DME on IDRiD)
  Epochs: 35
  Final DME QWK: 0.7891

✅ TRAINING COMPLETE!
Total time: 4.32 hours
Best DME QWK: 0.7891
======================================================================
```

Then model downloads to your computer! 🚀
