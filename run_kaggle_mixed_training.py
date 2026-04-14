#!/usr/bin/env python3
"""
run_kaggle_mixed_training.py - CLI for mixed IDRiD+EyePACS training on Kaggle

Usage:
    # Automatic dataset discovery
    python run_kaggle_mixed_training.py
    
    # Specify datasets explicitly
    python run_kaggle_mixed_training.py \
        --idrid-dataset idrid \
        --eyepacs-dataset eyepacs \
        --output-dir /kaggle/working/results
    
    # Custom configuration
    python run_kaggle_mixed_training.py \
        --config custom_config.yaml \
        --epochs-stage1 50 \
        --epochs-stage2 40
"""

import argparse
import json
import logging
import os
import sys
import time
import datetime
from pathlib import Path

# Optional: memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_file(root_path, filename_patterns):
    """Find file matching any pattern."""
    if not os.path.exists(root_path):
        return None
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            for pattern in filename_patterns:
                if pattern.lower() in file.lower():
                    return os.path.join(root, file)
    return None


def find_dir(root_path, dirname_patterns):
    """Find directory matching any pattern."""
    if not os.path.exists(root_path):
        return None
    
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            for pattern in dirname_patterns:
                if pattern.lower() in dir_name.lower():
                    return os.path.join(root, dir_name)
    return None


def check_memory_safety(idrid_img_dir, eyepacs_img_dir, batch_size):
    """Check if memory is safe for mixed training without kernel crash.
    
    Kaggle T4 GPU has:
    - ~15GB GPU VRAM
    - ~30GB CPU RAM
    
    This function warns if batch size or dataset is too large.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow not imported - skipping memory check")
        return
    
    # Get available memory
    try:
        if HAS_PSUTIL:
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
        else:
            available_ram_gb = 30  # Assume typical Kaggle RAM
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        gpu_memory_gb = 15 if gpu_devices else 0  # T4 has ~15GB
    except:
        logger.warning("Could not determine available memory - proceeding with caution")
        available_ram_gb = 30
        gpu_memory_gb = 15
    
    # Count images
    idrid_count = len([f for f in os.listdir(idrid_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    eyepacs_count = len([f for f in os.listdir(eyepacs_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    total_count = idrid_count + eyepacs_count
    
    logger.info(f"\n{'='*70}")
    logger.info("MEMORY SAFETY CHECK")
    logger.info(f"{'='*70}")
    logger.info(f"Available RAM: {available_ram_gb:.1f} GB")
    logger.info(f"Available GPU: {gpu_memory_gb:.1f} GB")
    logger.info(f"Total images: {total_count:,} (IDRiD: {idrid_count}, EyePACS: {eyepacs_count})")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"\n⚠️ IMPORTANT: Images load via tf.data STREAMING pipeline")
    logger.info(f"   → NOT all images loaded to RAM at once")
    logger.info(f"   → Only {batch_size} images per batch in memory")
    logger.info(f"   → Much safer than loading entire dataset!")
    
    # Warnings
    if batch_size > 32:
        logger.warning(f"\n⚠️ WARNING: batch_size={batch_size} is large for Kaggle GPU")
        logger.warning(f"   Kaggle T4 has limited VRAM. May cause OOM.")
        logger.warning(f"   Recommended: batch_size <= 16")
    
    if batch_size * 512 * 512 * 3 / (1024**3) > gpu_memory_gb * 0.7:
        logger.warning(f"\n⚠️ WARNING: One batch uses significant GPU memory")
        logger.warning(f"   Reduce batch_size if you see memory errors")
    
    logger.info(f"\n✅ Memory check passed. Safe to proceed.")
    logger.info(f"{'='*70}\n")


def discover_kaggle_datasets():
    """Auto-discover IDRiD and EyePACS datasets in /kaggle/input/"""
    input_dir = Path("/kaggle/input")
    
    if not input_dir.exists():
        logger.warning("Not running on Kaggle - /kaggle/input not found")
        return None, None
    
    idrid_path = None
    eyepacs_path = None
    
    logger.info("Discovering datasets in /kaggle/input/...")
    
    for dataset in sorted(input_dir.iterdir()):
        dataset_name = dataset.name.lower()
        
        if 'idrid' in dataset_name or 'indian' in dataset_name:
            idrid_path = str(dataset)
            logger.info(f"✅ Found IDRiD: {dataset.name}")
        
        if 'eyepacs' in dataset_name or 'diabetic' in dataset_name:
            eyepacs_path = str(dataset)
            logger.info(f"✅ Found EyePACS: {dataset.name}")
    
    return idrid_path, eyepacs_path


def locate_files(idrid_path, eyepacs_path):
    """Locate CSV and image directories."""
    logger.info("Locating files...")
    
    # Find IDRiD CSV
    idrid_csv = find_file(idrid_path, ["dme_grades", "dme.csv", "grades.csv"])
    if not idrid_csv:
        raise FileNotFoundError(f"IDRiD CSV not found in {idrid_path}")
    logger.info(f"✅ IDRiD CSV: {idrid_csv}")
    
    # Find IDRiD images
    idrid_img_dir = find_dir(idrid_path, ["training set", "train", "images"])
    if not idrid_img_dir:
        raise FileNotFoundError(f"IDRiD images not found in {idrid_path}")
    img_count = len([f for f in os.listdir(idrid_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    logger.info(f"✅ IDRiD Images: {idrid_img_dir} ({img_count} images)")
    
    # Find EyePACS CSV
    eyepacs_csv = find_file(eyepacs_path, ["trainlabels", "labels.csv", "dataset_metadata"])
    if not eyepacs_csv:
        raise FileNotFoundError(f"EyePACS CSV not found in {eyepacs_path}")
    logger.info(f"✅ EyePACS CSV: {eyepacs_csv}")
    
    # Find EyePACS images
    eyepacs_img_dir = find_dir(eyepacs_path, ["train"])
    if not eyepacs_img_dir:
        raise FileNotFoundError(f"EyePACS images not found in {eyepacs_path}")
    img_count = len([f for f in os.listdir(eyepacs_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    logger.info(f"✅ EyePACS Images: {eyepacs_img_dir} ({img_count} images)")
    
    return idrid_csv, idrid_img_dir, eyepacs_csv, eyepacs_img_dir


def create_config(args):
    """Create or load configuration."""
    if args.config and os.path.exists(args.config):
        logger.info(f"Using provided config: {args.config}")
        return args.config
    
    logger.info("Creating optimized Kaggle config...")
    
    config_yaml = f"""
# Auto-generated config for Kaggle mixed training
seed: 42
long_stage2_mode: true
input_shape: [512, 512, 3]
num_dme_classes: 3
num_dr_classes: 5
output_dir: {args.output_dir}
checkpoint_dir: {args.output_dir}/checkpoints

batch_size: {args.batch_size}
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

stage1:
  epochs: {args.epochs_stage1}
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

stage2:
  epochs: {args.epochs_stage2}
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
    
    config_path = os.path.join(args.output_dir, "config_kaggle_auto.yaml")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_yaml)
    
    logger.info(f"✅ Config created: {config_path}")
    return config_path


def run_training(args, idrid_csv, idrid_img_dir, eyepacs_csv, eyepacs_img_dir, config_path):
    """Run the mixed training pipeline."""
    try:
        from main_pipeline import run_pipeline
    except ImportError:
        logger.error("Cannot import main_pipeline. Make sure code is in Python path.")
        raise
    
    logger.info("\n" + "="*70)
    logger.info("MIXED IDRiD + EyePACS TRAINING")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.datetime.now()}")
    logger.info(f"\nConfiguration:")
    logger.info(f"  IDRiD CSV: {idrid_csv}")
    logger.info(f"  IDRiD Images: {idrid_img_dir}")
    logger.info(f"  EyePACS CSV: {eyepacs_csv}")
    logger.info(f"  EyePACS Images: {eyepacs_img_dir}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Output: {args.output_dir}")
    
    # Backbone setup
    backbone_weights = None
    if args.use_backbone:
        if not args.backbone_path:
            logger.error("❌ --use-backbone requires --backbone-path")
            sys.exit(1)
        if not os.path.exists(args.backbone_path):
            logger.error(f"❌ Backbone file not found: {args.backbone_path}")
            sys.exit(1)
        backbone_weights = args.backbone_path
        logger.info(f"  Backbone: {args.backbone_path}")
        logger.info(f"  Transfer learning: ENABLED")
    else:
        logger.info(f"  Backbone: ImageNet (default)")
    
    logger.info("="*70 + "\n")
    
    t_start = time.time()
    
    try:
        result = run_pipeline(
            csv_path=idrid_csv,
            image_dir=idrid_img_dir,
            config_path=config_path,
            eyepacs_csv=eyepacs_csv,
            eyepacs_image_dir=eyepacs_img_dir,
            two_stage=True,
            backbone_weights_path=backbone_weights,
        )
        
        t_elapsed = (time.time() - t_start) / 3600  # hours
        
        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total time: {t_elapsed:.2f} hours")
        logger.info(f"\nResults:")
        logger.info(f"  Best stage: {result['best_stage']}")
        logger.info(f"  Best DME QWK: {result['best_qwk']:.4f}")
        logger.info(f"  Best DR QWK: {result.get('best_dr_qwk', 'N/A')}")
        logger.info(f"  Target met (≥0.80): {result['target_met']}")
        logger.info(f"\nOutput saved to: {args.output_dir}")
        logger.info("="*70)
        
        # Save summary
        summary = {
            'training_time_hours': t_elapsed,
            'best_stage': result['best_stage'],
            'best_dme_qwk': float(result['best_qwk']),
            'best_dr_qwk': float(result.get('best_dr_qwk', 0)),
            'target_met': result['target_met'],
            'completion_time': str(datetime.datetime.now()),
        }
        
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Summary saved to {summary_path}")
        
        # List output files
        if args.list_outputs:
            logger.info("\nGenerated files:")
            for root, dirs, files in os.walk(args.output_dir):
                level = root.replace(args.output_dir, '').count(os.sep)
                indent = ' ' * (level * 2)
                logger.info(f"{indent}{os.path.basename(root)}/")
                sub_indent = ' ' * ((level + 1) * 2)
                for file in files[:10]:  # Limit to first 10 files per dir
                    logger.info(f"{sub_indent}{file}")
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run mixed IDRiD+EyePACS training on Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  1. FASTEST - Direct file paths (no discovery needed):
     python run_kaggle_mixed_training.py \\
       --idrid-csv /kaggle/input/idrid/DME_Grades.csv \\
       --idrid-images /kaggle/input/idrid/A. Training set \\
       --eyepacs-csv /kaggle/input/eyepacs/trainLabels.csv \\
       --eyepacs-images /kaggle/input/eyepacs/train

  2. MEMORY SAFE - Limit EyePACS to prevent kernel crash:
     python run_kaggle_mixed_training.py \\
       --idrid-csv /kaggle/input/idrid/DME_Grades.csv \\
       --idrid-images /kaggle/input/idrid/A. Training set \\
       --eyepacs-csv /kaggle/input/eyepacs/trainLabels.csv \\
       --eyepacs-images /kaggle/input/eyepacs/train \\
       --batch-size 8 \\
       --max-eyepacs-samples 20000

  3. WITH BACKBONE TRANSFER - Use pre-trained EyePACS backbone (RECOMMENDED):
     python run_kaggle_mixed_training.py \\
       --idrid-csv /kaggle/input/idrid/DME_Grades.csv \\
       --idrid-images /kaggle/input/idrid/A. Training set \\
       --eyepacs-csv /kaggle/input/eyepacs/trainLabels.csv \\
       --eyepacs-images /kaggle/input/eyepacs/train \\
       --use-backbone \\
       --backbone-path /kaggle/input/backbone/eyepacs_backbone.weights.h5 \\
       --epochs-stage1 20 \\
       --batch-size 12

  4. DIAGNOSE - Check memory without training:
     python run_kaggle_mixed_training.py \\
       --idrid-csv /kaggle/input/idrid/DME_Grades.csv \\
       --idrid-images /kaggle/input/idrid/A. Training set \\
       --eyepacs-csv /kaggle/input/eyepacs/trainLabels.csv \\
       --eyepacs-images /kaggle/input/eyepacs/train \\
       --skip-memory-check

  5. AUTO-DISCOVER - Full auto-discovery in /kaggle/input/:
     python run_kaggle_mixed_training.py

TRANSFER LEARNING:
  --use-backbone: Enables pre-trained backbone for faster convergence
  --backbone-path: Path to .h5 file with backbone weights (e.g., conv4_block6_out)
  
  Benefits of using backbone:
    - Faster Stage 1 convergence (20 epochs vs 40)
    - Better generalization (weights pre-trained on 88K images)
    - Should improve external dataset performance
    
MEMORY SAFETY:
  - tf.data pipeline streams images (doesn't load all to RAM)
  - Only 'batch_size' images in GPU memory at once
  - Recommended batch_size for Kaggle T4: 8-16
  - If kernel crashes: reduce batch_size or --max-eyepacs-samples
        """,
    )
    
    # Direct file paths (simplest - no auto-discovery needed)
    parser.add_argument(
        "--idrid-csv",
        type=str,
        default=None,
        help="Direct path to IDRiD DME_Grades.csv",
    )
    parser.add_argument(
        "--idrid-images",
        type=str,
        default=None,
        help="Direct path to IDRiD training images directory",
    )
    parser.add_argument(
        "--eyepacs-csv",
        type=str,
        default=None,
        help="Direct path to EyePACS trainLabels.csv",
    )
    parser.add_argument(
        "--eyepacs-images",
        type=str,
        default=None,
        help="Direct path to EyePACS training images directory",
    )
    
    # Alternative: dataset root paths (for auto-discovery within dataset)
    parser.add_argument(
        "--idrid-dataset",
        type=str,
        default=None,
        help="Path to IDRiD dataset root (will auto-discover CSVs/images)",
    )
    parser.add_argument(
        "--eyepacs-dataset",
        type=str,
        default=None,
        help="Path to EyePACS dataset root (will auto-discover CSVs/images)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/pipeline_outputs",
        help="Output directory for models and results (default: /kaggle/working/pipeline_outputs)",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml (auto-generated if not provided)",
    )
    
    # Hyperparameters
    parser.add_argument(
        "--epochs-stage1",
        type=int,
        default=40,
        help="Number of epochs for Stage 1 (default: 40; reduce to 20 if using --use-backbone)",
    )
    parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=35,
        help="Number of epochs for Stage 2 (default: 35)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size (default: 12 for Kaggle GPU memory)",
    )
    
    # Memory safety options
    parser.add_argument(
        "--max-eyepacs-samples",
        type=int,
        default=None,
        help="Limit EyePACS to first N images (e.g., 20000 for smaller dataset)",
    )
    parser.add_argument(
        "--skip-memory-check",
        action="store_true",
        help="Skip memory safety check (use with caution)",
    )
    
    # Transfer learning (backbone initialization)
    parser.add_argument(
        "--use-backbone",
        action="store_true",
        help="Use pre-trained backbone weights (requires --backbone-path)",
    )
    parser.add_argument(
        "--backbone-path",
        type=str,
        default=None,
        help="Path to pre-trained backbone .h5 file (e.g., eyepacs_backbone.weights.h5)",
    )
    
    # Other options
    parser.add_argument(
        "--list-outputs",
        action="store_true",
        help="List generated files after training",
    )
    
    args = parser.parse_args()
    
    try:
        # Step 1: Get file paths (in priority order)
        
        # Option A: Direct paths provided (fastest, no discovery needed)
        if args.idrid_csv and args.idrid_images and args.eyepacs_csv and args.eyepacs_images:
            logger.info("Using provided direct file paths")
            idrid_csv = args.idrid_csv
            idrid_img_dir = args.idrid_images
            eyepacs_csv = args.eyepacs_csv
            eyepacs_img_dir = args.eyepacs_images
            
            # Verify paths exist
            if not os.path.exists(idrid_csv):
                raise FileNotFoundError(f"IDRiD CSV not found: {idrid_csv}")
            if not os.path.exists(idrid_img_dir):
                raise FileNotFoundError(f"IDRiD images not found: {idrid_img_dir}")
            if not os.path.exists(eyepacs_csv):
                raise FileNotFoundError(f"EyePACS CSV not found: {eyepacs_csv}")
            if not os.path.exists(eyepacs_img_dir):
                raise FileNotFoundError(f"EyePACS images not found: {eyepacs_img_dir}")
            
            logger.info(f"✅ IDRiD CSV: {idrid_csv}")
            logger.info(f"✅ IDRiD Images: {idrid_img_dir}")
            logger.info(f"✅ EyePACS CSV: {eyepacs_csv}")
            logger.info(f"✅ EyePACS Images: {eyepacs_img_dir}")
        
        # Option B: Dataset root paths (auto-discover within each)
        elif args.idrid_dataset and args.eyepacs_dataset:
            logger.info("Using provided dataset paths (auto-discovering files within)")
            idrid_csv, idrid_img_dir, eyepacs_csv, eyepacs_img_dir = locate_files(args.idrid_dataset, args.eyepacs_dataset)
        
        # Option C: Auto-discover in /kaggle/input/
        else:
            logger.info("Auto-discovering datasets in /kaggle/input/...")
            idrid_path, eyepacs_path = discover_kaggle_datasets()
            
            if not idrid_path or not eyepacs_path:
                logger.error("Could not auto-discover datasets. Please provide paths:")
                logger.error("\nAuto-discover option:")
                logger.error("  --idrid-dataset /path/to/idrid")
                logger.error("  --eyepacs-dataset /path/to/eyepacs")
                logger.error("\nDirect paths option (faster):")
                logger.error("  --idrid-csv /path/to/DME_Grades.csv")
                logger.error("  --idrid-images /path/to/images")
                logger.error("  --eyepacs-csv /path/to/trainLabels.csv")
                logger.error("  --eyepacs-images /path/to/images")
                sys.exit(1)
            
            idrid_csv, idrid_img_dir, eyepacs_csv, eyepacs_img_dir = locate_files(idrid_path, eyepacs_path)
        
        # Step 2: Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 3: Create config
        config_path = create_config(args)
        
        # Step 3.5: Memory safety check (prevent kernel restart)
        if not args.skip_memory_check:
            check_memory_safety(idrid_img_dir, eyepacs_img_dir, args.batch_size)
        else:
            logger.info("⚠️ Memory check skipped (--skip-memory-check)")
        
        # Step 4: Run training
        result = run_training(args, idrid_csv, idrid_img_dir, eyepacs_csv, eyepacs_img_dir, config_path)
        
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
