"""
gradcam_visualization.py
=========================
Generates Grad-CAM saliency maps for the DR-ASPP-DRN model,
overlaid on fundus images.

For each image this script produces:
  • DME Grad-CAM heatmap  (what the DME head attends to)
  • DR  Grad-CAM heatmap  (what the DR  head attends to)
  • Side-by-side comparison panel (original | DR-CAM | DME-CAM)

The target layer is the LAST Conv layer of the ASPP module
(``aspp_proj``) — chosen because it is shared by both heads and
retains 64×64 spatial resolution (thanks to the dilation trick).

Usage
-----
# With a real trained model:
python gradcam_visualization.py \
    --model /path/to/best_stage2_model.keras \
    --csv   /path/to/IDRiD_labels.csv \
    --imgdir /path/to/images \
    --out   gradcam_outputs \
    --n     8

# Without a trained model (demo mode — random weights):
python gradcam_visualization.py --demo --out gradcam_demo
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# ── TF / Keras ───────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

# ════════════════════════════════════════════════════════════════════════════
# DR / DME class labels
# ════════════════════════════════════════════════════════════════════════════
DR_LABELS  = ["No DR", "Mild", "Moderate", "Severe NPDR", "Proliferative"]
DME_LABELS = ["No DME", "Mild", "Moderate"]

IDRID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IDRID_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Preprocessing (mirrors preprocess.py)
# ════════════════════════════════════════════════════════════════════════════

def _clahe_green(img_bgr):
    """Apply CLAHE to the green channel only (matches training pipeline)."""
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq   = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_image(img_path: str, size: int = 512) -> np.ndarray:
    """Load, crop borders, CLAHE, resize, normalise → (1, H, W, 3)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    # 10 % border crop
    t, b = int(h * 0.10), int(h * 0.90)
    l, r = int(w * 0.10), int(w * 0.90)
    img  = img[t:b, l:r]
    img  = _clahe_green(img)
    img  = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img  = (img - IDRID_MEAN) / IDRID_STD
    return img[np.newaxis]         # (1, H, W, 3)


def _load_raw_rgb(img_path: str, size: int = 512) -> np.ndarray:
    """Load image for display (not normalised)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


# ════════════════════════════════════════════════════════════════════════════
# Grad-CAM
# ════════════════════════════════════════════════════════════════════════════

def _find_target_layer(model: keras.Model) -> str:
    """
    Return the name of the best Grad-CAM target layer:
      1. aspp_proj_relu  (last ASPP relu — ideal)
      2. aspp_proj       (ASPP projection conv)
      3. conv4_block6_out (backbone output)
      4. last Conv2D layer in the model
    """
    preferred = ["aspp_proj_relu", "aspp_proj", "conv4_block6_out"]
    names = [l.name for l in model.layers]
    for p in preferred:
        if p in names:
            return p
    # Fallback: last Conv2D
    conv_layers = [l.name for l in model.layers
                   if isinstance(l, keras.layers.Conv2D)]
    return conv_layers[-1] if conv_layers else names[-2]


def gradcam_heatmap(
    model: keras.Model,
    img_tensor: np.ndarray,
    output_key: str,          # "dr_output" or "dme_risk"
    class_idx: int,
    layer_name: str,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap (H×W, float32, 0–1).

    Parameters
    ----------
    model      : the full multi-task model
    img_tensor : preprocessed image (1, H, W, 3)
    output_key : which head to differentiate ("dr_output" or "dme_risk")
    class_idx  : predicted class index
    layer_name : name of the convolutional target layer
    """
    # Sub-model that outputs (feature_map, predictions)
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [
            model.get_layer(layer_name).output,
            model.output[output_key] if isinstance(model.output, dict)
            else model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs     = tf.cast(img_tensor, tf.float32)
        conv_out, preds = grad_model(inputs, training=False)
        if isinstance(preds, dict):
            preds = preds[output_key]
        score = preds[:, class_idx]

    grads = tape.gradient(score, conv_out)              # (1, h, w, C)
    pooled = tf.reduce_mean(grads, axis=(1, 2))         # (1, C)
    conv_np  = conv_out[0].numpy()                      # (h, w, C)
    pooled_np = pooled[0].numpy()                       # (C,)

    cam = np.sum(conv_np * pooled_np[np.newaxis, np.newaxis, :], axis=-1)
    cam = np.maximum(cam, 0)                            # ReLU
    if cam.max() > 1e-8:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.astype(np.float32)


def overlay_heatmap(
    raw_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend Grad-CAM heatmap onto raw RGB image. Returns uint8 RGB."""
    h, w  = raw_img.shape[:2]
    heat  = cv2.resize(heatmap, (w, h))
    heat  = np.uint8(heat * 255)
    heat_color = cv2.applyColorMap(heat, colormap)          # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heat_color + (1 - alpha) * raw_img).astype(np.uint8)
    return blended


# ════════════════════════════════════════════════════════════════════════════
# Per-image visualisation
# ════════════════════════════════════════════════════════════════════════════

def visualise_single(
    model: keras.Model,
    img_path: str,
    layer_name: str,
    out_dir: str,
    img_id: str,
    true_dr:  int = None,
    true_dme: int = None,
):
    """Generate and save Grad-CAM panels for one image."""
    tensor  = preprocess_image(img_path)
    raw_rgb = _load_raw_rgb(img_path)

    # ── predictions ──────────────────────────────────────────────────────
    preds = model(tf.cast(tensor, tf.float32), training=False)
    if isinstance(preds, dict):
        dr_proba  = preds["dr_output"].numpy()[0]
        dme_proba = preds["dme_risk"].numpy()[0]
    else:
        dr_proba  = preds[0].numpy()[0]
        dme_proba = preds[1].numpy()[0] if len(preds) > 1 else np.zeros(3)

    pred_dr  = int(np.argmax(dr_proba))
    pred_dme = int(np.argmax(dme_proba))

    # ── Grad-CAM for both heads ───────────────────────────────────────────
    try:
        cam_dr  = gradcam_heatmap(model, tensor, "dr_output",  pred_dr,  layer_name)
        cam_dme = gradcam_heatmap(model, tensor, "dme_risk",   pred_dme, layer_name)
    except Exception as e:
        print(f"  ⚠  Grad-CAM failed for {img_id}: {e}")
        return

    overlay_dr  = overlay_heatmap(raw_rgb, cam_dr)
    overlay_dme = overlay_heatmap(raw_rgb, cam_dme)

    # ── build panel ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

    axes[0].imshow(raw_rgb)
    dr_true_str  = f"  (true: {DR_LABELS[true_dr]})"  if true_dr  is not None else ""
    dme_true_str = f"  (true: {DME_LABELS[true_dme]})" if true_dme is not None else ""
    axes[0].set_title(f"Original\n{img_id}", color="white", fontsize=11)

    axes[1].imshow(overlay_dr)
    conf_dr = dr_proba[pred_dr]
    axes[1].set_title(
        f"DR Grad-CAM\nPred: {DR_LABELS[pred_dr]}  ({conf_dr:.2f}){dr_true_str}",
        color="lightyellow", fontsize=11,
    )

    axes[2].imshow(overlay_dme)
    conf_dme = dme_proba[pred_dme]
    axes[2].set_title(
        f"DME Grad-CAM\nPred: {DME_LABELS[pred_dme]}  ({conf_dme:.2f}){dme_true_str}",
        color="lightcyan", fontsize=11,
    )

    plt.suptitle("DR-ASPP-DRN  |  Grad-CAM Explainability",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"gradcam_{img_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Summary grid (multiple images in one figure)
# ════════════════════════════════════════════════════════════════════════════

def make_summary_grid(results: list, out_dir: str):
    """
    results: list of (raw_rgb, overlay_dr, overlay_dme, pred_dr, pred_dme, img_id)
    """
    n   = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = axes[np.newaxis]
    fig.patch.set_facecolor("#1a1a2e")
    for row, (raw, ov_dr, ov_dme, pd, pm, iid) in enumerate(results):
        for ax in axes[row]:
            ax.axis("off"); ax.set_facecolor("#1a1a2e")
        axes[row, 0].imshow(raw)
        axes[row, 0].set_title(iid, color="white", fontsize=9)
        axes[row, 1].imshow(ov_dr)
        axes[row, 1].set_title(f"DR: {DR_LABELS[pd]}",  color="lightyellow", fontsize=9)
        axes[row, 2].imshow(ov_dme)
        axes[row, 2].set_title(f"DME: {DME_LABELS[pm]}", color="lightcyan",   fontsize=9)

    col_titles = ["Original", "DR Grad-CAM", "DME Grad-CAM"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title + "\n" + axes[0, col].get_title(),
                               color="white", fontsize=10, fontweight="bold")

    plt.suptitle("DR-ASPP-DRN  |  Grad-CAM Summary Grid",
                 color="white", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "gradcam_summary_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Summary grid saved: {grid_path}")


# ════════════════════════════════════════════════════════════════════════════
# Demo mode (no real images or trained model required)
# ════════════════════════════════════════════════════════════════════════════

def _demo_mode(out_dir: str, n: int = 4):
    """
    Run Grad-CAM on a randomly-initialised model with synthetic 'fundus' images.
    Useful for testing the pipeline before training is complete.
    """
    print("  ⚠  Demo mode: random model + synthetic images.")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from model import build_model, ResizeToMatch
        model = build_model(input_shape=(512, 512, 3), backbone_weights=None)
    except ImportError:
        # Minimal model if repo not on path
        print("  model.py not found — building minimal placeholder.")
        inp = keras.Input((512, 512, 3), name="input_image")
        x   = keras.layers.Conv2D(64, 3, padding="same", name="aspp_proj_relu")(inp)
        x   = keras.layers.GlobalAveragePooling2D()(x)
        dr  = keras.layers.Dense(5,  activation="softmax", name="dr_output")(x)
        dme = keras.layers.Dense(3,  activation="softmax", name="dme_risk")(x)
        model = keras.Model(inp, {"dr_output": dr, "dme_risk": dme})

    layer_name = _find_target_layer(model)
    print(f"  Target layer: {layer_name}")
    os.makedirs(out_dir, exist_ok=True)

    # synthetic retinal-looking images
    rng_demo = np.random.default_rng(0)
    grid_results = []
    for i in range(n):
        # green-tinted circular background
        H = W = 512
        raw = np.zeros((H, W, 3), np.uint8)
        cv2.circle(raw, (W // 2, H // 2), 220,
                   (int(rng_demo.integers(30, 80)),
                    int(rng_demo.integers(100, 180)),
                    int(rng_demo.integers(20, 60))), -1)
        # add random vessel-like lines
        for _ in range(rng_demo.integers(5, 15)):
            pt1 = (int(rng_demo.integers(100, 412)), int(rng_demo.integers(100, 412)))
            pt2 = (int(rng_demo.integers(100, 412)), int(rng_demo.integers(100, 412)))
            cv2.line(raw, pt1, pt2,
                     (int(rng_demo.integers(150, 220)), 80, 80), 2)

        # normalise for model
        tensor = (raw.astype(np.float32) / 255.0 - IDRID_MEAN) / IDRID_STD
        tensor = tensor[np.newaxis]

        preds = model(tf.cast(tensor, tf.float32), training=False)
        if isinstance(preds, dict):
            pred_dr  = int(np.argmax(preds["dr_output"].numpy()[0]))
            pred_dme = int(np.argmax(preds["dme_risk"].numpy()[0]))
        else:
            pred_dr = pred_dme = 0

        try:
            cam_dr  = gradcam_heatmap(model, tensor, "dr_output",  pred_dr,  layer_name)
            cam_dme = gradcam_heatmap(model, tensor, "dme_risk",   pred_dme, layer_name)
        except Exception as e:
            print(f"  Grad-CAM skipped: {e}")
            cam_dr = cam_dme = np.zeros((32, 32), np.float32)

        ov_dr  = overlay_heatmap(raw, cam_dr)
        ov_dme = overlay_heatmap(raw, cam_dme)
        img_id = f"synthetic_{i+1:02d}"
        grid_results.append((raw, ov_dr, ov_dme, pred_dr, pred_dme, img_id))

        # individual panel
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes: ax.axis("off")
        axes[0].imshow(raw);    axes[0].set_title("Synthetic Fundus", color="white")
        axes[1].imshow(ov_dr);  axes[1].set_title(f"DR Grad-CAM\n{DR_LABELS[pred_dr]}", color="lightyellow")
        axes[2].imshow(ov_dme); axes[2].set_title(f"DME Grad-CAM\n{DME_LABELS[pred_dme]}", color="lightcyan")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"gradcam_{img_id}.png"), dpi=120,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Panel {i+1}/{n} done.")

    make_summary_grid(grid_results, out_dir)
    print("\n  ✅ Demo Grad-CAM complete.\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for DR-ASPP-DRN")
    parser.add_argument("--model",   default=None)
    parser.add_argument("--csv",     default=None)
    parser.add_argument("--imgdir",  default=None)
    parser.add_argument("--out",     default="gradcam_outputs")
    parser.add_argument("--n",       type=int, default=8,
                        help="Number of images to visualise")
    parser.add_argument("--demo",    action="store_true")
    args = parser.parse_args()

    if args.demo or args.model is None:
        _demo_mode(args.out, n=args.n)
        return

    # ── real model path ───────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from model import ResizeToMatch
        custom = {"ResizeToMatch": ResizeToMatch}
    except ImportError:
        custom = {}

    model = keras.models.load_model(args.model, compile=False,
                                    custom_objects=custom)
    layer_name = _find_target_layer(model)
    print(f"  Target Grad-CAM layer: {layer_name}")
    os.makedirs(args.out, exist_ok=True)

    # ── load image paths & labels ─────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(args.csv)
    # Support common IDRiD column naming
    img_col = next((c for c in df.columns if "image" in c.lower()), df.columns[0])
    dr_col  = next((c for c in df.columns if "retinopathy" in c.lower() or c == "DR_grade"), None)
    dme_col = next((c for c in df.columns if "macular" in c.lower() or c == "DME_grade"), None)
    df = df.head(args.n)

    grid_results = []
    for _, row in df.iterrows():
        img_id = str(row[img_col]).replace(".jpg", "").replace(".JPG", "")
        # try common extensions
        img_path = None
        for ext in [".jpg", ".JPG", ".png", ".PNG"]:
            p = Path(args.imgdir) / (img_id + ext)
            if p.exists():
                img_path = p; break
        if img_path is None:
            print(f"  ⚠  Image not found: {img_id}")
            continue

        true_dr  = int(row[dr_col])  if dr_col  else None
        true_dme = int(row[dme_col]) if dme_col else None

        visualise_single(model, str(img_path), layer_name, args.out,
                         img_id, true_dr, true_dme)

        # collect for grid
        raw = _load_raw_rgb(str(img_path))
        tensor = preprocess_image(str(img_path))
        preds  = model(tf.cast(tensor, tf.float32), training=False)
        if isinstance(preds, dict):
            pd_dr  = int(np.argmax(preds["dr_output"].numpy()[0]))
            pd_dme = int(np.argmax(preds["dme_risk"].numpy()[0]))
        else:
            pd_dr = pd_dme = 0
        try:
            c_dr  = gradcam_heatmap(model, tensor, "dr_output",  pd_dr,  layer_name)
            c_dme = gradcam_heatmap(model, tensor, "dme_risk",   pd_dme, layer_name)
            ov_dr  = overlay_heatmap(raw, c_dr)
            ov_dme = overlay_heatmap(raw, c_dme)
            grid_results.append((raw, ov_dr, ov_dme, pd_dr, pd_dme, img_id))
        except Exception:
            pass

    if grid_results:
        make_summary_grid(grid_results, args.out)
    print("\n  ✅ Grad-CAM visualisation complete.\n")


if __name__ == "__main__":
    main()