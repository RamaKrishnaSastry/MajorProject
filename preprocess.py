"""
preprocess.py - Medical image preprocessing for fundus images.

Implements:
- Content-aware border cropping
- CLAHE on luminance only
- Resizing to model input size
- Normalization to [-1, 1]
- Integration with Keras / tf.data preprocessing
"""

import cv2
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Pure-NumPy / OpenCV helpers (used inside tf.py_function wrappers)
# ---------------------------------------------------------------------------

def crop_black_borders(image: np.ndarray, border_fraction: float = 0.05) -> np.ndarray:
    """Crop to the detected fundus content with a small fallback margin.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    border_fraction : float
        Fallback fraction of each side to crop if content detection fails.

    Returns
    -------
    np.ndarray
        Cropped image of the same dtype.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 0 and h > 0:
            return image[y : y + h, x : x + w]

    h, w = image.shape[:2]
    crop_h = int(h * border_fraction)
    crop_w = int(w * border_fraction)
    return image[crop_h : h - crop_h, crop_w : w - crop_w]


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to luminance only to preserve color relationships."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] to match ResNet50 preprocess_input expectations."""
    image = image.astype(np.float32)
    return image / 127.5 - 1.0


def resize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """Resize image to *target_size* using bicubic interpolation."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


def preprocess_image(
    image: np.ndarray,
    target_size: tuple = (512, 512),
    border_fraction: float = 0.05,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Full preprocessing pipeline for a single fundus image."""
    image = crop_black_borders(image, border_fraction)
    image = apply_clahe(image, clip_limit, grid_size)
    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image


def load_and_preprocess(
    image_path: str,
    target_size: tuple = (512, 512),
    border_fraction: float = 0.05,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Load an image from disk and run the full preprocessing pipeline."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return preprocess_image(image, target_size, border_fraction, clip_limit, grid_size)


# ---------------------------------------------------------------------------
# TensorFlow / tf.data integration
# ---------------------------------------------------------------------------

def _preprocess_tf_wrapper(
    image_path: tf.Tensor,
    target_size: tuple,
    border_fraction: float,
    clip_limit: float,
    grid_size: int,
) -> tf.Tensor:
    """Wrap *load_and_preprocess* for use inside tf.py_function."""

    def _fn(path):
        img = load_and_preprocess(
            path.numpy().decode("utf-8"),
            target_size=target_size,
            border_fraction=border_fraction,
            clip_limit=clip_limit,
            grid_size=grid_size,
        )
        return img.astype(np.float32)

    result = tf.py_function(_fn, [image_path], tf.float32)
    result.set_shape((*target_size, 3))
    return result


def make_preprocess_fn(
    target_size: tuple = (512, 512),
    border_fraction: float = 0.05,
    clip_limit: float = 2.0,
    grid_size: int = 8,
):
    """Return a preprocessing function suitable for tf.data.Dataset.map."""

    def preprocess_fn(image_path: tf.Tensor) -> tf.Tensor:
        return _preprocess_tf_wrapper(
            image_path, target_size, border_fraction, clip_limit, grid_size
        )

    return preprocess_fn
