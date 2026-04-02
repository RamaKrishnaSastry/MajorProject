"""
preprocess.py - Medical image preprocessing for fundus images.

Implements:
- Border cropping (black border removal)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Resizing to model input size
- Normalization to [0, 1]
- Integration with Keras / tf.data preprocessing
"""

import cv2
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Pure-NumPy / OpenCV helpers (used inside tf.py_function wrappers)
# ---------------------------------------------------------------------------

def crop_black_borders(image: np.ndarray, border_fraction: float = 0.10) -> np.ndarray:
    """Remove black circular border common in fundus photographs.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    border_fraction : float
        Fraction of each side to crop.  Default 0.10 (10 %).

    Returns
    -------
    np.ndarray
        Cropped image of the same dtype.
    """
    h, w = image.shape[:2]
    crop_h = int(h * border_fraction)
    crop_w = int(w * border_fraction)
    cropped = image[crop_h : h - crop_h, crop_w : w - crop_w]
    return cropped


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to the GREEN channel only.
    
    Per the DR-ASPP-DRN paper: the green channel provides the best vessel
    contrast for retinal fundus images and is enhanced using CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    # Extract green channel (index 1) — best contrast for retinal vessels
    green = image[:, :, 1]
    enhanced_green = clahe.apply(green)
    result = image.copy()
    result[:, :, 1] = enhanced_green  # replace only green channel
    return result


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] to match ResNet50 preprocess_input expectations.
    
    ResNet50 with ImageNet or EyePACS weights expects input in [-1, 1].
    Using /255 → [0,1] causes the backbone features to activate incorrectly.
    """
    image = image.astype(np.float32)
    image = image / 127.5 - 1.0  # maps [0, 255] → [-1, 1]
    return image

def resize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """Resize image to *target_size* using bicubic interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input image, any dtype.
    target_size : tuple
        (width, height) target dimensions.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


# def normalize_image(image: np.ndarray) -> np.ndarray:
#     """Normalize pixel values to the [0, 1] range.

#     Parameters
#     ----------
#     image : np.ndarray
#         Image array with values in [0, 255].

#     Returns
#     -------
#     np.ndarray
#         Float32 array with values in [0, 1].
#     """
#     return image.astype(np.float32) / 255.0


def preprocess_image(
    image: np.ndarray,
    target_size: tuple = (512, 512),
    border_fraction: float = 0.10,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Full preprocessing pipeline for a single fundus image.

    Steps applied in order:
    1. Crop black borders
    2. Apply CLAHE per channel
    3. Resize to *target_size*
    4. Normalize to [0, 1]

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image.
    target_size : tuple
        (width, height) for resizing.
    border_fraction : float
        Fraction of each side to crop.
    clip_limit : float
        CLAHE clip limit.
    grid_size : int
        CLAHE tile grid size.

    Returns
    -------
    np.ndarray
        Preprocessed float32 image of shape (*target_size, 3).
    """
    image = crop_black_borders(image, border_fraction)
    image = apply_clahe(image, clip_limit, grid_size)
    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image


def load_and_preprocess(
    image_path: str,
    target_size: tuple = (512, 512),
    border_fraction: float = 0.10,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Load an image from disk and run the full preprocessing pipeline.

    Parameters
    ----------
    image_path : str
        Path to the image file (JPEG / PNG / BMP …).
    target_size : tuple
        (width, height) for the output image.
    border_fraction : float
        Fraction of each side to crop.
    clip_limit : float
        CLAHE clip limit.
    grid_size : int
        CLAHE tile grid size.

    Returns
    -------
    np.ndarray
        Preprocessed float32 image of shape (*target_size, 3).

    Raises
    ------
    FileNotFoundError
        If the image cannot be read from *image_path*.
    """
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
    """Wraps *load_and_preprocess* for use inside tf.py_function."""

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
    border_fraction: float = 0.10,
    clip_limit: float = 2.0,
    grid_size: int = 8,
):
    """Return a preprocessing function suitable for use with *tf.data.Dataset.map*.

    Example
    -------
    >>> preprocess_fn = make_preprocess_fn()
    >>> ds = ds.map(lambda path, label: (preprocess_fn(path), label))

    Parameters
    ----------
    target_size : tuple
        Output image size (H, W).
    border_fraction : float
        Fraction of each side to crop.
    clip_limit : float
        CLAHE clip limit.
    grid_size : int
        CLAHE tile grid size.

    Returns
    -------
    callable
        A function ``(image_path_tensor) -> image_tensor``.
    """

    def preprocess_fn(image_path: tf.Tensor) -> tf.Tensor:
        return _preprocess_tf_wrapper(
            image_path, target_size, border_fraction, clip_limit, grid_size
        )

    return preprocess_fn
