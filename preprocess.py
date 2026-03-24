"""
Medical-aware fundus image preprocessing for Diabetic Retinopathy / DME detection.

Pipeline:
  1. Border cropping  – remove black acquisition borders
  2. CLAHE            – enhance vessel/lesion contrast on green channel
  3. Resize           – standardise to target_size (512×512 by default)
  4. Normalise        – scale to [0, 1]
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def crop_black_borders(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Remove dark border artifacts from a fundus image.

    Parameters
    ----------
    image:
        BGR or RGB image as a uint8 numpy array.
    threshold:
        Pixel intensity below which a border pixel is considered black.

    Returns
    -------
    np.ndarray
        Cropped image with black borders removed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the green channel of a BGR/RGB fundus image.

    The green channel has the best contrast for retinal vessels and lesions.
    CLAHE is applied only to the green channel; the result is merged back.

    Parameters
    ----------
    image:
        BGR uint8 numpy array.
    clip_limit:
        CLAHE contrast limit.
    grid_size:
        Tile grid size for CLAHE.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced BGR uint8 array.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    b, g, r = cv2.split(image)
    g_eq = clahe.apply(g)
    return cv2.merge([b, g_eq, r])


def preprocess_fundus_image(
    image_path: str,
    target_size: tuple = (512, 512),
) -> np.ndarray:
    """Load and preprocess a single fundus image from disk.

    Steps: load → border crop → CLAHE → resize → normalise [0,1].

    Parameters
    ----------
    image_path:
        Absolute path to the image file.
    target_size:
        (height, width) to resize to.

    Returns
    -------
    np.ndarray
        Float32 array of shape (*target_size, 3) in [0, 1].

    Raises
    ------
    ValueError
        If the image cannot be loaded from *image_path*.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img = crop_black_borders(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (target_size[1], target_size[0]),
                     interpolation=cv2.INTER_AREA)
    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def batch_preprocess(
    image_array: np.ndarray,
    target_size: tuple = (512, 512),
) -> np.ndarray:
    """Preprocess a batch of already-loaded images (uint8 BGR or RGB).

    Parameters
    ----------
    image_array:
        Array of shape (N, H, W, 3), dtype uint8.
    target_size:
        Target (height, width) for each image.

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, *target_size, 3) in [0, 1].
    """
    out = np.empty(
        (len(image_array), target_size[0], target_size[1], 3),
        dtype=np.float32,
    )
    for i, img in enumerate(image_array):
        # Ensure uint8
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = crop_black_borders(img)
        img = apply_clahe(img)
        img = cv2.resize(img, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_AREA)
        # If the input was RGB convert to BGR for apply_clahe then back, but
        # since we already applied CLAHE above using BGR convention, just
        # ensure output is RGB.
        # apply_clahe expects and returns BGR; convert to RGB for model input
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out[i] = img.astype(np.float32) / 255.0
    return out
