"""
preprocess.py
Medical-aware fundus image preprocessing optimized for DR/DME detection.

Pipeline: Load → Crop Black Borders → CLAHE → Resize → Normalize
"""

import cv2
import numpy as np
from pathlib import Path


def crop_black_borders(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """
    Remove black borders from a fundus image.

    Fundus cameras produce circular images on a black background.
    Cropping to the retinal disc removes acquisition artifacts and
    focuses the model on the actual retinal region, reducing false
    positives from border noise.

    Args:
        image: RGB image as numpy array (H, W, 3), values in [0, 255].
        threshold: Minimum pixel brightness to consider non-black.

    Returns:
        Cropped image containing the retinal region (H', W', 3).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return image[row_min:row_max + 1, col_min:col_max + 1]


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE is the standard contrast-enhancement technique in ophthalmology.
    Applied to the green channel which offers the best vessel/lesion
    contrast in fundus photography. Improves visibility of:
    - Microaneurysms (early DR)
    - Hard exudates (DME marker)
    - Haemorrhages

    Args:
        image: RGB image as numpy array (H, W, 3), values in [0, 255].
        clip_limit: CLAHE clip limit; higher = stronger contrast.
        grid_size: Tile grid size for local histogram computation.

    Returns:
        CLAHE-enhanced RGB image as numpy array (H, W, 3).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus_image(
    image_path: str,
    target_size: tuple = (512, 512),
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single fundus image.

    Steps:
        1. Load image from disk as RGB.
        2. Crop black borders.
        3. Apply CLAHE contrast enhancement.
        4. Resize to target_size.
        5. Normalize to [0, 1].

    Args:
        image_path: Path to the image file (JPEG/PNG).
        target_size: Output spatial dimensions (height, width).

    Returns:
        Preprocessed image as float32 numpy array of shape
        (*target_size, 3) with values in [0, 1].

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_black_borders(image)
    image = apply_clahe(image)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image = image.astype(np.float32) / 255.0
    return image


def batch_preprocess(
    image_array: np.ndarray,
    target_size: tuple = (512, 512),
) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a batch of images.

    Args:
        image_array: Array of uint8 images, shape (N, H, W, 3).
        target_size: Output spatial dimensions (height, width).

    Returns:
        Float32 array of preprocessed images, shape (N, *target_size, 3).
    """
    results = []
    for i, img in enumerate(image_array):
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = crop_black_borders(img)
        img = apply_clahe(img)
        img = cv2.resize(img, (target_size[1], target_size[0]))
        img = img.astype(np.float32) / 255.0
        results.append(img)
        if (i + 1) % 100 == 0:
            print(f"  Preprocessed {i + 1}/{len(image_array)} images")
    return np.array(results, dtype=np.float32)
