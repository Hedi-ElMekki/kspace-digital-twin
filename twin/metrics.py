"""Image comparison helpers."""

import numpy as np
from skimage.metrics import structural_similarity


def nrmse(test, reference, mask=None):
    """Normalised root-mean-square error: ||test - reference|| / ||reference||.

    0 = identical, 1 = as far off as an all-black image. Relative, so it does
    not depend on overall brightness, but a brightness change counts as error.
    With a boolean mask, only the pixels where mask is True are compared.
    """
    test, reference = np.asarray(test), np.asarray(reference)
    if mask is not None:
        test, reference = test[mask], reference[mask]
    reference_size = np.linalg.norm(reference)
    if reference_size == 0:
        raise ValueError("reference is all zeros: NRMSE is undefined")
    return np.linalg.norm(test - reference) / reference_size


def brain_mask(image, threshold=0.05):
    """True where |image| is above threshold * its maximum: tissue vs air."""
    magnitude = np.abs(image)
    return magnitude > threshold * magnitude.max()


def psnr(test, reference):
    """Peak signal-to-noise ratio in dB: 20 log10(max |reference| / RMS error)."""
    test, reference = np.asarray(test), np.asarray(reference)
    rmse = np.sqrt(np.mean(np.abs(test - reference) ** 2))
    return np.inf if rmse == 0 else 20 * np.log10(np.abs(reference).max() / rmse)


def ssim(test, reference):
    """Structural similarity (1 = identical): compares local brightness,
    contrast and structure, so unlike NRMSE it notices lost fine detail.
    """
    reference = np.asarray(reference, dtype=float)
    return structural_similarity(np.asarray(test, dtype=float), reference,
                                 data_range=reference.max() - reference.min())
