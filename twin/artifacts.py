"""Imaging artefacts, each made the way a scanner makes it: in k-space."""

import numpy as np

from twin.kspace import fft2c


def add_noise(kspace, sigma, rng=None):
    """Add complex Gaussian noise, std sigma on the real and on the imaginary part.

    fft2c is orthonormal, so the image gets the same sigma per channel. In a
    magnitude image, air then has a Rayleigh distribution: mean sigma sqrt(pi/2),
    std sigma sqrt((4 - pi)/2) ~ 0.655 sigma (not 0: |noise| is never negative).
    """
    rng = np.random.default_rng(rng)
    noise = rng.normal(0, sigma, kspace.shape) + 1j * rng.normal(0, sigma, kspace.shape)
    return kspace + noise


def _check_keep(shape, keep):
    if not 1 <= keep <= min(shape):
        raise ValueError(f"keep={keep} must be between 1 and {min(shape)}")


def truncate(kspace, keep):
    """Keep only the central keep x keep block of k-space (the rest is zero).

    A sharp cut in k-space makes ripples next to sharp edges in the image:
    Gibbs ringing.
    """
    _check_keep(kspace.shape, keep)
    rows, cols = kspace.shape
    out = np.zeros_like(kspace)
    r0, c0 = rows // 2 - keep // 2, cols // 2 - keep // 2
    out[r0:r0 + keep, c0:c0 + keep] = kspace[r0:r0 + keep, c0:c0 + keep]
    return out


def hamming(kspace, keep):
    """Truncate to the central keep x keep block, tapered by a Hamming window.

    The smooth taper removes most of the ringing, at the cost of a little blur.
    """
    _check_keep(kspace.shape, keep)
    rows, cols = kspace.shape
    window = np.zeros(kspace.shape)
    r0, c0 = rows // 2 - keep // 2, cols // 2 - keep // 2
    window[r0:r0 + keep, c0:c0 + keep] = np.outer(np.hamming(keep), np.hamming(keep))
    return kspace * window


def undersample_ky(kspace, factor):
    """Keep one phase-encode row in `factor` (the rows with even offset from
    k = 0 when factor = 2), zero the others: a field of view divided by factor,
    so the image folds onto itself along the phase-encode axis (aliasing).
    """
    if factor < 1:
        raise ValueError("factor must be at least 1")
    rows = kspace.shape[0]
    keep = (np.arange(rows) - rows // 2) % factor == 0
    return kspace * keep[:, None]


def motion(image, shift, rows):
    """k-space of an object that moved by `shift` = (dy, dx) pixels while the
    rows selected by the boolean array `rows` were acquired.

    Each row is measured at a different moment, so the inconsistency is spread
    along the phase-encode axis: ghosts appear along axis 0, whatever the
    direction of the movement.
    """
    still = fft2c(image)
    moved = fft2c(np.roll(image, shift, axis=(0, 1)))
    return np.where(rows[:, None], moved, still)


def motion_rows(n, when):
    """Boolean mask of the rows acquired after the movement.

    centre: the central 10 % of rows; outer: the outer 10 %; alternate: every
    other row (a periodic movement such as breathing).
    """
    offset = np.abs(np.arange(n) - n // 2)
    if when == "centre":
        return offset < 0.05 * n
    if when == "outer":
        return offset >= 0.45 * n
    if when == "alternate":
        return np.arange(n) % 2 == 1
    raise ValueError(f"unknown timing {when!r}")
