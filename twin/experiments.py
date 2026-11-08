"""Course experiments on k-space: gratings, low/high-pass filtering."""

import numpy as np

from twin.kspace import fft2c, ifft2c


def grating(n=256, cycles=8, axis=1):
    """n x n image of one stripe pattern: brightness 0.5 + 0.5 cos, between 0 and 1.

    cycles = number of bright/dark periods across the image. axis=1 makes the
    brightness change along columns (vertical stripes), axis=0 along rows.
    """
    position = np.arange(n) - n // 2
    profile = 0.5 + 0.5 * np.cos(2 * np.pi * cycles * position / n)
    image = np.tile(profile, (n, 1))
    return image if axis == 1 else image.T


def lowpass_mask(shape, radius):
    """Boolean disc in centred k-space: True within `radius` grid steps of k = 0."""
    ky = np.arange(shape[0]) - shape[0] // 2
    kx = np.arange(shape[1]) - shape[1] // 2
    return np.hypot(ky[:, None], kx[None, :]) <= radius


def split_low_high(image, radius):
    """Split an image into its low-frequency part (centre of k-space, inside
    `radius`) and its high-frequency part (everything else). low + high = image.
    """
    kspace = fft2c(image)
    low = lowpass_mask(kspace.shape, radius)
    return ifft2c(kspace * low), ifft2c(kspace * ~low)


def k_axis(n, fov_mm):
    """k value of each index along one centred k-space axis, in cycles per mm.

    Grid spacing dk = 1 / FOV: the slowest stripe that is not flat makes exactly
    one cycle across the field of view. The edge reaches 1 / (2 * pixel size).
    """
    return (np.arange(n) - n // 2) / fov_mm


def radial_profile(kspace):
    """Mean energy |k|^2 on rings of integer radius around k = 0 (index = radius)."""
    ky = np.arange(kspace.shape[0]) - kspace.shape[0] // 2
    kx = np.arange(kspace.shape[1]) - kspace.shape[1] // 2
    radius = np.rint(np.hypot(ky[:, None], kx[None, :])).astype(int)
    energy = np.bincount(radius.ravel(), weights=np.abs(kspace.ravel()) ** 2)
    return energy / np.bincount(radius.ravel())
