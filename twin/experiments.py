"""Course experiments on k-space: gratings, low/high-pass filtering."""

import numpy as np


def grating(n=256, cycles=8, axis=1):
    """n x n image of one stripe pattern: brightness 0.5 + 0.5 cos, between 0 and 1.

    cycles = number of bright/dark periods across the image. axis=1 makes the
    brightness change along columns (vertical stripes), axis=0 along rows.
    """
    position = np.arange(n) - n // 2
    profile = 0.5 + 0.5 * np.cos(2 * np.pi * cycles * position / n)
    image = np.tile(profile, (n, 1))
    return image if axis == 1 else image.T
