"""Centred FFT conventions for the k-space twin.

Convention (course slide 32): image -> k-space uses e^{-i2pi(kx x + ky y)},
k-space -> image uses e^{+i...}. k = 0 sits at the array centre, index N//2.
Axis 0 (rows) = ky = phase encode, axis 1 (columns) = kx = readout.
"""

import numpy as np


def fft2c(image):
    """Image -> centred k-space.

    ifftshift moves the image centre to index [0, 0], where np.fft expects
    the origin; fft2 does the sum; fftshift moves k = 0 back to the centre.
    norm="ortho" scales by 1/sqrt(N*M) so the transform preserves energy.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm="ortho"))


def log_mag(kspace):
    """Display helper: log(1 + |k|), so the bright centre does not hide the rest."""
    return np.log1p(np.abs(kspace))


def ifft2c(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho"))