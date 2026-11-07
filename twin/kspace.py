"""Centred FFT conventions for the k-space twin.

Convention: image -> k-space uses e^{-i2pi(kx x + ky y)} (the signal equation
S(k) = sum rho(x, y) e^{-i2pi(kx x + ky y)}), k-space -> image uses e^{+i...}.
k = 0 sits at the array centre, index N//2.
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


def dft2_explicit(image):
    """Image -> centred k-space by the literal sum S(k) = sum rho(x, y) e^{-i2pi(kx x + ky y)}.

    Slow on purpose (N^4 operations): it shows that fft2c computes exactly this
    physical sum. Positions x, y are in pixels with 0 at the centre; frequencies
    kx, ky are in cycles per pixel, from -1/2 to 1/2 - 1/N. Use N <= 64.
    """
    ny, nx = image.shape
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    ky = y / ny
    kx = x / nx
    # phase[ky, kx, y, x] = -2pi (kx x + ky y): one term of the sum per (k, pixel) pair
    phase = -2j * np.pi * (kx[None, :, None, None] * x[None, None, None, :]
                           + ky[:, None, None, None] * y[None, None, :, None])
    return np.sum(image[None, None, :, :] * np.exp(phase), axis=(2, 3)) / np.sqrt(nx * ny)


def log_mag(kspace):
    """Display helper: log(1 + |k|), so the bright centre does not hide the rest."""
    return np.log1p(np.abs(kspace))


def ifft2c(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho"))