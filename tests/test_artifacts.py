import numpy as np
import pytest

from twin.artifacts import add_noise, hamming, motion, motion_rows, truncate, undersample_ky
from twin.kspace import fft2c, ifft2c


def square(n=64, half=8):
    image = np.zeros((n, n))
    image[n // 2 - half:n // 2 + half, n // 2 - half:n // 2 + half] = 1
    return image


def test_noise_in_air_follows_rayleigh():
    sigma = 0.1
    air = np.abs(ifft2c(add_noise(np.zeros((256, 256), dtype=complex), sigma, rng=0)))
    assert air.mean() == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.02)
    assert air.std() == pytest.approx(0.655 * sigma, rel=0.03)


def test_truncation_rings_and_hamming_calms_it():
    image = square()
    sharp = np.abs(ifft2c(truncate(fft2c(image), 24)))
    smooth = np.abs(ifft2c(hamming(fft2c(image), 24)))
    overshoot = lambda im: im.max() - 1
    assert overshoot(sharp) > 0.05          # Gibbs: overshoot near the edges
    assert overshoot(smooth) < overshoot(sharp) / 5


def test_undersampling_by_two_folds_half_a_field_of_view():
    image = np.random.default_rng(0).random((64, 64))
    folded = ifft2c(undersample_ky(fft2c(image), 2))
    expected = 0.5 * (image + np.roll(image, 32, axis=0))
    np.testing.assert_allclose(folded, expected, atol=1e-12)


def test_motion_ghosts_spread_along_phase_encode_only():
    image = square()
    rec = np.abs(ifft2c(motion(image, (0, 3), motion_rows(64, "alternate"))))
    error = np.abs(rec - image)
    object_rows = slice(64 // 2 - 8, 64 // 2 + 8)
    object_cols = slice(64 // 2 - 8 - 3, 64 // 2 + 8 + 3)
    outside_rows = np.delete(error, np.r_[object_rows], axis=0)[:, object_cols]
    outside_cols = np.delete(error, np.r_[object_cols], axis=1)[object_rows, :]
    assert outside_rows.sum() > 10            # ghosts above and below the object
    assert outside_cols.sum() < 1e-9          # nothing to its left or right


def test_no_movement_means_no_motion_artefact():
    image = square()
    np.testing.assert_allclose(motion(image, (0, 0), motion_rows(64, "centre")), fft2c(image))
