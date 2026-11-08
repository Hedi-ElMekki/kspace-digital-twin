import numpy as np
import pytest

from twin.experiments import grating, k_axis, split_low_high
from twin.metrics import nrmse
from twin.kspace import fft2c


def test_horizontal_grating_gives_three_dots():
    K = np.abs(fft2c(grating(n=64, cycles=5, axis=0)))
    expected = np.zeros((64, 64))
    expected[32, 32] = 0.5 * 64        # average brightness at k = 0
    expected[32 - 5, 32] = 0.25 * 64   # stripes change along rows -> dots on the ky axis,
    expected[32 + 5, 32] = 0.25 * 64   # 5 cycles -> 5 steps from the centre, plus the mirror
    np.testing.assert_allclose(K, expected, atol=1e-9)


def test_low_plus_high_gives_back_the_image():
    image = np.random.default_rng(0).random((32, 32))
    low, high = split_low_high(image, radius=5)
    np.testing.assert_allclose(low + high, image, atol=1e-12)


def test_keeping_more_of_the_centre_lowers_the_error():
    image = np.random.default_rng(1).random((32, 32))
    errors = [nrmse(split_low_high(image, r)[0], image) for r in range(0, 26, 2)]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] == pytest.approx(0, abs=1e-12)  # radius 24 > corner distance 22.6: all of k-space


def test_k_axis_spacing_is_one_over_fov():
    k = k_axis(256, fov_mm=240)
    np.testing.assert_allclose(np.diff(k), 1 / 240)
    pixel_mm = 240 / 256
    assert -k[0] == pytest.approx(1 / (2 * pixel_mm))  # edge of k-space = finest stripe


def test_grating_dot_sits_at_cycles_over_fov():
    K = np.abs(fft2c(grating(n=256, cycles=8)))
    k = k_axis(256, fov_mm=240)
    assert k[np.argmax(K[128, 129:]) + 129] == pytest.approx(8 / 240)
