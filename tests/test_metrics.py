import numpy as np
import pytest

from twin.metrics import brain_mask, nrmse, psnr, ssim


def test_nrmse_worked_examples():
    ref = np.array([6.0, 8.0])
    assert nrmse(ref, ref) == pytest.approx(0)
    assert nrmse(np.array([6.0, 0.0]), ref) == pytest.approx(0.8)
    assert nrmse(np.array([3.0, 4.0]), ref) == pytest.approx(0.5)
    assert nrmse(np.zeros(2), ref) == pytest.approx(1)


def test_nrmse_refuses_an_all_zero_reference():
    with pytest.raises(ValueError):
        nrmse(np.ones(2), np.zeros(2))


def test_nrmse_mask_ignores_pixels_outside():
    ref = np.array([6.0, 8.0, 1.0])
    test = np.array([6.0, 8.0, 100.0])
    mask = np.array([True, True, False])
    assert nrmse(test, ref, mask) == pytest.approx(0)


def test_brain_mask_finds_a_disc():
    y, x = np.mgrid[:64, :64] - 32
    disc = np.hypot(x, y) < 20
    image = disc * 1.0 + 0.01 * np.random.default_rng(0).random((64, 64))
    np.testing.assert_array_equal(brain_mask(image), disc)


def test_psnr_and_ssim_of_identical_images():
    image = np.random.default_rng(0).random((32, 32))
    assert psnr(image, image) == np.inf
    assert ssim(image, image) == pytest.approx(1)


def test_psnr_known_value():
    ref = np.array([[1.0, 0.0], [0.0, 0.0]])
    test = ref + 0.1  # RMS error 0.1, peak 1 -> 20 dB
    assert psnr(test, ref) == pytest.approx(20)


def test_ssim_sees_blur_that_nrmse_underrates():
    from twin.experiments import split_low_high
    y, x = np.mgrid[:64, :64] - 32
    image = ((np.hypot(x, y) < 20) * 1.0) + 0.5 * (np.hypot(x - 5, y) < 3)
    blurred = np.abs(split_low_high(image, 6)[0])
    assert ssim(blurred, image) < 0.9
