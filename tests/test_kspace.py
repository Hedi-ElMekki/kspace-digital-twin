import numpy as np
import pytest

from twin.kspace import dft2_explicit, fft2c, hermitian_residual, ifft2c


@pytest.mark.parametrize("shape", [(8, 8), (7, 7), (8, 7)])
def test_round_trip_gives_back_the_image(shape):
    image = np.random.default_rng(0).random(shape)
    np.testing.assert_allclose(ifft2c(fft2c(image)), image, atol=1e-12)


def test_fft_equals_the_physical_sum():
    image = np.random.default_rng(1).random((32, 32))
    np.testing.assert_allclose(fft2c(image), dft2_explicit(image), atol=1e-10)


@pytest.mark.parametrize("shape", [(8, 8), (7, 7), (8, 7)])
def test_real_image_gives_hermitian_kspace(shape):
    image = np.random.default_rng(2).random(shape)
    assert hermitian_residual(fft2c(image)) < 1e-12


def test_phase_breaks_hermitian_symmetry():
    image = np.random.default_rng(3).random((16, 16))
    x = np.arange(16)[None, :]
    assert hermitian_residual(fft2c(image * np.exp(0.3j * x))) > 0.1


def test_centre_pixel_gives_flat_kspace():
    image = np.zeros((16,16))
    image[8, 8] = 1
    image_fft = fft2c(image)
    np.testing.assert_allclose(image_fft, 1/16, atol=1e-12)
