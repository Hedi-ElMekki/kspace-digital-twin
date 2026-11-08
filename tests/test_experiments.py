import numpy as np

from twin.experiments import grating
from twin.kspace import fft2c


def test_horizontal_grating_gives_three_dots():
    K = np.abs(fft2c(grating(n=64, cycles=5, axis=0)))
    expected = np.zeros((64, 64))
    expected[32, 32] = 0.5 * 64        # average brightness at k = 0
    expected[32 - 5, 32] = 0.25 * 64   # stripes change along rows -> dots on the ky axis,
    expected[32 + 5, 32] = 0.25 * 64   # 5 cycles -> 5 steps from the centre, plus the mirror
    np.testing.assert_allclose(K, expected, atol=1e-9)
