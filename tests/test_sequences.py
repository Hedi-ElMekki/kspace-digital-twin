import numpy as np
import pytest

from twin.kspace import fft2c, ifft2c
from twin.metrics import nrmse
from twin.sequences import acquire, row_order, scan_time


@pytest.mark.parametrize("n", [8, 9])
@pytest.mark.parametrize("kind", ["linear", "centric"])
def test_every_row_is_acquired_exactly_once(n, kind):
    assert sorted(row_order(n, kind)) == list(range(n))


def test_centric_starts_at_the_centre_and_goes_out():
    np.testing.assert_array_equal(row_order(8, "centric"), [4, 5, 3, 6, 2, 7, 1, 0])


def test_full_scan_does_not_depend_on_order():
    image = np.random.default_rng(0).random((16, 16))
    for kind in ("linear", "centric"):
        np.testing.assert_allclose(acquire(image, kind), fft2c(image), atol=1e-12)


def test_stopping_early_hurts_linear_much_more_than_centric():
    y, x = np.mgrid[:64, :64] - 32
    image = (np.hypot(x, y) < 20) * 1.0
    errors = {k: nrmse(np.abs(ifft2c(acquire(image, k, n_rows=6))), image) for k in ("linear", "centric")}
    assert errors["centric"] < 0.5 * errors["linear"]


def test_scan_time_is_tr_times_rows():
    assert scan_time(500, 256) == pytest.approx(128)
