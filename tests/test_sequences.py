import numpy as np
import pytest

from twin.kspace import fft2c, ifft2c
from twin.metrics import nrmse
from twin.contrast import param_maps, se_signal
from twin.metrics import brain_mask
from twin.sequences import (
    acquire, multi_echo, row_order, scan_time, t2_map, tse_kspace, tse_te_per_row,
)


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


def test_t2_map_recovers_the_true_t2_without_noise():
    labels = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.uint8)
    maps = param_maps(labels)
    tes = [20, 40, 60, 80]
    echoes = multi_echo(maps, tr=3000, tes=tes)
    mask = brain_mask(echoes[0])
    t2 = t2_map(echoes, tes, mask)
    np.testing.assert_allclose(t2[mask], maps["t2"][mask], rtol=1e-9)
    assert np.isnan(t2[1, 2])  # air


def test_multi_echo_signal_falls_with_te():
    maps = param_maps(np.array([[2]], dtype=np.uint8))
    s = multi_echo(maps, tr=3000, tes=[20, 40, 80])[:, 0, 0]
    assert s[0] > s[1] > s[2]


def test_t2_map_refuses_a_single_echo_time():
    echoes = np.ones((2, 1, 1))
    with pytest.raises(ValueError):
        t2_map(echoes, [30, 30], np.ones((1, 1), dtype=bool))


def brain_like_maps():
    y, x = np.mgrid[:32, :32] - 16
    labels = np.where(np.hypot(x, y) < 12, 2, 0).astype(np.uint8)
    labels[np.hypot(x, y) < 5] = 1
    return param_maps(labels)


def test_tse_with_one_echo_is_spin_echo():
    maps = brain_like_maps()
    se = fft2c(se_signal(maps["pd"], maps["t1"], maps["t2"], 3000, 20))
    np.testing.assert_allclose(tse_kspace(maps, 3000, etl=1, esp=20, te_eff=20), se, atol=1e-12)


def test_tse_centre_row_uses_the_effective_te():
    te = tse_te_per_row(32, etl=8, esp=10, te_eff=60)
    assert te[16] == 60
    assert sorted(set(te)) == [10, 20, 30, 40, 50, 60, 70, 80]


def test_longer_echo_train_blurs_more():
    # single-T2 object: the image is the true one convolved with the PSF of the
    # TE-per-row weighting; measure how much PSF energy leaks off the centre pixel
    def leak(etl):
        weight = np.exp(-tse_te_per_row(64, etl, esp=10, te_eff=10) / 90)
        psf = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(weight))))
        return 1 - psf.max() ** 2 / np.sum(psf ** 2)
    leaks = [leak(etl) for etl in (1, 4, 16, 32)]
    assert leaks[0] == pytest.approx(0, abs=1e-12)
    assert all(later > earlier for earlier, later in zip(leaks, leaks[1:]))


def test_tse_divides_scan_time_by_the_echo_train_length():
    assert scan_time(3000, 256 // 16) == pytest.approx(scan_time(3000, 256) / 16)
