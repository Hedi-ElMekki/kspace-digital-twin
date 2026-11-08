"""Acquisition: the scanner fills k-space one phase-encode row (ky) per TR."""

import numpy as np

from twin.contrast import se_signal
from twin.kspace import fft2c


def row_order(n, kind="linear"):
    """Row indices of an n-row k-space, in the order the scanner acquires them.

    linear:  top to bottom, ky = -n/2 ... n/2 - 1 (k = 0 comes halfway through).
    centric: centre out, ky = 0, +1, -1, +2, -2, ... (k = 0 comes first).
    """
    ky = np.arange(n) - n // 2
    if kind == "linear":
        return np.arange(n)
    if kind == "centric":
        return np.argsort(2 * np.abs(ky) - (ky > 0), kind="stable")
    raise ValueError(f"unknown order {kind!r}")


def acquire(image, kind="linear", n_rows=None):
    """k-space as the scanner has it after the first n_rows TRs (all rows if None).

    Rows not acquired yet are zero.
    """
    full = fft2c(image)
    order = row_order(full.shape[0], kind)
    acquired = np.zeros_like(full)
    rows = order[:n_rows]
    acquired[rows] = full[rows]
    return acquired


def scan_time(tr_ms, n_rows, averages=1):
    """Spin-echo scan time in seconds: one row per TR."""
    return tr_ms * n_rows * averages / 1000


def multi_echo(maps, tr, tes):
    """Spin-echo images at several echo times after one excitation per TR.

    maps: dict with pd, t1, t2 maps (from contrast.param_maps). Returns an
    array of shape (len(tes), rows, cols): one image per echo time.
    """
    tes = np.asarray(tes, dtype=float)[:, None, None]
    return se_signal(maps["pd"], maps["t1"], maps["t2"], tr, tes)


def t2_map(echoes, tes, mask):
    """T2 in ms for each pixel, from the decay S(TE) = S0 e^{-TE/T2}.

    Taking the log makes it a straight line, ln S = ln S0 - TE / T2, so the
    slope of a least-squares line through (TE, ln S) is -1/T2. Pixels outside
    mask (air, no signal) are NaN.
    """
    tes = np.asarray(tes, dtype=float)
    if np.ptp(tes) == 0:
        raise ValueError("need at least two different echo times")
    if np.any(echoes[:, mask] <= 0):
        raise ValueError("mask contains pixels with no signal: exclude them")
    log_s = np.log(echoes[:, mask])                  # (n_echoes, n_pixels in mask)
    te_c = tes - tes.mean()
    slope = (te_c @ (log_s - log_s.mean(axis=0))) / (te_c @ te_c)
    t2 = np.full(mask.shape, np.nan)
    t2[mask] = -1 / slope
    return t2


def tse_te_per_row(n, etl, esp, te_eff):
    """Echo time of each k-space row in a turbo spin echo.

    Each TR plays a train of etl echoes at TE = esp, 2 esp, ..., etl esp and
    fills one row in each of etl bands of n / etl consecutive rows, so n / etl
    TRs fill k-space. The band holding k = 0 gets echo round(te_eff / esp),
    which sets the contrast. Going out from the centre, bands below it take the
    later echoes and bands above it the earlier ones, so TE changes smoothly;
    when one side runs out, the remaining echoes continue outwards.
    """
    if n % etl:
        raise ValueError(f"{n} rows cannot be split into echo trains of {etl}")
    centre_echo = round(te_eff / esp)
    if not 1 <= centre_echo <= etl:
        raise ValueError(f"te_eff={te_eff} is not one of the {etl} echoes at {esp} ms spacing")
    rows_per_band = n // etl
    centre_band = (n // 2) // rows_per_band
    later = list(range(centre_echo + 1, etl + 1))
    earlier = list(range(centre_echo - 1, 0, -1))
    echo_of_band = {centre_band: centre_echo}
    for d in range(1, etl):
        for band, own, other in ((centre_band + d, later, earlier), (centre_band - d, earlier, later)):
            if 0 <= band < etl:
                echo_of_band[band] = (own or other).pop(0)
    return np.repeat([echo_of_band[b] * esp for b in range(etl)], rows_per_band).astype(float)


def tse_kspace(maps, tr, etl, esp, te_eff):
    """Turbo spin echo k-space: each row is measured at its own echo time."""
    n = maps["pd"].shape[0]
    te_rows = tse_te_per_row(n, etl, esp, te_eff)
    kspace = np.empty(maps["pd"].shape, dtype=complex)
    for te in np.unique(te_rows):
        rows = te_rows == te
        kspace[rows] = fft2c(se_signal(maps["pd"], maps["t1"], maps["t2"], tr, te))[rows]
    return kspace
