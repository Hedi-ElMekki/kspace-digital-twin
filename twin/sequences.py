"""Acquisition: the scanner fills k-space one phase-encode row (ky) per TR."""

import numpy as np

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
