"""Make the README figures in figures/ from the twin itself.

Run from the project root: python scripts/make_figures.py
The real-data figure needs the two M4Raw files in data/m4raw/ (see README).
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from twin.artifacts import add_noise, hamming, motion, motion_rows, truncate, undersample_ky  # noqa: E402
from twin.contrast import param_maps, se_signal  # noqa: E402
from twin.experiments import split_low_high  # noqa: E402
from twin.kspace import fft2c, ifft2c, log_mag  # noqa: E402
from twin.kspace_io import load_kspace, reconstruct  # noqa: E402
from twin.metrics import brain_mask, nrmse  # noqa: E402
from twin.sequences import acquire, multi_echo, scan_time, t2_map, tse_kspace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "figure.facecolor": "white"})

labels = np.load(ROOT / "data/derived/brainweb_z200_labels_256.npy")
maps = param_maps(labels)
crop = (slice(28, 228), slice(28, 228))


def se(tr, te):
    return se_signal(maps["pd"], maps["t1"], maps["t2"], tr, te)


def panel(ax, image, title, cmap="gray", **kwargs):
    ax.imshow(image, cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.axis("off")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


t1w, t2w, pdw = se(500, 15), se(2000, 120), se(2000, 15)

# 1. one brain, three contrasts, and what the scanner actually measures
fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
panel(axs[0], t1w[crop], "T1-weighted\nTR 500 ms · TE 15 ms")
panel(axs[1], t2w[crop], "T2-weighted\nTR 2000 ms · TE 120 ms")
panel(axs[2], pdw[crop], "PD-weighted\nTR 2000 ms · TE 15 ms")
panel(axs[3], log_mag(fft2c(t1w))[64:192, 64:192], "what the scanner measures:\nk-space (T1w, zoomed)", "magma")
save(fig, "readme_contrasts.png")

# 2. centre of k-space = contrast, edges = detail
low, high = split_low_high(t1w, 16)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
panel(axs[0], np.abs(low)[crop], "centre of k-space only\n(1 % of the points): contrast")
panel(axs[1], np.abs(high)[crop], "edges of k-space only:\ndetail and borders")
panel(axs[2], t1w[crop], "both together:\nthe full image")
save(fig, "readme_centre_edges.png")

# 3. filling order: the scan is stopped after 10 % of the rows
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, kind, name in zip(axs, ["linear", "centric"], ["top to bottom", "centre outwards"]):
    rec = np.abs(ifft2c(acquire(t1w, kind, 26)))
    panel(ax, rec[crop], f"rows filled {name}\nNRMSE {nrmse(rec, t1w):.2f}", vmin=0, vmax=t1w.max())
fig.suptitle(f"Scan stopped after 10 % of the rows ({scan_time(500, 26):.0f} s of {scan_time(500, 256):.0f} s)")
save(fig, "readme_filling_order.png")

# 4. turbo spin echo: same contrast, much faster
ref = se(3000, 100)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
for ax, etl in zip(axs, [1, 16, 32]):
    rec = ref if etl == 1 else np.abs(ifft2c(tse_kspace(maps, 3000, etl, 10, 100)))
    name = "spin echo" if etl == 1 else f"turbo spin echo, {etl} echoes per TR"
    panel(ax, rec[crop], f"{name}\n{scan_time(3000, 256 // etl):.0f} s", vmin=0, vmax=ref.max())
save(fig, "readme_tse.png")

# 5. multi-echo T2 map
tes = np.arange(20, 161, 20)
echoes = multi_echo(maps, 3000, tes)
mask = brain_mask(echoes[0]) & np.isin(labels, [1, 2, 3])
t2 = t2_map(echoes, tes, mask)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.12]})
panel(axs[0], echoes[0][crop], "echo at TE 20 ms", vmin=0, vmax=echoes[0].max())
panel(axs[1], echoes[-1][crop], "echo at TE 160 ms", vmin=0, vmax=echoes[0].max())
im = axs[2].imshow(np.where(mask, t2, np.nan)[crop], cmap="viridis", vmin=0, vmax=220)
axs[2].set_title("fitted T2 map (ms)")
axs[2].axis("off")
fig.colorbar(im, ax=axs[2], fraction=0.046)
save(fig, "readme_t2_map.png")

# 6. artefacts, each made in k-space
K = fft2c(t1w)
cases = [("noise", add_noise(K, 0.02, 0)), ("Gibbs ringing", truncate(K, 48)),
         ("ringing smoothed (Hamming)", hamming(K, 64)), ("aliasing (1 row in 2)", undersample_ky(K, 2)),
         ("motion during alternate rows", motion(t1w, (0, 3), motion_rows(256, "alternate"))),
         ("motion during central rows", motion(t1w, (0, 3), motion_rows(256, "centre")))]
fig, axs = plt.subplots(2, 3, figsize=(12, 8.6))
for ax, (title, k) in zip(axs.flat, cases):
    panel(ax, np.abs(ifft2c(k)), title, vmin=0, vmax=t1w.max())
save(fig, "readme_artefacts.png")

# 7. real raw data from a 0.3 T scanner (M4Raw)
files = [ROOT / f"data/m4raw/2022061203_{c}01.h5" for c in ("T1", "T2")]
if all(f.exists() for f in files):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    kspace = load_kspace(files[0])
    panel(axs[0], log_mag(kspace[0]), "real k-space (coil 1 of 4)", "magma")
    for ax, f, name in zip(axs[1:], files, ["T1", "T2"]):
        panel(ax, reconstruct(load_kspace(f)), f"our reconstruction, {name}-weighted")
    save(fig, "readme_real_data.png")
else:
    print("M4Raw files missing: real-data figure skipped")
