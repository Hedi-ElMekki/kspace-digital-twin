"""Run every validation check and write VALIDATION.md (+ figures/radial_profile.png).

Run from the project root: python scripts/validate.py
The M4Raw rows need data/m4raw/2022061203_T101.h5 and _T201.h5 (see README).
"""

from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from twin.artifacts import add_noise, undersample_ky  # noqa: E402
from twin.contrast import TISSUES, ernst_angle, gre_signal, param_maps, se_signal  # noqa: E402
from twin.experiments import grating, radial_profile, split_low_high  # noqa: E402
from twin.kspace import dft2_explicit, fft2c, hermitian_residual, ifft2c  # noqa: E402
from twin.kspace_io import load_kspace, reconstruct  # noqa: E402
from twin.metrics import brain_mask, nrmse, ssim  # noqa: E402
from twin.sequences import multi_echo, scan_time, t2_map, tse_kspace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
labels = np.load(ROOT / "data/derived/brainweb_z200_labels_256.npy")
maps = param_maps(labels)
t1w = se_signal(maps["pd"], maps["t1"], maps["t2"], 500, 15)
rows = []


def check(name, expected, measured, ok):
    rows.append((name, expected, measured, "✅" if ok else "❌"))


def report(name, measured, comment):
    """A measurement with no pass/fail threshold, reported as is."""
    rows.append((name, comment, measured, "ℹ️"))


def se(tissue, tr, te):
    t = TISSUES[tissue]
    return se_signal(t["pd"], t["t1"], t["t2"], tr, te)


# k-space conventions
err = np.abs(ifft2c(fft2c(t1w)) - t1w).max()
check("Round trip image → k → image (brain 256²)", "≈ 1e-15", f"{err:.1e}", err < 1e-12)
small = np.random.default_rng(0).random((32, 32))
err = np.abs(dft2_explicit(small) - fft2c(small)).max()
check("FFT = explicit signal sum (32²)", "≈ 1e-14", f"{err:.1e}", err < 1e-10)
res = hermitian_residual(fft2c(t1w))
check("Hermitian symmetry, twin (magnitude image)", "≈ 1e-16", f"{res:.1e}", res < 1e-12)
K = np.abs(fft2c(grating(256, 8)))
dots = sorted(int(c) - 128 for c in np.nonzero(K[128] > 1e-9)[0])
check("8-cycle grating → dots at kx", "−8, 0, +8", ", ".join(f"{d:+d}" if d else "0" for d in dots), dots == [-8, 0, 8])
low, high = split_low_high(t1w, 16)
err = np.abs(low + high - t1w).max()
check("Low-pass + high-pass = image", "≈ 1e-15", f"{err:.1e}", err < 1e-12)

# contrast
s = [se(n, 500, 15) for n in ("white matter", "grey matter", "CSF")]
check("T1w (500/15): WM > GM > CSF", "decreasing", " > ".join(f"{v:.3f}" for v in s), s[0] > s[1] > s[2])
s = [se(n, 2000, 120) for n in ("CSF", "grey matter", "white matter")]
check("T2w (2000/120): CSF > GM > WM", "decreasing", " > ".join(f"{v:.3f}" for v in s), s[0] > s[1] > s[2])
e = ernst_angle(30, 785)
angles = np.linspace(0.01, 90, 90001)
best = angles[np.argmax(gre_signal(1, 785, 65, 30, 5, angles))]
check("Ernst angle WM, TR 30 ms", f"numerical max {best:.2f}°", f"{e:.2f}°", abs(best - e) < 0.01)

# sequences
tes = np.arange(20, 161, 20)
echoes = multi_echo(maps, 3000, tes)
mask = brain_mask(echoes[0]) & np.isin(labels, [1, 2, 3])
t2 = t2_map(echoes, tes, mask)
err = np.nanmax(np.abs(t2[mask] - maps["t2"][mask]) / maps["t2"][mask])
check("T2 map, noiseless (CSF/GM/WM)", "rel. error ≤ 1e-9", f"{err:.1e}", err < 1e-9)
se_k = fft2c(se_signal(maps["pd"], maps["t1"], maps["t2"], 3000, 20))
err = np.abs(tse_kspace(maps, 3000, 1, 20, 20) - se_k).max()
check("TSE with ETL 1 = spin echo", "≈ 0", f"{err:.1e}", err < 1e-12)
check("Scan time SE 500 ms × 256", "128 s", f"{scan_time(500, 256):.0f} s", scan_time(500, 256) == 128)
ref_t2w = se_signal(maps["pd"], maps["t1"], maps["t2"], 3000, 100)
tse16 = np.abs(ifft2c(tse_kspace(maps, 3000, 16, 10, 100)))
report("TSE ETL 16 vs SE (TE 100), 16× faster", f"SSIM {ssim(tse16, ref_t2w):.3f}, NRMSE {nrmse(tse16, ref_t2w):.3f}",
       "same contrast, some lost detail (no threshold set)")

# artefacts
sigma = 0.05
air = np.abs(ifft2c(add_noise(np.zeros((256, 256), dtype=complex), sigma, rng=1)))
check("Noise in air: mean / σ", "√(π/2) = 1.253", f"{air.mean() / sigma:.3f}", abs(air.mean() / sigma - 1.2533) < 0.02)
check("Noise in air: std / σ", "0.655", f"{air.std() / sigma:.3f}", abs(air.std() / sigma - 0.655) < 0.02)
folded = ifft2c(undersample_ky(fft2c(t1w), 2))
err = np.abs(folded - 0.5 * (t1w + np.roll(t1w, 128, axis=0))).max()
check("Aliasing R = 2 = ½(I + I shifted N/2)", "≈ 0", f"{err:.1e}", err < 1e-12)

# real data
m4raw = {c: ROOT / f"data/m4raw/2022061203_{c}01.h5" for c in ("T1", "T2")}
if all(p.exists() for p in m4raw.values()):
    for contrast, path in m4raw.items():
        with h5py.File(path) as f:
            theirs = f["reconstruction_rss"][f["kspace"].shape[0] // 2]
        k = load_kspace(path)
        err = nrmse(reconstruct(k), theirs)
        check(f"M4Raw {contrast}: our reconstruction vs the dataset’s", "NRMSE ≈ 0", f"{err:.1e}", err < 1e-6)
    res = hermitian_residual(load_kspace(m4raw["T2"])[0])
    check("Hermitian symmetry, real k-space (M4Raw T2, coil 1)", "broken (real phase)", f"{res:.2f}", res > 0.5)
else:
    print("M4Raw files missing: real-data rows skipped")

# figure: radial energy profile, twin vs real
fig, ax = plt.subplots(figsize=(7, 4.5))
twin_profile = radial_profile(fft2c(t1w))
ax.semilogy(twin_profile / twin_profile.sum(), label="twin (BrainWeb, T1w 500/15)", lw=2)
if all(p.exists() for p in m4raw.values()):
    real = load_kspace(m4raw["T1"])
    real_profile = np.mean([radial_profile(c) for c in real], axis=0)
    report("Real k-space: radius of the energy peak (M4Raw T1)", f"{int(np.argmax(real_profile))} steps",
           "twin: 0. Real: peak not at k = 0, probably an off-centre echo (a phase ramp); not investigated")
    floor = np.median(real_profile[100:150]) / real_profile.sum()
    report("Real k-space: energy share per ring at radius 100–150", f"{floor:.1e}",
           f"twin: {np.median(twin_profile[100:150]) / twin_profile.sum():.1e}; the real floor is probably mostly noise")
    ax.semilogy(real_profile / real_profile.sum(), label="real (M4Raw T1, 4 coils)", lw=2)
ax.set_xlabel("radius in k-space (grid steps)")
ax.set_ylabel("share of the total energy")
ax.set_title("Where the energy sits in k-space")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "figures/radial_profile.png", dpi=110)

table = "\n".join(f"| {n} | {e} | {m} | {ok} |" for n, e, m, ok in rows)
passed = sum(r[3] == "✅" for r in rows)
checked = sum(r[3] != "ℹ️" for r in rows)
(ROOT / "VALIDATION.md").write_text(f"""# Validation

Generated by `scripts/validate.py`. {passed} / {checked} checks pass; ℹ️ rows are measurements without a threshold.

| Check | Expected | Measured | |
|---|---|---|---|
{table}

![Radial k-space energy profile](figures/radial_profile.png)

## Known limitations

- **Simulated contrast**: the anatomy is real (BrainWeb), but each tissue is a flat value from the signal
  equation: no partial volume, no texture, no noise unless added.
- **Magnitude only**: the twin's images have no phase, so its k-space is perfectly Hermitian; real k-space is not.
- **Parameter values**: T1 of WM, GM, CSF, fat and T2 of WM, GM, CSF are course values at 1.5 T. PD, T2*
  and all other tissues are placeholders. WM and GM share PD = 0.8 on purpose.
- **Idealised sequences**: TE ≪ TR, perfect RF pulses, no stimulated echoes in TSE, perfect spoiling in GRE.
- **Real data**: M4Raw is 0.3 T, not 1.5 T, and a different subject: it validates reconstruction, not contrast values.
- **NRMSE** under-rates blur; SSIM is reported where detail matters.
""")
print(f"{passed}/{checked} checks pass -> VALIDATION.md")
