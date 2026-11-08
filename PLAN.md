# k-space digital twin — project plan

Status: **plan approved 2026-11-06**. Resume a session with `claude --continue` from this folder.

## 1. Context (confirmed interview answers, 2026-11-06)

| # | Topic | Answer |
|---|---|---|
| 1 | Time | Build until **Sun 2026-11-08** (extended 2026-11-07 by the student: quality over speed; oral still Wed 2026-11-11) |
| 2 | Graded | **Live demo + oral defence** only, on **Wed 2026-11-11**. Notebook = GitHub/LinkedIn portfolio |
| 3 | Language | French for demo/oral; English for notebook; code always English |
| 4 | Env | Local `.venv`, Python 3.14.7: numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, scikit-image 0.26.0, jupyterlab, ipywidgets 8.1.9, streamlit 1.67.0 (smoke-tested OK), pytest, h5py 3.16.0, nibabel 5.4.2, pymupdf 2.1 |
| 5 | Data | Slide 29 image (from `Cours d’IRM-1.pdf`) + BrainWeb `subject04_crisp` (`data/brainweb/`, 362×434×362 uint8, labels 0–11, meanings to confirm visually) |
| 6 | Level | Python 3/3, numpy 1/3, Fourier 2/3, MRI physics 1.75/3 |
| 7 | Scope | Default core (below) + extra feature: complex k-space file → image |
| 8 | Tissue values | T1/T2 from slide 21 (verified). PD + extra BrainWeb tissues = `PLACEHOLDER`. GR = fat (inferred from slide 21 diagram, not stated). CSF T2 = 200 ms is the course value (literature is much longer) |
| 9 | Demo | **Streamlit = main demo** (local first, cloud optional). ipywidgets reduced to a few notebook cells. 256×256 / FOV 240 mm for figures, 128×128 interactive |
| 10 | Extras | None (GRE/Ernst promoted to core because the course covers it) |

Verified from slide 21: T1 at 1.5 T: GR 250, SB 785, SG 920, LCR 3000 ms; T2: SB 90, SG 100, LCR 200 ms. T1w example TR 500 / TE 15; T2w example TR 2000 / TE 120.

Conventions (from guide + slide 32): forward `e^{-i2π(kx·x+ky·y)}`, k = 0 at array centre, `log(1+|k|)` display, 1.5 T, **rows = ky (phase encode), columns = kx (readout)**.

## 2. Scope and time budget (build: ~18 h until Sat night)

| Guide phase | Decision | Budget |
|---|---|---|
| 0 Setup | Reduce: package skeleton, pytest, git, .gitignore | 0.5 h |
| 1 Data | Include: slide image + BrainWeb slice, visual label check | 1.5 h |
| 2 FFT conventions | Include | 1.0 h |
| 3 Experiments | Reduce: exp 1, 2, 5 | 1.0 h |
| 4 Forward model | Reduce: explicit DFT sum = FFT (32×32) | 0.25 h |
| 5 Tissue/contrast | Include (+ T2* placeholders for GRE) | 1.5 h |
| 6 Sequences | SE, multi-echo, TSE, minimal GRE + comparison table | 3.5 h |
| 7 Artefacts | Noise, Gibbs, aliasing, motion | 1.5 h |
| 8 Validation | Reduce (no fastMRI) | 0.75 h |
| 9 Demo | Streamlit app (main) + ipywidgets cells | 2.0 h |
| New | k-space file → image | 0.5 h |
| Portfolio | Notebook + README | 0.5 h |
| Buffer | | 3.5 h |
| **Total build** | | **18 h** |
| Oral prep (FR) | Sun–Tue: money shots, glossary, Q&A, 2 rehearsals | ~2–3 h, outside build budget |

## 3. Milestones

| # | Goal | Built (`twin/…`) | Verification |
|---|---|---|---|
| M1 | FFT conventions | `kspace.fft2c/ifft2c` (centred, norm="ortho"), `log_mag`, `dft2_matrix`, `metrics.nrmse`, `brain_mask` | round trip ~1e-15; DFT = FFT ~1e-12; Hermitian residual with index map `roll(flip(K),1,(0,1))`: ~1e-16 real image vs O(1) with synthetic phase |
| M2 | Real data | `phantom.load_slide_image`, `load_brainweb_slice` → 256×256 `.npy` (read MINC once) | shape/range; label overlay checked by student |
| M3 | Course experiments | `grating`, `lowpass_mask`, high = 1 − low, NRMSE vs radius, Δk = 1/FOV | n cycles/FOV; low+high = full exactly; NRMSE decreasing |
| M4 | Tissue + contrast | `contrast.TISSUES` (source-tagged), `param_maps`, `se_signal`, `gre_signal` | T1w: WM > GM > CSF; T2w 2000/120 CSF > GM depends on placeholder PD → also PD-independent check at TR 4000 |
| M5 | SE filling | `sequences.acquire(order)`, `scan_time` | centric 10 % lines ≪ NRMSE of linear 10 %; 500 ms × 256 = 128 s |
| M6 | Multi-echo + T2 map | `multi_echo`, `t2_map` | noiseless magnitude T2 = truth, rtol 1e-9 in tissue mask |
| M7 | TSE | `tse_kspace(ETL, esp, TE_eff)`: echo round(TE_eff/esp) at ky=0, Ny % ETL == 0, PSF on single-T2 object | ETL=1 == SE when TE = esp = TE_eff; PSF width ↑ with ETL; time ÷ ETL |
| M8 | GRE (minimal) | `gre_signal`, Ernst angle, SE/ME/TSE/GRE table | Ernst = arccos(e^{-TR/T1}) numerically |
| M9 | Artefacts | `artifacts.add_noise` (σ per real/imag channel), `truncate`, `hamming`, `undersample_ky` (zero-filled N×N), `motion(lines=centre/outer)` | air mean ≈ σ√(π/2), std ≈ 0.655σ; ½(I + I shifted N/2) in magnitude; ghosts on axis 0 only |
| M10 | k-space file → image | `io.load_kspace` (.npy/.npz/.h5), `reconstruct`, RSS | save/reload exact; 4-coil synthetic with Σ\|C\|² = 1 |
| M11 | Validation | radial profile, NRMSE/PSNR/SSIM, acceptance checklist | honest table |
| M12 | Streamlit app | `app.py`, 128×128, `st.cache_data`, ships small derived `.npy` | runs locally, all controls work |
| M13 | Notebook + README | single notebook, ipywidgets cells, limitations, unverified-values list | runs top to bottom |
| M14 | Oral prep (FR) | 5 money shots, FR glossary, Q&A, rehearsals | timed rehearsal |

## 4. Risks and assumptions
- Magnitude-only data → Hermitian k-space, no realistic phase.
- Crisp BrainWeb labels (no partial volume); label meanings confirmed only after visual check.
- PD and extra-tissue values are placeholders; GR = fat inferred; CSF T2 = 200 ms course value vs literature.
- Idealised SE formula: TE ≪ TR, no residual transverse magnetisation, perfect RF. TSE ignores stimulated echoes.
- PDw at 2000/15 shows T1 leakage (CSF darker than WM before PD) — explain it.
- FFT shift / axis-order pitfalls.
- Licensing: do not commit the raw BrainWeb volume or the slide image to a public repo; commit a download script + citation.

## 5. Fallback (drop first → last)
1. Cloud deployment  2. GRE  3. ipywidgets cells  4. Radial profile  5. TR/TE sweep grid  6. `.h5` loader  7. Motion  8. PSF plot.
Never dropped: M1, M4, M5, M12, M14.

## 6. Expert Reviewer changes (applied)
Oral prep added; single main demo (Streamlit); TSE TE_eff ordering fixed (centre-out only gives the shortest TE); T2w test made PD-independent; Hermitian index map, noise σ and aliasing wording corrected; NRMSE moved to M1; loader moved later; GRE promoted; licensing warning.

## 7. Working rules (agreed 2026-11-06)
1. **Commits after every step, made by the student.** The mentor only proposes the commands and the message. Commits stay small. No Co-Authored-By trailer: AI assistance is disclosed in the README and in the oral.
2. **Co-coding**: each milestone leaves one small piece (a function or a test) for the student to write, with a spec and a check.
3. **No moving on without understanding**: the next step starts only when the student types **"cbon"**.
4. Repo `kspace-digital-twin`: **private** until after the oral; commit email = GitHub noreply address.

## 8. Progress log
- 2026-11-06: interview done, plan approved, env + BrainWeb ready. M0 done (skeleton, .gitignore, git init).
- 2026-11-07: M1 steps 1–2 done and pushed: `fft2c` (mentor), `ifft2c` (student; fixed an odd-N shift bug), `dft2_explicit` (matches FFT to ~1e-14), figure `figures/m1_spike_from_stripes.png`.
  - Concepts already understood: a k-space point is one stripe pattern over the whole image (not a pixel); one slice = one k-space; a sharp pixel = all stripes lining up; centre = contrast, periphery = detail; shifts only reorder; FFT = the same sum, faster.
  - Teaching notes: explain with everyday analogies + figures (equalizer, hand clap, stripes adding up); student found formulas alone confusing.

- 2026-11-07 (cont.): M1 step 3 (Hermitian: `mirror_k`, `hermitian_residual`) and step 4 (`tests/test_kspace.py`, 9 passing; student wrote `test_centre_pixel_gives_flat_kspace`) done. Student got lost on the reasoning; made a 17-slide visual study deck (M1 concepts, analogies, figures from our code, FR glossary). Paused step 5 (NRMSE prediction) until the student has read it.
- 2026-11-08: M1 done: `metrics.nrmse` (optional mask) + `brain_mask` (threshold on |image|), 12 tests passing.
- 2026-11-08: M2 done: `phantom.load_brainweb`, `brainweb_slice` (pad to FOV 240 mm, nearest-neighbour to 256×256), `BRAINWEB_LABELS` (student identified 0 air, 2 GM, 3 WM; corrected 1 = CSF), `load_image` (centre crop, interpolating resize). Course MRI image extracted to `data/course/mri_axial.png` (gitignored); our high/low/full split reproduces the course figure. 16 tests passing.
- 2026-11-08: M3 done: `experiments.grating` (3 dots: k=0 + mirror pair at ±cycles), `lowpass_mask`, `split_low_high` (low+high = image exactly), NRMSE vs radius on the course image (NRMSE < 0.1 from radius 19 = 1.7 % of k-space, yet visibly blurry: NRMSE ignores fine detail), `k_axis` (Δk = 1/FOV; every 2nd row → FOV/2 → head wraps). 21 tests. Student asked for a faster pace: from here the mentor writes the code (incl. `test_horizontal_grating_gives_three_dots`, written by mentor at the student's request); student reads each step and types cbon.
- 2026-11-08: M4 done: `contrast.TISSUES` (course-tagged; rest placeholders), `param_maps`, `se_signal`, `gre_signal`. **Decision (student): WM and GM share PD = 0.8** — with the course T1s (785 vs 920 ms) a realistic PD gap made WM darker than GM on T1w. T1w 500/15: WM 0.319 > GM 0.289 > CSF 0.142; T2w 2000/120: CSF > GM > WM; PDw 2000/15: CSF darker (T1 leakage, CSF not recovered at TR 2000). 28 tests.
- 2026-11-08: M5 done: `sequences.row_order` (linear / centric), `acquire(image, kind, n_rows)`, `scan_time`. T1w brain stopped after 26 rows (13 s): linear NRMSE 0.97 vs centric 0.30; full scan 500 ms × 256 = 128 s. 36 tests.
- 2026-11-08: M6 done: `sequences.multi_echo`, `t2_map` (log-linear least squares, NaN outside mask, refuses one TE or zero-signal pixels). Noiseless brain, TR 3000, TE 20–160: CSF 200, GM 100, WM 90 ms recovered to ~1e-15. 39 tests. With noise the log fit is biased (revisit in M9).
- 2026-11-08: M7 done: `sequences.tse_te_per_row` (ETL contiguous bands; centre band = echo round(TE_eff/esp); later echoes below, earlier above, leftovers continue outwards → smooth TE, centric when TE_eff = first echo), `tse_kspace`. First version (closeness ranking) zigzagged TE between neighbouring bands → ringing; replaced. T2w TR 3000/TE_eff 100: SE 768 s, ETL 16 48 s (NRMSE 0.10), ETL 32 24 s (0.11). 43 tests.
- 2026-11-08: M12 v1 done (built early): `app.py` (French UI, 256×256 — fast enough, no need for 128), tabs Contraste / Espace k / Ordre de remplissage / TSE / Carte T2; ships `data/derived/brainweb_z200_labels_256.npy` made by `scripts/make_derived.py` (cite BrainWeb in README). `tests/test_app.py` drives every control with Streamlit AppTest. Run: `.venv/bin/streamlit run app.py --server.address localhost`. 46 tests.
- 2026-11-08: M10 done: `twin/kspace_io.py` (`save_kspace` .npy, `load_kspace` .npy/.npz/.h5 fastMRI middle slice, no pickles, `reconstruct` with RSS for coils). App tab « Importer »: image → k-space (+ download .npy) and k-space file → image; demo file `data/derived/demo_kspace_4coils.npy` (4 smooth coils, Σ|C|² = 1). Download → upload round trip error 6e-16. 54 tests.
- 2026-11-08: Real data: two M4Raw validation files (subject 2022061203, T1 + T2, 18×4×256×256) extracted from the 3 GB Zenodo zip with HTTP range requests (~25 MB) into `data/m4raw/` (gitignored, CC-BY 4.0). Our RSS matches the dataset's own `reconstruction_rss` to NRMSE 9e-8; real k-space Hermitian residual 1.41 (has phase); 61 zero-filled readout columns. README started with data sources and references.
- 2026-11-08: M9 done: `twin/artifacts.py` (`add_noise` σ per channel → Rayleigh air mean σ√(π/2), std 0.655σ; `truncate` Gibbs; `hamming`; `undersample_ky` → ½(I + I rolled N/2) exactly for R = 2; `motion` + `motion_rows` centre/outer/alternate, ghosts along axis 0 only). App tab « Artefacts ». 60 tests.

## 9. Resume point (for a fresh session)
**Current:** M8 (minimal GRE + Ernst angle + SE/ME/TSE/GRE table) and M11 (validation table, incl. M4Raw real-vs-twin), then M13 (notebook + README, AI disclosure).
