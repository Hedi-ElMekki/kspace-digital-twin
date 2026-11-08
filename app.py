"""Streamlit demo of the k-space digital twin. Run: streamlit run app.py"""

from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

from twin.artifacts import add_noise, hamming, motion, motion_rows, truncate, undersample_ky
from twin.contrast import TISSUES, param_maps, se_signal
from twin.experiments import lowpass_mask, split_low_high
from twin.kspace import fft2c, ifft2c, log_mag
from twin.kspace_io import load_kspace, reconstruct, save_kspace
from twin.metrics import brain_mask, nrmse
from twin.phantom import load_image
from twin.sequences import (
    acquire, multi_echo, scan_time, t2_map, tse_kspace, tse_te_per_row,
)

LABELS_FILE = Path(__file__).parent / "data/derived/brainweb_z200_labels_256.npy"
BRAIN_TISSUES = {1: "LCR", 2: "Substance grise", 3: "Substance blanche"}
PRESETS = {"T1 (TR 500, TE 15)": (500, 15), "T2 (TR 2000, TE 120)": (2000, 120),
           "DP (TR 2000, TE 15)": (2000, 15)}


@st.cache_data
def load_labels():
    return np.load(LABELS_FILE)


@st.cache_data
def load_maps():
    return param_maps(load_labels())


def to_rgb(image, cmap="gray", vmax=None):
    """Float image -> RGB array for st.image, scaled to [0, vmax]."""
    vmax = vmax or image.max() or 1
    return matplotlib.colormaps[cmap](np.clip(image / vmax, 0, 1))[..., :3]


def show_pair(image, kspace, image_caption, kspace_caption):
    left, right = st.columns(2)
    left.image(to_rgb(image), caption=image_caption, width="stretch")
    right.image(to_rgb(log_mag(kspace), "magma"), caption=kspace_caption, width="stretch")


def apply_preset():
    st.session_state.tr, st.session_state.te = PRESETS[st.session_state.preset]


st.set_page_config(page_title="Jumeau numérique de l’espace k", layout="wide")
st.title("Jumeau numérique de l’espace k")
st.caption("Fantôme BrainWeb (coupe axiale, 256 × 256, champ de vue 240 mm) · 1,5 T · écho de spin idéalisé")

maps = load_maps()
labels = load_labels()
tab_contrast, tab_kspace, tab_order, tab_tse, tab_t2, tab_artefacts, tab_import = st.tabs(
    ["Contraste", "Espace k", "Ordre de remplissage", "TSE", "Carte T2", "Artefacts", "Importer"])

with tab_contrast:
    st.subheader("TR et TE choisissent le contraste")
    st.selectbox("Préréglage", list(PRESETS), key="preset", on_change=apply_preset)
    st.session_state.setdefault("tr", 500)
    st.session_state.setdefault("te", 15)
    tr = st.slider("TR (ms)", 100, 5000, step=50, key="tr")
    te = st.slider("TE (ms)", 5, 250, step=5, key="te")
    image = se_signal(maps["pd"], maps["t1"], maps["t2"], tr, te)
    show_pair(image, fft2c(image), f"Image, TR {tr} ms, TE {te} ms", "Espace k, log(1 + |k|)")
    st.dataframe(pd.DataFrame(
        [{"Tissu": name, "T1 (ms)": TISSUES[en]["t1"], "T2 (ms)": TISSUES[en]["t2"],
          "Signal": round(float(se_signal(TISSUES[en]["pd"], TISSUES[en]["t1"], TISSUES[en]["t2"], tr, te)), 3)}
         for name, en in [("Substance blanche", "white matter"), ("Substance grise", "grey matter"), ("LCR", "CSF")]]),
        hide_index=True)
    st.caption("S = DP · (1 − e^(−TR/T1)) · e^(−TE/T2). TR court → pondération T1 ; TE long → pondération T2.")

reference = se_signal(maps["pd"], maps["t1"], maps["t2"], 500, 15)

with tab_kspace:
    st.subheader("Centre = contraste, périphérie = détails")
    radius = st.slider("Rayon gardé autour du centre (pas de k)", 1, 128, 16)
    part = st.radio("Garder", ["Le centre (basses fréquences)", "La périphérie (hautes fréquences)"], horizontal=True)
    low, high = split_low_high(reference, radius)
    keep_low = part.startswith("Le centre")
    kept = np.abs(low if keep_low else high)
    mask = lowpass_mask(reference.shape, radius)
    show_pair(kept, fft2c(reference) * (mask if keep_low else ~mask),
              f"Image reconstruite · NRMSE {nrmse(kept, reference):.2f}",
              f"Espace k gardé : {(mask if keep_low else ~mask).mean():.1%} des points")

with tab_order:
    st.subheader("Le scanner remplit une ligne de k par TR")
    kind = st.radio("Ordre", ["linear", "centric"], horizontal=True,
                    format_func={"linear": "Linéaire (haut → bas)", "centric": "Centrique (centre → bords)"}.get)
    percent = st.slider("Examen interrompu à (% des lignes)", 1, 100, 10)
    n_rows = max(1, round(256 * percent / 100))
    kspace = acquire(reference, kind, n_rows)
    image = np.abs(ifft2c(kspace))
    show_pair(image, kspace, f"Image après {n_rows} lignes · NRMSE {nrmse(image, reference):.2f}",
              f"Temps écoulé : {scan_time(500, n_rows):.0f} s sur {scan_time(500, 256):.0f} s (TR 500 ms)")

with tab_tse:
    st.subheader("Turbo spin écho : plusieurs lignes par TR")
    c1, c2 = st.columns(2)
    etl = c1.select_slider("Longueur du train d’échos (ETL)", [1, 2, 4, 8, 16, 32], value=16)
    esp = 10
    te_choices = [esp * e for e in range(1, etl + 1)]
    te_eff = c2.select_slider("TE effectif (ms)", te_choices, value=min(100, te_choices[-1]))
    tr_tse = 3000
    kspace = tse_kspace(maps, tr_tse, etl, esp, te_eff)
    image = np.abs(ifft2c(kspace))
    se_ref = se_signal(maps["pd"], maps["t1"], maps["t2"], tr_tse, te_eff)
    show_pair(image, kspace, f"TSE · NRMSE vs écho de spin au même TE : {nrmse(image, se_ref):.2f}",
              f"Durée : {scan_time(tr_tse, 256 // etl):.0f} s (écho de spin : {scan_time(tr_tse, 256):.0f} s)")
    st.caption(f"TR {tr_tse} ms, espacement des échos {esp} ms. TE de chaque ligne de k :")
    st.line_chart(pd.DataFrame({"TE (ms)": tse_te_per_row(256, etl, esp, te_eff)},
                               index=pd.Index(np.arange(256) - 128, name="ky")), height=220)

with tab_t2:
    st.subheader("Multi-écho : une vraie mesure de T2 par pixel")
    n_echoes = st.slider("Nombre d’échos (espacés de 20 ms)", 2, 12, 8)
    tes = 20 * np.arange(1, n_echoes + 1)
    echoes = multi_echo(maps, 3000, tes)
    mask = brain_mask(echoes[0]) & np.isin(labels, list(BRAIN_TISSUES))
    t2 = t2_map(echoes, tes, mask)
    left, right = st.columns(2)
    left.image(to_rgb(echoes[-1], vmax=echoes[0].max()), caption=f"Dernier écho, TE {tes[-1]} ms", width="stretch")
    right.image(to_rgb(np.nan_to_num(t2), "viridis", vmax=220), caption="Carte T2 (0 → 220 ms)", width="stretch")
    st.dataframe(pd.DataFrame([{"Tissu": name, "T2 mesuré (ms)": round(float(np.nanmedian(t2[labels == k])), 1)}
                               for k, name in BRAIN_TISSUES.items()]), hide_index=True)

with tab_artefacts:
    st.subheader("Les artefacts naissent dans l’espace k")
    artefact = st.radio("Artefact", ["Bruit", "Gibbs (troncature)", "Repliement", "Mouvement"], horizontal=True,
                        key="artefact")
    clean_k = fft2c(reference)
    if artefact == "Bruit":
        sigma = st.slider("Écart-type du bruit (par canal)", 0.0, 0.1, 0.02, step=0.005, format="%.3f")
        kspace = add_noise(clean_k, sigma, rng=0)
        note = (f"Dans l’air, le module du bruit n’a pas une moyenne nulle : σ·√(π/2) ≈ {sigma * 1.2533:.3f}. "
                "Le bruit est réparti uniformément sur tout l’espace k ; il domine la périphérie, où le signal est faible.")
    elif artefact == "Gibbs (troncature)":
        keep = st.slider("Taille du bloc central gardé (points)", 16, 256, 48, step=8)
        smooth = st.checkbox("Fenêtre de Hamming (adoucir la coupure)", key="hamming")
        kspace = hamming(clean_k, keep) if smooth else truncate(clean_k, keep)
        note = ("Une coupure nette de l’espace k crée des ondulations le long des bords francs (Gibbs). "
                "La fenêtre de Hamming les supprime presque toutes, au prix d’un léger flou.")
    elif artefact == "Repliement":
        factor = st.slider("Facteur de sous-échantillonnage en ky (une ligne sur R)", 1, 4, 2)
        kspace = undersample_ky(clean_k, factor)
        note = (f"Une ligne sur {factor} : Δk × {factor}, donc champ de vue ÷ {factor} "
                f"({240 // factor} mm). La tête ne tient plus et se replie, seulement selon le codage de phase.")
    else:
        shift_px = st.slider("Déplacement du patient (pixels, gauche-droite)", 0, 10, 3)
        when = st.radio("Pendant quelles lignes ?", ["centre", "outer", "alternate"], horizontal=True, key="when",
                        format_func={"centre": "Lignes centrales (10 %)", "outer": "Lignes périphériques (10 %)",
                                     "alternate": "Une ligne sur deux (respiration)"}.get)
        kspace = motion(reference, (0, shift_px), motion_rows(256, when))
        note = ("Chaque ligne est mesurée à un instant différent : l’incohérence se répartit le long du codage "
                "de phase (vertical), quelle que soit la direction du mouvement. Au centre : flou et fantômes forts ; "
                "en périphérie : presque invisible.")
    image = np.abs(ifft2c(kspace))
    show_pair(image, kspace, f"Image · NRMSE {nrmse(image, reference):.2f}", "Espace k, log(1 + |k|)")
    st.caption(note)

with tab_import:
    st.subheader("Image → espace k")
    upload = st.file_uploader("Une image (PNG, JPG) ; sinon le cerveau du jumeau en T1", type=["png", "jpg", "jpeg"])
    try:
        source = load_image(upload) if upload else reference
    except ValueError as error:
        st.error(str(error))
        source = reference
    kspace = fft2c(source)
    show_pair(np.abs(source), kspace, "Image", "Son espace k, log(1 + |k|)")
    buffer = BytesIO()
    save_kspace(buffer, kspace)
    st.download_button("Télécharger cet espace k (.npy)", buffer.getvalue(), "espace_k.npy")

    st.subheader("Espace k → image")
    upload_k = st.file_uploader("Un fichier d’espace k centré (.npy, .npz, .h5 fastMRI)", type=["npy", "npz", "h5"])
    if upload_k:
        try:
            kspace = load_kspace(upload_k, name=upload_k.name)
        except (ValueError, KeyError, OSError) as error:
            st.error(f"Fichier illisible : {error}")
        else:
            coils = 1 if kspace.ndim == 2 else kspace.shape[0]
            show_pair(reconstruct(kspace), kspace if coils == 1 else kspace[0],
                      f"Image reconstruite ({coils} antenne{'s' if coils > 1 else ''}"
                      f"{', racine de la somme des carrés' if coils > 1 else ''})",
                      f"Espace k {kspace.shape[-2]} × {kspace.shape[-1]}" + (" (antenne 1)" if coils > 1 else ""))
