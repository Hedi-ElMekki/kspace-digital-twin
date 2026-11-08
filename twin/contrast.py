"""Tissue parameters and signal equations at 1.5 T.

Every value is tagged: `course` lists the parameters taken from the MRI
course (T1 of WM, GM, CSF, fat; T2 of WM, GM, CSF). Everything else is a
PLACEHOLDER of plausible order of magnitude, to replace with sourced values.
Times in ms, PD relative to CSF = 1.

Deliberate simplification: WM and GM share the same PD (0.8). The course T1
values are close (785 vs 920 ms), so a realistic PD gap (GM holds more water)
would cancel the T1 contrast and make WM darker than GM on T1-weighted images.
With equal PD, WM/GM contrast comes from T1 and T2 alone.
"""

import numpy as np

from twin.phantom import BRAINWEB_LABELS

TISSUES = {
    "background":   dict(pd=0.0,  t1=np.inf, t2=np.inf, t2s=np.inf, course=()),
    "CSF":          dict(pd=1.0,  t1=3000,   t2=200,    t2s=150,    course=("t1", "t2")),
    "grey matter":  dict(pd=0.80, t1=920,    t2=100,    t2s=70,     course=("t1", "t2")),
    "white matter": dict(pd=0.80, t1=785,    t2=90,     t2s=65,     course=("t1", "t2")),
    "fat":          dict(pd=0.90, t1=250,    t2=70,     t2s=40,     course=("t1",)),
    "muscle":       dict(pd=0.75, t1=900,    t2=45,     t2s=30,     course=()),
    "muscle/skin":  dict(pd=0.75, t1=900,    t2=45,     t2s=30,     course=()),
    "skull":        dict(pd=0.05, t1=300,    t2=1,      t2s=0.5,    course=()),
    "vessels":      dict(pd=0.0,  t1=1400,   t2=150,    t2s=50,     course=()),  # flow void in SE
    "around fat":   dict(pd=0.90, t1=250,    t2=70,     t2s=40,     course=("t1",)),
    "dura mater":   dict(pd=0.60, t1=1000,   t2=80,     t2s=40,     course=()),
    "bone marrow":  dict(pd=0.80, t1=400,    t2=70,     t2s=30,     course=()),
}


def param_maps(labels):
    """Label image -> dict of parameter maps (pd, t1, t2, t2s), same shape.

    Builds one small lookup array per parameter (index = label) and indexes it
    with the whole label image at once: lookup[labels].
    """
    names = [BRAINWEB_LABELS[k] for k in range(len(BRAINWEB_LABELS))]
    return {p: np.array([TISSUES[n][p] for n in names], dtype=float)[labels]
            for p in ("pd", "t1", "t2", "t2s")}


def se_signal(pd, t1, t2, tr, te):
    """Spin echo: S = PD (1 - e^{-TR/T1}) e^{-TE/T2}. Assumes TE << TR."""
    return pd * (1 - np.exp(-tr / t1)) * np.exp(-te / t2)


def gre_signal(pd, t1, t2s, tr, te, flip_deg):
    """Spoiled gradient echo: S = PD sin(a) (1 - E1) / (1 - cos(a) E1) e^{-TE/T2*}, E1 = e^{-TR/T1}."""
    a = np.radians(flip_deg)
    e1 = np.exp(-tr / t1)
    return pd * np.sin(a) * (1 - e1) / (1 - np.cos(a) * e1) * np.exp(-te / t2s)
