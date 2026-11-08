import numpy as np
import pytest

from twin.contrast import TISSUES, ernst_angle, gre_signal, param_maps, se_signal


def se(tissue, tr, te, pd=None):
    t = TISSUES[tissue]
    return se_signal(t["pd"] if pd is None else pd, t["t1"], t["t2"], tr, te)


def test_course_values_are_untouched():
    assert [TISSUES[n]["t1"] for n in ("white matter", "grey matter", "CSF", "fat")] == [785, 920, 3000, 250]
    assert [TISSUES[n]["t2"] for n in ("white matter", "grey matter", "CSF")] == [90, 100, 200]


def test_t1_weighted_wm_brighter_than_gm_brighter_than_csf():
    assert se("white matter", 500, 15) > se("grey matter", 500, 15) > se("CSF", 500, 15)


def test_t2_weighted_csf_brightest_even_with_equal_pd():
    # long TR removes T1 effects, so the order comes from T2 alone, whatever PD is
    s = [se(n, 4000, 120, pd=1) for n in ("CSF", "grey matter", "white matter")]
    assert s[0] > s[1] > s[2]


def test_t2_star_never_longer_than_t2():
    assert all(t["t2s"] <= t["t2"] for t in TISSUES.values())


def test_background_gives_no_signal():
    maps = param_maps(np.zeros((2, 2), dtype=np.uint8))
    assert np.all(se_signal(maps["pd"], maps["t1"], maps["t2"], 500, 15) == 0)
    assert np.all(gre_signal(maps["pd"], maps["t1"], maps["t2s"], 500, 15, 30) == 0)


def test_gre_at_90_degrees_and_long_tr_is_pd_times_t2_star_decay():
    s = gre_signal(0.8, 1000, 50, tr=1e6, te=10, flip_deg=90)
    assert s == pytest.approx(0.8 * np.exp(-10 / 50))


def test_param_maps_follow_the_labels():
    labels = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    t1 = param_maps(labels)["t1"]
    np.testing.assert_array_equal(t1, [[3000, 920], [785, np.inf]])


@pytest.mark.parametrize("tr, t1", [(10, 785), (30, 920), (100, 3000), (500, 250)])
def test_ernst_angle_gives_the_most_gre_signal(tr, t1):
    angles = np.linspace(0.01, 90, 90001)
    best = angles[np.argmax(gre_signal(1, t1, 50, tr, 5, angles))]
    assert best == pytest.approx(ernst_angle(tr, t1), abs=0.01)


def test_short_tr_needs_a_small_flip_angle():
    assert ernst_angle(10, 785) < 10 < ernst_angle(500, 785)
