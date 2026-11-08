import numpy as np

from twin.phantom import BRAINWEB_SHAPE, brainweb_slice


def fake_volume():
    # left half label 2, right half label 3, background elsewhere
    volume = np.zeros(BRAINWEB_SHAPE, dtype=np.uint8)
    volume[:, 100:300, 80:181] = 2
    volume[:, 100:300, 181:280] = 3
    return volume


def test_slice_has_the_requested_grid():
    labels = brainweb_slice(fake_volume(), n=256)
    assert labels.shape == (256, 256)
    assert labels.dtype == np.uint8


def test_resize_never_invents_a_label():
    labels = brainweb_slice(fake_volume(), n=256)
    assert set(np.unique(labels)) == {0, 2, 3}
