import numpy as np
from skimage import io

from twin.phantom import BRAINWEB_SHAPE, brainweb_slice, load_image


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


def test_load_image_is_square_and_in_range(tmp_path):
    # wide colour image with a bright square in the middle
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[10:30, 20:40] = 200
    io.imsave(tmp_path / "wide.png", rgb)
    image = load_image(tmp_path / "wide.png", n=32)
    assert image.shape == (32, 32)
    assert image.min() == 0 and image.max() == 1
    assert image[16, 16] == 1  # the bright square stays centred after the crop
