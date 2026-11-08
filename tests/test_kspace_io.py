import h5py
import numpy as np
import pytest

from twin.kspace import fft2c
from twin.kspace_io import load_kspace, reconstruct, save_kspace


def complex_kspace(shape=(16, 16)):
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".h5"])
def test_save_then_load_is_exact(tmp_path, suffix):
    kspace = complex_kspace()
    path = tmp_path / f"k{suffix}"
    if suffix == ".npy":
        save_kspace(path, kspace)
    elif suffix == ".npz":
        np.savez(path, kspace=kspace)
    else:
        with h5py.File(path, "w") as f:
            f["kspace"] = kspace
    np.testing.assert_array_equal(load_kspace(path), kspace)


def test_fastmri_h5_gives_the_middle_slice(tmp_path):
    volume = np.stack([np.full((4, 8, 8), s, dtype=complex) for s in range(5)])  # slices, coils, rows, cols
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f["kspace"] = volume
    np.testing.assert_array_equal(load_kspace(tmp_path / "f.h5"), volume[2])


def test_open_file_needs_its_name(tmp_path):
    save_kspace(tmp_path / "k.npy", complex_kspace())
    with open(tmp_path / "k.npy", "rb") as f:
        assert load_kspace(f, name="upload.npy").shape == (16, 16)


def test_unknown_type_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_kspace(tmp_path / "k.txt")


def test_single_coil_reconstruct_is_the_magnitude_image():
    image = np.random.default_rng(1).random((16, 16))
    np.testing.assert_allclose(reconstruct(fft2c(image)), image, atol=1e-12)


def test_four_coils_combine_back_to_the_image():
    # smooth complex coil sensitivities normalised so sum |C|^2 = 1 at every pixel
    y, x = np.mgrid[:32, :32] - 16
    image = np.random.default_rng(2).random((32, 32))
    centres = [(-16, -16), (-16, 16), (16, -16), (16, 16)]
    blobs = np.stack([np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 400) * np.exp(1j * 0.1 * (x + cy))
                      for cy, cx in centres])
    coils = blobs / np.sqrt(np.sum(np.abs(blobs) ** 2, axis=0))
    kspace = np.stack([fft2c(c * image) for c in coils])
    np.testing.assert_allclose(reconstruct(kspace), image, atol=1e-12)
