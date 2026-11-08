"""Read and write k-space files, and turn k-space back into an image.

Arrays are centred k-space (k = 0 at index n // 2, as in kspace.fft2c).
2D = one coil (rows, cols); 3D = several coils (coils, rows, cols).
"""

from pathlib import Path

import h5py
import numpy as np

from twin.kspace import ifft2c


def save_kspace(file, kspace):
    """Write k-space as a .npy file (path or open binary file)."""
    np.save(file, np.asarray(kspace))


def load_kspace(file, name=None):
    """Read k-space from .npy, .npz or .h5 (fastMRI layout).

    file: a path or an open binary file; name: the file name when file is not
    a path (needed for the extension). .npz: the array named "kspace", else
    the first one. .h5: dataset "kspace"; fastMRI stores (slices, [coils,]
    rows, cols), so the middle slice is returned. Pickled objects are refused.
    """
    suffix = Path(name or str(file)).suffix.lower()
    if suffix == ".npy":
        kspace = np.load(file, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(file, allow_pickle=False) as archive:
            kspace = archive["kspace"] if "kspace" in archive else archive[archive.files[0]]
    elif suffix in (".h5", ".hdf5"):
        with h5py.File(file, "r") as f:
            data = f["kspace"]
            kspace = data[data.shape[0] // 2] if data.ndim >= 3 else data[()]
    else:
        raise ValueError(f"unsupported k-space file type {suffix!r} (use .npy, .npz or .h5)")
    if kspace.ndim not in (2, 3):
        raise ValueError(f"expected 2D or 3D k-space, got shape {kspace.shape}")
    return kspace


def reconstruct(kspace):
    """k-space -> magnitude image. Several coils are combined by root sum of
    squares: sqrt(sum over coils of |coil image|^2).
    """
    kspace = np.asarray(kspace)
    if kspace.ndim == 2:
        return np.abs(ifft2c(kspace))
    coil_images = np.stack([ifft2c(coil) for coil in kspace])
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
