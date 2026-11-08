"""Digital brain phantom: BrainWeb tissue labels on our image grid.

BrainWeb anatomical models are 362 x 434 x 362 voxels of 0.5 mm, one uint8
tissue label per voxel. Axis 0 indexes axial slices.
"""

import gzip

import numpy as np
from skimage.transform import resize

BRAINWEB_SHAPE = (362, 434, 362)
BRAINWEB_VOXEL_MM = 0.5

# Label meanings from the BrainWeb documentation, checked visually on slice 200
# (ventricles = CSF, cortical ribbon = grey matter, deep core = white matter).
BRAINWEB_LABELS = {
    0: "background",
    1: "CSF",
    2: "grey matter",
    3: "white matter",
    4: "fat",
    5: "muscle",
    6: "muscle/skin",
    7: "skull",
    8: "vessels",
    9: "around fat",
    10: "dura mater",
    11: "bone marrow",
}


def load_brainweb(path):
    """Read a gzipped raw BrainWeb label volume (uint8, no header)."""
    with gzip.open(path) as f:
        volume = np.frombuffer(f.read(), dtype=np.uint8)
    return volume.reshape(BRAINWEB_SHAPE)


def brainweb_slice(volume, z=200, fov_mm=240, n=256):
    """One axial slice as an n x n label image covering fov_mm x fov_mm.

    The slice is padded with background (label 0) to the field of view, then
    shrunk with nearest-neighbour (order=0): labels are names, so a pixel
    between white matter and grey matter must stay one of the two, never an
    average of them.
    """
    labels = volume[z]
    side = round(fov_mm / BRAINWEB_VOXEL_MM)
    pad_rows, pad_cols = side - labels.shape[0], side - labels.shape[1]
    if pad_rows < 0 or pad_cols < 0:
        raise ValueError(f"fov_mm={fov_mm} is smaller than the slice")
    labels = np.pad(labels, ((pad_rows // 2, pad_rows - pad_rows // 2),
                             (pad_cols // 2, pad_cols - pad_cols // 2)))
    small = resize(labels, (n, n), order=0, preserve_range=True, anti_aliasing=False)
    return small.astype(np.uint8)
