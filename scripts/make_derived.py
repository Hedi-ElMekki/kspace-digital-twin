"""Cut the demo's label slice out of the raw BrainWeb volume (run once).

The raw volume stays out of git; the app only needs this small derived file.
"""

from pathlib import Path

import numpy as np

from twin.phantom import brainweb_slice, load_brainweb

ROOT = Path(__file__).resolve().parents[1]

labels = brainweb_slice(load_brainweb(ROOT / "data/brainweb/subject04_crisp.mnc.gz"), z=200)
np.save(ROOT / "data/derived/brainweb_z200_labels_256.npy", labels)
print("saved", labels.shape, labels.dtype)

# a 4-coil k-space of the T1-weighted slice, to demo the k-space -> image upload
from twin.contrast import param_maps, se_signal  # noqa: E402
from twin.kspace import fft2c  # noqa: E402

maps = param_maps(labels)
image = se_signal(maps["pd"], maps["t1"], maps["t2"], 500, 15)
y, x = np.mgrid[:256, :256] - 128
blobs = np.stack([np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 90 ** 2)) for cy, cx in
                  [(-128, -128), (-128, 128), (128, -128), (128, 128)]])
coils = blobs / np.sqrt(np.sum(blobs ** 2, axis=0))  # sum |C|^2 = 1
kspace = np.stack([fft2c(c * image) for c in coils]).astype(np.complex64)
np.save(ROOT / "data/derived/demo_kspace_4coils.npy", kspace)
print("saved", kspace.shape, kspace.dtype)
