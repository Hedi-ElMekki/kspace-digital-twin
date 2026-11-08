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
