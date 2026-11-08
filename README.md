# k-space digital twin

A small MRI simulator that shows how a scanner fills k-space and how that shapes the image: tissue contrast (TR/TE), filling order, turbo spin echo, T2 mapping, and reconstruction of real multi-coil k-space files.

Run the demo: `streamlit run app.py` · Run the tests: `pytest`

*(Full README to come.)*

## Data and references

**BrainWeb digital brain phantoms**: the tissue labels of our simulated brain (subject 04, crisp labels, 0.5 mm voxels). The raw volume is not in this repo; `scripts/make_derived.py` cuts the slice the app ships with.
- Website and downloads: https://brainweb.bic.mni.mcgill.ca/brainweb/
- B. Aubert-Broche, M. Griffin, G. B. Pike, A. C. Evans, D. L. Collins, “Twenty new digital brain phantoms for creation of validation image data bases”, *IEEE Transactions on Medical Imaging* 25(11), 1410–1416, 2006.

**M4Raw**: real 0.3 T multi-coil brain k-space (fastMRI layout), used to test the k-space → image reconstruction on real data. License CC-BY 4.0. Files are not in this repo.
- Dataset: https://zenodo.org/records/8056074 (DOI 10.5281/zenodo.8056074)
- Paper: M. Lyu et al., “M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space dataset for low-field MRI research”, *Scientific Data* 10, 264, 2023. doi:10.1038/s41597-023-02181-4 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10172399/
- Code and tutorial: https://github.com/mylyu/M4Raw

**fastMRI**: the reference raw k-space dataset whose HDF5 layout our loader follows.
- Dataset (data-sharing agreement required): https://fastmri.med.nyu.edu
- J. Zbontar et al., “fastMRI: An Open Dataset and Benchmarks for Accelerated MRI”, arXiv:1811.08839, 2018. https://arxiv.org/abs/1811.08839
