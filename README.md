# SHRED-turbulence-sensing
Mapping surface height dynamics to flow physics in free-surface turbulent flow using the SHRED neural network algorithm

Code and notebooks for SHRED (SHallow REcurrent Decoder) applied to free-surface turbulence: reconstructing subsurface velocity fields from sparse surface-height sensors, with comparisons across DNS and laboratory (T-Tank) data.

📄 This repo accompanies:
“Mapping surface height dynamics to sub-surface flow physics in free-surface turbulent flow using a shallow recurrent decoder” (in submission).
Preprint: link • Archived code snapshot (Zenodo DOI): to be added

Overview
Goal — infer subsurface velocity fields from a few surface elevation time series.

Method — SHRED = LSTM (temporal encoder) + shallow decoder (spatial mapping) trained in a compressed SVD basis.

Data — DNS cases S1/S2 and experimental T-Tank cases E1/E2.

Outputs — reconstructions vs depth, temporal RMS tracking, PSD comparisons, and parameter sweeps over SVD rank/sensor count.

Repository layout
graphql
Copy
Edit
.
├─ Run turbulence sensing SHRED.ipynb   # main notebook: load artifacts, run/plot results
├─ models.py                            # SHRED model + training loop (based on pyshred)
├─ processdata3.py                      # SHRED runs, metrics (MSE/SSIM/PSNR/PSD), post-analysis
├─ plot_results3.py                     # figure helpers (depth profiles, PSD panels, etc.)
├─ utilities3.py                        # I/O for DNS/T-Tank, SVD helpers, geometry, misc
├─ figures/                             # saved figures (created by notebook/plot scripts)
├─ data/                                # (empty; your local raw data lives elsewhere)
├─ artifacts/                           # optional: precomputed SVDs / SHRED outputs
├─ requirements.txt / environment.yml   # dependencies (add one if missing)
└─ README.md
Large raw datasets are not tracked in git. See “Data & artifacts” below.

Installation
bash
Copy
Edit
# with conda (recommended)
conda env create -f environment.yml
conda activate shred-env

# or with pip
python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
Core deps: numpy, scipy, matplotlib, torch, h5py, scikit-image, cmocean, tqdm.

Data & artifacts
Raw data (DNS / T-Tank) must be stored outside the repo.

Precomputed artifacts (SVDs, SHRED test outputs) can be downloaded from Zenodo once available and placed under:

Copy
Edit
artifacts/
  ├─ SVD/
  └─ SHRED/
If you have raw data and want to recompute SVDs, see utilities3.py (e.g. save_svd_full) or use the main notebook.

Configure paths via one of:

environment variables:

bash
Copy
Edit
export SHRED_DATA_ROOT=/path/to/raw/data
export SHRED_ARTIFACTS_ROOT=/path/to/artifacts
export SHRED_FIGURES_ROOT=./figures
or a small config.yaml the code reads:

yaml
Copy
Edit
paths:
  data_root: "/path/to/raw/data"
  artifacts_root: "/path/to/artifacts"
  figures_root: "./figures"
(Where possible, we avoid hard-coded Windows paths and read from config.)

Quickstart
Put precomputed artifacts into artifacts/ (or compute SVDs locally).

Set the path variables (see above).

Open Run turbulence sensing SHRED.ipynb and run all cells:

loads SVDs + SHRED outputs

plots reconstructions vs ground truth at selected depths

computes error metrics (NMSE, SSIM, PSNR, PSD-error)

reproduces the key figures

Reproducing figures
Most paper figures are generated through functions in plot_results3.py and processdata3.py, called from the main notebook. Examples include:

Depth-dependent error profiles (DNS and T-Tank)

Instantaneous RMS profiles across cases S1/S2/E1/E2

PSD comparisons: ground truth vs SVD compression vs SHRED reconstruction

Parameter sweeps over rank r and number of sensors

If you only have artifacts (no raw data), the notebook will use those.

File guide (what lives where?)
models.py — SHRED network definition (LSTM + decoder) and training utilities (adapted from pyshred).

utilities3.py — data loaders for DNS/T-Tank, mesh/geometry helpers, SVD compute/load, reshaping utilities.

processdata3.py — SHRED run wrappers, error metrics (RMS/NMSE/SSIM/PSNR/PSD-error), ensemble averaging, figure-data preparation.

plot_results3.py — high-level plotting: multi-panel layouts, PSD panels with insets, depth profiles, etc.

Run turbulence sensing SHRED.ipynb — end-to-end demo + figure reproduction.

Notes on organization
Your current split is consistent and fine for a paper repo. If you want to polish:

Rename utilities3.py → utilities.py, processdata3.py → processdata.py, plot_results3.py → plot_results.py (drop the “3”).

Centralize all file paths in a tiny config reader (env vars or config.yaml).

Add brief module-level docstrings to each .py file.

(Optional later) move code into src/ as a small package; not necessary before submission.

Citing
If you use this code, please cite the paper and the archived code snapshot:

bibtex
Copy
Edit
@article{shred_turbulence_2025,
  title   = {Remote sensing of subsurface turbulence using SHRED},
  author  = {Moe, K. S. and …},
  journal = {…},
  year    = {2025},
  doi     = {…}
}

@software{shred_turbulence_code_2025,
  title   = {SHRED-turbulence-sensing: code and notebooks},
  author  = {Moe, K. S. and …},
  year    = {2025},
  version = {v1.0.0},
  doi     = {<Zenodo DOI>},
  url     = {https://github.com/krissmoe/SHRED-turbulence-sensing}
}
License
MIT (see LICENSE). Data may have separate terms.

Acknowledgements
Based on SHRED by Williams et al. We thank collaborators (NTNU, UW) and funding agencies for support, DNS datasets, and T-Tank experiments.


