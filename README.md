# SHRED-turbulence-sensing
Mapping surface height dynamics to flow physics in free-surface turbulent flow using the SHRED neural network algorithm

Code and notebooks for SHRED (SHallow REcurrent Decoder) applied to free-surface turbulence: reconstructing subsurface velocity fields from sparse surface-height sensors, with comparisons across DNS and laboratory (T-Tank) data.

📄 This repo accompanies:
“Mapping surface height dynamics to sub-surface flow physics in free-surface turbulent flow using a shallow recurrent decoder” (in submission).
Preprint: link • Archived code snapshot (Zenodo DOI): to be added

# Overview
Goal — infer subsurface velocity fields from a few surface elevation time series.

Method — SHRED = LSTM (temporal encoder) + shallow decoder (spatial mapping) trained in a compressed SVD basis. 
![SHRED architecture](Figures/SHRED%20architecture.png)


Data — DNS cases S1/S2 and experimental T-Tank cases E1/E2.

Outputs — reconstructions vs depth, temporal RMS tracking, PSD comparisons, and parameter sweeps over SVD rank/sensor count.

Example video showing ground truth fields, compressed fields and SHRED reconstructed fields, of the surface elevation profile (top), and two velocity fields at different depths (below):
[Watch: SHRED DNS demo](Figures/Shred%20DNS.mp4)



# Repository layout


- Run turbulence sensing SHRED.ipynb   # main notebook: load artifacts, run/plot results
 models.py                            # SHRED model + training loop (based on pyshred)
-  processdata.py                      # SHRED runs, metrics (MSE/SSIM/PSNR/PSD), post-analysis
-  lot_results.py                     # figure helpers (depth profiles, PSD panels, etc.)
-  utilities.py                        # I/O for DNS/T-Tank, SVD helpers, geometry, misc
-  figures/                             # saved figures (created by notebook/plot scripts)
-  data/                                # (empty; your local raw data lives elsewhere)
-  artifacts/                           # optional: precomputed SVDs / SHRED outputs
-  equirements.txt / environment.yml   # dependencies (add one if missing)
-  README.md
Large raw datasets are not tracked in git. See “Data & artifacts” below.

# Installation
bash
Copy
Edit

### with conda (recommended)
conda env create -f environment.yml
conda activate shred-env

### with pip
python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
Core deps: numpy, scipy, matplotlib, torch, h5py, scikit-image, cmocean, tqdm.

# Data & artifacts
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

# Quickstart
Put precomputed artifacts into artifacts/ (or compute SVDs locally).

Set the path variables (see above).

Open Run turbulence sensing SHRED.ipynb and run all cells:

loads SVDs + SHRED outputs

plots reconstructions vs ground truth at selected depths

computes error metrics (NMSE, SSIM, PSNR, PSD-error)

reproduces the key figures

# Reproducing figures
Most paper figures are generated through functions in plot_results.py and processdata.py, called from the main notebook. Examples include:


If you only have artifacts (no raw data), the notebook will use those.

# File guide
models.py — SHRED network definition (LSTM + decoder) and training utilities (adapted from pyshred).

utilities.py — data loaders for DNS/T-Tank, mesh/geometry helpers, SVD compute/load, reshaping utilities.

processdata.py — SHRED run wrappers, error metrics (RMS/NMSE/SSIM/PSNR/PSD-error), ensemble averaging, figure-data preparation.

plot_results.py — high-level plotting: multi-panel layouts, PSD panels with insets, depth profiles, etc.

Run turbulence sensing SHRED.ipynb — end-to-end demo + figure reproduction.


Citing
If you use this code, please cite the paper and the archived code as follows:

TBD

Acknowledgements
Based on SHRED by Williams et al. We thank collaborators (NTNU, UW) and funding agencies for support, DNS datasets, and T-Tank experiments.


