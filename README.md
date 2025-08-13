# SHRED-turbulence-sensing
Mapping surface height dynamics to flow physics in free-surface turbulent flow using the SHRED neural network algorithm

Code and notebooks for SHRED (SHallow REcurrent Decoder) applied to free-surface turbulence: reconstructing subsurface velocity fields from sparse surface-height sensors, with comparisons across DNS and laboratory (T-Tank) data.

📄 This repo accompanies:
“Mapping surface height dynamics to sub-surface flow physics in free-surface turbulent flow using a shallow recurrent decoder” (in submission).
Preprint: link • Archived code snapshot (Zenodo DOI): to be added

# Overview
### Goal: 
infer sub-surface turbulence velocity fields from 3 surface sensor points capturing the time series of the surface elevation. 

### Method 
We apply the SHallow REcurrent Decoder (SHRED), which combines an LSTM (temporal encoder) with a shallow decoder (spatial mapping). We train in a compressed SVD basis to ease the training while keeping the relevant turbulence. Figure below shows the general outline. 
![Figures/SHRED%20architecture.png](Figures/SHRED%20architecture.png)

We input time series of surface elevation from three randomly placed surface sensor points into a two-layer LSTM. The LSTM encodes these input sequences into a latent representation of their temporal dynamics. This latent vector is then passed to a shallow decoder network (SDN), which maps it onto the velocity fields across depth. We do this in compressed space, by feeding into the SDN the compressed $\bV$ matrices of the SVD decomposition for the surface elevation and the subsurface velocity fields. These fields are used in training and validation not to learn the sub-surface time dynamics but only to learn the mapping of the surface time dynamics onto the subsurface fields.



We quantify the reconstruction errors using five metrics:
- Comparison of planar RMS values of the velocity from (uncompressed) ground truth and reconstructed fields
- Normalized Mean Squared Error (NMSE) of reconstruction fields as compared to (uncompressed) ground truth
- Power Spectral Density Error (PSDE) of spectra calculated from reconstructed turbulent fields, as compared to (uncompressed) ground truth
- Structural Similarity Measure Index (SSIM) of reconstructed fields as compared to (uncompressed) ground truth
- Peak Signal-to-Noise Ratio (PSNR) of reconstructed fields as compared to (uncompressed) ground truth

Details about the error metrics can be found in the manuscript. The functions that calculate these can be found in 'Processdata.py'. 

### Data 
DNS cases S1/S2 and experimental T-Tank cases E1/E2. Details about the turbulence statistics can be found in the manuscript.  

### Outputs
Plots for manuscript are generated in 'Run turbulence sensing SHRED.ipynb'. This include plots for
- SVD modes for DNS and experiments (Fig. 3-4)
- Loss profile during SHRED training (Fig. 5)
- Comparison of test data snapshots of ground truth, compressed fields and SHRED reconstruction (Fig. 6-7)
- Case-by-case comparison of ground truth and reconstructed instantaneous velocity RMS profiles in depth (Fig. 8)
- Error metric plot for SHRED reconstructions, showing depth-dependence of metrics such as NMSE, PSDE, SSIM & PSNR (Fig. 9)
- Comparison of ground truth vs reconstruction of time series of planar RMS velocity for test snapshots (Fig. 10)
- Comparison of PSD spectra for ground truth, compressed and reconstructed fields for all cases (Fig. 11)
- Analysis of rank-dependence of PSD spectra (Fig. 12, in Appendix A)
- Analysis of rank-dependence of error metrics (Fig. 13, in Appendix A)

Example video showing ground truth fields, compressed fields and SHRED reconstructed fields, of the surface elevation profile (top), and two velocity fields at different depths (below):

[![Watch the video](Figures/SHRED_DNS.mp4)]



# Repository layout


- Run turbulence sensing SHRED.ipynb   # main notebook: load artifacts, run/plot results
 models.py                            # SHRED model + training loop (based on pyshred)
-  processdata.py                      # SHRED runs, metrics (MSE/SSIM/PSNR/PSD), post-analysis
-  plot_results.py                     # figure helpers (depth profiles, PSD panels, etc.)
-  utilities.py                        # I/O for DNS/T-Tank, SVD helpers, geometry, misc
-  figures/                             # saved figures (created by notebook/plot scripts)
-  data/                                # (empty; your local raw data lives elsewhere)
-  artifacts/                           # optional: precomputed SVDs / SHRED outputs
-  requirements.txt / environment.yml   # dependencies 
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


# Citing
If you use this code, please cite the paper and the archived code as follows:

TBD

# Acknowledgements
Based on SHRED by Williams et al. We thank collaborators (NTNU, UW) and funding agencies for support, DNS datasets, and T-Tank experiments.


