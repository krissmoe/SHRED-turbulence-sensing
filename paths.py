import os, yaml
from pathlib import Path




def _find_project_root():
    # Look upward for a marker; fallback to CWD
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / ".git").exists() or (p / ".project-root").exists():
            return p
    return Path.cwd()

#DEFINE PROJECT ROOTS 
PROJECT_ROOT = Path(os.environ.get("SHRED_PROJECT_ROOT", _find_project_root()))
DATA_ROOT    = Path(os.environ.get("SHRED_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT  = Path(os.environ.get("SHRED_OUTPUT_ROOT", PROJECT_ROOT / "output"))
PLOTS_ROOT = Path(os.environ.get("PLOTS_ROOT", PROJECT_ROOT / "plots"))

#DEFINE SPECIFIC PATHS RELATIVE TO ROOTS
DNS_RAW_DIR = DATA_ROOT / "DNS" / "raw"
EXP_RAW_DIR = DATA_ROOT / "exp" / "raw"
DNS_SVD_DIR  = DATA_ROOT / "DNS" / "SVD"
EXP_SVD_DIR  = DATA_ROOT / "exp" / "SVD"
SHRED_DIR    = OUTPUT_ROOT / "SHRED"
METRICS_DIR    = OUTPUT_ROOT / "metrics"
METRICS_DIR    = OUTPUT_ROOT / "metrics"
PLOTS_DIR = PLOTS_ROOT

def ensure_dirs():
    for d in (DNS_RAW_DIR, EXP_RAW_DIR, DNS_SVD_DIR, EXP_SVD_DIR, SHRED_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)