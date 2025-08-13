import os, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # repo root if file is at top-level
DATA_DIR = Path(os.getenv("SHRED_DATA_DIR", ROOT / "data"))  # external if env var set
ARTIFACTS_DIR = Path(os.getenv("SHRED_ARTIFACTS_DIR", ROOT / "artifacts"))
FIG_DIR = ROOT / "figures"

def data_path(*parts): return DATA_DIR.joinpath(*parts)
def artifacts_path(*parts): return ARTIFACTS_DIR.joinpath(*parts)
def fig_path(*parts): return FIG_DIR.joinpath(*parts)

def load_cfg(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(os.getenv("SHRED_DATA_DIR", cfg.get("data_root", ""))).expanduser()
    def expand(p):
        return Path(str(p)
                    .replace("${DATA_ROOT}", str(root))
                    .replace("${SHRED_DATA_DIR}", str(root))).expanduser()
    cfg["paths"] = {
        "dns": {k: expand(v) for k, v in cfg.get("dns", {}).items()},
        "exp": {k: expand(v) for k, v in cfg.get("exp", {}).items()},
        "outputs": {k: expand(v) for k, v in cfg.get("outputs", {}).items()},
    }
    return cfg

CFG = load_cfg()

def dns_file(case: str, plane: int):
    return CFG["paths"]["dns"][case] / f"u_layer{plane}.mat"

def exp_plane_dir(case: str, plane: str):
    return CFG["paths"]["exp"][case] / plane