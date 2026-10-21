"""Repository paths. Everything is relative to the repository root, whatever the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"                       # downloaded public files (not versioned)
PROCESSED = DATA / "processed"           # rebuilt dataset (not versioned)
DATASET_CSV = PROCESSED / "autogear_public_1hz.csv"
LEVELS_JSON = PROCESSED / "gear_levels.json"
RESULTS = ROOT / "results"               # small JSON results (versioned)
FIGURES = ROOT / "figures"               # figures used by the report (versioned)
FIRMWARE = ROOT / "firmware"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, RESULTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
