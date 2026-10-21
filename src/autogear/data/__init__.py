"""Public sources, download, and reconstruction of the 1-Hz dataset."""
from __future__ import annotations

import json


def load_segments():
    """Segments of the rebuilt dataset (run scripts 01 and 02 first)."""
    import pandas as pd

    from ..features import load_segments as _load
    from ..paths import DATASET_CSV, LEVELS_JSON
    if not DATASET_CSV.exists():
        raise FileNotFoundError("dataset missing: run scripts/01_download_sources.py and "
                                "scripts/02_build_dataset.py first")
    return _load(pd.read_csv(DATASET_CSV), json.loads(LEVELS_JSON.read_text()))
