"""What the retained model learned: split-gain shares and the steady-state shift map."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from .features import FEATURE_GROUPS, FEATURE_NAMES, N_FEATURES


def gain_shares(booster: xgb.Booster) -> dict[str, float]:
    """Share of the total split gain carried by each group of inputs."""
    score = booster.get_score(importance_type="total_gain")
    gain = np.zeros(N_FEATURES)
    for k, v in score.items():
        gain[int(k[1:]) if k.startswith("f") else FEATURE_NAMES.index(k)] = v
    total = gain.sum()
    return {g: float(gain[idx].sum() / total) for g, idx in FEATURE_GROUPS.items()}


def steady_state_inputs(speed_ms: float, slope: float) -> np.ndarray:
    """Inputs at constant speed and slope, zero acceleration, after >= 10 s of pedalling."""
    x = np.zeros(N_FEATURES, np.float32)
    x[[0, 1, 2]] = speed_ms
    x[[5, 6, 7]] = slope
    x[13] = 10
    return x


def shift_map(booster: xgb.Booster, speeds_kmh=(10, 15, 20, 25, 30, 35), slopes=(-0.05, 0.0, 0.03, 0.06)
              ) -> list[dict]:
    rows = []
    for sl in slopes:
        for v_kmh in speeds_kmh:
            v = v_kmh / 3.6
            dev = float(np.exp(booster.inplace_predict(steady_state_inputs(v, sl)[None, :])[0]))
            rows.append(dict(slope=sl, speed_kmh=v_kmh, development_m=dev, cadence_rpm=60 * v / dev))
    return rows


def tree_counts(booster: xgb.Booster) -> dict[str, int]:
    dump = booster.get_dump(dump_format="json")
    return dict(trees=len(dump), decision_nodes=sum(t.count('"split"') for t in dump),
                leaves=sum(t.count('"leaf"') for t in dump))
