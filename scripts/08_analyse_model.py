"""Train the retained model on the whole dataset (H = 3 s) and describe what it learned.

Output: results/model_analysis.json (tree counts, split-gain shares, steady-state shift map)
and results/gbdt_all_data.json (the XGBoost model, used by script 09).
"""
import json

from autogear.analysis import gain_shares, shift_map, tree_counts
from autogear.data import load_segments
from autogear.paths import RESULTS, ensure_dirs
from autogear.policies import make
from autogear.policies.targets import tree_storage_bytes

if __name__ == "__main__":
    ensure_dirs()
    model = make("GBDT").fit(load_segments(), horizon=3)
    booster = model.model
    booster.save_model(str(RESULTS / "gbdt_all_data.json"))
    res = dict(horizon_s=3, **tree_counts(booster), storage_bytes_convention=tree_storage_bytes(booster),
               gain_shares=gain_shares(booster), shift_map=shift_map(booster))
    (RESULTS / "model_analysis.json").write_text(json.dumps(res, indent=1))
    print(f"{res['trees']} trees, {res['decision_nodes']} decision nodes, {res['leaves']} leaves, "
          f"{res['storage_bytes_convention']} bytes (8 per node, 4 per leaf)")
    print("split-gain shares:", {k: f"{v * 100:.1f} %" for k, v in res["gain_shares"].items()})
    for r in res["shift_map"]:
        print(f"slope {r['slope'] * 100:+.0f} %  {r['speed_kmh']:2d} km/h  ->  {r['development_m']:.2f} m, "
              f"{r['cadence_rpm']:.0f} rpm")
