"""Export the retained model to C and write test vectors for the host test of the firmware.

1. Load the model trained on the whole dataset (script 08), convert it to complete trees of depth 4
   and check that the reference C-layout inference equals XGBoost's predictions.
2. Write firmware/autogear_ml/src/ag_model_data.{h,c}.
3. Write firmware/host_test/vectors.csv: for a few rides, the raw per-second signals, the Python
   inputs, the Python target and the Python closed-loop gears (h = 0.6, k = 3 s), so that
   firmware/host_test can check the C port against the Python reference.
"""
import json

import numpy as np
import xgboost as xgb

from autogear.controller import control
from autogear.data import load_segments
from autogear.export_c import from_booster, write_c
from autogear.features import FEATURE_NAMES
from autogear.paths import FIRMWARE, RESULTS, ensure_dirs

TEST_RIDES = ("inria_4", "edge810_2013", "di2_edge1040_2022")
SETTING = dict(h=0.6, k=3, d=0)

if __name__ == "__main__":
    ensure_dirs()
    booster = xgb.Booster()
    booster.load_model(str(RESULTS / "gbdt_all_data.json"))
    forest = from_booster(booster, depth=4)

    segs = load_segments()
    X = np.vstack([s.X for s in segs])
    ref = booster.inplace_predict(X)
    mine = forest.predict(X)
    err = float(np.max(np.abs(ref - mine)))
    print(f"complete-tree layout vs XGBoost: max |difference| = {err:.2e} over {len(X)} inputs")
    assert err < 1e-5, "export does not reproduce the model"

    meta = {"model": "100 boosted regression trees, depth 4, horizon 3 s, trained on 18.9 h",
            "inputs": len(FEATURE_NAMES), "flash bytes (arrays)": forest.flash_bytes(),
            "controller setting": "h = 0.6, k = 3 s"}
    h, c = write_c(forest, FIRMWARE / "autogear_ml" / "src", FEATURE_NAMES, meta)
    print(f"wrote {h.relative_to(FIRMWARE.parent)} and {c.name}: {forest.flash_bytes()} bytes of arrays")

    rows = []
    for k, s in enumerate(s for s in segs if s.ride in TEST_RIDES):
        target = booster.inplace_predict(s.X)
        g = control(s, target, **SETTING)
        rows.append(",".join(["S", str(k), str(len(s.levels)), str(int(s.gear[0])),
                              *[f"{v:.9g}" for v in s.levels]]))
        for t in range(s.n):
            rows.append(",".join(["D", f"{s.cadence[t]:.9g}", str(int(s.gear[t])), str(int(s.coasting[t])),
                                  f"{s.X[t, 5]:.9g}", f"{s.X[t, 8]:.9g}",
                                  *[f"{v:.9g}" for v in s.X[t]], f"{target[t]:.9g}", str(int(g[t]))]))
    out = FIRMWARE / "host_test" / "vectors.csv"
    out.write_text("\n".join(rows) + "\n")
    (RESULTS / "firmware_export.json").write_text(json.dumps(
        {"max_abs_difference_vs_xgboost": err, "flash_bytes_arrays": forest.flash_bytes(),
         "trees": forest.n_trees, "depth": forest.depth, "test_rides": TEST_RIDES,
         "test_seconds": int(sum(1 for r in rows if r.startswith("D")))}, indent=1))
    print(f"wrote {out.relative_to(FIRMWARE.parent)} ({sum(1 for r in rows if r.startswith('D'))} seconds)")
