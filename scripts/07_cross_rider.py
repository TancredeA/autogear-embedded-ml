"""Cross-rider protocol (no personalisation): train on five riders, test on the sixth.

Decision parameters are selected by an inner leave-one-rider-out over the five training riders,
with the same constraints as the within-rider protocol. Output: results/cross_rider.json
"""
import json

from autogear.data import load_segments
from autogear.paths import RESULTS, ensure_dirs
from autogear.protocol import run_cross_rider

if __name__ == "__main__":
    ensure_dirs()
    res = run_cross_rider(load_segments(), kinds=("R0", "R1", "RT", "GBDT"))
    (RESULTS / "cross_rider.json").write_text(json.dumps(res, indent=1, default=float))
