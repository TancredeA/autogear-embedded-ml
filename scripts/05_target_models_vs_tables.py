"""Main comparison (Sections 5.1-5.2 of the report), 14 within-rider tests, closed-loop replay.

Decision tables R0, R1, RT (RT uses the pedalling power, i.e. torque) and target models GBDT
(retained), LIN, TAB, MLP, RF, all with the same controller and the same nested selection.
Output: results/target_policies.json

Usage: python scripts/05_target_models_vs_tables.py [KIND ...]   (default: all eight)
"""
import json
import sys

from autogear.data import load_segments
from autogear.paths import RESULTS, ensure_dirs
from autogear.policies import ALL_TARGET_KINDS
from autogear.protocol import run_within_rider

if __name__ == "__main__":
    ensure_dirs()
    kinds = tuple(sys.argv[1:]) or ALL_TARGET_KINDS
    res = run_within_rider(load_segments(), kinds)
    name = "target_policies.json" if kinds == ALL_TARGET_KINDS else f"target_policies_{'_'.join(kinds)}.json"
    (RESULTS / name).write_text(json.dumps(res, indent=1, default=float))
