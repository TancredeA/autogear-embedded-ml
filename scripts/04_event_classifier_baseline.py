"""First approach (Section 3 of the report): two-stage classifier of shift events, 14 tests.

Closed-loop replay with its own gears, plus the offline scores used during AutoGear
(stage-1 PR-AUC and event F1 within +-2 s). Output: results/event_classifier.json
"""
import json

from autogear.data import load_segments
from autogear.paths import RESULTS, ensure_dirs
from autogear.protocol import run_event_baseline

if __name__ == "__main__":
    ensure_dirs()
    res = run_event_baseline(load_segments())
    (RESULTS / "event_classifier.json").write_text(json.dumps(res, indent=1, default=float))
