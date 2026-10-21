"""Shift timing (Section 5.4 of the report): what sets the delay of the policy's shifts?

1. Horizon H x persistence k sweep for the boosted trees, with h = 0.6 fixed (no selection).
2. Decision tables at their selected setting (h = 0.3, k = 8) and at the model's (h = 0.6, k = 3).
3. Asymmetric controller: shorter persistence for downshifts.
Delays: each rider shift is matched to the nearest policy shift of the same direction within
+-10 s; positive = the policy is late. Output: results/timing_sweep.json
"""
import json

import numpy as np

from autogear.controller import control
from autogear.data import load_segments
from autogear.paths import RESULTS, ensure_dirs
from autogear.policies import make
from autogear.protocol import within_rider_tests
from autogear.replay import aggregate, segment_metrics, shift_delays


def run(tests, targets, ctrl) -> dict:
    ms, up, down, matched, total = [], [], [], 0, 0
    for t, tg in zip(tests, targets):
        for s in t.test:
            g = ctrl(s, tg[id(s)])
            ms.append(segment_metrics(s, g))
            o, m, n = shift_delays(s, g)
            up += o["up"]
            down += o["down"]
            matched += m
            total += n
    a = aggregate(ms)
    return dict(agree_pm1=a["agree_pm1"], exact=a["exact"], shifts_per_h=a["shifts_per_h"],
                reversal=a["reversal"], matched_share=matched / total,
                delay_up_median=float(np.median(up)), delay_down_median=float(np.median(down)),
                delay_up_mean=float(np.mean(up)), delay_down_mean=float(np.mean(down)),
                delay_mean=float(np.mean(up + down)))


def targets_for(kind, horizon, tests):
    out = []
    for t in tests:
        m = make(kind).fit(t.train, horizon)
        out.append({id(s): m.predict(s) for s in t.test})
    return out


def show(name, r):
    print(f"{name:34s} agree {r['agree_pm1'] * 100:5.1f}  shifts/h {r['shifts_per_h']:4.0f}  "
          f"rev {r['reversal'] * 100:4.1f}  delay mean {r['delay_mean']:+.1f} s "
          f"(up med {r['delay_up_median']:+.0f}, down med {r['delay_down_median']:+.0f})", flush=True)


if __name__ == "__main__":
    ensure_dirs()
    tests = within_rider_tests(load_segments())
    res = {"horizon_x_persistence": {}, "tables": {}, "asymmetric": {}}
    for H in (0, 1, 2, 3, 5, 8):
        tg = targets_for("GBDT", H, tests)
        for k in (1, 3, 5):
            r = run(tests, tg, lambda s, l: control(s, l, h=0.6, k=k, d=0))
            res["horizon_x_persistence"][f"H{H}_k{k}"] = {"H": H, "k": k, **r}
            show(f"GBDT H={H} k={k}", r)
        if H == 3:
            for kd, ku in ((3, 3), (1, 3), (2, 3), (1, 5)):
                r = run(tests, tg, lambda s, l: control(s, l, h=0.6, k=ku, d=0, k_down=kd))
                res["asymmetric"][f"up{ku}_down{kd}"] = {"k_up": ku, "k_down": kd, **r}
                show(f"GBDT H=3 k_up={ku} k_down={kd}", r)
    for kind in ("R0", "R1", "RT"):
        tg = targets_for(kind, None, tests)
        for h, k in ((0.3, 8), (0.6, 3)):
            r = run(tests, tg, lambda s, l: control(s, l, h=h, k=k, d=0))
            res["tables"][f"{kind}_h{h}_k{k}"] = {"kind": kind, "h": h, "k": k, **r}
            show(f"{kind} h={h} k={k}", r)
    (RESULTS / "timing_sweep.json").write_text(json.dumps(res, indent=1))
