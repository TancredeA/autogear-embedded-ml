"""Offline event-level scores, the way the first model was judged during AutoGear.

A true event is a second where the rider shifts. Predicted positives that follow each other (same
sign, gap <= merge_gap) are merged into one predicted event, anchored on the highest score. Events
are matched greedily in time order within +-tol seconds; with `use_sign`, directions must agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_events(action: np.ndarray, t: np.ndarray) -> list[tuple[int, int]]:
    return [(int(ti), int(np.sign(a))) for a, ti in zip(action, t) if a != 0]


def predicted_events(action: np.ndarray, t: np.ndarray, score: np.ndarray | None = None,
                     merge_gap: int = 2) -> list[tuple[int, int]]:
    order = np.argsort(t)
    a, t = np.asarray(action)[order], np.asarray(t)[order]
    s = np.asarray(score)[order] if score is not None else None
    events, i, n = [], 0, len(a)
    while i < n:
        if a[i] == 0:
            i += 1
            continue
        sign = int(np.sign(a[i]))
        j = last = i
        while j + 1 < n and (t[j + 1] - t[last]) <= merge_gap:
            j += 1
            if a[j] == sign:
                last = j
        run = [k for k in range(i, last + 1) if a[k] == sign]
        anchor = run[int(np.argmax(s[run]))] if s is not None else run[len(run) // 2]
        events.append((int(t[anchor]), sign))
        i = last + 1
    return events


def match(true_ev, pred_ev, tol: int = 2, use_sign: bool = True) -> tuple[int, int, int]:
    true_ev, pred_ev = sorted(true_ev), sorted(pred_ev)
    used = [False] * len(pred_ev)
    tp = 0
    for tt, ts in true_ev:
        best, best_d = -1, tol + 1
        for k, (pt, ps) in enumerate(pred_ev):
            if used[k] or (use_sign and ps != ts):
                continue
            d = abs(pt - tt)
            if d <= tol and d < best_d:
                best, best_d = k, d
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, used.count(False), len(true_ev) - tp


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def event_scores(true_action, pred_action, t, segment, score=None, tol: int = 2,
                 merge_gap: int = 2) -> dict:
    """Precision, recall and F1 summed over segments; 'detect' ignores direction, 'action' does not."""
    df = pd.DataFrame({"ta": true_action, "pa": pred_action, "t": t, "seg": segment})
    if score is not None:
        df["sc"] = score
    totals = {"detect": [0, 0, 0], "action": [0, 0, 0]}
    for _, g in df.groupby("seg", sort=False):
        te = true_events(g.ta.to_numpy(), g.t.to_numpy())
        pe = predicted_events(g.pa.to_numpy(), g.t.to_numpy(),
                              g.sc.to_numpy() if score is not None else None, merge_gap)
        for key, use_sign in (("detect", False), ("action", True)):
            tp, fp, fn = match(te, pe, tol=tol, use_sign=use_sign)
            totals[key] = [totals[key][0] + tp, totals[key][1] + fp, totals[key][2] + fn]
    out = {}
    for key, (tp, fp, fn) in totals.items():
        p, r, f = prf(tp, fp, fn)
        out[key] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f)
    return out
