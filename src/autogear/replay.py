"""Closed-loop replay: the measures of a policy that chose its own gears on a recorded ride.

The policy's cadence is recomputed at unchanged speed, c_P(t) = c(t) L(g*(t)) / L(g_P(t)), while
the rider pedals. Five measures are aggregated, weighted by duration:

agree_pm1      share of seconds where the policy's gear is at most one step from the rider's
exact          share of seconds with exactly the rider's gear
shifts_per_h   shifts per hour (compare with the rider's own rate)
reversal       share of shifts undone within 4 s
cadence_band   share of pedalling time with c_P inside the rider's own 10-90 % cadence deciles
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .features import Segment


def segment_metrics(seg: Segment, g: np.ndarray) -> dict:
    gE, c, co, L = seg.gear, seg.cadence, seg.coasting, seg.levels
    ped = (co == 0) & (c > C.CONTROL_MIN_CADENCE_RPM)
    cP = np.where(ped, c * L[gE - 1] / L[g - 1], c)
    shifts = np.flatnonzero(np.diff(g) != 0) + 1
    rev = 0
    for i in shifts:
        nxt = shifts[(shifts > i) & (shifts <= i + C.REVERSAL_WINDOW_S)]
        if len(nxt) and np.sign(g[nxt[0]] - g[nxt[0] - 1]) != np.sign(g[i] - g[i - 1]):
            rev += 1
    lo, hi = np.percentile(c[ped], [10, 90]) if ped.sum() > 30 else (0, 999)
    return dict(n=seg.n, a1=float(np.sum(np.abs(g - gE) <= 1)), ex=float(np.sum(g == gE)),
                sh=len(shifts), rev=rev, band=float(np.sum(ped & (cP >= lo) & (cP <= hi))),
                ped=float(ped.sum()))


def aggregate(ms: list[dict]) -> dict:
    n = sum(m["n"] for m in ms)
    s = sum(m["sh"] for m in ms)
    return dict(agree_pm1=sum(m["a1"] for m in ms) / n, exact=sum(m["ex"] for m in ms) / n,
                shifts_per_h=s / (n / 3600),
                reversal=(sum(m["rev"] for m in ms) / s if s else 0.0),
                cadence_band=sum(m["band"] for m in ms) / max(1, sum(m["ped"] for m in ms)),
                hours=n / 3600)


def rider_rate(segs: list[Segment]) -> float:
    """The rider's own shifts per hour on these segments."""
    n = sum(s.n for s in segs)
    return sum(int((np.diff(s.gear) != 0).sum()) for s in segs) / (n / 3600)


def evaluate(segs: list[Segment], gears: dict[int, np.ndarray]) -> dict:
    """Aggregate measures; `gears` maps id(segment) to the policy's gear sequence."""
    return aggregate([segment_metrics(s, gears[id(s)]) for s in segs])


def reference_rows(segs: list[Segment]) -> dict:
    """The rider (reference) and a policy that never shifts."""
    return dict(rider=aggregate([segment_metrics(s, s.gear) for s in segs]),
                never_shift=aggregate([segment_metrics(s, np.full(s.n, s.gear[0])) for s in segs]))


def shift_delays(seg: Segment, g: np.ndarray, window: int = 10) -> tuple[dict, int, int]:
    """Delay (s) of the policy's shifts behind the rider's.

    Each rider shift is matched to the nearest policy shift in the same direction within +-window s.
    Positive = the policy is late. Returns ({'up': [...], 'down': [...]}, matched, total).
    """
    gE = seg.gear
    out = {"up": [], "down": []}
    matched = total = 0
    t_rider = np.flatnonzero(np.diff(gE) != 0) + 1
    t_pol = np.flatnonzero(np.diff(g) != 0) + 1
    d_pol = np.sign(g[t_pol] - g[t_pol - 1])
    for t in t_rider:
        d = np.sign(gE[t] - gE[t - 1])
        total += 1
        cand = t_pol[(d_pol == d) & (np.abs(t_pol - t) <= window)]
        if len(cand):
            matched += 1
            out["up" if d > 0 else "down"].append(int(cand[np.argmin(np.abs(cand - t))] - t))
    return out, matched, total
