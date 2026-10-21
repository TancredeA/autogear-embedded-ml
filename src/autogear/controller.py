"""Hysteresis controller shared by every target-based policy (learned models and decision tables).

Given a target log-development l_hat(t), the controller shifts up one step when, for k consecutive
seconds and while the rider pedals,

    l_hat(s) - log L(g)  >=  (1/2 + h) * (log L(g+1) - log L(g)),

shifts down in the symmetric case and holds otherwise. Equivalently, gear g is kept as long as the
target stays in the dead band L(g) (L(g-1)/L(g))^(1/2+h) < L_hat < L(g) (L(g+1)/L(g))^(1/2+h).
Mechanical rules: one step at a time, no shift while coasting or below 20 rpm. An optional
anti-reversal delay d forbids undoing a shift within d seconds.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .features import Segment


def control(seg: Segment, target: np.ndarray, h: float, k: int, d: int = 0,
            k_down: int | None = None) -> np.ndarray:
    """Gear sequence chosen by the controller on a segment (closed loop on its own gear).

    `k_down`, if given, is a separate persistence for downshifts (asymmetric controller).
    """
    k_up, k_dn = k, (k if k_down is None else k_down)
    n, lg, K = seg.n, seg.log_levels, len(seg.levels)
    g = np.empty(n, int)
    g[0] = seg.gear[0]
    up = dn = 0
    last_dir, last_t = 0, -10 ** 9
    for t in range(n - 1):
        gt, step = g[t], 0
        pedalling = seg.coasting[t] == 0 and seg.cadence[t] > C.CONTROL_MIN_CADENCE_RPM
        if pedalling and np.isfinite(target[t]):
            up = up + 1 if gt < K and target[t] - lg[gt - 1] >= (0.5 + h) * (lg[gt] - lg[gt - 1]) else 0
            dn = dn + 1 if gt > 1 and lg[gt - 1] - target[t] >= (0.5 + h) * (lg[gt - 1] - lg[gt - 2]) else 0
            if up >= k_up:
                step, up = 1, 0
            elif dn >= k_dn:
                step, dn = -1, 0
            if step and last_dir and step != last_dir and t - last_t < d:
                step = 0
        if step:
            last_dir, last_t = step, t
        g[t + 1] = gt + step
    return g
