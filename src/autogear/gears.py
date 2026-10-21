"""Recover the engaged gear from speed and cadence.

While the rider pedals, the development d(t) = 60 v(t) / c(t) (metres per crank revolution) equals
the development of the engaged gear. Its log-values cluster around one level per gear used.

1. `gear_levels` finds those levels as the peaks of the density of log d.
2. `reconstruct_viterbi` chooses the gear sequence that trades fidelity to the measured ratio
   against a cost for each change (lambda), plus a cost per extra step in a multi-gear jump.
3. `validate_against_true_gears` compares the reconstruction with real Di2 gears.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from . import config as C


def development(cadence: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pedalling mask, development in m/rev, NaN when not pedalling)."""
    ped = (cadence >= C.PEDAL_MIN_CADENCE_RPM) & (speed >= C.PEDAL_MIN_SPEED_MS)
    dev = np.where(ped, 60 * speed / np.maximum(cadence, 1), np.nan)
    return ped, dev


def gear_levels(dev: np.ndarray, min_sep: float = C.LEVEL_MIN_SEPARATION,
                min_frac: float = C.LEVEL_MIN_FRACTION) -> np.ndarray:
    """Peaks of the (smoothed) density of log-development, i.e. the developments of the gears used."""
    x = np.log(dev[np.isfinite(dev)])
    bins = np.arange(x.min() - 0.05, x.max() + 0.05, 0.004)
    h, e = np.histogram(x, bins)
    h = gaussian_filter1d(h.astype(float), 2.5)
    centres = (e[:-1] + e[1:]) / 2
    peaks, _ = find_peaks(h, distance=int(min_sep / 0.004), height=h.max() * 0.02)
    keep = [centres[p] for p in peaks if (np.abs(x - centres[p]) < min_sep / 2).mean() >= min_frac]
    return np.exp(np.array(sorted(keep)))


def reconstruct_viterbi(cadence: np.ndarray, speed: np.ndarray, sigma: float = C.VITERBI_SIGMA,
                        lam: float = C.VITERBI_SWITCH_COST, jump: float = C.VITERBI_JUMP_COST
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Most likely gear sequence (1..K) and the K gear levels (m/rev).

    States are the detected gears. While pedalling, the emission cost is the squared log-distance
    to the gear level over 2 sigma^2 (capped); while coasting it is zero, so the gear is held.
    Changing gear costs lam, plus `jump` for each step beyond the first.
    """
    cadence = np.asarray(cadence, float)
    speed = np.asarray(speed, float)
    ped, dev = development(cadence, speed)
    levels = gear_levels(dev[ped])
    log_levels = np.log(levels)
    K, n = len(levels), len(cadence)
    emission = np.zeros((n, K))
    x = np.log(dev)
    emission[ped] = ((x[ped][:, None] - log_levels[None, :]) ** 2) / (2 * sigma ** 2)
    emission = np.minimum(emission, C.VITERBI_EMISSION_CAP)
    k = np.arange(K)
    steps = np.abs(k[:, None] - k[None, :])
    transition = np.where(steps == 0, 0.0, lam + jump * (steps - 1))
    cost = emission[0].copy()
    back = np.zeros((n, K), int)
    for t in range(1, n):
        total = cost[:, None] + transition          # (previous, current)
        back[t] = np.argmin(total, 0)
        cost = total[back[t], k] + emission[t]
    g = np.zeros(n, int)
    g[-1] = int(np.argmin(cost))
    for t in range(n - 1, 0, -1):
        g[t - 1] = back[t, g[t]]
    return g + 1, levels


def levels_from_true_gears(cadence: np.ndarray, speed: np.ndarray, teeth: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Gear index (1 = smallest development) and levels from real rear-sprocket teeth (Di2)."""
    cad = pd.Series(np.asarray(cadence, float))
    spd = pd.Series(np.asarray(speed, float))
    teeth = pd.Series(teeth)
    ped = (cad >= C.PEDAL_MIN_CADENCE_RPM) & (spd >= C.PEDAL_MIN_SPEED_MS)
    dev = 60 * spd / cad.replace(0, np.nan)
    med = dev[ped].groupby(teeth[ped]).median().sort_values()
    rank = {t: i + 1 for i, t in enumerate(med.index)}
    gear = teeth.map(rank).ffill().bfill().astype(int).values
    return gear, med.values


def shift_events(gear: np.ndarray) -> np.ndarray:
    """Indices t where gear[t] != gear[t-1]."""
    return np.flatnonzero(np.diff(gear) != 0) + 1


def validate_against_true_gears(cadence: np.ndarray, speed: np.ndarray, true_gear: np.ndarray,
                                true_levels: np.ndarray, true_shift_times: np.ndarray,
                                tol_s: int = 3) -> dict:
    """Compare the Viterbi reconstruction (run as if the gears were unknown) with true gears.

    state_accuracy: share of pedalling seconds where the reconstructed development is within 5 % of
    the true one. Events: each true shift (time in s from the first record) is matched one-to-one to
    the nearest reconstructed shift within +-tol_s seconds; precision and recall follow.
    """
    g_hat, lv_hat = reconstruct_viterbi(cadence, speed)
    ped, _ = development(np.asarray(cadence, float), np.asarray(speed, float))
    d_true = true_levels[true_gear - 1]
    d_hat = lv_hat[g_hat - 1]
    ok = np.abs(np.log(d_hat / d_true)) < np.log(1.05)
    t_hat = shift_events(g_hat)
    used = np.zeros(len(t_hat), bool)
    tp = 0
    for t in np.asarray(true_shift_times, float):
        cand = np.flatnonzero(~used & (np.abs(t_hat - t) <= tol_s))
        if len(cand):
            used[cand[np.argmin(np.abs(t_hat[cand] - t))]] = True
            tp += 1
    return dict(state_accuracy=float(ok[ped].mean()), state_accuracy_all_seconds=float(ok.mean()),
                event_precision=tp / max(1, len(t_hat)), event_recall=tp / max(1, len(true_shift_times)),
                true_shifts=int(len(true_shift_times)), reconstructed_shifts=int(len(t_hat)),
                minutes=len(cadence) / 60, levels_detected=len(lv_hat), levels_true=len(true_levels))
