"""The inputs the device can compute, and the per-segment arrays used by the replay.

The device does not measure speed. While the rider pedals it derives it from the cadence and the
development of the engaged gear, v_hat = c * L(g) / 60; while coasting it integrates the
longitudinal acceleration. Because the replay recomputes the policy's cadence as
c_P = c * L(g*) / L(g_P), the product c_P * L(g_P) equals c * L(g*): v_hat, and hence every input
below, is the same whatever gear the policy selects. The learned model is outside the loop.

Inputs (14, in this order, all causal):
    0  v_hat                          5  slope                   9  mean acc 3 s
    1  mean v_hat 5 s                 6  mean slope 5 s         10  mean acc 5 s
    2  mean v_hat 10 s                7  mean slope 15 s        11  coasting flag
    3  v_hat(t) - v_hat(t-3)          8  acc                    12  share of coasting over 5 s
    4  v_hat(t) - v_hat(t-8)                                    13  seconds since pedalling resumed (cap 10)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

FEATURE_NAMES = ["v_hat", "v_hat_mean5", "v_hat_mean10", "v_hat_diff3", "v_hat_diff8",
                 "slope", "slope_mean5", "slope_mean15", "acc", "acc_mean3", "acc_mean5",
                 "coasting", "coasting_share5", "restart_s"]
FEATURE_GROUPS = {"speed": [0, 1, 2, 3, 4], "slope": [5, 6, 7], "acceleration": [8, 9, 10],
                  "restart counter": [13], "coasting": [11, 12]}
N_FEATURES = len(FEATURE_NAMES)


def causal_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean over the last w samples (fewer at the start of a segment)."""
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def lagged_diff(x: np.ndarray, lag: int) -> np.ndarray:
    """x(t) - x(t - lag), with x(t - lag) = x(0) at the start of a segment."""
    return x - np.r_[np.full(lag, x[0]), x[:-lag]]


def derived_speed(cadence: np.ndarray, gear: np.ndarray, levels: np.ndarray, coasting: np.ndarray,
                  acc: np.ndarray) -> np.ndarray:
    """v_hat: c * L(g) / 60 while pedalling (c > 20 rpm), else integrate the acceleration."""
    n = len(cadence)
    v = np.zeros(n)
    for t in range(n):
        if coasting[t] == 0 and cadence[t] > C.CONTROL_MIN_CADENCE_RPM:
            v[t] = cadence[t] * levels[gear[t] - 1] / 60
        else:
            v[t] = max(0.0, (v[t - 1] if t else 0.0) + acc[t])
    return v


def restart_counter(coasting: np.ndarray, cap: int = 10) -> np.ndarray:
    tp = np.zeros(len(coasting))
    for t in range(len(coasting)):
        tp[t] = 0 if coasting[t] else min(cap, (tp[t - 1] if t else 0) + 1)
    return tp


def device_features(v_hat: np.ndarray, slope: np.ndarray, acc: np.ndarray,
                    coasting: np.ndarray) -> np.ndarray:
    """The 14 inputs of the target models, as float32 (the precision used on the device)."""
    co = coasting.astype(float)
    return np.column_stack([
        v_hat, causal_mean(v_hat, 5), causal_mean(v_hat, 10), lagged_diff(v_hat, 3), lagged_diff(v_hat, 8),
        slope, causal_mean(slope, 5), causal_mean(slope, 15),
        acc, causal_mean(acc, 3), causal_mean(acc, 5),
        co, causal_mean(co, 5), restart_counter(coasting)]).astype(np.float32)


@dataclass(eq=False)
class Segment:
    """A continuous piece of ride, with everything the policies and the replay need."""
    ride: str
    rider: str
    levels: np.ndarray          # developments of the gears of this bicycle (m/rev), ascending
    cadence: np.ndarray         # rider's cadence (rpm)
    gear: np.ndarray            # rider's gear g*(t), 1..K
    coasting: np.ndarray        # 0/1
    v_hat: np.ndarray           # derived speed (m/s)
    X: np.ndarray               # (n, 14) device inputs
    slope5: np.ndarray          # 5-s mean slope, input of decision tables R1 and RT
    power5: np.ndarray          # 5-s mean power (W), input of the torque-aware table RT only
    log_levels: np.ndarray = field(init=False)

    def __post_init__(self):
        self.log_levels = np.log(self.levels)

    @property
    def n(self) -> int:
        return len(self.cadence)

    @property
    def hours(self) -> float:
        return self.n / 3600

    def slice(self, a: int, b: int) -> "Segment":
        return Segment(self.ride, self.rider, self.levels, self.cadence[a:b], self.gear[a:b],
                       self.coasting[a:b], self.v_hat[a:b], self.X[a:b], self.slope5[a:b], self.power5[a:b])


def make_segment(seg: pd.DataFrame, levels: np.ndarray) -> Segment:
    c = seg.cadence.values.astype(float)
    g = seg.gear.values.astype(int)
    co = seg.coasting.values.astype(int)
    acc = seg.acc.values.astype(float)
    slope = seg.slope.values.astype(float)
    power = seg.power.fillna(0).values.astype(float)
    v = derived_speed(c, g, levels, co, acc)
    return Segment(ride=seg.ride.iloc[0], rider=seg.rider.iloc[0], levels=levels, cadence=c, gear=g,
                   coasting=co, v_hat=v, X=device_features(v, slope, acc, co),
                   slope5=causal_mean(slope, 5), power5=causal_mean(power, 5))


def load_segments(data: pd.DataFrame, levels: dict[str, list[float]]) -> list[Segment]:
    lv = {k: np.array(v) for k, v in levels.items()}
    return [make_segment(s.reset_index(drop=True), lv[s.ride.iloc[0]])
            for _, s in data.groupby("segment", sort=False)]
