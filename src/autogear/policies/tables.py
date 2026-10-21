"""Decision tables: hand-written rules on the current state, each giving a target cadence c*.

R0  c* = a                          constant cadence
R1  c* = a + b * slope              cadence adapted to the slope
RT  c* = a + b * slope + e * P      ... and to the pedalling power P = torque x angular velocity

RT is the decision table that a torque sensor would allow; the product does without it. The
constants are fitted by least squares to the rider's cadence on the training rides. The target
development is then L* = 60 v_hat / c*, and the shared controller turns it into shifts.
"""
from __future__ import annotations

import numpy as np

from ..features import Segment

KINDS = ("R0", "R1", "RT")
PEDAL_FIT_MIN_CADENCE = 40.0


class DecisionTable:
    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.coef: np.ndarray | None = None

    def _design(self, s: Segment, mask: np.ndarray | slice = slice(None)) -> np.ndarray:
        cols = [np.ones(s.n)[mask]]
        if self.kind in ("R1", "RT"):
            cols.append(s.slope5[mask])
        if self.kind == "RT":
            cols.append(s.power5[mask])
        return np.column_stack(cols)

    def fit(self, segs: list[Segment], horizon: int | None = None) -> "DecisionTable":
        A, y = [], []
        for s in segs:
            ped = (s.coasting == 0) & (s.cadence > PEDAL_FIT_MIN_CADENCE)
            A.append(self._design(s, ped))
            y.append(s.cadence[ped])
        self.coef = np.linalg.lstsq(np.vstack(A), np.concatenate(y), rcond=None)[0]
        return self

    def target_cadence(self, s: Segment) -> np.ndarray:
        return np.clip(self._design(s) @ self.coef, 50, 120)

    def predict(self, s: Segment) -> np.ndarray:
        """Target log-development (NaN when the derived speed is below 0.5 m/s)."""
        c_star = self.target_cadence(s)
        return np.where(s.v_hat > 0.5, np.log(60 * np.maximum(s.v_hat, 0.5) / c_star), np.nan)

    def size_bytes(self) -> int:
        return 4 * len(self.coef)
