"""Target models: regress the log-development of the gear the rider will want H seconds ahead.

    l_hat(t) = f(z(t)) ~ E[ log L(g*(t + H)) | z(t) ]

All models use the same 14 device inputs (features.py), none of which depends on the gear chosen.

GBDT  100 boosted regression trees of depth <= 4 (the retained model)
LIN   ridge regression on standardised inputs
TAB   learned speed x slope table (median target per cell): a shift schedule fitted to data
MLP   two hidden layers of 16 units, standardised inputs, early stopping
RF    random forest, 50 trees of depth <= 6
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .. import config as C
from ..features import Segment

KINDS = ("GBDT", "LIN", "TAB", "MLP", "RF")


def training_set(segs: list[Segment], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (z(t), log L(g*(t + H))) within each segment."""
    X = np.vstack([s.X[:s.n - horizon] for s in segs])
    y = np.concatenate([s.log_levels[s.gear[horizon:] - 1] for s in segs])
    return X, y


def tree_storage_bytes(booster: xgb.Booster) -> int:
    """Size under the convention of the report: 8 bytes per decision node, 4 per leaf."""
    dump = booster.get_dump(dump_format="json")
    return (sum(t.count('"split"') for t in dump) * C.BYTES_PER_TREE_NODE
            + sum(t.count('"leaf"') for t in dump) * C.BYTES_PER_FLOAT)


class SpeedSlopeTable:
    """Median target per cell of 1 m/s of speed by 1 % of 15-s mean slope (17 x 17 cells)."""

    def _cells(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.clip(X[:, 0].astype(int), 0, 16)
        j = np.clip(np.round((X[:, 7] + 0.08) / 0.01).astype(int), 0, 16)
        return i, j

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpeedSlopeTable":
        i, j = self._cells(X)
        T = np.full((17, 17), np.nan)
        overall = np.median(y)
        for a in range(17):
            row = y[i == a]
            row_median = np.median(row) if len(row) else np.nan
            for b in range(17):
                m = (i == a) & (j == b)
                T[a, b] = np.median(y[m]) if m.sum() >= 20 else row_median
        for a in range(17):                      # empty cells: nearest filled speed
            for b in range(17):
                if np.isnan(T[a, b]):
                    filled = [(abs(a2 - a), T[a2, b]) for a2 in range(17) if not np.isnan(T[a2, b])]
                    T[a, b] = min(filled)[1] if filled else overall
        self.T = np.round(T * 100) / 100         # quantised to 0.01 in log: one byte per cell
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        i, j = self._cells(X)
        return self.T[i, j]


class TargetModel:
    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.model = None
        self.scaler: StandardScaler | None = None

    def fit(self, segs: list[Segment], horizon: int) -> "TargetModel":
        if self.kind == "GBDT":
            X, y = training_set(segs, horizon)
            self.model = xgb.train(C.GBDT_PARAMS, xgb.DMatrix(X, y), C.GBDT_ROUNDS)
            return self
        X, y = training_set(segs, horizon)
        if self.kind == "LIN":
            self.scaler = StandardScaler().fit(X)
            self.model = Ridge(alpha=1.0).fit(self.scaler.transform(X), y)
        elif self.kind == "MLP":
            self.scaler = StandardScaler().fit(X)
            self.model = MLPRegressor(hidden_layer_sizes=(16, 16), alpha=1e-3, max_iter=400,
                                      early_stopping=True, random_state=0).fit(self.scaler.transform(X), y)
        elif self.kind == "RF":
            self.model = RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=20,
                                               n_jobs=1, random_state=0).fit(X, y)
        elif self.kind == "TAB":
            self.model = SpeedSlopeTable().fit(X, y)
        return self

    def predict(self, s: Segment) -> np.ndarray:
        if self.kind == "GBDT":
            return self.model.inplace_predict(s.X)
        X = self.scaler.transform(s.X) if self.scaler is not None else s.X
        return self.model.predict(X)

    def size_bytes(self) -> int:
        F = C.BYTES_PER_FLOAT
        if self.kind == "GBDT":
            return tree_storage_bytes(self.model)
        if self.kind == "LIN":
            return (self.model.coef_.size + 1 + 2 * self.scaler.mean_.size) * F
        if self.kind == "MLP":
            m = self.model
            return (sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
                    + 2 * self.scaler.mean_.size) * F
        if self.kind == "RF":
            return sum(e.tree_.node_count for e in self.model.estimators_) * C.BYTES_PER_TREE_NODE
        if self.kind == "TAB":
            return self.model.T.size
        raise ValueError(self.kind)
