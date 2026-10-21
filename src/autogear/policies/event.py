"""First approach (AutoGear): a two-stage classifier of shift events on a raw 30 x 5 window.

At every second, predict the rider's action a(t) in {-1, 0, +1}.
Stage 1: "shift or not", boosted trees, positive class weighted by n(neg)/n(pos).
Stage 2: direction, trained on shift seconds only.
Inputs: the last 30 s of [cadence, slope, acceleration, engaged gear, coasting], flattened (150).
At run time a shift is commanded when the stage-1 probability exceeds theta, in the direction of
stage 2, and no new shift is allowed during a refractory period. Cadence and gear are inputs, so
the model sees the consequences of its own choices: it is inside the loop.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from .. import config as C
from ..event_metrics import event_scores
from ..features import Segment

W = C.EVENT_WINDOW_S


def _channels(s: Segment, cadence: np.ndarray, gear: np.ndarray) -> np.ndarray:
    return np.column_stack([cadence, s.X[:, 5], s.X[:, 8], gear.astype(float),
                            s.coasting.astype(float)]).astype(np.float32)


def windows(s: Segment, cadence: np.ndarray, gear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = _channels(s, cadence, gear)
    idx = np.arange(W - 1, s.n)
    if not len(idx):
        return np.zeros((0, 5 * W), np.float32), idx
    return np.stack([F[t - W + 1:t + 1].reshape(-1) for t in idx]), idx


def actions(s: Segment) -> np.ndarray:
    return np.sign(np.r_[np.diff(s.gear), 0])


def training_set(segs: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for s in segs:
        Wd, idx = windows(s, s.cadence, s.gear)
        if len(idx):
            Xs.append(Wd)
            ys.append(actions(s)[idx])
    return np.vstack(Xs), np.concatenate(ys)


class EventClassifier:
    def fit(self, segs: list[Segment]) -> "EventClassifier":
        X, a = training_set(segs)
        y1 = (a != 0).astype(int)
        cut = int(len(X) * (1 - C.EVENT_VALIDATION_SHARE))
        spw = (y1 == 0).sum() / max(1, y1.sum())
        self.stage1 = xgb.train({**C.EVENT_PARAMS, "objective": "binary:logistic", "eval_metric": "aucpr",
                                 "scale_pos_weight": spw}, xgb.DMatrix(X[:cut], y1[:cut]), C.EVENT_MAX_ROUNDS,
                                evals=[(xgb.DMatrix(X[cut:], y1[cut:]), "v")],
                                early_stopping_rounds=C.EVENT_EARLY_STOPPING, verbose_eval=False)
        m = a != 0
        self.stage2 = xgb.train({**C.EVENT_PARAMS, "objective": "binary:logistic"},
                                xgb.DMatrix(X[m], (a[m] > 0).astype(int)), C.EVENT_DIRECTION_ROUNDS)
        self.it1 = (0, self.stage1.best_iteration + 1)
        return self

    def ride(self, s: Segment, th: float, cool: int) -> np.ndarray:
        """Closed loop: the classifier chooses its own gears and sees its own cadence."""
        n, L, gE, c, co = s.n, s.levels, s.gear, s.cadence, s.coasting
        K = len(L)
        slope, acc = s.X[:, 5], s.X[:, 8]
        g = np.empty(n, int)
        g[0] = gE[0]
        cP = np.zeros(n)
        last = -10 ** 9
        for t in range(n):
            ped = co[t] == 0 and c[t] > C.CONTROL_MIN_CADENCE_RPM
            cP[t] = c[t] * L[gE[t] - 1] / L[g[t] - 1] if ped else c[t]
            if t == n - 1:
                break
            step = 0
            if t >= W - 1 and ped and t - last >= cool:
                lo = t - W + 1
                x = np.column_stack([cP[lo:t + 1], slope[lo:t + 1], acc[lo:t + 1], g[lo:t + 1].astype(float),
                                     co[lo:t + 1].astype(float)]).astype(np.float32).reshape(1, -1)
                if float(self.stage1.inplace_predict(x, iteration_range=self.it1)[0]) >= th:
                    step = 1 if float(self.stage2.inplace_predict(x)[0]) >= 0.5 else -1
                    if not 1 <= g[t] + step <= K:
                        step = 0
                    if step:
                        last = t
            g[t + 1] = g[t] + step
        return g

    def offline_scores(self, segs: list[Segment], th: float) -> dict:
        """Scores on the rider's own trajectory (inputs always come from the rider's gears)."""
        ys, ps, pas, ts, sg = [], [], [], [], []
        for k, s in enumerate(segs):
            Wd, idx = windows(s, s.cadence, s.gear)
            if not len(idx):
                continue
            p1 = self.stage1.inplace_predict(Wd, iteration_range=self.it1)
            p2 = self.stage2.inplace_predict(Wd)
            ys.append(actions(s)[idx])
            ps.append(p1)
            pas.append(np.where(p1 >= th, np.where(p2 >= 0.5, 1, -1), 0))
            ts.append(idx)
            sg.append(np.full(len(idx), k))
        y, p = np.concatenate(ys), np.concatenate(ps)
        ev = event_scores(y, np.concatenate(pas), np.concatenate(ts), np.concatenate(sg), score=p,
                          tol=C.EVENT_F1_TOLERANCE_S, merge_gap=2)
        return dict(pr_auc=float(average_precision_score((y != 0).astype(int), p)),
                    prevalence=float((y != 0).mean()), event_f1=ev["action"]["f1"],
                    event_precision=ev["action"]["precision"], event_recall=ev["action"]["recall"])

    def size_bytes(self) -> int:
        total = 0
        for b, n_trees in ((self.stage1, self.stage1.best_iteration + 1), (self.stage2, None)):
            dump = b.get_dump(dump_format="json")
            dump = dump[:n_trees] if n_trees else dump
            total += (sum(t.count('"split"') for t in dump) * C.BYTES_PER_TREE_NODE
                      + sum(t.count('"leaf"') for t in dump) * C.BYTES_PER_FLOAT)
        return total
