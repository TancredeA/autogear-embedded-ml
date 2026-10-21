"""Evaluation protocols.

Within-rider protocol (the product scenario: a model is fitted on a reference rider and tested on
rides, or parts of rides, by that rider that it has not seen):
  - Inria rider (4 rides): each ride is tested in turn, the other three train (4 tests). Decision
    parameters are chosen by nested cross-validation over those three rides.
  - Five single-ride riders: the ride is split in two halves in time; one half trains, the other
    tests, and vice versa (10 tests). Decision parameters are those chosen on the four Inria rides,
    frozen, never retuned.
  -> 14 tests, 18.9 h.

Selection rule: among settings that shift at most 1.2 x as often as the rider and reverse at most
8 % of their shifts, the one with the best agreement within one gear.

Cross-rider protocol (no personalisation): leave one rider out, select by inner leave-one-rider-out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import config as C
from .controller import control
from .features import Segment
from .policies import ALL_TARGET_KINDS, TABLE_KINDS, make
from .replay import aggregate, reference_rows, rider_rate, segment_metrics

INRIA = "inria"


@dataclass
class Test:
    name: str
    rider: str
    group: str                  # "inria" or "single-ride"
    train: list[Segment]
    test: list[Segment]
    frozen: bool                # True when decision parameters come from the Inria selection

    @property
    def hours(self) -> float:
        return sum(s.n for s in self.test) / 3600


def split_halves(segs: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    """First and second halves of a ride, in time (a long segment is cut in two if needed)."""
    n = sum(s.n for s in segs)
    acc, A, B = 0, [], []
    for s in segs:
        (A if acc < n / 2 else B).append(s)
        acc += s.n
    if not B:
        s = A.pop()
        h = s.n // 2
        A.append(s.slice(0, h))
        B.append(s.slice(h, s.n))
    return A, B


def inria_rides(segs: list[Segment]) -> list[str]:
    return sorted({s.ride for s in segs if s.rider == INRIA})


def within_rider_tests(segs: list[Segment]) -> list[Test]:
    tests = []
    rides = inria_rides(segs)
    for r in rides:
        tests.append(Test(r, INRIA, "inria", [s for s in segs if s.rider == INRIA and s.ride != r],
                          [s for s in segs if s.ride == r], frozen=False))
    for rider in sorted({s.rider for s in segs} - {INRIA}):
        A, B = split_halves([s for s in segs if s.rider == rider])
        tests.append(Test(f"{rider} A>B", rider, "single-ride", A, B, frozen=True))
        tests.append(Test(f"{rider} B>A", rider, "single-ride", B, A, frozen=True))
    return tests


# ---------------------------------------------------------------------------------------------
# Target-based policies
# ---------------------------------------------------------------------------------------------
def horizons(kind: str) -> list:
    return [None] if kind in TABLE_KINDS else list(C.HORIZONS_S)


def fit_predict(kind: str, horizon, train: list[Segment], test: list[Segment]) -> tuple[dict, int]:
    model = make(kind).fit(train, horizon)
    return {id(s): model.predict(s) for s in test}, model.size_bytes()


def select_setting(candidates: dict, folds: dict[str, list[Segment]], grid=C.CONTROLLER_GRID):
    """candidates: {horizon: {id(seg): target}} computed out-of-fold. Returns (horizon, params)."""
    segs = [s for v in folds.values() for s in v]
    cap = C.MAX_RATE_RATIO * rider_rate(segs)
    best = None
    for key, targets in candidates.items():
        for p in grid:
            r = aggregate([segment_metrics(s, control(s, targets[id(s)], **p)) for s in segs])
            ok = r["shifts_per_h"] <= cap and r["reversal"] <= C.MAX_REVERSAL_SHARE
            score = r["agree_pm1"] - (0 if ok else 1)
            if best is None or score > best[0]:
                best = (score, key, p)
    return best[1], best[2]


def choose(kind: str, folds: dict[str, list[Segment]]):
    """Nested selection: every fold is predicted by a model trained on the other folds."""
    candidates = {}
    for key in horizons(kind):
        targets = {}
        for name, held_out in folds.items():
            train = [s for other, v in folds.items() if other != name for s in v]
            targets.update(fit_predict(kind, key, train, held_out)[0])
        candidates[key] = targets
    return select_setting(candidates, folds)


def evaluate_target_policy(kind: str, key, params: dict, train, test) -> dict:
    targets, size = fit_predict(kind, key, train, test)
    res = aggregate([segment_metrics(s, control(s, targets[id(s)], **params)) for s in test])
    return {**res, "params": {**params, "H": key}, "size_bytes": size}


def run_within_rider(segs: list[Segment], kinds=ALL_TARGET_KINDS,
                     log: Callable[[str], None] = print) -> dict:
    tests = within_rider_tests(segs)
    by_ride = lambda rides: {r: [s for s in segs if s.ride == r] for r in rides}
    rides = inria_rides(segs)
    frozen = {k: choose(k, by_ride(rides)) for k in kinds}
    log(f"frozen settings: { {k: (v[0], v[1]) for k, v in frozen.items()} }")
    out = {}
    for t in tests:
        row = {"rider": t.rider, "group": t.group, "hours": t.hours, **reference_rows(t.test)}
        for k in kinds:
            key, p = frozen[k] if t.frozen else choose(k, by_ride([r for r in rides if r != t.name]))
            row[k] = evaluate_target_policy(k, key, p, t.train, t.test)
        out[t.name] = row
        log(f"{t.name:24s} " + " ".join(f"{k}={row[k]['agree_pm1'] * 100:.1f}" for k in kinds))
    return {"tests": out, "frozen": {k: {"H": v[0], **v[1]} for k, v in frozen.items()}}


def run_cross_rider(segs: list[Segment], kinds=("R0", "R1", "RT", "GBDT"),
                    log: Callable[[str], None] = print) -> dict:
    """Leave one rider out: no personalisation."""
    riders = list(dict.fromkeys(s.rider for s in segs))      # order of appearance in the dataset
    out = {}
    for r in riders:
        train = [s for s in segs if s.rider != r]
        test = [s for s in segs if s.rider == r]
        folds = {ir: [s for s in train if s.rider == ir] for ir in riders if ir != r}
        row = {"hours": sum(s.n for s in test) / 3600, **reference_rows(test)}
        for k in kinds:
            key, p = choose(k, folds)
            row[k] = evaluate_target_policy(k, key, p, train, test)
        out[r] = row
        log(f"{r:24s} " + " ".join(f"{k}={row[k]['agree_pm1'] * 100:.1f}" for k in kinds))
    return {"tests": out}


# ---------------------------------------------------------------------------------------------
# Event classifier (first approach)
# ---------------------------------------------------------------------------------------------
def choose_event(folds: dict[str, list[Segment]]) -> dict:
    from .policies.event import EventClassifier
    models = {name: EventClassifier().fit([s for o, v in folds.items() if o != name for s in v])
              for name in folds}
    segs = [s for v in folds.values() for s in v]
    owner = {id(s): name for name, v in folds.items() for s in v}
    cap = C.MAX_RATE_RATIO * rider_rate(segs)
    best = None
    for p in C.EVENT_GRID:
        r = aggregate([segment_metrics(s, models[owner[id(s)]].ride(s, **p)) for s in segs])
        ok = r["shifts_per_h"] <= cap and r["reversal"] <= C.MAX_REVERSAL_SHARE
        score = r["agree_pm1"] - (0 if ok else 1)
        if best is None or score > best[0]:
            best = (score, p)
    return best[1]


def run_event_baseline(segs: list[Segment], log: Callable[[str], None] = print) -> dict:
    from .policies.event import EventClassifier
    tests = within_rider_tests(segs)
    rides = inria_rides(segs)
    by_ride = lambda rr: {r: [s for s in segs if s.ride == r] for r in rr}
    frozen = choose_event(by_ride(rides))
    log(f"frozen event setting: {frozen}")
    out = {}
    for t in tests:
        p = frozen if t.frozen else choose_event(by_ride([r for r in rides if r != t.name]))
        clf = EventClassifier().fit(t.train)
        res = aggregate([segment_metrics(s, clf.ride(s, **p)) for s in t.test])
        out[t.name] = {"rider": t.rider, "group": t.group, "hours": t.hours,
                       "EV": {**res, "params": p, "size_bytes": clf.size_bytes(),
                              "offline": clf.offline_scores(t.test, p["th"])}}
        log(f"{t.name:24s} EV={res['agree_pm1'] * 100:.1f} offline F1={out[t.name]['EV']['offline']['event_f1']:.2f}")
    return {"tests": out, "frozen": frozen}


def weighted(tests: dict, kind: str, metric: str = "agree_pm1", group: str | None = None) -> float:
    rows = [r for r in tests.values() if group is None or r.get("group") == group]
    h = np.array([r["hours"] for r in rows])
    v = np.array([r[kind][metric] for r in rows])
    return float((h * v).sum() / h.sum())
