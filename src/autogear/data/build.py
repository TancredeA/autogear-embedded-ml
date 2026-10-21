"""Rebuild, from public files, the 1-Hz dataset the prototype would have recorded.

One row per second and per ride, with only quantities the device can measure or derive:
cadence, engaged gear (reconstructed, or true for the Di2 ride), slope, longitudinal acceleration
and a coasting flag. Measured speed is kept only to recover the gear; power is kept only for the
torque-aware decision table (RT) and is never an input of a learned model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .. import config as C
from ..gears import levels_from_true_gears, reconstruct_viterbi
from .fit import read_fit
from .sources import FIT_SOURCES, INRIA

COLUMNS = ["t", "cadence", "speed", "slope", "acc", "power", "gear", "n_gears", "D", "coasting",
           "ride", "rider", "segment", "action"]


def segment_ids(t: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Cut at timestamp gaps > SEGMENT_MAX_GAP_S and at stops of SEGMENT_MAX_STOP_S seconds."""
    seg = np.zeros(len(t), int)
    k = stop = 0
    for i in range(1, len(t)):
        stop = stop + 1 if not moving[i] else 0
        if t[i] - t[i - 1] > C.SEGMENT_MAX_GAP_S or stop == C.SEGMENT_MAX_STOP_S:
            k += 1
        seg[i] = k
    return seg


def finish_ride(df: pd.DataFrame, ride: str, rider: str, true_teeth: np.ndarray | None = None
                ) -> tuple[pd.DataFrame, np.ndarray]:
    """Add gear, development, coasting flag and segment ids; drop stops and short segments."""
    df = df.copy()
    df["cadence"] = df.cadence.fillna(0).clip(0, 160)
    df["speed"] = df.speed.fillna(0).clip(0, 25)
    if true_teeth is None:
        gear, levels = reconstruct_viterbi(df.cadence.values, df.speed.values)
    else:
        gear, levels = levels_from_true_gears(df.cadence.values, df.speed.values, true_teeth)
    df["gear"] = gear
    df["n_gears"] = len(levels)
    df["D"] = levels[gear - 1]
    df["coasting"] = (df.cadence < C.COASTING_MAX_CADENCE_RPM).astype(int)
    df["ride"] = ride
    df["rider"] = rider
    seg = segment_ids(df.t.values, (df.speed > 0.5).values)
    df["segment"] = [f"{ride}_{s}" for s in seg]
    df = df[(df.speed > 0.5) | (df.cadence > 0)]
    df = df.groupby("segment").filter(lambda x: len(x) >= C.SEGMENT_MIN_LEN_S)
    return df, levels


def load_inria(path: Path) -> list[tuple[str, pd.DataFrame]]:
    """The Inria file holds four rides by one cyclist, one per day."""
    br = pd.read_csv(path, parse_dates=["timestamp"])
    br["t"] = (br.timestamp - br.timestamp.iloc[0]).dt.total_seconds()
    rides = []
    for i, (_, sub) in enumerate(br.groupby(br.timestamp.dt.date)):
        sub = sub.reset_index(drop=True)
        rides.append((f"inria_{i + 1}", pd.DataFrame({
            "t": sub.t, "cadence": sub.cadence, "speed": sub.speed,
            "slope": sub.slope.clip(-C.SLOPE_CLIP, C.SLOPE_CLIP),
            "acc": sub.acceleration.clip(-C.ACC_CLIP, C.ACC_CLIP), "power": sub.power})))
    return rides


def load_fit_ride(path: Path) -> tuple[pd.DataFrame, np.ndarray | None]:
    """1-Hz signals of a FIT file, and the true rear-sprocket teeth if it carries Di2 events."""
    r, events = read_fit(path)
    r = r.dropna(subset=["timestamp"]).reset_index(drop=True)
    # altitude spikes: more than 20 m away from a 61-s rolling median
    alt = pd.Series(r.alt.values)
    med = alt.rolling(C.ALTITUDE_MEDIAN_WINDOW_S, center=True, min_periods=5).median()
    alt[(alt - med).abs() > C.ALTITUDE_SPIKE_M] = np.nan
    r["alt"] = alt.interpolate().bfill().ffill().values
    teeth = None
    if len(events):
        events = events.assign(timestamp=pd.to_datetime(events.timestamp)).sort_values("timestamp")
        teeth = pd.merge_asof(r[["timestamp"]], events[["timestamp", "rear_gear"]],
                              on="timestamp").rear_gear.values
    t = (r.timestamp - r.timestamp.iloc[0]).dt.total_seconds().values
    # slope: derivative of altitude over distance, both smoothed over ~10 s
    a = pd.Series(r.alt.values).interpolate().bfill().ffill().values
    d = pd.Series(r.distance.values).interpolate().bfill().ffill().values
    da = np.gradient(uniform_filter1d(a, C.SLOPE_SMOOTHING_S))
    dd = np.gradient(uniform_filter1d(d, C.SLOPE_SMOOTHING_S))
    slope = np.where(dd > 0.5, da / np.maximum(dd, 0.5), 0.0)
    v = pd.Series(r.speed.values).fillna(0).values
    acc = np.gradient(uniform_filter1d(v, 3))
    x = pd.DataFrame({
        "t": t, "cadence": r.cadence.values, "speed": v,
        "slope": np.clip(uniform_filter1d(slope, C.SLOPE_POST_SMOOTHING_S), -C.SLOPE_CLIP, C.SLOPE_CLIP),
        "acc": np.clip(acc, -C.ACC_CLIP, C.ACC_CLIP), "power": r.power.values})
    return x, teeth


def build_dataset(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Assemble all rides. Returns the dataset and the gear levels of each ride (m/rev)."""
    rows, levels = [], {}
    for ride, df in load_inria(raw_dir / INRIA.filename):
        out, lv = finish_ride(df, ride, "inria")
        rows.append(out)
        levels[ride] = lv
    for src in FIT_SOURCES:
        df, teeth = load_fit_ride(raw_dir / src.filename)
        out, lv = finish_ride(df, src.key, src.rider, true_teeth=teeth)
        rows.append(out)
        levels[src.key] = lv
    data = pd.concat(rows, ignore_index=True)
    nxt = data.groupby("segment").gear.shift(-1).fillna(data.gear)
    data["action"] = np.sign(nxt - data.gear).astype(int)   # shift between t and t+1
    return data[COLUMNS], {k: [round(float(x), 3) for x in v] for k, v in levels.items()}


def summary(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("ride", sort=False).agg(
        hours=("t", lambda s: len(s) / 3600), gears=("n_gears", "first"),
        shifts_per_hour=("action", lambda a: (a != 0).sum() / (len(a) / 3600)),
        coasting=("coasting", "mean"), power_available=("power", lambda p: p.notna().mean())).round(3)
