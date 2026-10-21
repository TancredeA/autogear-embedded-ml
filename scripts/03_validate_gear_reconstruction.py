"""Validate the gear reconstruction on the only ride with real gear labels (Shimano Di2).

1. Physics check: for a fixed chainring and wheel, development x sprocket teeth is constant, so the
   speed-to-cadence ratio does identify the gear.
2. Reconstruction check: run the Viterbi reconstruction on the whole ride as if the gears were
   unknown, and compare with the true gears (state accuracy within 5 % of development on pedalling
   seconds; shift events matched within +-3 s).
"""
import json

import numpy as np

from autogear import config as C
from autogear.data.build import load_fit_ride
from autogear.data.fit import read_fit
from autogear.data.sources import DI2_RIDE, FIT_SOURCES
from autogear.gears import development, levels_from_true_gears, validate_against_true_gears
from autogear.paths import RAW, RESULTS, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    path = RAW / next(s for s in FIT_SOURCES if s.key == DI2_RIDE).filename
    ride, teeth = load_fit_ride(path)
    cadence = ride.cadence.fillna(0).clip(0, 160).values
    speed = ride.speed.fillna(0).clip(0, 25).values

    # true shifts: rear_gear_change events where the sprocket actually changes
    records, events = read_fit(path)
    t0 = records.dropna(subset=["timestamp"]).timestamp.iloc[0]
    events = events.sort_values("timestamp")
    changed = events.rear_gear.ne(events.rear_gear.shift()) & events.rear_gear.shift().notna()
    shift_times = (events.timestamp[changed] - t0).dt.total_seconds().values

    true_gear, true_levels = levels_from_true_gears(cadence, speed, teeth)
    res = validate_against_true_gears(cadence, speed, true_gear, true_levels, shift_times)

    ped, dev = development(cadence, speed)
    teeth_ped = teeth[ped]
    product = {}
    for t in np.unique(teeth_ped[~np.isnan(teeth_ped.astype(float))]):
        m = teeth_ped == t
        if m.mean() >= 0.05:                                  # well-used sprockets
            product[int(t)] = round(float(np.median(dev[ped][m] * t)), 2)
    res["development_x_teeth"] = product
    (RESULTS / "gear_reconstruction_validation.json").write_text(json.dumps(res, indent=1))
    print(f"development x teeth on well-used sprockets: {product}")
    print(f"state accuracy {res['state_accuracy'] * 100:.1f} % of pedalling seconds; events: precision "
          f"{res['event_precision']:.2f}, recall {res['event_recall']:.2f} within +-3 s; "
          f"{res['true_shifts']} true rear shifts in {res['minutes']:.0f} min")
