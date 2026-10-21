"""Minimal FIT reader: 1-Hz records and Di2 gear-change events."""
from __future__ import annotations

from pathlib import Path

import fitdecode
import numpy as np
import pandas as pd

_RECORD_FIELDS = ("timestamp", "distance", "enhanced_altitude", "altitude", "enhanced_speed",
                  "speed", "power", "cadence")
_GEAR_FIELDS = ("timestamp", "rear_gear_num", "rear_gear", "front_gear_num", "front_gear")


def read_fit(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (records, gear_events).

    records: timestamp, cadence (rpm), speed (m/s), alt (m), power (W), distance (m)
    gear_events: timestamp, rear_gear_num, rear_gear (teeth), front_gear_num, front_gear
    """
    recs, events = [], []
    with fitdecode.FitReader(str(path), check_crc=fitdecode.CrcCheck.DISABLED) as reader:
        for msg in reader:
            if not isinstance(msg, fitdecode.FitDataMessage):
                continue
            if msg.name == "record":
                recs.append({k: msg.get_value(k, fallback=None) for k in _RECORD_FIELDS})
            elif msg.name == "event" and msg.get_value("event", fallback=None) in (
                    "rear_gear_change", "front_gear_change"):
                events.append({k: msg.get_value(k, fallback=None) for k in _GEAR_FIELDS})
    r = pd.DataFrame(recs)
    alt = r["enhanced_altitude"].fillna(r["altitude"]) if "enhanced_altitude" in r else r["altitude"]
    spd = r["enhanced_speed"].fillna(r["speed"]) if "enhanced_speed" in r else r["speed"]
    records = pd.DataFrame({
        "timestamp": pd.to_datetime(r.timestamp),
        "cadence": r.cadence.astype(float),
        "speed": spd.astype(float),
        "alt": alt.astype(float),
        "power": r.power.astype(float) if "power" in r else np.nan,
        "distance": r.distance.astype(float),
    })
    return records, pd.DataFrame(events)
