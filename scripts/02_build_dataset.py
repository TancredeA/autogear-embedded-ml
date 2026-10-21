"""Rebuild the 1-Hz dataset (data/processed/) from the downloaded sources.

Output: one row per second, 18.9 h, 6 riders, 9 rides. See data/README.md for the columns.
"""
import json

from autogear.data.build import build_dataset, summary
from autogear.paths import DATASET_CSV, LEVELS_JSON, RAW, RESULTS, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    data, levels = build_dataset(RAW)
    data.to_csv(DATASET_CSV, index=False)
    LEVELS_JSON.write_text(json.dumps(levels, indent=1))
    s = summary(data)
    print(s.to_string())
    stats = dict(hours=round(len(data) / 3600, 2), rides=int(data.ride.nunique()),
                 riders=int(data.rider.nunique()), segments=int(data.segment.nunique()),
                 shift_share=float((data.action != 0).mean()),
                 per_ride=s.reset_index().to_dict(orient="records"))
    (RESULTS / "dataset_summary.json").write_text(json.dumps(stats, indent=1))
    print(f"total {stats['hours']} h, {stats['riders']} riders, {stats['rides']} rides, "
          f"shift share {stats['shift_share'] * 100:.2f} %")
