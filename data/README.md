# Data

Nothing in this folder is versioned. `scripts/01_download_sources.py` downloads six public files,
each pinned to a commit and checked against its SHA-256 (see `src/autogear/data/sources.py`), and
`scripts/02_build_dataset.py` rebuilds `processed/autogear_public_1hz.csv` (18.9 h, 68,082 rows).

## Sources and licences

| Ride(s) | File | Repository | Licence |
|---|---|---|---|
| `inria_1` ... `inria_4` | `datasets/bike_rides.csv` | [INRIA/scikit-learn-mooc](https://github.com/INRIA/scikit-learn-mooc) | CC-BY 4.0 |
| `elemnt_2019` | `tests/files/2019-02-17-062644-ELEMNT-297E-195-0.fit` | [dtcooper/python-fitparse](https://github.com/dtcooper/python-fitparse) | MIT |
| `edge530_2020` | `testdata/bpg/garmin.fit` | [tormoder/fit](https://github.com/tormoder/fit) | MIT |
| `edge810_2013` | `tests/files/Edge810-Vector-2013-08-16-15-35-10.fit` | [dtcooper/python-fitparse](https://github.com/dtcooper/python-fitparse) | MIT |
| `edge1030_2020` | `testdata/from_official_sdk/activity_lowbattery.fit` | [muktihari/fit](https://github.com/muktihari/fit) | BSD-3-Clause (Garmin SDK file) |
| `di2_edge1040_2022` | `tests/fits/WithGearChangeData.fit` | [garmin/fit-python-sdk](https://github.com/garmin/fit-python-sdk) | Garmin FIT Protocol License |

The Di2 file carries real Shimano `rear_gear_change` events: it is the only ride with true gear
labels and serves to validate the gear reconstruction (`scripts/03_validate_gear_reconstruction.py`).

## Columns of the rebuilt dataset (one row per second)

| Column | Meaning | On the Shifter Pro |
|---|---|---|
| `t` | time since the start of the file (s) | — |
| `cadence` | crank cadence (rpm) | Hall sensor on the crank |
| `speed` | measured speed (m/s), **only used to recover the gear**, never an input | not measured |
| `slope` | road slope (fraction), altitude over distance on 10 s | barometer or IMU pitch |
| `acc` | longitudinal acceleration (m/s²) | IMU, gravity removed |
| `power` | pedalling power (W), **only for the torque-aware decision table RT** | absent by design |
| `gear`, `n_gears`, `D` | engaged gear (1 = smallest development), number of gears, development (m/rev) | servo state |
| `coasting` | 1 when cadence < 15 rpm | Hall sensor |
| `ride`, `rider`, `segment` | identifiers | — |
| `action` | shift between t and t+1 (−1, 0, +1) | — |

`processed/gear_levels.json` holds the development of each gear of each bicycle (m/rev).
