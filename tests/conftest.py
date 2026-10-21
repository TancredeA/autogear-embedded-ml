import numpy as np
import pandas as pd
import pytest

from autogear.features import make_segment

LEVELS = np.array([3.94, 4.46, 5.11, 5.65, 6.37, 7.18, 7.78])


def synthetic_ride(n=900, seed=0):
    """A ride whose rider shifts to hold ~80 rpm on a rolling road (gears known)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    speed = 6 + 2.5 * np.sin(t / 60) + 0.2 * rng.normal(size=n).cumsum() / np.sqrt(n)
    slope = 0.03 * np.sin(t / 90)
    gear = np.clip(np.searchsorted(LEVELS, 60 * speed / 80), 1, len(LEVELS))
    for i in range(1, n):                     # a rider does not shift every second
        if abs(gear[i] - gear[i - 1]) and i % 7:
            gear[i] = gear[i - 1]
    cadence = 60 * speed / LEVELS[gear - 1]
    coasting = np.zeros(n, int)
    coasting[300:330] = 1
    cadence[300:330] = 0
    return pd.DataFrame({"t": t, "cadence": cadence, "speed": speed, "slope": slope,
                         "acc": np.gradient(speed), "power": 150.0, "gear": gear,
                         "coasting": coasting, "ride": "synthetic", "rider": "synthetic"})


@pytest.fixture
def ride():
    return synthetic_ride()


@pytest.fixture
def segment(ride):
    return make_segment(ride, LEVELS)
