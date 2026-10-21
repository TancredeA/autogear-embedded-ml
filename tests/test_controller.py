import numpy as np

from autogear.controller import control
from conftest import LEVELS


def test_one_step_at_a_time_and_never_while_coasting(segment):
    target = np.full(segment.n, np.log(LEVELS[-1]) + 1.0)     # wants the largest gear, far away
    g = control(segment, target, h=0.6, k=1)
    assert np.all(np.abs(np.diff(g)) <= 1)
    coasting_steps = np.flatnonzero(np.diff(g) != 0)
    assert not np.any(segment.coasting[coasting_steps])


def test_dead_band_holds_the_gear(segment):
    g0 = segment.gear[0]
    lg = np.log(LEVELS)
    # inside the dead band: half a step + margin away is never reached
    target = np.full(segment.n, lg[g0 - 1] + 0.4 * (lg[g0] - lg[g0 - 1]))
    g = control(segment, target, h=0.6, k=1)
    assert np.all(g == g0)


def test_persistence_delays_the_shift(segment):
    g0 = segment.gear[0]
    target = np.full(segment.n, np.log(LEVELS[min(g0, len(LEVELS) - 1)]) + 0.2)
    first = {k: int(np.flatnonzero(np.diff(control(segment, target, h=0.3, k=k)))[0]) for k in (1, 3, 5)}
    assert first[1] < first[3] < first[5]
    assert first[3] - first[1] == 2 and first[5] - first[3] == 2


def test_asymmetric_persistence(segment):
    g0 = max(segment.gear[0], 3)
    target = np.full(segment.n, np.log(LEVELS[0]) - 0.5)       # wants the smallest gear
    seg = segment
    seg.gear[0] = g0
    slow = np.flatnonzero(np.diff(control(seg, target, h=0.3, k=3)))[0]
    fast = np.flatnonzero(np.diff(control(seg, target, h=0.3, k=3, k_down=1)))[0]
    assert fast < slow
