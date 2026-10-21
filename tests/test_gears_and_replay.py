import numpy as np

from autogear.gears import reconstruct_viterbi
from autogear.replay import aggregate, segment_metrics


def test_viterbi_recovers_known_gears(ride):
    g_hat, levels = reconstruct_viterbi(ride.cadence.values, ride.speed.values)
    used = np.unique(ride.gear)
    assert len(levels) == len(used)
    true_dev = np.array([3.94, 4.46, 5.11, 5.65, 6.37, 7.18, 7.78])[ride.gear.values - 1]
    est_dev = levels[g_hat - 1]
    ped = ride.coasting.values == 0
    assert np.mean(np.abs(np.log(est_dev / true_dev))[ped] < np.log(1.05)) > 0.95


def test_rider_agrees_with_itself(segment):
    r = aggregate([segment_metrics(segment, segment.gear)])
    assert r["agree_pm1"] == 1.0 and r["exact"] == 1.0


def test_reversal_is_a_shift_undone_within_four_seconds(segment):
    g = np.full(segment.n, 3)
    g[100:103] = 4           # up at 100, down at 103: reversal
    g[200:260] = 4           # up at 200, down at 260: not a reversal
    m = segment_metrics(segment, g)
    assert m["sh"] == 4 and m["rev"] == 1
