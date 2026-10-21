import numpy as np

from autogear.features import causal_mean, derived_speed, device_features, lagged_diff
from conftest import LEVELS


def test_causal_mean_uses_fewer_samples_at_start():
    x = np.array([2.0, 4.0, 6.0, 8.0])
    assert np.allclose(causal_mean(x, 3), [2.0, 3.0, 4.0, 6.0])


def test_lagged_diff_pads_with_first_value():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    assert np.allclose(lagged_diff(x, 2), [0.0, 1.0, 3.0, 5.0])


def test_inputs_do_not_depend_on_the_gear_chosen(ride):
    """The learned model is outside the loop: any gear sequence gives the same inputs."""
    c, g, co = ride.cadence.values, ride.gear.values, ride.coasting.values
    acc, slope = ride.acc.values, ride.slope.values
    rng = np.random.default_rng(1)
    g_policy = np.clip(g + rng.integers(-2, 3, size=len(g)), 1, len(LEVELS))
    ped = (co == 0) & (c > 20)
    c_policy = np.where(ped, c * LEVELS[g - 1] / LEVELS[g_policy - 1], c)   # replay rule
    v_rider = derived_speed(c, g, LEVELS, co, acc)
    v_policy = derived_speed(c_policy, g_policy, LEVELS, co, acc)
    ped_both = ped & (c_policy > 20)
    assert np.allclose(v_rider[ped_both], v_policy[ped_both])
    X1 = device_features(v_rider, slope, acc, co)
    assert X1.shape == (len(c), 14) and X1.dtype == np.float32


def test_derived_speed_integrates_acceleration_while_coasting():
    c = np.array([80.0, 0.0, 0.0])
    g = np.array([3, 3, 3])
    co = np.array([0, 1, 1])
    acc = np.array([0.0, -0.5, -0.5])
    v = derived_speed(c, g, LEVELS, co, acc)
    assert np.isclose(v[0], 80 * LEVELS[2] / 60)
    assert np.allclose(np.diff(v), [-0.5, -0.5])
