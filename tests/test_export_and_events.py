import shutil
import subprocess

import numpy as np
import pytest
import xgboost as xgb

from autogear.event_metrics import event_scores
from autogear.export_c import _c_float, from_booster, write_c


@pytest.fixture(scope="module")
def booster():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 14)).astype(np.float32)
    y = 0.3 * X[:, 0] + np.sin(X[:, 5]) + 0.1 * rng.normal(size=3000)
    params = dict(max_depth=4, learning_rate=0.1, objective="reg:squarederror", tree_method="hist",
                  min_child_weight=200, seed=0)            # large leaves: many pruned branches
    return xgb.train(params, xgb.DMatrix(X, y), 50), X


def test_complete_tree_layout_reproduces_xgboost(booster):
    b, X = booster
    forest = from_booster(b, depth=4)
    assert np.isinf(forest.thresholds).any()               # some branches were padded
    assert np.max(np.abs(forest.predict(X) - b.inplace_predict(X))) < 1e-5
    assert forest.flash_bytes() == 50 * (15 * 4 + 15 + 16 * 4) + 4


def test_c_float_literals():
    assert _c_float(10.0) == "10.0f" and _c_float(np.inf) == "INFINITY"
    assert float(_c_float(np.float32(0.1))[:-1]) == pytest.approx(0.1)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_generated_c_compiles(booster, tmp_path):
    b, _ = booster
    write_c(from_booster(b, depth=4), tmp_path, [f"f{i}" for i in range(14)], {"test": "yes"})
    subprocess.run(["cc", "-std=c99", "-Wall", "-Werror", "-c", str(tmp_path / "ag_model_data.c"),
                    "-o", str(tmp_path / "m.o")], check=True)


def test_event_f1_tolerance_and_direction():
    t = np.arange(20)
    true = np.zeros(20, int)
    true[[5, 15]] = [1, -1]
    pred = np.zeros(20, int)
    pred[[6, 15]] = [1, 1]                                  # first: on time; second: wrong direction
    s = event_scores(true, pred, t, np.zeros(20, int), tol=2)
    assert s["detect"]["tp"] == 2 and s["action"]["tp"] == 1
    assert s["action"]["f1"] == pytest.approx(0.5)
