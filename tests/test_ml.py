"""Tests for the machine-learning layer. Uses the real dataset (subsampled where needed for speed)."""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ["NASSAU_ML_FAST"] = "1"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
import data_processing as dp  # noqa: E402
import ml_models as ml  # noqa: E402

CSV = ROOT / "data" / "Nassau_Candy_Distributor.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="dataset missing")


@pytest.fixture(scope="module")
def table():
    full, _ = dp.clean_data(dp.read_csv_safely(CSV), reference_date="2026-11-07")
    ship = dp.build_shipments(dp.apply_basis(dp.analysis_rows(full), cfg.ANALYSIS_BASIS_RELATIVE), 5)
    return ml.prepare_ml_table(ship, 5)


def test_prepare_table_has_no_leaky_columns_used_as_features(table):
    used = set(ml.CAT_ALL + ml.NUM_ALL)
    assert not used & {"Lead Time", "Delayed", "Ship Date", "Raw Lead Time", "Relative Lead Time"}
    assert set(table["Delayed"].unique()) <= {0, 1} and table["Lead Time"].notna().all()


def test_group_folds_never_split_a_customer(table):
    for tr, te in ml._folds(table):
        assert not set(table.iloc[tr]["Customer ID"]) & set(table.iloc[te]["Customer ID"])


def test_baseline_regressor_is_exactly_the_mode_mean(table):
    est, cols = ml.regression_models()[ml.BASELINE_REG]
    p = ml._pipe(est, cols).fit(table[cols], table["Lead Time"])
    pred = pd.Series(p.predict(table[cols]), index=table.index)
    expected = table.groupby("Ship Mode")["Lead Time"].transform("mean")
    assert np.allclose(pred, expected)


def test_cv_tables_structure_and_sanity(table):
    sub = table.sample(2500, random_state=0).reset_index(drop=True)
    reg = ml.cross_validate_regression(sub)["table"]
    clf = ml.cross_validate_classification(sub)["table"]
    assert reg.loc[reg.Model == ml.BASELINE_REG, "Beats baseline?"].iloc[0] == "(baseline)"
    assert reg.loc[reg.Model == "Global mean", "R²"].iloc[0] < 0.05          # no skill without ship mode
    assert reg.loc[reg.Model == ml.BASELINE_REG, "R²"].iloc[0] > 0.4         # ship mode explains most variance
    assert 0.55 < clf.loc[clf.Model == ml.BASELINE_CLF, "ROC AUC"].iloc[0] < 0.95


def test_importance_ranks_ship_mode_first(table):
    sub = table.sample(3000, random_state=1).reset_index(drop=True)
    assert ml.feature_importance(sub, "regression", n_repeats=3).iloc[0]["Feature"] == "Ship Mode"


def test_what_if_orders_modes_correctly(table):
    pred = ml.fit_predictor(table, "Ridge, all features", "Logistic, all features")
    fast = ml.predict_what_if(pred, "Lot's O' Nuts", "New York", "Same Day", "Chocolate", 6, 2, 2, 16.0)
    slow = ml.predict_what_if(pred, "Lot's O' Nuts", "New York", "Standard Class", "Chocolate", 6, 2, 2, 16.0)
    assert fast["lead"] < slow["lead"] and fast["p_delay"] < slow["p_delay"]
    assert 0 <= fast["p_delay"] <= 1 and fast["low"] < fast["lead"] < fast["high"]


def test_clustering_outputs(table):
    c = ml.cluster_routes(table)
    assert c["k"] >= 2 and c["table"]["Cluster"].nunique() == c["k"] and (c["table"]["Shipments"] >= cfg.MIN_SHIPMENTS_FOR_RANKING).all()
    assert -1 <= c["silhouette"]["Silhouette"].max() <= 1


def test_leap_days_spanned_helper():
    o = pd.Series(pd.to_datetime(["2028-01-01", "2027-01-01", "2028-03-01", "2024-01-01"]))
    s = pd.Series(pd.to_datetime(["2028-03-01", "2027-12-01", "2028-06-01", "2028-12-31"]))
    assert dp.leap_days_spanned(o, s).tolist() == [1, 0, 0, 2]
