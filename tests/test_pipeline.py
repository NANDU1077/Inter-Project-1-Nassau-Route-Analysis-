"""Unit + integration tests. Small hand-built frames are used ONLY to test logic; the
real-data tests run against the actual CSV when it is present."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as an  # noqa: E402
import config as cfg  # noqa: E402
import data_processing as dp  # noqa: E402

CSV = ROOT / "data" / "Nassau_Candy_Distributor.csv"


def make_raw(rows):
    cols = cfg.REQUIRED_COLUMNS
    base = dict(zip(cols, [1, "O1", "01-01-2024", "05-01-2024", "Standard Class", 1, "United States", "Austin", "texas", "78701",
                           "Chocolate", "Gulf", "P1", "Wonka Bar - Milk Chocolate", 10.0, 2, 6.0, 4.0]))
    out = []
    for i, r in enumerate(rows, 1):
        d = dict(base); d["Row ID"] = i; d["Order ID"] = f"O{i}"; d.update(r); out.append(d)
    return pd.DataFrame(out)


# ---------------- parsing / lead time ----------------
def test_date_parsing_day_first_and_lead_time():
    df, rep = dp.clean_data(make_raw([{"Order Date": "30-06-2024", "Ship Date": "04-07-2024"}]))
    assert rep.date_format == "%d-%m-%Y"
    assert df.loc[0, "Raw Lead Time"] == 4


def test_unparseable_date_excluded_not_crash():
    df, rep = dp.clean_data(make_raw([{}, {"Ship Date": "not a date"}]))
    assert rep.invalid_ship_dates == 1 and df["Excluded"].sum() == 1
    assert df.loc[1, "Exclusion Reason"].startswith("Invalid")


def test_negative_lead_excluded_and_reported():
    df, rep = dp.clean_data(make_raw([{"Order Date": "10-01-2024", "Ship Date": "05-01-2024"}, {}]))
    assert rep.negative_lead_rows == 1 and rep.kept_rows == 1


def test_extreme_lead_flagged_but_kept():
    df, rep = dp.clean_data(make_raw([{"Ship Date": "01-01-2026"}]))
    assert bool(df.loc[0, "Flag Extreme Lead Time"]) and not bool(df.loc[0, "Excluded"])


# ---------------- factory mapping ----------------
def test_every_mapped_product_resolves_including_spacing_variant():
    for name, fac in cfg.PRODUCT_FACTORY_MAP.items():
        assert dp.map_product_to_factory(name) == fac
    assert dp.map_product_to_factory("Wonka Bar -Scrumdiddlyumptious") == "Lot's O' Nuts"
    assert dp.map_product_to_factory("wonka bar - milk chocolate") == "Wicked Choccy's"


def test_unmapped_product_reported_and_excluded():
    df, rep = dp.clean_data(make_raw([{"Product Name": "Mystery Gum"}, {}]))
    assert rep.unmapped_products == ["Mystery Gum"] and rep.excluded_rows == 1


def test_duplicate_flag():
    r = make_raw([{}, {}])
    r["Order ID"] = "SAME"; r.loc[:, "Row ID"] = [1, 2]
    df, rep = dp.clean_data(r)
    assert rep.duplicate_rows == 1


# ---------------- aggregation ----------------
def _shipments():
    raw = make_raw([
        {"Order Date": "01-01-2024", "Ship Date": "03-01-2024", "State/Province": "Texas"},
        {"Order Date": "01-01-2024", "Ship Date": "05-01-2024", "State/Province": "Texas"},
        {"Order Date": "01-01-2024", "Ship Date": "09-01-2024", "State/Province": "Texas"},
        {"Order Date": "01-01-2024", "Ship Date": "04-01-2024", "State/Province": "Ohio"},
        {"Order Date": "01-01-2024", "Ship Date": "04-01-2024", "State/Province": "Ohio"},
    ])
    df, _ = dp.clean_data(raw)
    df = dp.apply_basis(dp.analysis_rows(df), cfg.ANALYSIS_BASIS_RECORDED)
    return df, dp.build_shipments(df, threshold=3)


def test_route_aggregation_and_score():
    rows, ship = _shipments()
    rt = an.route_table(ship, "state", min_shipments=2)
    tx = rt[rt["Destination"] == "Texas"].iloc[0]
    assert tx["Shipments"] == 3 and tx["Avg Lead Time"] == pytest.approx(np.mean([2, 4, 8]))
    assert tx["Delay Rate"] == pytest.approx(2 / 3)           # 4 and 8 > 3
    oh = rt[rt["Destination"] == "Ohio"].iloc[0]
    assert oh["Efficiency Score"] == 100 and tx["Efficiency Score"] == 0
    assert rt["Efficiency Score"].between(0, 100).all()


def test_shipment_vs_order_counts_multi_factory_order():
    raw = make_raw([{"Order ID": "X"}, {"Order ID": "X", "Product Name": "Laffy Taffy"}])
    raw["Row ID"] = [1, 2]
    df, _ = dp.clean_data(raw)
    df = dp.apply_basis(dp.analysis_rows(df), cfg.ANALYSIS_BASIS_RECORDED)
    ship = dp.build_shipments(df, 3)
    k = an.compute_kpis(df, ship)
    assert k["orders"] == 1 and k["shipments"] == 2 and k["rows"] == 2


def test_min_shipments_unranked():
    _, ship = _shipments()
    rt = an.route_table(ship, "state", min_shipments=4)
    assert rt["Efficiency Score"].isna().all() and (rt["Performance"] == "Not ranked").all()


# ---------------- filters / empty ----------------
def test_filters_and_empty_state():
    rows, _ = _shipments()
    assert len(dp.apply_filters(rows, {"states": ["Texas"]})) == 3
    empty = dp.apply_filters(rows, {"states": []})
    assert empty.empty
    assert dp.build_shipments(empty, 3).empty
    assert an.compute_kpis(empty, pd.DataFrame()) == {}
    assert an.route_table(pd.DataFrame()).empty


def test_extreme_toggle_removes_flagged():
    df, _ = dp.clean_data(make_raw([{"Ship Date": "01-01-2026"}, {}]))
    assert len(dp.apply_filters(dp.analysis_rows(df), {"include_extreme": False})) == 1


# ---------------- real data ----------------
@pytest.mark.skipif(not CSV.exists(), reason="dataset not present")
def test_real_dataset_end_to_end():
    full, rep = dp.clean_data(dp.read_csv_safely(CSV))
    assert rep.raw_rows == 10194 and rep.unmapped_rows == 0 and rep.invalid_order_dates == 0
    assert rep.negative_lead_rows == 0
    assert set(full["Factory"].unique()) <= set(cfg.FACTORY_COORDS)
    assert full["Customer State"].isin(cfg.STATE_COORDS).all()          # every state has map coords
    # the data-quality problem must be surfaced, not hidden
    assert rep.extreme_lead_rows / rep.raw_rows > cfg.EXTREME_SHARE_CRITICAL
    rows = dp.apply_basis(dp.analysis_rows(full), cfg.ANALYSIS_BASIS_RELATIVE)
    assert rows["Lead Time"].min() == 0 and rows["Lead Time"].max() < 30
    ship = dp.build_shipments(rows, 5)
    assert len(ship) <= len(rows) and ship["Order ID"].nunique() == rows["Order ID"].nunique()
    rt = an.route_table(ship, "state")
    assert rt["Shipments"].sum() == len(ship)


@pytest.mark.skipif(not CSV.exists(), reason="dataset not present")
def test_shipmode_ordering_check_runs():
    full, _ = dp.clean_data(dp.read_csv_safely(CSV))
    chk = an.shipmode_ordering_check(dp.analysis_rows(full))
    assert set(chk["Ship Mode"]) == {"Same Day", "First Class", "Second Class", "Standard Class"}
    assert an.ordering_is_monotonic(chk, "Mean Relative Lead Time") in (True, False)


def test_mode_group_and_heatmap_build():
    import visualizations as viz
    rows, ship = _shipments()
    assert set(ship["Mode Group"]) == {"Standard"}
    stats = an.state_map_stats(ship, min_shipments=1)
    assert viz.us_heatmap(stats, "Average lead time") is not None
    assert viz.us_heatmap(pd.DataFrame(), "Delay rate") is not None
    assert set(cfg.STATE_ABBR.values()).__len__() == len(cfg.STATE_ABBR)
    assert all(s in cfg.STATE_ABBR or cfg.STATE_COORDS[s][1] < 0 for s in cfg.STATE_COORDS)
