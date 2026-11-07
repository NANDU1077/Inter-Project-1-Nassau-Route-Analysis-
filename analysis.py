"""Aggregations, route efficiency scoring and data-backed recommendations.

Units of analysis
  * ROW       = one order line in the CSV (used for product / sales / profit-by-product views)
  * ORDER     = unique Order ID
  * SHIPMENT  = unique (Order ID, Factory); lead-time statistics are computed on shipments
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from data_processing import build_shipments  # noqa: F401  (re-export for convenience)


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def compute_kpis(rows: pd.DataFrame, ship: pd.DataFrame) -> dict:
    if ship.empty:
        return {}
    sales, gp = float(ship["Sales"].sum()), float(ship["Gross Profit"].sum())
    return {
        "rows": int(len(rows)),
        "shipments": int(len(ship)),
        "orders": int(ship["Order ID"].nunique()),
        "customers": int(ship["Customer ID"].nunique()),
        "avg_lead": float(ship["Lead Time"].mean()),
        "median_lead": float(ship["Lead Time"].median()),
        "std_lead": float(ship["Lead Time"].std(ddof=1)) if len(ship) > 1 else float("nan"),
        "delay_rate": float(ship["Delayed"].mean()),
        "sales": sales,
        "gross_profit": gp,
        "margin": gp / sales if sales else float("nan"),
        "extreme_share": float(ship["Any Extreme"].mean()),
    }


# --------------------------------------------------------------------------- #
# Generic grouped performance
# --------------------------------------------------------------------------- #
def group_performance(ship: pd.DataFrame, by) -> pd.DataFrame:
    if ship.empty:
        return pd.DataFrame()
    by = [by] if isinstance(by, str) else list(by)
    g = ship.groupby(by, observed=True).agg(
        Shipments=("Order ID", "size"),
        Orders=("Order ID", "nunique"),
        **{
            "Avg Lead Time": ("Lead Time", "mean"),
            "Median Lead Time": ("Lead Time", "median"),
            "Lead Time Std Dev": ("Lead Time", "std"),
            "Delay Rate": ("Delayed", "mean"),
            "Total Sales": ("Sales", "sum"),
            "Total Cost": ("Cost", "sum"),
            "Total Gross Profit": ("Gross Profit", "sum"),
            "Avg Sales / Shipment": ("Sales", "mean"),
            "Avg Cost / Shipment": ("Cost", "mean"),
            "Avg Gross Profit / Shipment": ("Gross Profit", "mean"),
        },
    ).reset_index()
    g["Gross Margin"] = np.where(g["Total Sales"] > 0, g["Total Gross Profit"] / g["Total Sales"], np.nan)
    return g


def _minmax_score(values: pd.Series) -> pd.Series:
    """100 = fastest (lowest avg lead time), 0 = slowest. Constant input -> 50."""
    lo, hi = values.min(), values.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(50.0, index=values.index)
    return 100.0 * (hi - values) / (hi - lo)


def performance_category(score) -> str:
    if pd.isna(score):
        return "Not ranked"
    if score >= cfg.SCORE_EFFICIENT:
        return "Efficient"
    if score < cfg.SCORE_POOR:
        return "Inefficient"
    return "Moderate"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def route_table(ship: pd.DataFrame, level: str = "state", min_shipments: int | None = None) -> pd.DataFrame:
    """Route = Factory -> Customer State ('state') or Factory -> Customer Region ('region').

    Efficiency score: min-max normalisation of *average lead time* across routes that have
    at least ``min_shipments`` shipments (100 = fastest, 0 = slowest). Smaller routes are
    reported but not scored, to avoid ranking on a handful of shipments.
    """
    min_shipments = cfg.MIN_SHIPMENTS_FOR_RANKING if min_shipments is None else min_shipments
    if ship.empty:
        return pd.DataFrame()
    key = "Customer State" if level == "state" else "Customer Region"
    t = group_performance(ship, ["Factory", key])
    t = t.rename(columns={key: "Destination"})
    t.insert(0, "Route", t["Factory"] + " → " + t["Destination"])
    if level == "state":
        reg = ship.groupby("Customer State")["Customer Region"].agg(lambda s: s.mode().iat[0])
        t["Customer Region"] = t["Destination"].map(reg)
    t["Eligible"] = t["Shipments"] >= min_shipments
    t["Efficiency Score"] = np.nan
    if t["Eligible"].any():
        t.loc[t["Eligible"], "Efficiency Score"] = _minmax_score(t.loc[t["Eligible"], "Avg Lead Time"])
    t["Performance"] = t["Efficiency Score"].map(performance_category)
    return t.sort_values(["Efficiency Score", "Shipments"], ascending=[False, False]).reset_index(drop=True)


def top_bottom_routes(routes: pd.DataFrame, n: int = 10):
    """Return (top n most efficient, bottom n least efficient) among ranking-eligible routes."""
    el = routes[routes["Eligible"]] if not routes.empty else routes
    if el.empty:
        return el, el
    top = el.sort_values(["Avg Lead Time", "Shipments"], ascending=[True, False]).head(n)
    bottom = el.sort_values(["Avg Lead Time", "Shipments"], ascending=[False, False]).head(n)
    return top.reset_index(drop=True), bottom.reset_index(drop=True)


def high_volume_poor_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """Eligible routes with volume >= HIGH_VOLUME_QUANTILE and avg lead time above the
    median of eligible routes' averages."""
    el = routes[routes["Eligible"]] if not routes.empty else routes
    if el.empty:
        return el
    vol_cut = el["Shipments"].quantile(cfg.HIGH_VOLUME_QUANTILE)
    lt_cut = el["Avg Lead Time"].median()
    out = el[(el["Shipments"] >= vol_cut) & (el["Avg Lead Time"] > lt_cut)]
    return out.sort_values(["Shipments", "Avg Lead Time"], ascending=False).reset_index(drop=True)


def state_map_stats(ship: pd.DataFrame, min_shipments: int | None = None) -> pd.DataFrame:
    """Per-destination stats for the map, with coordinates (state centroids from config)."""
    min_shipments = cfg.MIN_SHIPMENTS_FOR_RANKING if min_shipments is None else min_shipments
    if ship.empty:
        return pd.DataFrame()
    t = group_performance(ship, "Customer State")
    meta = ship.groupby("Customer State").agg(
        Region=("Customer Region", lambda s: s.mode().iat[0]),
        Country=("Country", lambda s: s.mode().iat[0]),
        Main_Factory=("Factory", lambda s: s.mode().iat[0]),
        Factories=("Factory", "nunique"),
    ).reset_index()
    t = t.merge(meta, on="Customer State")
    t["Latitude"] = t["Customer State"].map(lambda s: cfg.STATE_COORDS.get(s, (np.nan, np.nan))[0])
    t["Longitude"] = t["Customer State"].map(lambda s: cfg.STATE_COORDS.get(s, (np.nan, np.nan))[1])
    t["Eligible"] = t["Shipments"] >= min_shipments
    t["Efficiency Score"] = np.nan
    if t["Eligible"].any():
        t.loc[t["Eligible"], "Efficiency Score"] = _minmax_score(t.loc[t["Eligible"], "Avg Lead Time"])
    return t


def bottleneck_table(group_df: pd.DataFrame, label_col: str, n: int = 5) -> pd.DataFrame:
    """Groups with the highest average lead time among those meeting the minimum volume."""
    if group_df.empty:
        return group_df
    g = group_df[group_df["Shipments"] >= cfg.MIN_SHIPMENTS_FOR_RANKING]
    return g.sort_values("Avg Lead Time", ascending=False).head(n)[
        [label_col, "Shipments", "Avg Lead Time", "Median Lead Time", "Delay Rate"]
    ].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Product / division (ROW level: each order line carries its shipment's lead time)
# --------------------------------------------------------------------------- #
def product_performance(rows: pd.DataFrame, threshold: float, by="Product Name") -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    by = [by] if isinstance(by, str) else list(by)
    r = rows.assign(Delayed=rows["Lead Time"] > threshold)
    g = r.groupby(by, observed=True).agg(
        Lines=("Row ID", "size"),
        Orders=("Order ID", "nunique"),
        Units=("Units", "sum"),
        **{
            "Avg Lead Time": ("Lead Time", "mean"),
            "Median Lead Time": ("Lead Time", "median"),
            "Delay Rate": ("Delayed", "mean"),
            "Total Sales": ("Sales", "sum"),
            "Total Cost": ("Cost", "sum"),
            "Total Gross Profit": ("Gross Profit", "sum"),
        },
    ).reset_index()
    g["Gross Margin"] = np.where(g["Total Sales"] > 0, g["Total Gross Profit"] / g["Total Sales"], np.nan)
    if "Product Name" in by:
        fac = r.groupby("Product Name")["Factory"].first()
        g["Factory"] = g["Product Name"].map(fac)
    return g.sort_values("Total Sales", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Ship-mode cost-time view (descriptive; there is NO shipping-cost field in the data)
# --------------------------------------------------------------------------- #
def shipmode_tradeoff(ship: pd.DataFrame) -> pd.DataFrame:
    t = group_performance(ship, "Ship Mode")
    if t.empty:
        return t
    total = t["Shipments"].sum()
    t["Share of Shipments"] = t["Shipments"] / total
    units = ship.groupby("Ship Mode")["Units"].sum()
    t["Cost per Unit (product)"] = t["Ship Mode"].map(
        (ship.groupby("Ship Mode")["Cost"].sum() / units.replace(0, np.nan)))
    return t.sort_values("Avg Lead Time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Drill-down helpers
# --------------------------------------------------------------------------- #
def monthly_trend(ship: pd.DataFrame) -> pd.DataFrame:
    if ship.empty:
        return pd.DataFrame()
    return ship.groupby("Order Month").agg(
        Shipments=("Order ID", "size"), **{"Avg Lead Time": ("Lead Time", "mean")}).reset_index()


# --------------------------------------------------------------------------- #
# Recommendations (templated from computed results; observed vs assumption labelled)
# --------------------------------------------------------------------------- #
def _fmt(x, d=1):
    return "n/a" if pd.isna(x) else f"{x:,.{d}f}"


def generate_recommendations(ship, routes, factory_perf, region_perf, mode_perf, hv_poor, basis_label, kpis) -> list[dict]:
    recs: list[dict] = []
    if ship.empty or not kpis:
        return recs
    unit = "relative days" if basis_label == cfg.ANALYSIS_BASIS_RELATIVE else "days"

    # 1. data integrity
    if kpis["extreme_share"] > 0:
        recs.append({
            "kind": "Observed", "priority": "Critical",
            "title": "Validate ship-date integrity before acting on absolute lead times",
            "text": (f"{kpis['extreme_share']:.1%} of shipments in the current selection have a recorded lead time above "
                     f"{cfg.MAX_PLAUSIBLE_LEAD_DAYS} days. Recorded values should be confirmed with the source system "
                     "(ERP/WMS export, date-format or timezone handling) before they are used for operational decisions."),
        })

    # 2. which routes to investigate first
    if not hv_poor.empty:
        top = hv_poor.head(3)
        names = "; ".join(f"{r.Route} ({r.Shipments:,} shipments, avg {_fmt(r['Avg Lead Time'])} {unit})"
                          for _, r in top.iterrows())
        recs.append({"kind": "Observed", "priority": "High",
                     "title": "Investigate high-volume routes with above-median lead time first",
                     "text": f"{len(hv_poor)} route(s) combine high volume with above-median average lead time. Highest volume: {names}. "
                             "These routes affect the most shipments, so a review there has the largest potential reach."})
    el = routes[routes["Eligible"]] if not routes.empty else routes
    if not el.empty:
        w = el.sort_values("Avg Lead Time", ascending=False).iloc[0]
        b = el.sort_values("Avg Lead Time").iloc[0]
        recs.append({"kind": "Observed", "priority": "Medium",
                     "title": "Slowest vs fastest ranked route",
                     "text": (f"Slowest ranked route: {w.Route} (avg {_fmt(w['Avg Lead Time'])} {unit}, {w.Shipments:,} shipments). "
                              f"Fastest: {b.Route} (avg {_fmt(b['Avg Lead Time'])} {unit}, {b.Shipments:,} shipments). "
                              "The data cannot say why they differ; an operational review (carrier, packing, dispatch timing) is the next step.")})

    # 3. factory / region bottlenecks
    for df, label, what in [(factory_perf, "Factory", "factory"), (region_perf, "Customer Region", "customer region")]:
        d = df[df["Shipments"] >= cfg.MIN_SHIPMENTS_FOR_RANKING] if not df.empty else df
        if len(d) >= 2:
            hi, lo = d.sort_values("Avg Lead Time").iloc[-1], d.sort_values("Avg Lead Time").iloc[0]
            gap = hi["Avg Lead Time"] - lo["Avg Lead Time"]
            rel = gap / lo["Avg Lead Time"] if lo["Avg Lead Time"] else float("nan")
            recs.append({"kind": "Observed", "priority": "Medium",
                         "title": f"Highest-lead-time {what}: {hi[label]}",
                         "text": (f"{hi[label]} averages {_fmt(hi['Avg Lead Time'])} {unit} vs {_fmt(lo['Avg Lead Time'])} for {lo[label]} "
                                  f"(gap {_fmt(gap)} {unit}, {rel:.1%} relative). Treat small gaps as descriptive rather than a confirmed bottleneck.")})

    # 4. ship modes
    if not mode_perf.empty and len(mode_perf) >= 2:
        best = mode_perf.sort_values("Median Lead Time").iloc[0]
        txt = (f"{best['Ship Mode']} has the lowest median lead time ({_fmt(best['Median Lead Time'])} {unit}).")
        pm = mode_perf.set_index("Ship Mode")
        if "Same Day" in pm.index and "Standard Class" in pm.index:
            d = pm.loc["Standard Class", "Avg Lead Time"] - pm.loc["Same Day", "Avg Lead Time"]
            txt += (f" Same Day vs Standard Class average difference: {_fmt(d)} {unit}. "
                    "If premium modes are not clearly faster than Standard Class, check whether ship dates reflect actual dispatch.")
        recs.append({"kind": "Observed", "priority": "Medium", "title": "Ship-mode comparison", "text": txt})

    # 5. delay concentration
    if not el.empty and kpis["delay_rate"] > 0:
        delayed = el.assign(Delayed_N=el["Delay Rate"] * el["Shipments"]).sort_values("Delayed_N", ascending=False)
        tot = (ship["Delayed"]).sum()
        top3 = delayed.head(3)
        share = top3["Delayed_N"].sum() / tot if tot else float("nan")
        recs.append({"kind": "Observed", "priority": "Medium",
                     "title": "Where delayed shipments concentrate",
                     "text": (f"The three routes with the most delayed shipments ({', '.join(top3['Route'])}) account for "
                              f"{share:.1%} of delayed shipments at the current threshold.")})

    recs.append({"kind": "Assumption", "priority": "Info",
                 "title": "Interpretation limits",
                 "text": ("Findings are descriptive. The dataset contains no carrier, distance, weather or shipping-cost fields, so causes "
                          "of delay are not identified. Straight-line factory-to-state distance is not used in any metric.")})
    return recs


# --------------------------------------------------------------------------- #
# Plausibility check used on the Data Quality page
# --------------------------------------------------------------------------- #
SERVICE_ORDER = ["Same Day", "First Class", "Second Class", "Standard Class"]


def shipmode_ordering_check(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean recorded vs relative lead time per ship mode (row level, analysed rows only),
    with whether each basis reproduces the expected service ordering
    (Same Day < First < Second < Standard)."""
    if rows.empty:
        return pd.DataFrame()
    t = rows.groupby("Ship Mode").agg(
        Rows=("Row ID", "size"),
        **{"Mean Recorded Lead Time": ("Raw Lead Time", "mean"),
           "Mean Relative Lead Time": ("Relative Lead Time", "mean")}).reset_index()
    t["_o"] = t["Ship Mode"].map({m: i for i, m in enumerate(SERVICE_ORDER)})
    t = t.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return t


def ordering_is_monotonic(t: pd.DataFrame, col: str) -> bool | None:
    known = t[t["Ship Mode"].isin(SERVICE_ORDER)]
    if len(known) < 3:
        return None
    return bool(known[col].is_monotonic_increasing)
