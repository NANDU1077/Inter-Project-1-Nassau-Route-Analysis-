"""Factory-to-Customer Shipping Route Efficiency Analysis - Nassau Candy Distributor.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
import config as cfg
import data_processing as dp
import ml_models as ml
import visualizations as viz

st.set_page_config(page_title="Nassau Candy | Shipping Route Efficiency", page_icon="🚚", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background:#F3F6FA; border:1px solid #E1E8F0; border-radius:8px; padding:10px 14px;}
[data-testid="stMetricLabel"] p {font-size:0.80rem; color:#5A6B7B;}
h1 {font-size:1.7rem !important;} h2 {font-size:1.3rem !important;} h3 {font-size:1.1rem !important;}
.rec {border-left:5px solid #1F4E79; background:#F7F9FC; padding:8px 14px; margin-bottom:8px; border-radius:4px;}
.rec.Critical {border-left-color:#C0392B;} .rec.High {border-left-color:#E07B39;}
.rec.Info {border-left-color:#9AA5B1;}
.tag {font-size:0.72rem; font-weight:600; padding:1px 7px; border-radius:10px; background:#E1E8F0; margin-right:6px;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and validating dataset…")
def load_from_path(path: str, mtime: float):
    return dp.clean_data(dp.read_csv_safely(path))


@st.cache_data(show_spinner="Loading and validating uploaded file…")
def load_from_bytes(content: bytes):
    return dp.clean_data(dp.read_csv_safely(io.BytesIO(content)))


def fmt_days(x, unit):
    return "n/a" if pd.isna(x) else f"{x:,.1f} {unit}"


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:.1%}"


TABLE_FORMATS = {
    "Avg Lead Time": st.column_config.NumberColumn("Avg Lead Time", format="%.1f"),
    "Median Lead Time": st.column_config.NumberColumn("Median Lead Time", format="%.1f"),
    "Lead Time Std Dev": st.column_config.NumberColumn("Lead Time Std Dev", format="%.2f"),
    "Delay Rate": st.column_config.NumberColumn("Delay Rate", format="percent"),
    "Gross Margin": st.column_config.NumberColumn("Gross Margin", format="percent"),
    "Share of Shipments": st.column_config.NumberColumn("Share of Shipments", format="percent"),
    "Total Sales": st.column_config.NumberColumn("Total Sales", format="$%.0f"),
    "Total Cost": st.column_config.NumberColumn("Total Cost", format="$%.0f"),
    "Total Gross Profit": st.column_config.NumberColumn("Total Gross Profit", format="$%.0f"),
    "Avg Sales / Shipment": st.column_config.NumberColumn("Avg Sales / Shipment", format="$%.2f"),
    "Avg Cost / Shipment": st.column_config.NumberColumn("Avg Cost / Shipment", format="$%.2f"),
    "Avg Gross Profit / Shipment": st.column_config.NumberColumn("Avg GP / Shipment", format="$%.2f"),
    "Cost per Unit (product)": st.column_config.NumberColumn("Product Cost / Unit", format="$%.2f"),
    "Efficiency Score": st.column_config.ProgressColumn("Efficiency Score", min_value=0, max_value=100, format="%.0f"),
    "Sales": st.column_config.NumberColumn("Sales", format="$%.2f"),
    "Cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
    "Gross Profit": st.column_config.NumberColumn("Gross Profit", format="$%.2f"),
    "Lead Time": st.column_config.NumberColumn("Lead Time", format="%.0f"),
    "Order Date": st.column_config.DateColumn("Order Date"),
    "Ship Date": st.column_config.DateColumn("Ship Date"),
}


def show_table(df: pd.DataFrame, cols=None, height=None):
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    d = df[cols] if cols else df
    kw = {"height": height} if height else {}
    st.dataframe(d, use_container_width=True, hide_index=True,
                 column_config={c: TABLE_FORMATS[c] for c in d.columns if c in TABLE_FORMATS}, **kw)


ROUTE_COLS = ["Route", "Customer Region", "Shipments", "Avg Lead Time", "Median Lead Time", "Lead Time Std Dev",
              "Delay Rate", "Efficiency Score", "Performance", "Total Sales", "Total Gross Profit",
              "Avg Sales / Shipment", "Avg Cost / Shipment", "Avg Gross Profit / Shipment"]


def route_cols(t: pd.DataFrame):
    return [c for c in ROUTE_COLS if c in t.columns]


# --------------------------------------------------------------------------- #
# Sidebar: data source, basis, filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🚚 Nassau Candy")
PAGES = ["A. Executive Overview", "B. Route Efficiency", "C. Geographic Map", "D. Ship Mode Comparison",
         "E. Route Drill-Down", "F. Data Quality", "G. Lead-Time & Delay Prediction"]
page = st.sidebar.radio("Dashboard section", PAGES)

with st.sidebar.expander("📁 Data source", expanded=False):
    default_path = next((c for c in cfg.CSV_CANDIDATES if c and Path(c).is_file()), cfg.CSV_CANDIDATES[1])
    csv_path = st.text_input("CSV path", value=default_path, help="Or set the NASSAU_CSV_PATH environment variable.")
    uploaded = st.file_uploader("…or upload a CSV", type="csv")

try:
    if uploaded is not None:
        full_df, report = load_from_bytes(uploaded.getvalue())
        source_label = f"uploaded file: {uploaded.name}"
    else:
        p = dp.resolve_csv_path(csv_path)
        full_df, report = load_from_path(str(p), p.stat().st_mtime)
        source_label = str(p)
except FileNotFoundError as e:
    st.error(f"📂 {e}")
    st.stop()
except Exception as e:  # noqa: BLE001 - show any parsing problem helpfully
    st.error(f"❌ Could not load/validate the dataset: {e}")
    st.stop()

base = dp.analysis_rows(full_df)
if base.empty:
    st.error("All rows were excluded during validation. See the Data Quality page for the reasons.")
    st.stop()

st.sidebar.markdown("### Analysis basis")
basis = st.sidebar.radio("Lead-time basis", [cfg.ANALYSIS_BASIS_RECORDED, cfg.ANALYSIS_BASIS_RELATIVE],
                         label_visibility="collapsed",
                         help="Recorded = exactly Ship Date − Order Date. Relative = an assumption-based diagnostic "
                              "that removes whole-year offsets; see Data Quality page before relying on it.")
unit = "relative days" if basis == cfg.ANALYSIS_BASIS_RELATIVE else "days"
base = dp.apply_basis(base, basis)

st.sidebar.markdown("### Filters")
dmin, dmax = base["Order Date"].min().date(), base["Order Date"].max().date()
dr = st.sidebar.date_input("Order date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(dr, (tuple, list)) and len(dr) == 2:
    date_range = (dr[0], dr[1])
else:
    d0 = dr[0] if isinstance(dr, (tuple, list)) and dr else dr
    date_range = (d0, d0)

regions = st.sidebar.multiselect("Customer region", sorted(base["Customer Region"].unique()),
                                 default=sorted(base["Customer Region"].unique()))
states = st.sidebar.multiselect("Customer state / province", sorted(base["Customer State"].unique()),
                                default=sorted(base["Customer State"].unique()))
factories = st.sidebar.multiselect("Factory", sorted(base["Factory"].unique()), default=sorted(base["Factory"].unique()))
modes = st.sidebar.multiselect("Ship mode", sorted(base["Ship Mode"].unique()), default=sorted(base["Ship Mode"].unique()))
divisions = st.sidebar.multiselect("Product division", sorted(base["Division"].unique()),
                                   default=sorted(base["Division"].unique()))
prod_opts = sorted(base.loc[base["Division"].isin(divisions), "Product Name"].unique())
products = st.sidebar.multiselect("Product name", prod_opts, default=prod_opts)

lt_valid = base["Lead Time"].dropna()
lt_max = int(math.ceil(lt_valid.max())) if len(lt_valid) else 1
lt_default = int(round(lt_valid.quantile(0.75))) if len(lt_valid) else 0
bkey = "rel" if basis == cfg.ANALYSIS_BASIS_RELATIVE else "rec"
threshold = st.sidebar.slider(f"Delay threshold ({unit})", 0, max(lt_max, 1), lt_default, key=f"thr_{bkey}",
                              help="A shipment is 'delayed' if its lead time is above this value. "
                                   "Default = 75th percentile of the data; set it to your own service target.")

extreme_share_all = float(base["Flag Extreme Lead Time"].mean())
all_extreme = extreme_share_all > cfg.EXTREME_SHARE_CRITICAL
include_extreme = st.sidebar.checkbox(
    "Include flagged extreme lead times", value=True,
    help=f"Extreme = recorded lead time > {cfg.MAX_PLAUSIBLE_LEAD_DAYS} days. "
         + ("Unchecking removes most/all data because the issue is dataset-wide." if all_extreme else ""))
excl_dups = st.sidebar.checkbox("Exclude possible duplicate rows", value=False,
                                help="Rows identical to an earlier row in every column except Row ID.")

filters = dict(date_range=date_range, regions=regions, states=states, factories=factories, ship_modes=modes,
               divisions=divisions, products=products, include_extreme=include_extreme, exclude_duplicates=excl_dups)
rows = dp.apply_filters(base, filters)
ship = dp.build_shipments(rows, threshold) if not rows.empty else pd.DataFrame()

st.sidebar.caption(f"Selection: **{len(rows):,}** rows · **{len(ship):,}** shipments")


# --------------------------------------------------------------------------- #
# Reusable banners
# --------------------------------------------------------------------------- #
def dq_banner():
    r = report
    share = r.extreme_lead_rows / max(r.raw_rows, 1)
    if share > cfg.EXTREME_SHARE_CRITICAL:
        s = r.raw_lead_stats
        st.error(
            f"🚨 **Critical data-quality warning:** {share:.1%} of rows ({r.extreme_lead_rows:,} of {r.raw_rows:,}) have a recorded "
            f"lead time above {cfg.MAX_PLAUSIBLE_LEAD_DAYS} days (min {s.get('min', float('nan')):,.0f}, median "
            f"{s.get('median', float('nan')):,.0f}, max {s.get('max', float('nan')):,.0f} days). "
            f"{r.future_ship_rows:,} rows ({r.future_ship_rows / max(r.raw_rows, 1):.1%}) have Ship Dates later than today "
            f"({r.reference_date:%d %b %Y}). These values are not plausible as real transit times, so **absolute lead times should not be "
            "taken at face value**. They are shown as recorded, not hidden; use the relative basis only as an exploratory comparison. "
            "See the Data Quality page.")
    elif r.extreme_lead_rows:
        st.warning(f"⚠️ {r.extreme_lead_rows:,} rows ({share:.1%}) have lead times above {cfg.MAX_PLAUSIBLE_LEAD_DAYS} days.")
    if r.excluded_rows:
        st.warning(f"{r.excluded_rows:,} rows were excluded from analysis during validation (invalid dates, negative lead time or unmapped products).")
    if basis == cfg.ANALYSIS_BASIS_RELATIVE:
        st.info("ℹ️ **Relative lead time basis is active.** Values are an *assumption-based* transformation (whole 365-day cycles removed, "
                "dataset minimum subtracted). They are useful only for comparing routes with each other, not as real transit times.")


def metric_note():
    st.caption("**Metric bases:** lead-time statistics, delay rate and route/mode/region/state tables use **shipments** "
               "(unique Order ID × Factory). Sales, cost and gross profit are summed over order lines. 'Orders' = unique Order IDs. "
               "Product tables use **order lines (rows)**.")


# --------------------------------------------------------------------------- #
# F. Data Quality (independent of filters)
# --------------------------------------------------------------------------- #
def page_data_quality():
    st.title("F. Data Quality")
    st.caption(f"Source: `{source_label}` · date format detected: `{report.date_format}` · reference date for 'future' checks: "
               f"{report.reference_date:%Y-%m-%d}. All figures here describe the **full raw dataset**, before sidebar filters.")
    dq_banner()
    r = report
    c = st.columns(5)
    c[0].metric("Raw rows", f"{r.raw_rows:,}")
    c[1].metric("Analysed rows", f"{r.kept_rows:,}")
    c[2].metric("Excluded rows", f"{r.excluded_rows:,}")
    c[3].metric("Extreme lead (flagged)", f"{r.extreme_lead_rows:,}")
    c[4].metric("Possible duplicates", f"{r.duplicate_rows:,}")

    st.subheader("Summary of records removed or flagged")
    summary = pd.DataFrame([
        ("Unparseable Order Date", r.invalid_order_dates, "Excluded"),
        ("Unparseable Ship Date", r.invalid_ship_dates, "Excluded"),
        ("Ship Date earlier than Order Date", r.negative_lead_rows, "Excluded"),
        ("Product with no factory mapping", r.unmapped_rows, "Excluded (no route can be defined)"),
        (f"Lead time > {cfg.MAX_PLAUSIBLE_LEAD_DAYS} days", r.extreme_lead_rows, "Flagged, kept (toggle in sidebar)"),
        ("Ship Date later than today", r.future_ship_rows, "Flagged, kept"),
        ("Rows identical to an earlier row (excl. Row ID)", r.duplicate_rows, "Flagged, kept (optional exclusion in sidebar)"),
        ("Rows sharing an Order ID with another row", r.duplicate_order_id_rows, "Informational: multi-line orders are normal"),
        ("Orders with inconsistent Ship Date / Ship Mode across lines", r.inconsistent_orders, "Informational"),
        ("Non-US customer rows (Canada)", r.non_us_rows, "Kept; shown on map at province centroids"),
    ], columns=["Check", "Rows / orders", "Treatment"])
    show_table(summary)
    if r.unknown_filled:
        st.write("Missing values filled with 'Unknown':", r.unknown_filled)

    t1, t2, t3, t4 = st.tabs(["Missing values", "Suspicious lead times", "Factory mapping", "Duplicates"])
    with t1:
        mv = r.missing_by_column.rename("Missing").to_frame()
        mv["% of rows"] = mv["Missing"] / max(r.raw_rows, 1)
        st.dataframe(mv.reset_index(names="Column"), hide_index=True, use_container_width=True,
                     column_config={"% of rows": st.column_config.NumberColumn(format="percent")})
        if mv["Missing"].sum() == 0:
            st.success("No missing values in any column of the raw file.")
    with t2:
        valid = full_df.loc[~full_df["Excluded"]]
        st.plotly_chart(viz.leadtime_hist(valid["Raw Lead Time"], "Recorded lead time distribution (Ship Date − Order Date)",
                                          nbins=80, cut=cfg.MAX_PLAUSIBLE_LEAD_DAYS), use_container_width=True)
        s = r.raw_lead_stats
        if s:
            st.write(f"Recorded lead time: min **{s['min']:,.0f}**, p25 {s['p25']:,.0f}, median **{s['median']:,.0f}**, "
                     f"p75 {s['p75']:,.0f}, p95 {s['p95']:,.0f}, max **{s['max']:,.0f}** days.")
        st.markdown("**Order year × Ship year (row counts)**")
        ct = pd.crosstab(valid["Order Date"].dt.year.rename("Order year"), valid["Ship Date"].dt.year.rename("Ship year"))
        st.dataframe(ct, use_container_width=True)
        if r.year_cycle_span is not None:
            st.markdown(
                f"**Pattern observed:** after removing whole 365-day cycles, recorded lead times vary over only "
                f"**{r.year_cycle_span:,.0f} days** (remainders of {r.year_cycle_min_residual:,.0f}–"
                f"{r.year_cycle_min_residual + r.year_cycle_span:,.0f} days). Lead times therefore cluster at roughly one-year spacing.\n\n"
                "**Interpretation (assumption, not proven):** this is consistent with Ship Dates carrying a constant offset plus whole-year "
                "shifts, i.e. a date-processing/export problem, rather than genuine multi-year shipping. The *Relative* basis in the sidebar "
                "subtracts those year cycles and the minimum remainder so that routes can be compared with each other. It is "
                "an exploratory transformation: it cannot recover true transit times, and leap days add ±1 day of noise. "
                "Confirm with the data owner before using either basis for decisions.")
        chk = an.shipmode_ordering_check(valid)
        if not chk.empty:
            st.markdown("**Plausibility check: do ship modes behave like their service levels?**")
            show_table(chk)
            ok_rec = an.ordering_is_monotonic(chk, "Mean Recorded Lead Time")
            ok_rel = an.ordering_is_monotonic(chk, "Mean Relative Lead Time")
            if ok_rec is not None:
                st.markdown(
                    f"- Recorded basis reproduces the expected order (Same Day < First < Second < Standard): **{ok_rec}**\n"
                    f"- Relative basis reproduces the expected order: **{ok_rel}**\n\n"
                    "If only the relative basis does, that supports (but does not prove) the view that the recorded dates are shifted by a "
                    "systematic offset while carrying real ordering information.")
        if "Leap Days Spanned" in valid and valid["Leap Days Spanned"].nunique() > 1:
            lg = valid.groupby("Leap Days Spanned")["Relative Lead Time"].agg(["mean", "size"])
            st.markdown("**Calendar artifact in the relative basis (leap days)**")
            st.dataframe(lg.rename(columns={"mean": "Mean relative lead time", "size": "Rows"}).reset_index(), hide_index=True,
                         use_container_width=True)
            st.markdown("Rows whose Order→Ship interval spans a 29 February sit about "
                        f"**{lg['mean'].iloc[-1] - lg['mean'].iloc[0]:.2f} days** higher on the relative basis than rows that do not, because the "
                        "365-day cycle ignores leap days. Relative lead times therefore carry roughly ±1 day of calendar noise, so they are "
                        "reliable for coarse comparisons (modes, large route gaps) but not for fine, single-day differences, and they should "
                        "not be compared across order years.")
        ext = valid[valid["Flag Extreme Lead Time"]].sort_values("Raw Lead Time", ascending=False).head(200)
        st.markdown("**Most extreme records (top 200)**")
        show_table(ext, ["Row ID", "Order ID", "Order Date", "Ship Date", "Raw Lead Time", "Ship Mode", "Customer State"], height=260)
    with t3:
        mp = full_df.groupby(["Product Name", "Factory"], dropna=False).size().rename("Rows").reset_index()
        mp["Factory"] = mp["Factory"].fillna("⚠️ UNMAPPED")
        show_table(mp)
        if r.unmapped_products:
            st.error(f"Unmapped products: {r.unmapped_products}")
        else:
            st.success(f"All {full_df['Product Name'].nunique()} distinct product names map to a factory.")
    with t4:
        dups = full_df[full_df["Flag Possible Duplicate"]]
        st.write(f"{len(dups):,} rows are identical to an earlier row in every column except Row ID. "
                 "They could be genuine repeat order lines or accidental double-entries; they are flagged, not removed, unless you tick "
                 "'Exclude possible duplicate rows'.")
        show_table(dups, ["Row ID", "Order ID", "Order Date", "Product Name", "Customer ID", "Sales", "Units"], height=300)



# --------------------------------------------------------------------------- #
# G. Predictive modelling (machine learning)  -- independent of sidebar filters
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_ml_cached(ship_df: pd.DataFrame, threshold: float):
    return ml.run_all(ship_df, threshold)


@st.cache_resource(show_spinner=False)
def fit_predictor_cached(_d: pd.DataFrame, reg_name: str, clf_name: str, threshold: float, _reg_oof):
    return ml.fit_predictor(_d, reg_name, clf_name, _reg_oof)


def page_ml():
    st.title("G. Lead-Time & Delay Prediction")
    st.caption("Can shipment lead time, or the chance of a delay, be predicted from route, ship mode, product and order timing? "
               "Models are validated with **5-fold cross-validation grouped by customer** (no customer appears in both training and test data) "
               "and every model must beat a one-line **baseline: the average for its ship mode**. Models use the *relative* lead time "
               "(the recorded one is dominated by the date offset), always on all validated shipments, independent of the sidebar filters.")
    ml_ship = dp.build_shipments(dp.apply_basis(dp.analysis_rows(full_df), cfg.ANALYSIS_BASIS_RELATIVE), 5)
    c1, c2 = st.columns([1, 2])
    thr = c1.slider("Delay threshold (relative days)", 1, 9, 5, help="Classification target: lead time above this value.")
    go = c2.button("▶ Train and evaluate models (about 30-90 s the first time)", type="primary")
    if go:
        st.session_state["ml_thr"] = thr
    if "ml_thr" not in st.session_state:
        st.info("Press the button to train the models. Results are cached, so re-opening this page is instant.")
        return
    t = st.session_state["ml_thr"]
    if t != thr:
        st.warning(f"Showing results for threshold {t}. Press the button to retrain with {thr}.")
    try:
        with st.spinner("Training and cross-validating models…"):
            R = run_ml_cached(ml_ship, float(t))
    except Exception as e:  # noqa: BLE001
        st.error(f"Model training failed: {e}")
        return
    d, reg, clf = R["data"], R["reg"]["table"], R["clf"]["table"]
    br, bc = reg[reg["Model"] == ml.BASELINE_REG].iloc[0], clf[clf["Model"] == ml.BASELINE_CLF].iloc[0]
    best_r, best_c = reg.sort_values("MAE").iloc[0], clf.sort_values("ROC AUC", ascending=False).iloc[0]
    k = st.columns(4)
    k[0].metric("Shipments modelled", f"{len(d):,}")
    k[1].metric("Baseline MAE (rel. days)", f"{br['MAE']:.3f}", help="Predicting each shipment's lead time as the average for its ship mode.")
    k[2].metric("Best model MAE", f"{best_r['MAE']:.3f}", f"{best_r['MAE'] - br['MAE']:+.3f} vs baseline", delta_color="inverse")
    k[3].metric("Best model delay AUC", f"{best_c['ROC AUC']:.3f}", f"{best_c['ROC AUC'] - bc['ROC AUC']:+.3f} vs baseline")

    yes_r = reg.loc[reg["Beats baseline?"] == "Yes", "Model"].tolist(); yes_c = clf.loc[clf["Beats baseline?"] == "Yes", "Model"].tolist()
    ro_r = reg[reg["Model"].str.contains("route-only")].iloc[0]; ro_c = clf[clf["Model"].str.contains("route-only")].iloc[0]
    st.subheader("What the models show")
    st.markdown(
        f"- **Lead-time regression:** the ship-mode baseline reaches R² = {br['R²']:.2f} (MAE {br['MAE']:.2f}). Models that significantly beat it "
        f"(95% bootstrap CI): **{', '.join(yes_r) if yes_r else 'none'}**.\n"
        f"- **Delay classification (> {t} relative days):** baseline AUC = {bc['ROC AUC']:.3f}. Significantly better: "
        f"**{', '.join(yes_c) if yes_c else 'none'}**"
        + (f" (gain of {best_c['ROC AUC'] - bc['ROC AUC']:+.3f} AUC, small)." if yes_c else ".") + "\n"
        f"- **Route information alone** (factory, state, region; no ship mode): R² = {ro_r['R²']:.3f}, AUC = {ro_c['ROC AUC']:.3f}. "
        + ("That is essentially no predictive power: route effects exist in the data but are too small and noisy to predict new shipments." if ro_r["R²"] < 0.05
           else "Route information carries some predictive signal.") + "\n"
        "- Cross-validation compares models fairly on unseen customers; a model that matches the baseline is *not* better, however complex.")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(viz.ml_comparison_chart(reg, "MAE", ml.BASELINE_REG, True), use_container_width=True)
        show_table(reg[["Model", "MAE", "RMSE", "R²", "MAE vs baseline", "95% CI low", "95% CI high", "Beats baseline?"]].round(4))
    with b:
        st.plotly_chart(viz.ml_comparison_chart(clf, "ROC AUC", ml.BASELINE_CLF, False), use_container_width=True)
        show_table(clf[["Model", "ROC AUC", "Avg precision", "Brier score", "AUC vs baseline", "95% CI low", "95% CI high", "Beats baseline?"]].round(4))
    st.caption("MAE/RMSE are in relative days (lower is better). AUC: 0.5 = chance, 1 = perfect. 'vs baseline' columns show the paired difference with a 95% bootstrap interval.")

    st.subheader("Which inputs matter? (permutation importance, hold-out customers)")
    a, b = st.columns(2)
    a.plotly_chart(viz.importance_chart(R["imp_reg"], "Lead-time model: increase in MAE when a feature is shuffled", "MAE increase (relative days)"), use_container_width=True)
    b.plotly_chart(viz.importance_chart(R["imp_clf"], "Delay model: decrease in AUC when a feature is shuffled", "AUC decrease"), use_container_width=True)
    st.caption("Features whose importance is near zero (or negative) do not help the model on unseen customers. Error bars are one standard deviation over 10 shuffles.")

    st.subheader("What-if predictor")
    st.caption("Predictions come from the selected models. Cross-validation shows that route inputs (factory, state) add little beyond ship mode, "
               "so treat differences between routes in this tool as small and uncertain. The range is an 80% interval from out-of-fold errors, in relative days.")
    mcol = st.columns(2)
    reg_name = mcol[0].selectbox("Lead-time model", reg["Model"].tolist(), index=int(reg["Model"].tolist().index(best_r["Model"])))
    clf_name = mcol[1].selectbox("Delay model", clf["Model"].tolist(), index=int(clf["Model"].tolist().index(best_c["Model"])))
    pred = fit_predictor_cached(d, reg_name, clf_name, float(t), R["reg"]["oof"][reg_name])
    ic = st.columns(4)
    fac = ic[0].selectbox("Factory", sorted(d["Factory"].unique()))
    sta = ic[1].selectbox("Customer state", sorted(d["Customer State"].unique()), index=sorted(d["Customer State"].unique()).index("New York") if "New York" in set(d["Customer State"]) else 0)
    mode = ic[2].selectbox("Ship mode", ["Same Day", "First Class", "Second Class", "Standard Class"])
    div = ic[3].selectbox("Division", sorted(d["Division"].unique()))
    ic2 = st.columns(4)
    mon = ic2[0].slider("Order month", 1, 12, 6)
    dow = ic2[1].slider("Order weekday (0 = Mon)", 0, 6, 2)
    units = ic2[2].number_input("Units", 1, 50, int(d["Units"].median()))
    sales = ic2[3].number_input("Sales ($)", 1.0, 500.0, float(round(d["Sales"].median(), 2)))
    out = ml.predict_what_if(pred, fac, sta, mode, div, mon, dow, units, sales)
    o = st.columns(3)
    o[0].metric("Predicted lead time", f"{max(out['lead'], 0):.2f} relative days")
    o[1].metric("80% range", f"{max(out['low'], 0):.1f} to {max(out['high'], 0):.1f}")
    o[2].metric(f"Chance of > {t} relative days", f"{out['p_delay']:.0%}")

    st.subheader("Route segments (unsupervised: K-means)")
    cl = R["clusters"]
    if cl["k"] > 1:
        sil = cl["silhouette"]; best_sil = sil["Silhouette"].max()
        st.caption(f"Ranked routes (≥ {cfg.MIN_SHIPMENTS_FOR_RANKING} shipments) are grouped on ship-mode-adjusted mean lead time, spread, volume and order value. "
                   f"Best k = {cl['k']} (silhouette {best_sil:.2f}; below ~0.5 means weak, exploratory structure).")
        a, b = st.columns([3, 2])
        a.plotly_chart(viz.cluster_scatter(cl["table"]), use_container_width=True)
        with b:
            show_table(cl["profile"].round(2))
            show_table(sil.round(3))
        with st.expander("Routes by segment"):
            show_table(cl["table"].sort_values(["Cluster", "Shipments"], ascending=[True, False])[["Cluster", "Route", "Shipments", "Adj mean lead", "Adj lead SD", "Avg sales / shipment"]].round(2), height=320)
    else:
        st.info("Not enough ranked routes to cluster.")

# --------------------------------------------------------------------------- #
# Guard for empty selections
# --------------------------------------------------------------------------- #
if page.startswith("F."):
    page_data_quality()
    st.stop()
if page.startswith("G."):
    page_ml()
    st.stop()

if ship.empty:
    st.title(page[3:])
    st.warning("🔍 The current filters return no shipments. Widen the date range, re-select regions/states/factories/modes/products, "
               "or tick 'Include flagged extreme lead times'.")
    st.stop()

kpis = an.compute_kpis(rows, ship)
routes_state = an.route_table(ship, "state")
routes_region = an.route_table(ship, "region")
factory_perf = an.group_performance(ship, "Factory")
region_perf = an.group_performance(ship, "Customer Region")
state_perf = an.group_performance(ship, "Customer State")
mode_perf = an.shipmode_tradeoff(ship)
hv_poor = an.high_volume_poor_routes(routes_state)


# --------------------------------------------------------------------------- #
# A. Executive overview
# --------------------------------------------------------------------------- #
def page_exec():
    st.title("Factory-to-Customer Shipping Route Efficiency")
    st.caption("Executive overview for Nassau Candy Distributor · all figures respond to the sidebar filters")
    dq_banner()
    metric_note()

    c = st.columns(5)
    c[0].metric("Shipments", f"{kpis['shipments']:,}", help="Unique Order ID × Factory combinations in the selection.")
    c[1].metric("Orders", f"{kpis['orders']:,}", help="Unique Order IDs.")
    c[2].metric("Avg lead time", fmt_days(kpis["avg_lead"], unit), help="Mean of shipment lead times.")
    c[3].metric("Median lead time", fmt_days(kpis["median_lead"], unit), help="Less sensitive to outliers than the mean.")
    c[4].metric("Lead-time std dev", fmt_days(kpis["std_lead"], unit), help="Variability across shipments.")
    c = st.columns(5)
    c[0].metric("Delay frequency", pct(kpis["delay_rate"]), help=f"Share of shipments with lead time above {threshold} {unit}.")
    c[1].metric("Total sales", f"${kpis['sales']:,.0f}")
    c[2].metric("Total gross profit", f"${kpis['gross_profit']:,.0f}")
    c[3].metric("Gross margin", pct(kpis["margin"]))
    c[4].metric("Customers", f"{kpis['customers']:,}")

    st.subheader("Best and worst routes")
    st.caption(f"Routes are Factory → Customer State. Only routes with ≥ {cfg.MIN_SHIPMENTS_FOR_RANKING} shipments are ranked. "
               "Efficiency score = min-max scaled average lead time (100 = fastest ranked route, 0 = slowest).")
    top, bottom = an.top_bottom_routes(routes_state, 5)
    a, b = st.columns(2)
    with a:
        st.markdown("**🟢 Most efficient**")
        show_table(top, ["Route", "Shipments", "Avg Lead Time", "Delay Rate", "Efficiency Score"])
    with b:
        st.markdown("**🔴 Least efficient**")
        show_table(bottom, ["Route", "Shipments", "Avg Lead Time", "Delay Rate", "Efficiency Score"])

    a, b = st.columns(2)
    a.plotly_chart(viz.group_bar(factory_perf, "Factory", "Average lead time by factory", "Factory", cfg.FACTORY_COLORS, unit),
                   use_container_width=True)
    a.caption("Descriptive comparison of factories; differences may be small relative to lead-time variability.")
    b.plotly_chart(viz.monthly_trend_chart(an.monthly_trend(ship), unit), use_container_width=True)
    b.caption("Bars: shipment volume by order month. Line: average lead time.")

    st.subheader("Recommendations")
    st.caption("Generated from the numbers in the current selection. 'Observed' = what the data shows; 'Assumption' = a limit on interpretation.")
    for r in an.generate_recommendations(ship, routes_state, factory_perf, region_perf, mode_perf, hv_poor, basis, kpis):
        st.markdown(f"<div class='rec {r['priority']}'><span class='tag'>{r['kind']}</span><span class='tag'>{r['priority']}</span>"
                    f"<b>{r['title']}</b><br>{r['text']}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# B. Route efficiency
# --------------------------------------------------------------------------- #
def page_routes():
    st.title("B. Route Efficiency Overview")
    st.caption("A **route** is Factory → Customer State (or Factory → Customer Region). Metrics are computed on shipments.")
    metric_note()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    level = c1.radio("Route level", ["State", "Region"], horizontal=True)
    min_ship = c2.number_input("Min shipments to rank", 1, 1000, cfg.MIN_SHIPMENTS_FOR_RANKING,
                               help="Routes below this volume are listed but not scored/ranked.")
    perf_sel = c3.multiselect("Performance class", ["Efficient", "Moderate", "Inefficient", "Not ranked"],
                              default=["Efficient", "Moderate", "Inefficient", "Not ranked"])
    search = c4.text_input("Search route", placeholder="e.g. Sugar Shack or Texas")
    routes = an.route_table(ship, level.lower(), int(min_ship))
    view = routes[routes["Performance"].isin(perf_sel)]
    if search:
        view = view[view["Route"].str.contains(search, case=False, regex=False)]

    top, bottom = an.top_bottom_routes(routes, 10)
    st.subheader("Route leaderboard")
    st.caption("Sorted by efficiency score. Efficiency score = 100 × (slowest avg − route avg) / (slowest avg − fastest avg) "
               "among ranked routes. Std dev measures lead-time variability within the route.")
    show_table(view, route_cols(view), height=380)

    a, b = st.columns(2)
    a.plotly_chart(viz.route_bar(top, "Top 10 most efficient routes", unit=unit), use_container_width=True)
    b.plotly_chart(viz.route_bar(bottom.iloc[::-1] if not bottom.empty else bottom, "Bottom 10 least efficient routes", unit=unit),
                   use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Top 10 most efficient**")
        show_table(top, ["Route", "Shipments", "Avg Lead Time", "Lead Time Std Dev", "Delay Rate", "Efficiency Score"])
    with b:
        st.markdown("**Bottom 10 least efficient**")
        show_table(bottom, ["Route", "Shipments", "Avg Lead Time", "Lead Time Std Dev", "Delay Rate", "Efficiency Score"])

    st.subheader("Volume vs. performance")
    st.caption("Routes in the upper-right (above median volume *and* above median lead time) deserve review first.")
    a, b = st.columns([3, 2])
    a.plotly_chart(viz.volume_vs_leadtime(routes[routes["Eligible"]], unit), use_container_width=True)
    b.plotly_chart(viz.efficiency_hist(routes), use_container_width=True)

    st.subheader("High-volume routes with poor performance")
    st.caption(f"Ranked routes at or above the {cfg.HIGH_VOLUME_QUANTILE:.0%} volume quantile whose average lead time is above the median of ranked routes.")
    hv = an.high_volume_poor_routes(routes)
    show_table(hv, route_cols(hv))
    if hv.empty:
        st.info("No route meets both criteria in the current selection.")


# --------------------------------------------------------------------------- #
# C. Geographic map
# --------------------------------------------------------------------------- #
def page_map():
    st.title("C. Geographic Shipping Map")
    st.caption("Diamonds are factories. Bubbles are customer states/provinces (positioned at approximate centroids): size = shipments, "
               "colour = chosen metric. Lines connect factories to states for the highest-volume routes. Hover for details.")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    metric = c1.selectbox("Colour by", ["Average lead time", "Efficiency score", "Delay rate"])
    show_lines = c2.checkbox("Show factory → state routes", value=True)
    n_lines = c3.slider("Routes drawn (top by volume)", 5, 100, 30, step=5, disabled=not show_lines)
    stats = an.state_map_stats(ship)
    lines = routes_state.sort_values("Shipments", ascending=False).head(n_lines)
    tab_heat, tab_routes = st.tabs(["US heatmap (states)", "Bubbles + factory routes (incl. Canada)"])
    with tab_heat:
        st.plotly_chart(viz.us_heatmap(stats, metric, unit), use_container_width=True)
        n_ca = int((stats["Country"] != "United States").sum()) if not stats.empty else 0
        if n_ca:
            st.caption(f"{n_ca} Canadian province(s) are not drawn on the US heatmap; see the bubble map.")
    with tab_routes:
        st.plotly_chart(viz.route_map(stats, lines, metric, unit, show_lines), use_container_width=True)
    st.caption("Efficiency score on this map is computed per destination state (all factories combined) among states with enough shipments; "
               "low-volume states show 'n/a'.")

    st.subheader("Bottleneck states and regions")
    st.caption(f"Highest average lead time among groups with ≥ {cfg.MIN_SHIPMENTS_FOR_RANKING} shipments. "
               "These are observed slow spots, not diagnosed causes.")
    a, b = st.columns(2)
    with a:
        st.markdown("**States / provinces**")
        show_table(an.bottleneck_table(state_perf, "Customer State", 10))
    with b:
        st.markdown("**Customer regions**")
        show_table(an.bottleneck_table(region_perf, "Customer Region", 10))
    a, b = st.columns(2)
    a.plotly_chart(viz.group_bar(region_perf, "Customer Region", "Average lead time by customer region", unit=unit),
                   use_container_width=True)
    b.plotly_chart(viz.group_bar(factory_perf, "Factory", "Average lead time by factory", "Factory", cfg.FACTORY_COLORS, unit),
                   use_container_width=True)
    with st.expander("Factory-to-region routes"):
        show_table(routes_region, route_cols(routes_region))
    with st.expander("All state statistics"):
        show_table(stats.drop(columns=["Latitude", "Longitude", "Eligible"], errors="ignore"))


# --------------------------------------------------------------------------- #
# D. Ship mode comparison
# --------------------------------------------------------------------------- #
def page_modes():
    st.title("D. Ship Mode Comparison")
    st.caption("Ship modes found in the data are compared on shipments. **The dataset has no shipping-cost field**: Sales, Cost and Gross Profit "
               "describe the *products*, not freight, so cost–time trade-offs below are descriptive only.")
    metric_note()
    show_table(mode_perf, ["Ship Mode", "Shipments", "Share of Shipments", "Avg Lead Time", "Median Lead Time", "Lead Time Std Dev", "Delay Rate",
                           "Total Sales", "Total Gross Profit", "Gross Margin", "Avg Sales / Shipment", "Cost per Unit (product)"])
    a, b = st.columns(2)
    a.plotly_chart(viz.shipmode_grouped(mode_perf, unit), use_container_width=True)
    b.plotly_chart(viz.leadtime_box(ship, "Ship Mode", "Lead-time spread by ship mode", cfg.SHIPMODE_COLORS, unit),
                   use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(viz.delay_rate_bar(mode_perf, "Ship Mode", "Delay frequency by ship mode", "Ship Mode", cfg.SHIPMODE_COLORS),
                   use_container_width=True)
    b.plotly_chart(viz.volume_bar(mode_perf, "Ship Mode", "Shipment volume by ship mode", "Ship Mode", cfg.SHIPMODE_COLORS),
                   use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(viz.tradeoff_scatter(mode_perf, "Avg Sales / Shipment", "Lead time vs. average sales per shipment", unit),
                   use_container_width=True)
    b.plotly_chart(viz.tradeoff_scatter(mode_perf, "Gross Margin", "Lead time vs. gross margin", unit), use_container_width=True)

    st.subheader("Standard vs. expedited")
    st.caption("Grouping assumption: Standard Class = Standard; First Class, Second Class and Same Day = Expedited "
               "(the data itself has no 'expedited' label).")
    grp = an.group_performance(ship, "Mode Group")
    show_table(grp, ["Mode Group", "Shipments", "Avg Lead Time", "Median Lead Time", "Lead Time Std Dev", "Delay Rate",
                     "Total Sales", "Total Gross Profit", "Gross Margin", "Avg Sales / Shipment"])
    a2, b2 = st.columns(2)
    a2.plotly_chart(viz.group_bar(grp, "Mode Group", "Average lead time: Standard vs. Expedited", "Mode Group",
                                  cfg.MODE_GROUP_COLORS, unit), use_container_width=True)
    b2.plotly_chart(viz.delay_rate_bar(grp, "Mode Group", "Delay frequency: Standard vs. Expedited", "Mode Group",
                                       cfg.MODE_GROUP_COLORS), use_container_width=True)

    st.subheader("How to read the trade-off")
    fastest = mode_perf.sort_values("Median Lead Time").iloc[0]
    slowest = mode_perf.sort_values("Median Lead Time").iloc[-1]
    lines = [f"**Observed:** {fastest['Ship Mode']} has the lowest median lead time ({fmt_days(fastest['Median Lead Time'], unit)}); "
             f"{slowest['Ship Mode']} the highest ({fmt_days(slowest['Median Lead Time'], unit)})."]
    pm = mode_perf.set_index("Ship Mode")
    if len(pm) > 1:
        m_hi, m_lo = pm["Gross Margin"].idxmax(), pm["Gross Margin"].idxmin()
        lines.append(f"**Observed:** gross margin ranges from {pct(pm.loc[m_lo, 'Gross Margin'])} ({m_lo}) to {pct(pm.loc[m_hi, 'Gross Margin'])} ({m_hi}).")
    spread = (mode_perf["Avg Lead Time"].max() - mode_perf["Avg Lead Time"].min())
    lines.append(f"**Observed:** average lead time differs by {spread:,.1f} {unit} across modes, compared with an overall standard deviation of "
                 f"{fmt_days(kpis['std_lead'], unit)}. When the between-mode gap is small relative to within-mode spread, ship mode explains little "
                 "of the variation in this data.")
    lines.append("**Not available:** freight cost per mode. Any statement about paying more for faster delivery would be an assumption.")
    st.markdown("\n\n".join(f"- {l}" for l in lines))


# --------------------------------------------------------------------------- #
# E. Drill-down
# --------------------------------------------------------------------------- #
def page_drill():
    st.title("E. Route Drill-Down")
    st.caption("Narrow the (already sidebar-filtered) data to a specific route. Pick 'All' to leave a dimension open.")
    cols = st.columns(5)

    def sel(col, label, series):
        opts = ["All"] + sorted(series.dropna().unique().tolist())
        return col.selectbox(label, opts)

    f = sel(cols[0], "Factory", rows["Factory"])
    s = sel(cols[1], "Customer state", rows["Customer State"])
    r = sel(cols[2], "Customer region", rows["Customer Region"])
    p = sel(cols[3], "Product", rows["Product Name"])
    m = sel(cols[4], "Ship mode", rows["Ship Mode"])
    sub = rows
    for val, col in [(f, "Factory"), (s, "Customer State"), (r, "Customer Region"), (p, "Product Name"), (m, "Ship Mode")]:
        if val != "All":
            sub = sub[sub[col] == val]
    sub_ship = dp.build_shipments(sub, threshold) if not sub.empty else pd.DataFrame()
    if sub_ship.empty:
        st.warning("No shipments match this combination (e.g. the chosen product is not made at the chosen factory). Change a selector.")
        return
    k = an.compute_kpis(sub, sub_ship)
    c = st.columns(6)
    c[0].metric("Shipments", f"{k['shipments']:,}")
    c[1].metric("Avg lead time", fmt_days(k["avg_lead"], unit))
    c[2].metric("Median", fmt_days(k["median_lead"], unit))
    c[3].metric("Std dev", fmt_days(k["std_lead"], unit))
    c[4].metric("Delay frequency", pct(k["delay_rate"]))
    c[5].metric("Sales / GP", f"${k['sales']:,.0f} / ${k['gross_profit']:,.0f}")
    st.caption("Compared with the full selection: avg " + fmt_days(kpis["avg_lead"], unit) + ", delay frequency " + pct(kpis["delay_rate"]) + ".")

    st.subheader("Route-level metrics")
    sub_routes = an.route_table(sub_ship, "state", 1)
    show_table(sub_routes, route_cols(sub_routes), height=260)

    a, b = st.columns(2)
    a.plotly_chart(viz.leadtime_hist(sub_ship["Lead Time"], "Lead-time distribution (shipments)", nbins=40, unit=unit, cut=threshold),
                   use_container_width=True)
    b.plotly_chart(viz.leadtime_over_time(sub_ship, unit), use_container_width=True)

    st.subheader("State-level performance insights")
    st.caption("States within the current drill-down selection (all factories serving them). Highest average lead time first.")
    sp = an.group_performance(sub_ship, "Customer State")
    show_table(sp.sort_values("Avg Lead Time", ascending=False),
               ["Customer State", "Shipments", "Avg Lead Time", "Median Lead Time", "Lead Time Std Dev", "Delay Rate",
                "Total Sales", "Total Gross Profit"], height=260)

    st.subheader("Order-level shipment timeline")
    st.caption("Each bar spans Order Date → Ship Date for one shipment. With recorded dates, bars can be years long - see Data Quality.")
    n_show = st.slider("Shipments shown", 10, 200, 40, step=10)
    st.plotly_chart(viz.timeline_chart(sub_ship, n_show), use_container_width=True)

    t1, t2, t3 = st.tabs(["Shipments", "Order lines (product & customer detail)", "Product summary"])
    with t1:
        show_table(sub_ship.sort_values("Order Date"),
                   ["Order ID", "Factory", "Customer State", "Customer Region", "Ship Mode", "Order Date", "Ship Date", "Lead Time",
                    "Delayed", "Lines", "Units", "Sales", "Gross Profit", "Products"], height=380)
    with t2:
        show_table(sub.sort_values("Order Date"),
                   ["Row ID", "Order ID", "Customer ID", "City", "Customer State", "Postal Code", "Product Name", "Division", "Factory",
                    "Ship Mode", "Order Date", "Ship Date", "Lead Time", "Units", "Sales", "Cost", "Gross Profit"], height=380)
    with t3:
        pp = an.product_performance(sub, threshold)
        show_table(pp)


# --------------------------------------------------------------------------- #
# Product / division section lives under Route Efficiency? -> shown in Executive extras
# --------------------------------------------------------------------------- #
def page_products_extra():
    st.subheader("Product and division performance")
    st.caption("Based on **order lines (rows)**: each line inherits its shipment's lead time.")
    a, b = st.columns(2)
    with a:
        st.markdown("**By division**")
        dv = an.product_performance(rows, threshold, "Division")
        show_table(dv)
    with b:
        st.markdown("**By product**")
        pp = an.product_performance(rows, threshold)
        show_table(pp, ["Product Name", "Factory", "Lines", "Orders", "Units", "Avg Lead Time", "Delay Rate", "Total Sales",
                        "Total Gross Profit", "Gross Margin"])


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
if page.startswith("A."):
    page_exec()
    page_products_extra()
elif page.startswith("B."):
    page_routes()
elif page.startswith("C."):
    page_map()
elif page.startswith("D."):
    page_modes()
elif page.startswith("E."):
    page_drill()
