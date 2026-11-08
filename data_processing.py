"""Data loading, validation, cleaning and feature engineering.

Cleaning policy (documented in README as well):
  * Rows are NEVER silently dropped. Each row receives boolean flag columns and, if it
    is excluded from analysis, a human-readable ``Exclusion Reason``.
  * Excluded: unparseable Order/Ship Date, Ship Date earlier than Order Date,
    product with no factory mapping (no route can be defined).
  * Flagged but kept: extreme lead times, ship dates in the future, possible duplicate
    rows, rows with missing non-critical fields (filled with "Unknown").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


# --------------------------------------------------------------------------- #
# Path / file loading
# --------------------------------------------------------------------------- #
def resolve_csv_path(user_path: str | None = None) -> Path:
    """Return the first existing CSV path (user supplied first), else raise a helpful error."""
    candidates = [user_path] if user_path else []
    candidates += cfg.CSV_CANDIDATES
    tried = []
    for c in candidates:
        if not c:
            continue
        p = Path(str(c).strip().strip('"'))
        tried.append(str(p))
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find the dataset CSV. Paths tried:\n  - " + "\n  - ".join(tried)
        + "\nSet the path in the sidebar, upload the file, or set NASSAU_CSV_PATH."
    )


def read_csv_safely(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, trying common encodings."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc, dtype={"Postal Code": "string"})
        except UnicodeDecodeError as e:  # try next encoding
            last_err = e
    raise ValueError(f"Could not decode the CSV with common encodings: {last_err}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_product_key(name) -> str:
    """Normalise product names so '-Scrumdiddlyumptious' and '- Scrumdiddlyumptious' match."""
    s = str(name).replace("\u2019", "'").strip()
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


_NORMALISED_MAP = {normalize_product_key(k): v for k, v in cfg.PRODUCT_FACTORY_MAP.items()}


def map_product_to_factory(name):
    return _NORMALISED_MAP.get(normalize_product_key(name), np.nan)


def parse_dates(*series: pd.Series):
    """Parse one or more date columns with a *shared* format chosen by parse success.

    Returns (list_of_parsed_series, chosen_format).
    """
    cleaned = [s.astype("string").str.strip() for s in series]
    combined = pd.concat(cleaned, ignore_index=True)
    best_fmt, best_ok = None, -1
    for fmt in cfg.DATE_FORMATS:
        ok = pd.to_datetime(combined, format=fmt, errors="coerce").notna().sum()
        if ok > best_ok:  # strict '>' => ties keep the earlier (day-first) format
            best_fmt, best_ok = fmt, ok
    out = []
    for s in cleaned:
        parsed = pd.to_datetime(s, format=best_fmt, errors="coerce")
        out.append(parsed)
    return out, best_fmt


def _title_case_geo(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.str.title().str.replace(r"\bOf\b", "of", regex=True).str.replace(r"\bAnd\b", "and", regex=True)


@dataclass
class CleaningReport:
    """Everything the Data Quality page shows. All values are computed, none hard-coded."""
    raw_rows: int = 0
    date_format: str = ""
    missing_by_column: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    invalid_order_dates: int = 0
    invalid_ship_dates: int = 0
    negative_lead_rows: int = 0
    extreme_lead_rows: int = 0
    future_ship_rows: int = 0
    unmapped_rows: int = 0
    unmapped_products: list = field(default_factory=list)
    duplicate_rows: int = 0
    duplicate_order_id_rows: int = 0
    inconsistent_orders: int = 0
    non_us_rows: int = 0
    excluded_rows: int = 0
    kept_rows: int = 0
    reference_date: pd.Timestamp | None = None
    raw_lead_stats: dict = field(default_factory=dict)
    year_cycle_min_residual: float | None = None
    year_cycle_span: float | None = None
    missing_columns: list = field(default_factory=list)
    unknown_filled: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)



def leap_days_spanned(order: pd.Series, ship: pd.Series) -> pd.Series:
    """Number of 29-Feb dates falling within [Order Date, Ship Date] (diagnostic for the relative-lead-time basis)."""
    out = pd.Series(0, index=order.index, dtype="int64")
    ok = order.notna() & ship.notna()
    if not ok.any():
        return out
    for y in range(int(order[ok].dt.year.min()), int(ship[ok].dt.year.max()) + 1):
        if pd.Timestamp(year=y, month=1, day=1).is_leap_year:
            day = pd.Timestamp(year=y, month=2, day=29)
            out += ((order <= day) & (ship >= day) & ok).astype("int64")
    return out

# --------------------------------------------------------------------------- #
# Main cleaning pipeline
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame, reference_date=None):
    """Validate, flag and enrich the raw frame. Returns (full_df, CleaningReport).

    ``full_df`` contains ALL rows; use ``analysis_rows(full_df)`` for rows kept.
    """
    rep = CleaningReport(raw_rows=len(raw))
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in cfg.REQUIRED_COLUMNS if c not in df.columns]
    rep.missing_columns = missing_cols
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    rep.reference_date = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp.today().normalize()
    rep.missing_by_column = df.isna().sum()

    # ---- text standardisation ---------------------------------------------
    for c in ["Order ID", "Product ID", "Product Name", "Ship Mode", "Division", "City"]:
        df[c] = df[c].astype("string").str.strip()
    for c in ["Country/Region", "State/Province", "Region"]:
        df[c] = _title_case_geo(df[c])
    df["Ship Mode"] = df["Ship Mode"].str.title()

    # Missing non-critical fields -> "Unknown" (transparent, counted)
    for c in ["Ship Mode", "Country/Region", "State/Province", "Region", "Division", "City"]:
        n = int(df[c].isna().sum() + (df[c] == "").sum())
        if n:
            rep.unknown_filled[c] = n
            df[c] = df[c].replace("", pd.NA).fillna("Unknown")

    for c in ["Sales", "Units", "Gross Profit", "Cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- dates ------------------------------------------------------------
    (od, sd), fmt = parse_dates(df["Order Date"], df["Ship Date"])
    df["Order Date"], df["Ship Date"] = od, sd
    rep.date_format = fmt
    df["Flag Invalid Order Date"] = od.isna()
    df["Flag Invalid Ship Date"] = sd.isna()
    rep.invalid_order_dates = int(df["Flag Invalid Order Date"].sum())
    rep.invalid_ship_dates = int(df["Flag Invalid Ship Date"].sum())

    df["Raw Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days.astype("float")
    df["Leap Days Spanned"] = leap_days_spanned(df["Order Date"], df["Ship Date"])
    df["Flag Negative Lead Time"] = df["Raw Lead Time"] < 0
    df["Flag Extreme Lead Time"] = df["Raw Lead Time"] > cfg.MAX_PLAUSIBLE_LEAD_DAYS
    df["Flag Future Ship Date"] = df["Ship Date"] > rep.reference_date
    rep.negative_lead_rows = int(df["Flag Negative Lead Time"].sum())
    rep.extreme_lead_rows = int(df["Flag Extreme Lead Time"].sum())
    rep.future_ship_rows = int(df["Flag Future Ship Date"].sum())

    # ---- factory mapping --------------------------------------------------
    df["Factory"] = df["Product Name"].map(map_product_to_factory)
    df["Flag Unmapped Product"] = df["Factory"].isna()
    rep.unmapped_rows = int(df["Flag Unmapped Product"].sum())
    rep.unmapped_products = sorted(df.loc[df["Flag Unmapped Product"], "Product Name"].dropna().unique().tolist())
    df["Factory Latitude"] = df["Factory"].map(lambda f: cfg.FACTORY_COORDS.get(f, (np.nan, np.nan))[0])
    df["Factory Longitude"] = df["Factory"].map(lambda f: cfg.FACTORY_COORDS.get(f, (np.nan, np.nan))[1])

    # ---- duplicates / consistency ----------------------------------------
    business_cols = [c for c in df.columns if c in cfg.REQUIRED_COLUMNS and c != "Row ID"]
    df["Flag Possible Duplicate"] = df.duplicated(subset=business_cols, keep="first")
    rep.duplicate_rows = int(df["Flag Possible Duplicate"].sum())
    rep.duplicate_order_id_rows = int(df.duplicated(subset=["Order ID"], keep=False).sum())
    chk = df.groupby("Order ID").agg(n_sd=("Ship Date", "nunique"), n_sm=("Ship Mode", "nunique"))
    rep.inconsistent_orders = int(((chk["n_sd"] > 1) | (chk["n_sm"] > 1)).sum())
    rep.non_us_rows = int((df["Country/Region"] != "United States").sum())

    # ---- exclusions -------------------------------------------------------
    reason = pd.Series("", index=df.index, dtype="object")
    reason[df["Flag Invalid Order Date"] | df["Flag Invalid Ship Date"]] = "Invalid/unparseable date"
    reason[df["Flag Negative Lead Time"]] = "Ship Date earlier than Order Date"
    reason[df["Flag Unmapped Product"] & (reason == "")] = "Product has no factory mapping"
    df["Exclusion Reason"] = reason
    df["Excluded"] = reason != ""
    rep.excluded_rows = int(df["Excluded"].sum())
    rep.kept_rows = int((~df["Excluded"]).sum())

    # ---- features ---------------------------------------------------------
    df["Customer State"] = df["State/Province"]
    df["Customer Region"] = df["Region"]
    df["Route"] = df["Factory"].astype("string") + " → " + df["Customer State"].astype("string")
    df["Factory-Region Route"] = df["Factory"].astype("string") + " → " + df["Customer Region"].astype("string")
    df["Shipment ID"] = df["Order ID"].astype("string") + " | " + df["Factory"].astype("string")
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()

    # Relative lead time (ASSUMPTION-based diagnostic; see README / Data Quality page):
    # remove whole 365-day cycles, then subtract the dataset minimum of the remainder.
    valid = df["Raw Lead Time"].where(~df["Excluded"])
    resid = valid % cfg.YEAR_CYCLE_DAYS
    if resid.notna().any():
        rep.year_cycle_min_residual = float(resid.min())
        rep.year_cycle_span = float(resid.max() - resid.min())
        df["Relative Lead Time"] = (df["Raw Lead Time"] % cfg.YEAR_CYCLE_DAYS) - resid.min()
    else:
        df["Relative Lead Time"] = np.nan

    s = valid.dropna()
    if len(s):
        rep.raw_lead_stats = {
            "min": float(s.min()), "p25": float(s.quantile(.25)), "median": float(s.median()),
            "p75": float(s.quantile(.75)), "p95": float(s.quantile(.95)), "max": float(s.max()),
            "mean": float(s.mean()), "std": float(s.std()),
        }
    return df, rep


def analysis_rows(full_df: pd.DataFrame) -> pd.DataFrame:
    return full_df.loc[~full_df["Excluded"]].copy()


def apply_basis(df: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Create the 'Lead Time' column used by every metric, according to the chosen basis."""
    out = df.copy()
    col = "Relative Lead Time" if basis == cfg.ANALYSIS_BASIS_RELATIVE else "Raw Lead Time"
    out["Lead Time"] = out[col]
    return out


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Apply sidebar filters. ``f`` keys are optional; a list filter that is present but
    empty returns zero rows (the UI handles the empty state)."""
    m = pd.Series(True, index=df.index)
    if f.get("date_range"):
        start, end = f["date_range"]
        m &= (df["Order Date"] >= pd.Timestamp(start)) & (df["Order Date"] <= pd.Timestamp(end))
    for key, col in [("regions", "Customer Region"), ("states", "Customer State"), ("factories", "Factory"),
                     ("ship_modes", "Ship Mode"), ("divisions", "Division"), ("products", "Product Name")]:
        if f.get(key) is not None:
            m &= df[col].isin(f[key])
    if not f.get("include_extreme", True):
        m &= ~df["Flag Extreme Lead Time"]
    if f.get("exclude_duplicates", False):
        m &= ~df["Flag Possible Duplicate"]
    return df.loc[m]


def build_shipments(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Collapse order lines into SHIPMENTS = unique (Order ID, Factory).

    An order containing products from two factories is two physical shipments. Lead-time
    statistics are computed on this table so multi-line orders are not over-weighted.
    """
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["Order ID", "Factory"], as_index=False, sort=False).agg(
        **{
            "Order Date": ("Order Date", "min"),
            "Ship Date": ("Ship Date", "max"),
            "Ship Mode": ("Ship Mode", "first"),
            "Customer ID": ("Customer ID", "first"),
            "City": ("City", "first"),
            "Customer State": ("Customer State", "first"),
            "Customer Region": ("Customer Region", "first"),
            "Country": ("Country/Region", "first"),
            "Division": ("Division", "first"),
            "Lead Time": ("Lead Time", "first"),
            "Lines": ("Row ID", "count"),
            "Units": ("Units", "sum"),
            "Sales": ("Sales", "sum"),
            "Cost": ("Cost", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Products": ("Product Name", lambda s: ", ".join(sorted(set(s.dropna())))),
            "Any Extreme": ("Flag Extreme Lead Time", "max"),
        }
    )
    g["Route"] = g["Factory"] + " → " + g["Customer State"]
    g["Factory-Region Route"] = g["Factory"] + " → " + g["Customer Region"]
    g["Mode Group"] = g["Ship Mode"].map(cfg.MODE_GROUPS).fillna("Other")
    g["Delayed"] = g["Lead Time"] > threshold
    g["Order Month"] = g["Order Date"].dt.to_period("M").dt.to_timestamp()
    return g
