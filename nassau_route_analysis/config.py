"""Central configuration: paths, factory/product reference data, thresholds, colours.

Everything here is *reference data or tunable settings* - never analytical results.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Data source (first existing candidate wins; override with NASSAU_CSV_PATH)
# --------------------------------------------------------------------------- #
CSV_CANDIDATES = [
    os.environ.get("NASSAU_CSV_PATH", ""),
    str(BASE_DIR / "data" / "Nassau_Candy_Distributor.csv"),
    str(BASE_DIR / "data" / "Nassau Candy Distributor.csv"),
    r"C:\Users\nandu\OneDrive\Desktop\Intern1\Nassau Candy Distributor.csv",
]

REQUIRED_COLUMNS = [
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
    "Country/Region", "City", "State/Province", "Postal Code", "Division", "Region",
    "Product ID", "Product Name", "Sales", "Units", "Gross Profit", "Cost",
]

# Candidate date formats, tried in this order; the one that parses the most values wins.
# Ties (fully ambiguous data) resolve to the earliest entry, i.e. day-first.
DATE_FORMATS = ["%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]

# --------------------------------------------------------------------------- #
# Factories & product mapping (from the project brief)
# --------------------------------------------------------------------------- #
FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

# --------------------------------------------------------------------------- #
# Approximate state / province centroids (map display only)
# --------------------------------------------------------------------------- #
STATE_COORDS = {
    "Alabama": (32.81, -86.79), "Arizona": (34.29, -111.66), "Arkansas": (34.89, -92.44),
    "California": (37.18, -119.47), "Colorado": (38.99, -105.55), "Connecticut": (41.62, -72.73),
    "Delaware": (38.99, -75.51), "District of Columbia": (38.90, -77.03), "Florida": (28.63, -82.45),
    "Georgia": (32.64, -83.44), "Idaho": (44.35, -114.61), "Illinois": (40.04, -89.20),
    "Indiana": (39.89, -86.28), "Iowa": (42.07, -93.50), "Kansas": (38.48, -98.38),
    "Kentucky": (37.53, -85.30), "Louisiana": (31.07, -92.00), "Maine": (45.37, -69.24),
    "Maryland": (39.05, -76.79), "Massachusetts": (42.26, -71.81), "Michigan": (44.35, -85.41),
    "Minnesota": (46.28, -94.31), "Mississippi": (32.74, -89.67), "Missouri": (38.36, -92.46),
    "Montana": (47.05, -109.64), "Nebraska": (41.54, -99.80), "Nevada": (39.33, -116.63),
    "New Hampshire": (43.68, -71.58), "New Jersey": (40.19, -74.67), "New Mexico": (34.41, -106.11),
    "New York": (42.95, -75.53), "North Carolina": (35.56, -79.39), "North Dakota": (47.45, -100.47),
    "Ohio": (40.29, -82.79), "Oklahoma": (35.59, -97.49), "Oregon": (43.93, -120.56),
    "Pennsylvania": (40.88, -77.80), "Rhode Island": (41.68, -71.51), "South Carolina": (33.91, -80.90),
    "South Dakota": (44.44, -100.23), "Tennessee": (35.86, -86.35), "Texas": (31.48, -99.33),
    "Utah": (39.31, -111.67), "Vermont": (44.07, -72.67), "Virginia": (37.52, -78.85),
    "Washington": (47.38, -120.45), "West Virginia": (38.64, -80.62), "Wisconsin": (44.62, -89.99),
    "Wyoming": (42.99, -107.55), "Hawaii": (20.8, -156.3), "Alaska": (64.0, -152.0),
    # Canada
    "Alberta": (55.0, -115.0), "British Columbia": (53.7, -127.6), "Manitoba": (54.8, -98.8),
    "New Brunswick": (46.5, -66.2), "Newfoundland and Labrador": (53.1, -60.0),
    "Nova Scotia": (45.0, -63.0), "Ontario": (50.0, -85.0), "Prince Edward Island": (46.4, -63.3),
    "Quebec": (52.0, -72.0), "Saskatchewan": (54.0, -106.0),
}

# --------------------------------------------------------------------------- #
# Analysis settings
# --------------------------------------------------------------------------- #
# Lead times above this are flagged "extreme" (kept, flagged, never silently dropped).
MAX_PLAUSIBLE_LEAD_DAYS = 60
# Routes with fewer shipments than this are shown but not ranked / scored.
MIN_SHIPMENTS_FOR_RANKING = 5
# "High volume" route = shipments at/above this quantile of route volumes.
HIGH_VOLUME_QUANTILE = 0.75
# If more than this share of shipments is flagged extreme, exclude-by-default is disabled
# (it would empty the dashboard) and a critical warning is shown instead.
EXTREME_SHARE_CRITICAL = 0.50
# Efficiency-score category cut-offs (0-100 scale)
SCORE_EFFICIENT = 66.0
SCORE_POOR = 33.0

ANALYSIS_BASIS_RECORDED = "Recorded lead time (Ship Date - Order Date)"
ANALYSIS_BASIS_RELATIVE = "Relative lead time (assumption-based, see Data Quality)"
YEAR_CYCLE_DAYS = 365

# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
FACTORY_COLORS = {
    "Lot's O' Nuts": "#E07B39",
    "Wicked Choccy's": "#7B4B94",
    "Sugar Shack": "#2A9D8F",
    "Secret Factory": "#1F4E79",
    "The Other Factory": "#C0392B",
}
PERF_COLORS = {"Efficient": "#2E8B57", "Moderate": "#E9A820", "Inefficient": "#C0392B", "Not ranked": "#9AA5B1"}
SHIPMODE_COLORS = {
    "Standard Class": "#1F4E79", "Second Class": "#2A9D8F",
    "First Class": "#E9A820", "Same Day": "#C0392B",
}


# US state -> postal abbreviation (for the choropleth "heatmap"; Canada is shown on the bubble map only)
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# Ship-mode grouping for the "Standard vs Expedited" comparison in the project brief.
# ASSUMPTION: the data does not label modes as expedited; Standard Class = Standard, every faster-named mode = Expedited.
MODE_GROUPS = {"Standard Class": "Standard", "Second Class": "Expedited", "First Class": "Expedited", "Same Day": "Expedited"}
MODE_GROUP_COLORS = {"Standard": "#1F4E79", "Expedited": "#E07B39", "Other": "#9AA5B1"}
