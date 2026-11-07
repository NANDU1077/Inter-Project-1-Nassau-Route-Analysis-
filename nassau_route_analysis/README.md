# Factory-to-Customer Shipping Route Efficiency Analysis — Nassau Candy Distributor

An interactive Streamlit dashboard that measures factory → customer shipping performance, ranks routes,
maps bottlenecks, compares ship modes, and surfaces data-quality problems. No numbers are hard-coded:
every KPI, ranking and recommendation is computed from the CSV and responds to the sidebar filters.

## ⚠️ Most important finding about the dataset

**Every one of the 10,194 rows has a recorded lead time of 904–1,642 days (≈2.5–4.5 years), and ship dates run to 2030
(most are later than today).** These cannot be real transit times. What the pipeline shows:

* Lead times fall into three tight clusters about 365 days apart. After removing whole 365-day cycles, they vary over only ~11 days.
* This is consistent with Ship Dates carrying a constant offset plus whole-year shifts (a date export/processing issue). **This is an inference, not a proven cause.**
* Under the *recorded* values, ship modes are indistinguishable (Same Day is not faster than Standard Class). Under the *relative* transformation they separate in the expected order (Same Day < First < Second < Standard). That supports, but does not prove, the offset interpretation.

How the app handles this (nothing is silently hidden or "fixed"):

1. Rows are flagged (`Flag Extreme Lead Time` = lead > 60 days, `Flag Future Ship Date`) and kept. A critical banner is shown on every page.
2. The sidebar **Lead-time basis** lets you choose:
   * **Recorded** (default): exactly `Ship Date − Order Date`.
   * **Relative** (assumption-based, exploratory): `(lead time mod 365) − dataset minimum of that remainder`. Useful only for comparing routes with each other; it cannot recover true transit times (leap days add ±1 day noise).
3. "Include flagged extreme lead times" is on by default because turning it off removes essentially all data. Turn it off and the app shows a clear "no shipments" message rather than crashing.

**Confirm the Ship Date source with the data owner before using either basis for operational decisions.**

## Quick start

```bash
cd nassau_route_analysis
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Data location (first existing wins): sidebar *CSV path* / upload → env var `NASSAU_CSV_PATH` →
`data/Nassau_Candy_Distributor.csv` (bundled copy) → `data/Nassau Candy Distributor.csv` →
`C:\Users\nandu\OneDrive\Desktop\Intern1\Nassau Candy Distributor.csv`.
To point at your own copy: `set NASSAU_CSV_PATH=C:\path\to\file.csv` (Windows) before running, or paste the path in the sidebar.

Run tests: `python -m pytest -q` (24 tests: unit logic, real-data checks, and headless runs of every dashboard page).

## Project structure

```
nassau_route_analysis/
├── app.py                # Streamlit UI: sidebar, six pages, banners, recommendations
├── data_processing.py    # load, parse dates, validate, flag, engineer features, filters, shipments
├── analysis.py           # KPIs, route tables, efficiency score, rankings, recommendations
├── visualizations.py     # Plotly charts + interactive factory→state map
├── config.py             # paths, factory coordinates, product→factory map, thresholds, colours
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── data/Nassau_Candy_Distributor.csv
└── tests/ (test_pipeline.py, test_app_smoke.py)
```

## Units of analysis (what each metric is based on)

| Term | Definition |
|---|---|
| **Row / order line** | One CSV row. Used for product & division tables. |
| **Order** | Unique `Order ID`. |
| **Shipment** | Unique (`Order ID`, `Factory`). An order with products from two factories = two shipments. |

Lead-time statistics, delay frequency, route / state / region / factory / ship-mode tables use **shipments**
(so multi-line orders are not over-weighted). Sales, cost and gross profit are summed from order lines.

## Cleaning decisions

| Issue | Decision |
|---|---|
| Date format | Candidates (`%d-%m-%Y`, `%m-%d-%Y`, …) are tried; the one parsing the most values wins (ties → day-first). Detected: `%d-%m-%Y`. |
| Unparseable date | Row **excluded**, counted, reason recorded in `Exclusion Reason`. |
| Ship Date < Order Date | Row **excluded** and reported. |
| Lead time > 60 days (`MAX_PLAUSIBLE_LEAD_DAYS`) | **Flagged, kept**; toggle in sidebar; warning banner. |
| Ship Date after today | **Flagged, kept.** |
| Product not in factory map | Row **excluded** (no route possible) and listed on Data Quality page. Names are normalised so `Wonka Bar -Scrumdiddlyumptious` matches. |
| Missing geography / ship mode / division | Filled with `Unknown`, counted. (None in this file.) |
| Rows identical except Row ID (313) | **Flagged, kept**; optional sidebar exclusion (could be genuine repeat lines). |
| Shared Order IDs (multi-line orders) | Informational; handled by the shipment definition. |
| Canada (200 rows) | Kept; plotted at approximate province centroids. |
| Text | Whitespace trimmed; geography title-cased. |

## Feature engineering

Lead time, Factory + coordinates, Customer State, Customer Region (`Region` column), Route (`Factory → State`),
Factory-Region route, Delay indicator (`lead time > sidebar threshold`; default threshold = 75th percentile, so set your own SLA),
route lead-time standard deviation, shipment volume, average sales/cost/gross profit per shipment.

**Route efficiency score** = `100 × (slowest avg − route avg) / (slowest avg − fastest avg)`, min-max over routes with
≥ 5 shipments (`MIN_SHIPMENTS_FOR_RANKING`). 100 = fastest ranked route, 0 = slowest. Smaller routes are listed but
not scored, to avoid ranking on a handful of shipments. Scores are relative to the current filter selection.
Categories: ≥ 66 Efficient, 33–66 Moderate, < 33 Inefficient (`config.py`).

**High-volume poor performers**: ranked routes with volume ≥ the 75th percentile that also have above-median average lead time.

## Dashboard pages

* **A. Executive Overview** – KPIs, best/worst routes, trend, auto-generated recommendations, product & division tables.
* **B. Route Efficiency** – leaderboard (state or region level), top/bottom 10, volume-vs-lead-time scatter, high-volume/poor routes.
* **C. Geographic Map** – US state heatmap (choropleth) plus a bubble map with factories (diamonds), top routes as lines and Canadian provinces; colour = lead time / score / delay rate; bottleneck tables.
* **D. Ship Mode Comparison** – average/median lead time, delay frequency, volume, sales, gross profit, descriptive trade-off, and a Standard vs. Expedited grouping (assumption: First, Second Class and Same Day = Expedited).
* **E. Route Drill-Down** – factory/state/region/product/ship-mode selectors, route metrics, state-level insights, shipment timeline, order-line detail.
* **F. Data Quality** – missing values, invalid dates, unmapped products, suspicious lead times, duplicates, removal/flag summary, plausibility check.

## Limitations

* The data has **no shipping cost, carrier, distance or delivery-date field**. `Cost` and `Gross Profit` describe products, not freight, so the cost-time view is descriptive only; no shipping costs are invented.
* Recommendations describe observed patterns. They do not claim causes of delay.
* Map positions use approximate state/province centroids, not customer addresses.
* Routes are Factory → State; a state served by one factory has a single route.

## Notes on the technical brief
* Dataset claims checked: factory coordinates and all 15 product→factory mappings match the brief; `Gross Profit = Sales − Cost` holds on every row.
* The brief lists *Fizzy Lifting Drinks* under division "Other"; the CSV has "Sugar". The CSV value is used (it does not affect factory mapping).
* The brief describes US customers; the CSV also has 200 Canadian rows, kept and shown on the bubble map.
* The brief says "remove invalid or negative lead times": negative/unparseable ones are excluded; implausibly *large* ones are flagged and kept (not removed), because here the issue affects every row.
* Not part of the code: the research paper and executive summary listed under the brief's deliverables.
