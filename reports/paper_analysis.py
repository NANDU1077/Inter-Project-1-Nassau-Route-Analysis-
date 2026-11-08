"""Reproduces every statistic and figure in the research paper from the source CSV.

Usage (from the project root):   python reports/paper_analysis.py [path/to/file.csv]
Outputs: reports/output/results.json, ml_results.json and fig1..fig9.png
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as an, config as cfg, data_processing as dp, ml_models as ml  # noqa: E402

CSV = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "Nassau_Candy_Distributor.csv")
OUT = ROOT / "reports" / "output"; OUT.mkdir(parents=True, exist_ok=True)
REFERENCE_DATE = "2026-11-07"   # date used in the paper for "future ship date" checks
THRESHOLD = 5                    # illustrative delay threshold (relative days)
FC = cfg.FACTORY_COLORS
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

full, rep = dp.clean_data(dp.read_csv_safely(CSV), reference_date=REFERENCE_DATE)
rows = dp.apply_basis(dp.analysis_rows(full), cfg.ANALYSIS_BASIS_RELATIVE)
ship = dp.build_shipments(rows, THRESHOLD)
rec_ship = dp.build_shipments(dp.apply_basis(dp.analysis_rows(full), cfg.ANALYSIS_BASIS_RECORDED), 1638)
R = {}

# ---- overview / data quality ----
R["rows"], R["shipments"], R["customers"] = len(full), len(ship), int(ship["Customer ID"].nunique())
R["future_rows"], R["dups"], R["raw_stats"] = rep.future_ship_rows, rep.duplicate_rows, rep.raw_lead_stats
R["residual_range"] = [rep.year_cycle_min_residual, rep.year_cycle_min_residual + rep.year_cycle_span]
R["sales"], R["gp"] = float(ship.Sales.sum()), float(ship["Gross Profit"].sum())
R["rel"] = dict(mean=ship["Lead Time"].mean(), median=ship["Lead Time"].median(), sd=ship["Lead Time"].std(), delay=ship["Delayed"].mean())
R["share_standard"] = float((ship["Mode Group"] == "Standard").mean())

# ---- group tables ----
R["mode_relative"] = an.shipmode_tradeoff(ship).round(3).to_dict("records")
R["mode_recorded"] = an.group_performance(rec_ship, "Ship Mode")[["Ship Mode", "Avg Lead Time"]].round(1).to_dict("records")
R["modegroup"] = an.group_performance(ship, "Mode Group").round(3).to_dict("records")
R["factory"] = an.group_performance(ship, "Factory").round(3).to_dict("records")
R["region"] = an.group_performance(ship, "Customer Region").round(3).to_dict("records")
rt = an.route_table(ship, "state"); top, bot = an.top_bottom_routes(rt, 10); hv = an.high_volume_poor_routes(rt)
R["n_routes"], R["n_ranked"], R["hv_n"] = len(rt), int(rt.Eligible.sum()), len(hv)
R["top10"], R["bot10"], R["hv"] = top.round(3).to_dict("records"), bot.round(3).to_dict("records"), hv.round(3).to_dict("records")

# ---- statistical checks ----
ship["Mode-adj"] = ship["Lead Time"] - ship.groupby("Ship Mode")["Lead Time"].transform("mean")
for name, col in [("factory", "Factory"), ("region", "Customer Region"), ("mode", "Ship Mode")]:
    for key, v in [("", "Lead Time"), ("_adj", "Mode-adj")]:
        H, p = stats.kruskal(*[g[v].values for _, g in ship.groupby(col)])
        R[f"kw_{name}{key}"] = dict(H=float(H), p=float(p))
ss_b = sum(len(g) * (g["Lead Time"].mean() - ship["Lead Time"].mean()) ** 2 for _, g in ship.groupby("Ship Mode"))
R["eta_mode"] = float(ss_b / ((ship["Lead Time"] - ship["Lead Time"].mean()) ** 2).sum())
ra = ship.groupby(["Factory", "Customer State"]).agg(n=("Order ID", "size"), adj=("Mode-adj", "mean"), raw=("Lead Time", "mean")).reset_index()
ra = ra[ra.n >= cfg.MIN_SHIPMENTS_FOR_RANKING]; ra["Route"] = ra.Factory + " → " + ra["Customer State"]
R["rank_corr_raw_vs_adj"] = float(stats.spearmanr(ra.raw, ra.adj)[0])
R["bottom10_overlap_adj"] = len(set(bot.Route) & set(ra.sort_values("adj", ascending=False).head(10).Route))
R["adj_slow"] = ra.sort_values("adj", ascending=False).head(5)[["Route", "n", "raw", "adj"]].round(2).to_dict("records")
R["adj_fast"] = ra.sort_values("adj").head(5)[["Route", "n", "raw", "adj"]].round(2).to_dict("records")
rng = np.random.default_rng(0)
sub = ship.merge(ra[["Factory", "Customer State"]], on=["Factory", "Customer State"]); sims = []
for _ in range(200):
    s = sub.copy(); s["perm"] = s.groupby("Ship Mode")["Lead Time"].transform(lambda x: rng.permutation(x.values))
    s["pa"] = s["perm"] - s.groupby("Ship Mode")["perm"].transform("mean")
    g = s.groupby(["Factory", "Customer State"])["pa"].agg(["mean", "size"]); sims.append(g[g["size"] >= cfg.MIN_SHIPMENTS_FOR_RANKING]["mean"].std())
R["perm_obs_sd"], R["perm_null_mean"], R["perm_null_p95"] = float(ra.adj.std()), float(np.mean(sims)), float(np.percentile(sims, 95))
R["perm_n_ge_observed"] = int((np.array(sims) >= ra.adj.std()).sum())
json.dump(R, open(OUT / "results.json", "w"), indent=1, default=str)

# ---- figures ----
order = ["Same Day", "First Class", "Second Class", "Standard Class"]
fig, ax = plt.subplots(figsize=(6.5, 2.8)); ax.hist(full["Raw Lead Time"], bins=np.arange(880, 1670, 10), color="#1F4E79")
ax.axvline(60, color="#C0392B", ls="--"); ax.set_xlim(0, 1700); ax.set_xlabel("Recorded lead time, days"); ax.set_ylabel("Order lines")
fig.tight_layout(); fig.savefig(OUT / "fig1.png", dpi=200); plt.close()
rec = rec_ship.groupby("Ship Mode")["Lead Time"].mean().reindex(order); rel = ship.groupby("Ship Mode")["Lead Time"].mean().reindex(order)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8)); ax[0].bar(order, rec.values, color="#9AA5B1"); ax[0].set_ylim(0, 1600); ax[0].set_title("Recorded basis (days)")
ax[1].bar(order, rel.values, color=[cfg.SHIPMODE_COLORS[o] for o in order]); ax[1].set_ylim(0, 6.2); ax[1].set_title("Relative basis")
for a in ax: a.tick_params(axis="x", rotation=25)
fig.tight_layout(); fig.savefig(OUT / "fig2.png", dpi=200); plt.close()
fp = an.group_performance(ship, "Factory").set_index("Factory").sort_values("Avg Lead Time"); sh = ship.assign(s=ship["Mode Group"] == "Standard").groupby("Factory")["s"].mean()
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9)); ax[0].barh(fp.index, fp["Avg Lead Time"], color=[FC[f] for f in fp.index]); ax[0].set_xlim(3.5, 4.9)
ax[1].barh(fp.index, sh.reindex(fp.index) * 100, color=[FC[f] for f in fp.index]); ax[1].set_yticklabels([]); ax[1].set_xlim(40, 75)
fig.tight_layout(); fig.savefig(OUT / "fig3.png", dpi=200); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
for a, d in [(ax[0], top), (ax[1], bot)]:
    d = d.iloc[::-1]; a.barh([f"{r} (n={n})".replace(" → ", "→") for r, n in zip(d.Route, d.Shipments)], d["Avg Lead Time"], color=[FC[f] for f in d.Factory]); a.tick_params(axis="y", labelsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "fig4.png", dpi=200); plt.close()
el = rt[rt.Eligible]; fig, ax = plt.subplots(figsize=(6.5, 3.3))
for f, g in el.groupby("Factory"): ax.scatter(g.Shipments, g["Avg Lead Time"], s=18, color=FC[f], label=f)
ax.scatter(hv.Shipments, hv["Avg Lead Time"], s=90, facecolors="none", edgecolors="black"); ax.set_xscale("log"); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig5.png", dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for f, g in ra.groupby("Factory"): ax.scatter(g.raw, g.adj, s=14 + g.n / 6, color=FC[f], label=f)
ax.axhline(0, color="#9AA5B1", lw=.8); ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig6.png", dpi=200); plt.close()
print("wrote", OUT)


# =========================== machine learning + artifact robustness ===========================
import warnings; warnings.filterwarnings("ignore")
M = {}
# ---- leap-day artifact diagnostics and re-test of route-level variation ----
v = dp.analysis_rows(full)
lg = v.groupby("Leap Days Spanned")["Relative Lead Time"].agg(["mean", "size"])
M["leap"] = {"share_spanning": float((v["Leap Days Spanned"] > 0).mean()), "mean_by_leapdays": {int(k): float(x) for k, x in lg["mean"].items()},
             "gap": float(lg["mean"].iloc[-1] - lg["mean"].iloc[0])}
ship["crosses"] = (ship["Order Date"].map(lambda d: 0)).astype(int)  # placeholder, replaced below
ls = dp.leap_days_spanned(ship["Order Date"], ship["Ship Date"]); ship["crosses"] = (ls > 0).astype(int); ship["oy"] = ship["Order Date"].dt.year
def route_test(strata, label):
    ship["adj"] = ship["Lead Time"] - ship.groupby(strata)["Lead Time"].transform("mean")
    r_ = ship.groupby(["Factory", "Customer State"]).agg(n=("Order ID", "size"), adj=("adj", "mean")).reset_index(); r_ = r_[r_.n >= cfg.MIN_SHIPMENTS_FOR_RANKING]
    obs = r_.adj.std(); rg = np.random.default_rng(0); sb = ship.merge(r_[["Factory", "Customer State"]], on=["Factory", "Customer State"]); sm = []
    for _ in range(200):
        t_ = sb.copy(); t_["p"] = t_.groupby(strata)["Lead Time"].transform(lambda x: rg.permutation(x.values)); t_["pa"] = t_["p"] - t_.groupby(strata)["p"].transform("mean")
        g_ = t_.groupby(["Factory", "Customer State"])["pa"].agg(["mean", "size"]); sm.append(g_[g_["size"] >= cfg.MIN_SHIPMENTS_FOR_RANKING]["mean"].std())
    kf = stats.kruskal(*[g["adj"].values for _, g in ship.groupby("Factory")])[1]; kr = stats.kruskal(*[g["adj"].values for _, g in ship.groupby("Customer Region")])[1]
    return {"adjusted_for": label, "obs_sd": float(obs), "null_mean": float(np.mean(sm)), "null_p95": float(np.percentile(sm, 95)),
            "perms_ge_obs": int((np.array(sm) >= obs).sum()), "kw_factory_p": float(kf), "kw_region_p": float(kr)}
M["robust"] = [route_test(["Ship Mode"], "ship mode"), route_test(["Ship Mode", "crosses"], "ship mode x leap-day span"),
               route_test(["Ship Mode", "oy"], "ship mode x order year"), route_test(["Ship Mode", "oy", "crosses"], "ship mode x order year x leap-day span")]

# ---- machine learning ----
res = ml.run_all(ship, THRESHOLD)
M["ml"] = {"n": len(res["data"]), "delay_rate": float(res["data"]["Delayed"].mean()),
           "reg": res["reg"]["table"].round(4).to_dict("records"), "clf": res["clf"]["table"].round(4).to_dict("records"),
           "imp_reg": res["imp_reg"].round(4).to_dict("records"), "imp_clf": res["imp_clf"].round(4).to_dict("records"),
           "k": res["clusters"]["k"], "silhouette": res["clusters"]["silhouette"].round(3).to_dict("records"),
           "profile": res["clusters"]["profile"].round(3).to_dict("records")}
json.dump(M, open(OUT / "ml_results.json", "w"), indent=1, default=str)

rt_, ct_ = res["reg"]["table"].copy(), res["clf"]["table"].copy()
def short(n): return n.replace(", all features", " (all)").replace("Ship-mode mean (baseline)", "Ship-mode mean\n(baseline)").replace("Ship-mode rate (baseline)", "Ship-mode rate\n(baseline)").replace(" (no ship mode)", "\n(no ship mode)").replace("Prior (always base rate)", "Prior")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
for a, t, m, base, asc, ttl in [(ax[0], rt_, "MAE", ml.BASELINE_REG, False, "Lead-time error (MAE, lower is better)"), (ax[1], ct_, "ROC AUC", ml.BASELINE_CLF, True, "Delay classification (AUC, higher is better)")]:
    t = t.sort_values(m, ascending=asc); cols = ["#C0392B" if x == base else ("#9AA5B1" if "route-only" in x else "#1F4E79") for x in t["Model"]]
    a.barh([short(x) for x in t["Model"]], t[m], color=cols); a.axvline(t.loc[t["Model"] == base, m].iloc[0], color="#C0392B", ls="--", lw=.8); a.set_title(ttl, fontsize=8.5); a.tick_params(axis="y", labelsize=6.5)
    if m == "ROC AUC": a.set_xlim(0.4, 0.8)
fig.tight_layout(); fig.savefig(OUT / "fig7.png", dpi=200); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for a, imp, ttl in [(ax[0], res["imp_reg"], "Lead-time model: MAE increase"), (ax[1], res["imp_clf"], "Delay model: AUC decrease")]:
    i_ = imp.iloc[::-1]; a.barh(i_["Feature"], i_["Importance"], xerr=i_["Std"], color="#1F4E79"); a.axvline(0, color="#9AA5B1", lw=.8); a.set_title(ttl, fontsize=8.5); a.tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig8.png", dpi=200); plt.close()
cl = res["clusters"]["table"]; fig, ax = plt.subplots(figsize=(6.5, 3.2))
for c_, g in cl.groupby("Cluster"): ax.scatter(g["Adj mean lead"], g["Adj lead SD"], s=12 + g["Shipments"] / 8, label=f"Cluster {c_} (n={len(g)} routes)", alpha=.75)
ax.axvline(0, color="#9AA5B1", ls=":"); ax.set_xlabel("Mean lead vs. expectation for ship-mode mix (days)"); ax.set_ylabel("Lead-time spread within route (days)"); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig9.png", dpi=200); plt.close()
print("wrote ML outputs")
