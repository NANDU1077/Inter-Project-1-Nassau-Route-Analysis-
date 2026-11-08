"""Machine-learning layer: predict shipment lead time / delay and segment routes.

Design choices (all documented in the README and paper):
  * Target uses the RELATIVE lead time (the recorded one is dominated by a date offset and is not learnable).
  * Validation = GroupKFold by Customer ID, so a customer never appears in both train and test folds.
    A time-based split is deliberately NOT used: the relative lead time contains a calendar artifact
    that differs by order year, which would confound a time split.
  * Every model is compared with a one-line baseline (mean lead time per ship mode) and with route-only models
    (no ship mode) to isolate how much route information helps.
  * No Ship Date / recorded lead time is used as a feature (that would leak the target).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg

CAT_ALL = ["Factory", "Customer State", "Customer Region", "Ship Mode", "Division"]
NUM_ALL = ["Order Month Num", "Order Weekday", "Units", "Log Sales"]
ROUTE_ONLY = ["Factory", "Customer State", "Customer Region"]
BASELINE_REG = "Ship-mode mean (baseline)"
BASELINE_CLF = "Ship-mode rate (baseline)"
SEED = 0


def _fast() -> bool:
    return os.environ.get("NASSAU_ML_FAST", "") == "1"


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare_ml_table(ship: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Shipment-level modelling table. ``ship`` must come from data_processing.build_shipments
    using the RELATIVE lead-time basis."""
    d = ship.dropna(subset=["Lead Time"]).copy()
    d["Order Month Num"] = d["Order Date"].dt.month
    d["Order Weekday"] = d["Order Date"].dt.dayofweek
    d["Log Sales"] = np.log1p(d["Sales"].clip(lower=0))
    d["Delayed"] = (d["Lead Time"] > threshold).astype(int)
    return d.reset_index(drop=True)


def _pre(cols):
    cat = [c for c in cols if c in CAT_ALL]
    num = [c for c in cols if c in NUM_ALL]
    parts = []
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if num:
        parts.append(("num", StandardScaler(), num))
    return ColumnTransformer(parts)


def _trees():
    return 30 if _fast() else 100


def regression_models() -> dict:
    return {
        "Global mean": (DummyRegressor(strategy="mean"), ["Ship Mode"]),
        BASELINE_REG: (LinearRegression(), ["Ship Mode"]),  # exactly the per-mode mean
        "Ridge, route-only (no ship mode)": (Ridge(alpha=1.0), ROUTE_ONLY),
        "Ridge, all features": (Ridge(alpha=1.0), CAT_ALL + NUM_ALL),
        "Random forest, all features": (RandomForestRegressor(n_estimators=_trees(), min_samples_leaf=5, random_state=SEED, n_jobs=-1), CAT_ALL + NUM_ALL),
        "Gradient boosting, all features": (HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=SEED), CAT_ALL + NUM_ALL),
    }


def classification_models() -> dict:
    return {
        "Prior (always base rate)": (DummyClassifier(strategy="prior"), ["Ship Mode"]),
        BASELINE_CLF: (LogisticRegression(max_iter=2000), ["Ship Mode"]),
        "Logistic, route-only (no ship mode)": (LogisticRegression(max_iter=2000), ROUTE_ONLY),
        "Logistic, all features": (LogisticRegression(max_iter=2000), CAT_ALL + NUM_ALL),
        "Random forest, all features": (RandomForestClassifier(n_estimators=_trees(), min_samples_leaf=5, random_state=SEED, n_jobs=-1), CAT_ALL + NUM_ALL),
        "Gradient boosting, all features": (HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=SEED), CAT_ALL + NUM_ALL),
    }


def _pipe(est, cols):
    return Pipeline([("pre", _pre(cols)), ("m", est)])


def _folds(d: pd.DataFrame, n_splits=5):
    return list(GroupKFold(n_splits=n_splits).split(d, groups=d["Customer ID"]))


def _boot_ci(diffs: np.ndarray, n=1000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- #
# Cross-validated comparison
# --------------------------------------------------------------------------- #
def cross_validate_regression(d: pd.DataFrame, n_splits=5) -> dict:
    y = d["Lead Time"].to_numpy()
    folds = _folds(d, n_splits)
    rows, oof_all = [], {}
    for name, (est, cols) in regression_models().items():
        oof = np.zeros(len(d)); fold_mae = []
        for tr, te in folds:
            p = _pipe(est, cols).fit(d.iloc[tr][cols], y[tr])
            oof[te] = p.predict(d.iloc[te][cols]); fold_mae.append(mean_absolute_error(y[te], oof[te]))
        oof_all[name] = oof
        rows.append({"Model": name, "MAE": mean_absolute_error(y, oof), "RMSE": mean_squared_error(y, oof) ** 0.5,
                     "R²": r2_score(y, oof), "MAE fold std": float(np.std(fold_mae))})
    t = pd.DataFrame(rows)
    base = np.abs(y - oof_all[BASELINE_REG])
    lo, hi, dm = [], [], []
    for name in t["Model"]:
        diff = np.abs(y - oof_all[name]) - base            # >0 means worse than baseline
        a, b = _boot_ci(diff); lo.append(a); hi.append(b); dm.append(float(diff.mean()))
    t["MAE vs baseline"], t["95% CI low"], t["95% CI high"] = dm, lo, hi
    t["Beats baseline?"] = np.where(t["95% CI high"] < 0, "Yes", np.where(t["95% CI low"] > 0, "No (worse)", "No (no sig. difference)"))
    t.loc[t["Model"] == BASELINE_REG, "Beats baseline?"] = "(baseline)"
    return {"table": t, "oof": oof_all}


def cross_validate_classification(d: pd.DataFrame, n_splits=5) -> dict:
    y = d["Delayed"].to_numpy()
    folds = _folds(d, n_splits)
    rows, oof_all = [], {}
    for name, (est, cols) in classification_models().items():
        oof = np.zeros(len(d)); fold_auc = []
        for tr, te in folds:
            p = _pipe(est, cols).fit(d.iloc[tr][cols], y[tr])
            oof[te] = p.predict_proba(d.iloc[te][cols])[:, 1]
            fold_auc.append(roc_auc_score(y[te], oof[te]) if len(set(y[te])) > 1 else np.nan)
        oof_all[name] = oof
        rows.append({"Model": name, "ROC AUC": roc_auc_score(y, oof), "Avg precision": average_precision_score(y, oof),
                     "Brier score": brier_score_loss(y, oof), "AUC fold std": float(np.nanstd(fold_auc))})
    t = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED); n = len(y); B = 300 if not _fast() else 100
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    base_auc = np.array([roc_auc_score(y[i], oof_all[BASELINE_CLF][i]) for i in idxs])
    lo, hi = [], []
    for name in t["Model"]:
        aucs = np.array([roc_auc_score(y[i], oof_all[name][i]) for i in idxs]) - base_auc
        lo.append(float(np.percentile(aucs, 2.5))); hi.append(float(np.percentile(aucs, 97.5)))
    t["AUC vs baseline"] = t["ROC AUC"] - t.loc[t["Model"] == BASELINE_CLF, "ROC AUC"].iloc[0]
    t["95% CI low"], t["95% CI high"] = lo, hi
    t["Beats baseline?"] = np.where(t["95% CI low"] > 0, "Yes", np.where(t["95% CI high"] < 0, "No (worse)", "No (no sig. difference)"))
    t.loc[t["Model"] == BASELINE_CLF, "Beats baseline?"] = "(baseline)"
    return {"table": t, "oof": oof_all}


# --------------------------------------------------------------------------- #
# Feature importance (grouped hold-out, permutation of ORIGINAL columns)
# --------------------------------------------------------------------------- #
def feature_importance(d: pd.DataFrame, task="regression", n_repeats=10) -> pd.DataFrame:
    cols = CAT_ALL + NUM_ALL
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(d, groups=d["Customer ID"]))
    if task == "regression":
        est, y, scoring = RandomForestRegressor(n_estimators=_trees(), min_samples_leaf=5, random_state=SEED, n_jobs=-1), d["Lead Time"].to_numpy(), "neg_mean_absolute_error"
    else:
        est, y, scoring = RandomForestClassifier(n_estimators=_trees(), min_samples_leaf=5, random_state=SEED, n_jobs=-1), d["Delayed"].to_numpy(), "roc_auc"
    p = _pipe(est, cols).fit(d.iloc[tr][cols], y[tr])
    pi = permutation_importance(p, d.iloc[te][cols], y[te], n_repeats=n_repeats, random_state=SEED, scoring=scoring, n_jobs=1)
    out = pd.DataFrame({"Feature": cols, "Importance": pi.importances_mean, "Std": pi.importances_std})
    out["Metric change if shuffled"] = "MAE increase (relative days)" if task == "regression" else "AUC decrease"
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# What-if predictor
# --------------------------------------------------------------------------- #
def fit_predictor(d: pd.DataFrame, reg_name: str, clf_name: str, reg_oof=None) -> dict:
    rm, cm = regression_models()[reg_name], classification_models()[clf_name]
    reg = _pipe(*rm[:1], rm[1]).fit(d[rm[1]], d["Lead Time"])
    clf = _pipe(*cm[:1], cm[1]).fit(d[cm[1]], d["Delayed"])
    resid = (d["Lead Time"].to_numpy() - reg_oof) if reg_oof is not None else (d["Lead Time"].to_numpy() - reg.predict(d[rm[1]]))
    return {"reg": reg, "clf": clf, "reg_cols": rm[1], "clf_cols": cm[1], "q10": float(np.quantile(resid, .10)), "q90": float(np.quantile(resid, .90)),
            "state_region": d.groupby("Customer State")["Customer Region"].agg(lambda s: s.mode().iat[0]).to_dict()}


def predict_what_if(pred: dict, factory, state, ship_mode, division, month, weekday, units, sales) -> dict:
    row = pd.DataFrame([{"Factory": factory, "Customer State": state, "Customer Region": pred["state_region"].get(state, "Unknown"),
                         "Ship Mode": ship_mode, "Division": division, "Order Month Num": month, "Order Weekday": weekday,
                         "Units": units, "Log Sales": float(np.log1p(max(sales, 0)))}])
    lead = float(pred["reg"].predict(row[pred["reg_cols"]])[0])
    p = float(pred["clf"].predict_proba(row[pred["clf_cols"]])[0, 1])
    return {"lead": lead, "low": lead + pred["q10"], "high": lead + pred["q90"], "p_delay": p}


# --------------------------------------------------------------------------- #
# Unsupervised: route segmentation
# --------------------------------------------------------------------------- #
def cluster_routes(d: pd.DataFrame, k_range=range(2, 7)) -> dict:
    """K-means on ranked routes using mode-adjusted behaviour (so ship-mode mix does not drive clusters)."""
    d = d.copy()
    d["adj"] = d["Lead Time"] - d.groupby("Ship Mode")["Lead Time"].transform("mean")
    g = d.groupby(["Factory", "Customer State"]).agg(Shipments=("Order ID", "size"), **{"Adj mean lead": ("adj", "mean"),
        "Adj lead SD": ("adj", "std"), "Avg sales / shipment": ("Sales", "mean")}).reset_index()
    g = g[g["Shipments"] >= cfg.MIN_SHIPMENTS_FOR_RANKING].dropna().reset_index(drop=True)
    if len(g) < 8:
        return {"table": g.assign(Cluster=0), "silhouette": pd.DataFrame(), "k": 1, "profile": pd.DataFrame()}
    g["Route"] = g["Factory"] + " → " + g["Customer State"]
    X = pd.DataFrame({"Adj mean lead": g["Adj mean lead"], "Adj lead SD": g["Adj lead SD"],
                      "Log shipments": np.log(g["Shipments"]), "Log avg sales": np.log1p(g["Avg sales / shipment"])})
    Xs = StandardScaler().fit_transform(X)
    sil = []
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xs); sil.append({"k": k, "Silhouette": silhouette_score(Xs, lab)})
    sil = pd.DataFrame(sil); best_k = int(sil.loc[sil["Silhouette"].idxmax(), "k"])
    g["Cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(Xs)
    prof = g.groupby("Cluster").agg(Routes=("Route", "size"), Shipments=("Shipments", "sum"), **{"Adj mean lead": ("Adj mean lead", "mean"),
        "Adj lead SD": ("Adj lead SD", "mean"), "Avg sales / shipment": ("Avg sales / shipment", "mean")}).reset_index()
    return {"table": g, "silhouette": sil, "k": best_k, "profile": prof}


# --------------------------------------------------------------------------- #
# Convenience: run everything once
# --------------------------------------------------------------------------- #
def run_all(ship: pd.DataFrame, threshold: float) -> dict:
    d = prepare_ml_table(ship, threshold)
    reg, clf = cross_validate_regression(d), cross_validate_classification(d)
    best_reg = reg["table"].sort_values("MAE").iloc[0]["Model"]
    return {"data": d, "reg": reg, "clf": clf, "imp_reg": feature_importance(d, "regression"), "imp_clf": feature_importance(d, "classification"),
            "clusters": cluster_routes(d), "best_reg": best_reg}
