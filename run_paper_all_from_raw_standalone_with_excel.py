#!/usr/bin/env python3
# -*- coding: utf-8 -*-F

from __future__ import annotations

import os
import math
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import networkx as nx
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parent
DESCRIPTIVE_INPUT = str(ROOT / "price_files" / "dataproduct_industry_analysis_list_format_anymous.xlsx")
MAIN_RAW_INPUT = str(ROOT / "export_neo4j" / "api_data_snapshot.csv")
OUT_DESC = str(ROOT / "paper_result_final/paper_results_descriptives")
OUT_MAIN = str(ROOT / "paper_result_final/paper_results_main_appendix")
OUT_FIG = str(ROOT / "paper_result_final/paper_results_figures")

COL_NAME = "name"
COL_PRICE = "price"
COL_SUPPLIER = "supplier"
TOPK_SUPPLIERS_FOR_BOXPLOT = 30
MIN_LISTINGS_PER_SUPPLIER = 5
HIST_BINS = 60

# ================= Configurations =================
SEED = 42
N_SPLITS = 5
GROUP_COL = "supplier"

VECTOR_SIZE, WALK_LENGTH, NUM_WALKS, WINDOW_SIZE, EPOCHS, MIN_COUNT = 32, 15, 20, 3, 50, 1
TEXT_DIM, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE = 64, 50000, 2, (1, 2)
K_NEIGHBORS, EVIDENCE_MAX, KNN_SIM_POW = 50, 5, 2.0
MIN_PRICE, Z_95, DEFAULT_SIGMA_OBS = 1e-8, 1.96, 0.45
RIDGE_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
RHO_CANDIDATES, DELTA_CANDIDATES = [1.05, 1.10, 1.20, 1.30, 1.50], [0.05, 0.10, 0.15, 0.20]
TUNE_MAX_POINTS = 250

# OUT_MAIN = os.path.join(ROOT, "paper_results_main_appendix")


# ================= Utility & Data =================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_png_pdf(fig: plt.Figure, stem: str, outdir: str) -> tuple[str, str]:
    ensure_dir(outdir)
    png = os.path.join(outdir, f"{stem}.png")
    pdf = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def read_descriptive_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    df.columns = [str(c).strip() for c in df.columns]
    for c in [COL_NAME, COL_PRICE, COL_SUPPLIER]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df[COL_PRICE] = pd.to_numeric(df[COL_PRICE], errors="coerce")
    df = df.dropna(subset=[COL_PRICE]).copy()
    df = df[df[COL_PRICE] > 0].copy()
    df[COL_SUPPLIER] = df[COL_SUPPLIER].astype(str).str.strip()
    df["log_price"] = np.log(df[COL_PRICE].astype(float))
    df["product_id"] = df[COL_SUPPLIER] + "||" + df[COL_NAME].astype(str)
    return df.reset_index(drop=True)


def descriptive_table_1(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"Statistic": "Number of API listings (posted quotes)", "Value": int(df.shape[0])},
        {"Statistic": "Number of suppliers", "Value": int(df[COL_SUPPLIER].nunique())},
    ])


def descriptive_table_2(df: pd.DataFrame) -> pd.DataFrame:
    def row(series: pd.Series, label: str) -> dict:
        s = series.dropna().astype(float)
        return {
            "Var.": label,
            "Obs.": int(s.count()),
            "Mean": float(s.mean()),
            "SD": float(s.std(ddof=1)) if s.count() > 1 else np.nan,
            "Min": float(s.min()),
            "P25": float(s.quantile(0.25)),
            "Median": float(s.median()),
            "P75": float(s.quantile(0.75)),
            "Max": float(s.max()),
        }
    return pd.DataFrame([row(df[COL_PRICE], "Quote"), row(df["log_price"], "ln(Quote)")])


def supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(COL_SUPPLIER)["log_price"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean_log": g.mean(),
        "median_log": g.median(),
        "std_log": g.std(ddof=1),
        "q25_log": g.quantile(0.25),
        "q75_log": g.quantile(0.75),
    }).reset_index()
    out["iqr_log"] = out["q75_log"] - out["q25_log"]
    return out.sort_values("median_log", ascending=True)


def descriptive_table_3(df: pd.DataFrame) -> pd.DataFrame:
    y = df["log_price"].astype(float).values
    y_mean = float(np.mean(y))
    g = df.groupby(COL_SUPPLIER)["log_price"]
    n_j = g.size().values.astype(float)
    y_j = g.mean().values.astype(float)
    ssb = float(np.sum(n_j * (y_j - y_mean) ** 2))
    sst = float(np.sum((y - y_mean) ** 2))
    try:
        import statsmodels.formula.api as smf
        d = df[[COL_SUPPLIER, "log_price"]].dropna().copy()
        m = smf.mixedlm("log_price ~ 1", d, groups=d[COL_SUPPLIER]).fit(reml=True)
        var_sup = float(m.cov_re.iloc[0, 0])
        var_res = float(m.scale)
        icc = var_sup / (var_sup + var_res) if (var_sup + var_res) > 0 else np.nan
    except Exception:
        var_sup, var_res, icc = np.nan, np.nan, np.nan
    return pd.DataFrame([
        {"Measure": "Supplier fixed-effect share, R² = SSB/SST", "Estimate": ssb / sst if sst > 0 else np.nan},
        {"Measure": "Between-supplier sum of squares, SSB", "Estimate": ssb},
        {"Measure": "Total sum of squares, SST", "Estimate": sst},
        {"Measure": "Intra-class correlation (random intercept), ICC", "Estimate": icc},
        {"Measure": "Between-supplier variance, σᵤ²", "Estimate": var_sup},
        {"Measure": "Residual (within-supplier) variance, σₑ²", "Estimate": var_res},
    ])


def apply_descriptive_style() -> None:
    sns.set_theme(style="ticks", font_scale=1.10)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def build_fig2_panel(df: pd.DataFrame, outdir: str) -> tuple[str, str]:
    # Panel (a) plots the raw posted-quote histogram truncated at the 99th percentile for display.
    # Panel (b) plots the natural-log histogram of the same prices on the full positive sample.
    apply_descriptive_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    p99 = df[COL_PRICE].quantile(0.99)
    raw_plot = df.loc[df[COL_PRICE] <= p99, COL_PRICE].dropna().values
    sns.histplot(raw_plot, bins=HIST_BINS, color="#7f8c8d", edgecolor="white", alpha=0.85, kde=False, ax=axes[0])
    axes[0].set_xlabel(f"Raw price (CNY, truncated at 99th percentile: {p99:.2f})", fontweight="bold")
    axes[0].set_ylabel("Frequency", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(ax=axes[0])
    axes[0].text(0.01, 0.99, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    sns.histplot(df["log_price"].dropna().values, bins=HIST_BINS, color="#2c3e50", edgecolor="white", alpha=0.85, kde=True, line_kws={"linewidth": 1.8}, ax=axes[1])
    axes[1].set_xlabel("ln(price in CNY)", fontweight="bold")
    axes[1].set_ylabel("Frequency", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(ax=axes[1])
    axes[1].text(0.01, 0.99, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontweight="bold")

    fig.tight_layout(w_pad=0.8)
    return save_png_pdf(fig, "Fig2_Distribution_of_Posted_Quotes", outdir)


def build_fig3_panel(df: pd.DataFrame, outdir: str) -> tuple[str, str]:
    # Panel (a) plots supplier-level boxplots for top suppliers ordered by the median log price.
    # Panel (b) plots the within-supplier deviation histogram after subtracting each supplier median.
    apply_descriptive_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    ss = supplier_summary(df)
    ss = ss[ss["n"] >= MIN_LISTINGS_PER_SUPPLIER].copy()
    ss_top = ss.sort_values("n", ascending=False).head(TOPK_SUPPLIERS_FOR_BOXPLOT).sort_values("median_log", ascending=True)
    suppliers = ss_top[COL_SUPPLIER].tolist()
    data = [df.loc[df[COL_SUPPLIER] == s, "log_price"].values for s in suppliers]
    axes[0].boxplot(
        data, showfliers=False, patch_artist=True, widths=0.6,
        medianprops=dict(color="#c0392b", linewidth=1.8),
        boxprops=dict(facecolor="#ecf0f1", color="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1, linestyle="--"),
        capprops=dict(color="black", linewidth=1),
    )
    axes[0].set_xticks(range(1, len(suppliers) + 1))
    axes[0].set_xticklabels(suppliers, rotation=45, ha="right", fontsize=8.5)
    axes[0].set_ylabel("ln(price in CNY)", fontweight="bold")
    axes[0].set_xlabel("Anonymized supplier ID (top by count)", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(ax=axes[0], trim=True)
    axes[0].text(0.01, 0.99, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    tmp = df.join(df.groupby(COL_SUPPLIER)["log_price"].median().rename("supplier_median_log"), on=COL_SUPPLIER)
    tmp["within_log"] = tmp["log_price"] - tmp["supplier_median_log"]
    sns.histplot(tmp["within_log"].dropna().values, bins=HIST_BINS, color="#34495e", edgecolor="white", alpha=0.85, kde=True, line_kws={"linewidth": 1.8}, ax=axes[1])
    axes[1].axvline(0, color="#c0392b", linestyle=":", linewidth=1.4)
    axes[1].set_xlabel("Within-supplier deviation (ln(price) − supplier median)", fontweight="bold")
    axes[1].set_ylabel("Frequency", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(ax=axes[1])
    axes[1].text(0.01, 0.99, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontweight="bold")

    fig.tight_layout(w_pad=0.9)
    return save_png_pdf(fig, "Fig3_Descriptive_Evidence_of_Supplier_Anchoring", outdir)


def run_descriptive_block() -> pd.DataFrame:
    ensure_dir(OUT_DESC)
    df = read_descriptive_data(DESCRIPTIVE_INPUT)
    descriptive_table_1(df).to_csv(os.path.join(OUT_DESC, "Table_1_Sample_Composition.csv"), index=False, encoding="utf-8-sig")
    descriptive_table_2(df).to_csv(os.path.join(OUT_DESC, "Table_2_Distribution_of_Posted_Quotes.csv"), index=False, encoding="utf-8-sig")
    descriptive_table_3(df).to_csv(os.path.join(OUT_DESC, "Table_3_Supplier_Anchoring_Strength.csv"), index=False, encoding="utf-8-sig")
    supplier_summary(df).to_csv(os.path.join(OUT_DESC, "supplier_summary.csv"), index=False, encoding="utf-8-sig")
    build_fig2_panel(df, OUT_FIG)
    build_fig3_panel(df, OUT_FIG)
    return df


def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))


def safe_log_price(p):
    return math.log(max(MIN_PRICE, float(p)))


def l2norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def wape(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a - b)) / max(1e-9, np.sum(np.abs(a))))


def safe_list(x):
    s = str(x).strip() if pd.notna(x) else ""
    if not s or s.lower() == "nan":
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(t).strip() for t in v if str(t).strip()]
    except Exception:
        pass
    return [t.strip() for t in s.replace("，", ",").split(",") if t.strip()]


def fetch_from_csv(csv_path=MAIN_RAW_INPUT):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['src_list'] = df['src_list'].apply(safe_list)
    df['app_list'] = df['app_list'].apply(safe_list)
    df['supplier'] = df['supplier'].fillna("").astype(str)

    df = df[df['price'].apply(lambda x: pd.notna(x) and float(x) > 0)]
    df = df[df['supplier'] != ""]

    df["y"] = df["price"].apply(safe_log_price)
    df["text"] = (df["name"].fillna("").astype(str) + " " + df["desc"].fillna("").astype(str)).str.strip()
    return df.reset_index(drop=True)


# ================= Embeddings & Feature Engineering =================
def train_graph_embeddings(df_train):
    total_docs = len(df_train)
    ind_counts = {}
    for _, r in df_train.iterrows():
        for ind in sorted(list(set(r["src_list"] + r["app_list"]))):
            ind_counts[ind] = ind_counts.get(ind, 0) + 1

    ind_weights = {ind: math.log(total_docs / (cnt + 1)) + 1.0 for ind, cnt in ind_counts.items()}

    G = nx.Graph()
    for i, r in df_train.reset_index(drop=True).iterrows():
        prod = f"PROD_{i}"
        if r["supplier"]:
            G.add_edge(f"SUP_{r['supplier']}", prod, weight=3.0)
        for s in r["src_list"]:
            G.add_edge(f"SRC_{s}", prod, weight=ind_weights.get(s, 1.0))
        for a in r["app_list"]:
            G.add_edge(f"APP_{a}", prod, weight=ind_weights.get(a, 1.0))

    nodes = sorted(list(G.nodes()))
    walks = []
    for _ in range(NUM_WALKS):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < WALK_LENGTH:
                cur = walk[-1]
                nbrs = sorted(list(G.neighbors(cur)))
                if not nbrs:
                    break
                ws = [G[cur][nb]["weight"] for nb in nbrs]
                walk.append(random.choices(nbrs, weights=ws, k=1)[0])
            walks.append(walk)

    model = Word2Vec(sentences=walks, vector_size=VECTOR_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, sg=1, workers=1, seed=SEED)
    emb = {}
    for node in G.nodes():
        if node not in model.wv:
            continue
        if node.startswith("SUP_"):
            emb[("supplier", node[4:])] = model.wv[node]
        elif node.startswith("SRC_"):
            emb[("src_ind", node[4:])] = model.wv[node]
        elif node.startswith("APP_"):
            emb[("app_ind", node[4:])] = model.wv[node]
    return emb


def build_graph_feature(emb, supplier, src_list, app_list, normalize=False):
    def get_vec(k, name):
        return np.asarray(emb.get((k, str(name).strip()), np.zeros(VECTOR_SIZE, dtype=np.float32)), dtype=np.float32)

    def mean_vec(k, lst):
        vs = [get_vec(k, x) for x in lst if np.linalg.norm(get_vec(k, x)) > 1e-12]
        return np.mean(vs, axis=0) if vs else np.zeros(VECTOR_SIZE, dtype=np.float32)

    x = np.concatenate([get_vec("supplier", supplier), mean_vec("src_ind", src_list), mean_vec("app_ind", app_list)], axis=0).astype(np.float32)
    return l2norm(x) if normalize else x


def train_text_embedder(text_train):
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF, ngram_range=TFIDF_NGRAM_RANGE)
    X = vec.fit_transform(text_train)
    svd = TruncatedSVD(n_components=min(TEXT_DIM, max(2, X.shape[1] - 1)), random_state=SEED)
    svd.fit(X)

    def _transform(texts):
        Zt = svd.transform(vec.transform(texts))
        if Zt.shape[1] < TEXT_DIM:
            Zt = np.concatenate([Zt, np.zeros((Zt.shape[0], TEXT_DIM - Zt.shape[1]))], axis=1)
        return Zt.astype(np.float32)

    return _transform


# ================= Core Algorithms =================
def fit_ridge_prior_with_inner_cv(X, y):
    inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
    best_alpha, best_rmse = None, 1e18
    for a in RIDGE_ALPHAS:
        rmses = []
        for tr, va in inner.split(X):
            model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a, random_state=SEED))]).fit(X[tr], y[tr])
            rmses.append(np.sqrt(np.mean((y[va] - model.predict(X[va])) ** 2)))
        if np.mean(rmses) < best_rmse:
            best_rmse, best_alpha = np.mean(rmses), a
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha, random_state=SEED))]).fit(X, y)


def calibrate_sigma_obs(train_y, train_x_knn):
    n = len(train_x_knn)
    if n < 10:
        return DEFAULT_SIGMA_OBS
    nn = NearestNeighbors(n_neighbors=min(5, n), metric="cosine").fit(train_x_knn)
    dists, idxs = nn.kneighbors(train_x_knn, return_distance=True)
    diffs = [abs(train_y[i] - train_y[int(idxs[i][t])]) for i in range(n) for t in range(len(idxs[i])) if int(idxs[i][t]) != i]
    if len(diffs) < 10:
        return DEFAULT_SIGMA_OBS
    return clip(float(np.median(diffs)) / 0.6745, 0.10, 1.50)


def normal_normal_posterior(mu0, sigma0, ybar, m, sigma_obs):
    if m <= 0 or np.isnan(ybar):
        return float(mu0), float(sigma0)
    tau0, tau = 1.0 / (float(sigma0) ** 2), 1.0 / (float(sigma_obs) ** 2)
    tau_post = tau0 + m * tau
    return (tau0 * float(mu0) + (m * tau) * float(ybar)) / tau_post, math.sqrt(1.0 / tau_post)


def gap_trim_count(sims_sorted, rho, delta):
    m = min(EVIDENCE_MAX, len(sims_sorted))
    for k in range(1, m):
        if sims_sorted[k] <= 1e-12 or (sims_sorted[k - 1] / sims_sorted[k]) >= rho or (sims_sorted[k - 1] - sims_sorted[k]) >= delta:
            return k
    return m


def tune_rho_delta(mu0_train, sigma0, sigma_obs, y_train, Xtr_knn, nn_index):
    n = len(Xtr_knn)
    if n <= 5:
        return 1.20, 0.10
    rng = np.random.RandomState(SEED)
    tune_idx = rng.choice(np.arange(n), size=min(n, TUNE_MAX_POINTS), replace=False)
    dists, idxs = nn_index.kneighbors(Xtr_knn[tune_idx], return_distance=True)
    sims = 1.0 - dists

    best_rho, best_delta, best_rmse = 1.20, 0.10, 1e18
    for rho in RHO_CANDIDATES:
        for delta in DELTA_CANDIDATES:
            preds = []
            for row_i, i in enumerate(tune_idx):
                ns, ny = [], []
                for t in range(len(idxs[row_i])):
                    if int(idxs[row_i][t]) != i and sims[row_i][t] > 0:
                        ns.append(sims[row_i][t])
                        ny.append(y_train[int(idxs[row_i][t])])
                    if len(ns) >= EVIDENCE_MAX:
                        break
                if not ns:
                    preds.append(mu0_train[i])
                    continue
                order = np.argsort(-np.asarray(ns))
                ns = [ns[t] for t in order]
                ny = [ny[t] for t in order]
                m = gap_trim_count(ns, rho, delta)
                preds.append(normal_normal_posterior(mu0_train[i], sigma0, np.mean(ny[:m]), m, sigma_obs)[0])
            rmse = np.sqrt(np.mean((y_train[tune_idx] - preds) ** 2))
            if rmse < best_rmse:
                best_rmse, best_rho, best_delta = rmse, rho, delta
    return best_rho, best_delta


def lambda_from_precision(m, sigma0, sigma_obs):
    if m <= 0:
        return 0.0
    tau0, tau = 1.0 / (float(sigma0) ** 2), 1.0 / (float(sigma_obs) ** 2)
    return float((m * tau) / (tau0 + m * tau))


def pct_abs_err(y_true_price, y_pred_log):
    pred_price = np.maximum(MIN_PRICE, np.exp(np.asarray(y_pred_log, dtype=float)))
    y_true_price = np.asarray(y_true_price, dtype=float)
    return np.abs(pred_price - y_true_price) / np.maximum(MIN_PRICE, y_true_price)


def assign_lambda_bin(x):
    if pd.isna(x):
        return "NA"
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        if lo <= x < hi or (lab == labels[-1] and abs(x - 1.0) < 1e-12):
            return lab
    return labels[-1]


# ================= Unified Pipeline =================
def run_unified_cv(df, splitter, cv_name):
    print(f"\n--- Executing {cv_name} ---")
    results = []
    raw_preds = []
    diag_rows = []

    for fold, (tr_idx, te_idx) in enumerate(splitter, start=1):
        print(f"  Fold {fold}/{N_SPLITS}...")
        tr_df = df.iloc[tr_idx].copy().reset_index(drop=True)
        te_df = df.iloc[te_idx].copy().reset_index(drop=True)

        emb = train_graph_embeddings(tr_df)
        tf = train_text_embedder(tr_df["text"].tolist())

        X_tr = np.concatenate([
            np.stack([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False) for _, r in tr_df.iterrows()]),
            tf(tr_df["text"].tolist())
        ], axis=1)
        X_te = np.concatenate([
            np.stack([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False) for _, r in te_df.iterrows()]),
            tf(te_df["text"].tolist())
        ], axis=1)

        y_tr, y_te = tr_df["y"].values.astype(float), te_df["y"].values.astype(float)
        p_te = te_df["price"].values.astype(float)

        prior_model = fit_ridge_prior_with_inner_cv(X_tr, y_tr)
        mu0_tr, mu0_te = np.asarray(prior_model.predict(X_tr)), np.asarray(prior_model.predict(X_te))
        sigma0 = clip(float(np.sqrt(np.mean((y_tr - mu0_tr) ** 2))), 0.10, 1.50)

        X_tr_knn = np.vstack([
            l2norm(np.concatenate([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], True), l2norm(z)]))
            for (_, r), z in zip(tr_df.iterrows(), tf(tr_df["text"].tolist()))
        ]).astype(np.float32)
        X_te_knn = np.vstack([
            l2norm(np.concatenate([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], True), l2norm(z)]))
            for (_, r), z in zip(te_df.iterrows(), tf(te_df["text"].tolist()))
        ]).astype(np.float32)

        nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(tr_df)), metric="cosine").fit(X_tr_knn)
        dists_te, idxs_te = nn.kneighbors(X_te_knn, return_distance=True)
        sims_te = 1.0 - dists_te

        sigma_obs = calibrate_sigma_obs(y_tr, X_tr_knn)
        rho, delta = tune_rho_delta(mu0_tr, sigma0, sigma_obs, y_tr, X_tr_knn, nn)

        val_resids = np.abs(y_tr - mu0_tr)
        q95 = np.percentile(val_resids, 95)

        preds = {
            "Prior (Ridge)": {"pred": mu0_te.copy(), "low": mu0_te - Z_95 * sigma0, "high": mu0_te + Z_95 * sigma0},
            "KNN (Mean)": {"pred": [], "low": [], "high": []},
            "KNN (GapTrim)": {"pred": [], "low": [], "high": []},
            "Bayes (Mean)": {"pred": [], "low": [], "high": []},
            "Bayes (GapTrim)": {"pred": [], "low": [], "high": []},
            "Plus Bayes (Robust Cov)": {"pred": [], "low": [], "high": []},
            "Stacking (Conformal)": {"pred": [], "low": [], "high": []}
        }

        for i in range(len(te_df)):
            ns, ny = [], []
            for t in range(len(idxs_te[i])):
                if sims_te[i][t] > 0:
                    ns.append(float(sims_te[i][t]))
                    ny.append(float(y_tr[int(idxs_te[i][t])]))
                if len(ns) >= EVIDENCE_MAX:
                    break

            mean_n = len(ns)
            mean_ybar = np.mean(ny) if mean_n > 0 else np.nan
            mean_sim = np.mean(ns) if mean_n > 0 else np.nan
            lambda_mean = lambda_from_precision(mean_n, sigma0, sigma_obs)

            if mean_n > 0:
                order = np.argsort(-np.asarray(ns))
                ns = [ns[x] for x in order]
                ny = [ny[x] for x in order]

                ws_mean = np.asarray([max(1e-6, s) ** KNN_SIM_POW for s in ns])
                knn_m = float(np.sum(ws_mean * ny) / np.sum(ws_mean))
                preds["KNN (Mean)"]["pred"].append(knn_m)
                preds["KNN (Mean)"]["low"].append(knn_m)
                preds["KNN (Mean)"]["high"].append(knn_m)

                mu_bm, sig_bm = normal_normal_posterior(mu0_te[i], sigma0, np.mean(ny), len(ny), sigma_obs)
                preds["Bayes (Mean)"]["pred"].append(mu_bm)
                preds["Bayes (Mean)"]["low"].append(mu_bm - Z_95 * math.sqrt(sig_bm ** 2 + sigma_obs ** 2))
                preds["Bayes (Mean)"]["high"].append(mu_bm + Z_95 * math.sqrt(sig_bm ** 2 + sigma_obs ** 2))

                m_trim = gap_trim_count(ns, rho, delta)
                gap_ybar = np.mean(ny[:m_trim]) if m_trim > 0 else np.nan
                gap_sim = np.mean(ns[:m_trim]) if m_trim > 0 else np.nan
                lambda_gap = lambda_from_precision(m_trim, sigma0, sigma_obs)

                ws_gap = np.asarray([max(1e-6, s) ** KNN_SIM_POW for s in ns[:m_trim]])
                knn_g = float(np.sum(ws_gap * ny[:m_trim]) / np.sum(ws_gap))
                preds["KNN (GapTrim)"]["pred"].append(knn_g)
                preds["KNN (GapTrim)"]["low"].append(knn_g)
                preds["KNN (GapTrim)"]["high"].append(knn_g)

                mu_bg, sig_bg = normal_normal_posterior(mu0_te[i], sigma0, np.mean(ny[:m_trim]), m_trim, sigma_obs)
                preds["Bayes (GapTrim)"]["pred"].append(mu_bg)
                preds["Bayes (GapTrim)"]["low"].append(mu_bg - Z_95 * math.sqrt(sig_bg ** 2 + sigma_obs ** 2))
                preds["Bayes (GapTrim)"]["high"].append(mu_bg + Z_95 * math.sqrt(sig_bg ** 2 + sigma_obs ** 2))

                mu_plus, sig_plus = normal_normal_posterior(mu0_te[i], sigma0, np.mean(ny[:m_trim]), m_trim, sigma_obs * 1.5)
                preds["Plus Bayes (Robust Cov)"]["pred"].append(mu_plus)
                preds["Plus Bayes (Robust Cov)"]["low"].append(mu_plus - Z_95 * math.sqrt(sig_plus ** 2 + (sigma_obs * 1.5) ** 2))
                preds["Plus Bayes (Robust Cov)"]["high"].append(mu_plus + Z_95 * math.sqrt(sig_plus ** 2 + (sigma_obs * 1.5) ** 2))

                mu_stack = 0.5 * mu0_te[i] + 0.5 * knn_g
                preds["Stacking (Conformal)"]["pred"].append(mu_stack)
                preds["Stacking (Conformal)"]["low"].append(mu_stack - q95)
                preds["Stacking (Conformal)"]["high"].append(mu_stack + q95)
            else:
                m_trim = 0
                gap_ybar = np.nan
                gap_sim = np.nan
                lambda_gap = 0.0
                knn_m = mu0_te[i]
                knn_g = mu0_te[i]
                mu_bm, sig_bm = mu0_te[i], sigma0
                mu_bg, sig_bg = mu0_te[i], sigma0
                for k in preds.keys():
                    if k != "Prior (Ridge)":
                        preds[k]["pred"].append(mu0_te[i])
                        preds[k]["low"].append(mu0_te[i] - Z_95 * sigma0)
                        preds[k]["high"].append(mu0_te[i] + Z_95 * sigma0)

            # store diagnostics row
            diag_rows.append({
                "CV Protocol": cv_name,
                "Fold": fold,
                "supplier": te_df.loc[i, "supplier"],
                "y_true": float(y_te[i]),
                "price_true": float(p_te[i]),
                "mu0": float(mu0_te[i]),
                "sigma0": float(sigma0),
                "sigma_obs": float(sigma_obs),
                "rho": float(rho),
                "delta": float(delta),
                "q95": float(q95),
                "n_mean": int(mean_n),
                "n_gap": int(m_trim),
                "avg_sim_mean": float(mean_sim) if pd.notna(mean_sim) else np.nan,
                "avg_sim_gap": float(gap_sim) if pd.notna(gap_sim) else np.nan,
                "ybar_mean": float(mean_ybar) if pd.notna(mean_ybar) else np.nan,
                "ybar_gap": float(gap_ybar) if pd.notna(gap_ybar) else np.nan,
                "lambda_mean": float(lambda_mean),
                "lambda_gap": float(lambda_gap),
                "knn_mean_pred": float(preds["KNN (Mean)"]["pred"][-1]),
                "knn_gap_pred": float(preds["KNN (GapTrim)"]["pred"][-1]),
                "bayes_mean_pred": float(preds["Bayes (Mean)"]["pred"][-1]),
                "bayes_gap_pred": float(preds["Bayes (GapTrim)"]["pred"][-1]),
                "bayes_mean_low": float(preds["Bayes (Mean)"]["low"][-1]),
                "bayes_mean_high": float(preds["Bayes (Mean)"]["high"][-1]),
                "bayes_gap_low": float(preds["Bayes (GapTrim)"]["low"][-1]),
                "bayes_gap_high": float(preds["Bayes (GapTrim)"]["high"][-1]),
                "prior_local_gap_mean": abs(float(mean_ybar) - float(mu0_te[i])) if pd.notna(mean_ybar) else np.nan,
                "prior_local_gap_gap": abs(float(gap_ybar) - float(mu0_te[i])) if pd.notna(gap_ybar) else np.nan,
                "post_prior_dist_mean": abs(float(preds["Bayes (Mean)"]["pred"][-1]) - float(mu0_te[i])),
                "post_local_dist_mean": abs(float(preds["Bayes (Mean)"]["pred"][-1]) - float(mean_ybar)) if pd.notna(mean_ybar) else np.nan,
                "post_prior_dist_gap": abs(float(preds["Bayes (GapTrim)"]["pred"][-1]) - float(mu0_te[i])),
                "post_local_dist_gap": abs(float(preds["Bayes (GapTrim)"]["pred"][-1]) - float(gap_ybar)) if pd.notna(gap_ybar) else np.nan,
            })

        # Wide export for plotting and method metrics
        df_out = te_df.copy()
        df_out["CV Protocol"] = cv_name
        df_out["Fold"] = fold
        for method, values in preds.items():
            y_pred = np.array(values["pred"], dtype=float)
            low = np.array(values["low"], dtype=float)
            high = np.array(values["high"], dtype=float)
            df_out[method] = y_pred
            df_out[f"{method}_low"] = low
            df_out[f"{method}_high"] = high

            p_pred = np.maximum(MIN_PRICE, np.exp(y_pred))
            results.append({
                "CV Protocol": cv_name,
                "Method": method,
                "R2": r2_score(y_te, y_pred),
                "RMSE": np.sqrt(np.mean((y_te - y_pred) ** 2)),
                "WAPE": wape(p_te, p_pred),
                "Coverage": np.mean((y_te >= low) & (y_te <= high)) if "KNN" not in method else np.nan,
                "Width": np.mean(high - low) if "KNN" not in method else np.nan
            })
        raw_preds.append(df_out)

    return pd.DataFrame(results), pd.concat(raw_preds, ignore_index=True), pd.DataFrame(diag_rows)


# ================= Diagnostics Summaries =================
def build_tail_risk_table(detail_df):
    method_cols = [
        "Prior (Ridge)", "KNN (Mean)", "KNN (GapTrim)",
        "Bayes (Mean)", "Bayes (GapTrim)", "Plus Bayes (Robust Cov)", "Stacking (Conformal)"
    ]
    rows = []
    for cv_name in sorted(detail_df["CV Protocol"].unique()):
        dcv = detail_df[detail_df["CV Protocol"] == cv_name].copy()
        y_true = dcv["y"].values.astype(float)
        price_true = dcv["price"].values.astype(float)
        for method in method_cols:
            pred = dcv[method].values.astype(float)
            log_ae = np.abs(y_true - pred)
            ape = pct_abs_err(price_true, pred)
            rows.append({
                "CV Protocol": cv_name,
                "Method": method,
                "Median LogAE": np.median(log_ae),
                "P90 LogAE": np.percentile(log_ae, 90),
                "P95 LogAE": np.percentile(log_ae, 95),
                "P99 LogAE": np.percentile(log_ae, 99),
                "Max LogAE": np.max(log_ae),
                "Median APE": np.median(ape),
                "P90 APE": np.percentile(ape, 90),
                "P95 APE": np.percentile(ape, 95),
                "P99 APE": np.percentile(ape, 99),
                "Max APE": np.max(ape),
            })
    return pd.DataFrame(rows)


def build_shrinkage_diagnostics(diag_df):
    rows = []
    for cv_name in sorted(diag_df["CV Protocol"].unique()):
        dcv = diag_df[diag_df["CV Protocol"] == cv_name].copy()
        for variant in ["Mean", "Gap"]:
            lam = dcv[f"lambda_{variant.lower()}"] if variant == "Mean" else dcv["lambda_gap"]
            nret = dcv[f"n_{variant.lower()}"] if variant == "Mean" else dcv["n_gap"]
            gap = dcv[f"prior_local_gap_{variant.lower()}"] if variant == "Mean" else dcv["prior_local_gap_gap"]
            post_prior = dcv[f"post_prior_dist_{variant.lower()}"] if variant == "Mean" else dcv["post_prior_dist_gap"]
            post_local = dcv[f"post_local_dist_{variant.lower()}"] if variant == "Mean" else dcv["post_local_dist_gap"]
            low = dcv[f"bayes_{variant.lower()}_low"] if variant == "Mean" else dcv["bayes_gap_low"]
            high = dcv[f"bayes_{variant.lower()}_high"] if variant == "Mean" else dcv["bayes_gap_high"]
            pred = dcv[f"bayes_{variant.lower()}_pred"] if variant == "Mean" else dcv["bayes_gap_pred"]
            y_true = dcv["y_true"]

            rows.append({
                "CV Protocol": cv_name,
                "Variant": variant,
                "N": len(dcv),
                "Lambda Mean": np.nanmean(lam),
                "Lambda P25": np.nanpercentile(lam, 25),
                "Lambda Median": np.nanpercentile(lam, 50),
                "Lambda P75": np.nanpercentile(lam, 75),
                "Share Lambda < 0.5": np.mean(lam < 0.5),
                "Retained Neighbors Mean": np.nanmean(nret),
                "Prior-Local Gap Mean": np.nanmean(gap),
                "Prior-Local Gap P90": np.nanpercentile(gap.dropna(), 90) if gap.notna().any() else np.nan,
                "Posterior Closer to Prior Share": np.nanmean(post_prior < post_local),
                "PI95 Coverage": np.mean((y_true >= low) & (y_true <= high)),
                "Avg Width": np.mean(high - low),
                "RMSE": np.sqrt(np.mean((y_true - pred) ** 2)),
            })

            tmp = dcv.copy()
            tmp["lambda_bin"] = lam.apply(assign_lambda_bin)
            for lb, g in tmp.groupby("lambda_bin"):
                rows.append({
                    "CV Protocol": cv_name,
                    "Variant": f"{variant} | {lb}",
                    "N": len(g),
                    "Lambda Mean": np.nanmean(g[f"lambda_{variant.lower()}"] if variant == "Mean" else g["lambda_gap"]),
                    "Lambda P25": np.nan,
                    "Lambda Median": np.nan,
                    "Lambda P75": np.nan,
                    "Share Lambda < 0.5": np.mean((g[f"lambda_{variant.lower()}"] if variant == "Mean" else g["lambda_gap"]) < 0.5),
                    "Retained Neighbors Mean": np.nanmean(g[f"n_{variant.lower()}"] if variant == "Mean" else g["n_gap"]),
                    "Prior-Local Gap Mean": np.nanmean(g[f"prior_local_gap_{variant.lower()}"] if variant == "Mean" else g["prior_local_gap_gap"]),
                    "Prior-Local Gap P90": np.nan,
                    "Posterior Closer to Prior Share": np.nanmean((g[f"post_prior_dist_{variant.lower()}"] if variant == "Mean" else g["post_prior_dist_gap"]) < (g[f"post_local_dist_{variant.lower()}"] if variant == "Mean" else g["post_local_dist_gap"])),
                    "PI95 Coverage": np.mean((g["y_true"] >= (g[f"bayes_{variant.lower()}_low"] if variant == "Mean" else g["bayes_gap_low"])) & (g["y_true"] <= (g[f"bayes_{variant.lower()}_high"] if variant == "Mean" else g["bayes_gap_high"]))),
                    "Avg Width": np.mean((g[f"bayes_{variant.lower()}_high"] if variant == "Mean" else g["bayes_gap_high"]) - (g[f"bayes_{variant.lower()}_low"] if variant == "Mean" else g["bayes_gap_low"])),
                    "RMSE": np.sqrt(np.mean((g["y_true"] - (g[f"bayes_{variant.lower()}_pred"] if variant == "Mean" else g["bayes_gap_pred"])) ** 2)),
                })
    return pd.DataFrame(rows)


def build_high_disagreement_table(diag_df):
    """
    Refined Appendix B3: mechanism-focused rather than exhaustive.
    Keep only the columns that directly support the high-disagreement
    shrinkage argument: subset size, mean lambda, WAPE, and upper-tail errors.
    """
    rows = []
    for cv_name in ["KFold", "GroupKFold"]:
        dcv = diag_df[diag_df["CV Protocol"] == cv_name].copy()
        if len(dcv) == 0:
            continue
        for variant in ["Mean", "Gap"]:
            dis = dcv["prior_local_gap_mean"] if variant == "Mean" else dcv["prior_local_gap_gap"]
            if dis.notna().sum() < 20:
                continue
            q75 = np.nanpercentile(dis, 75)
            q90 = np.nanpercentile(dis, 90)
            subset_defs = [
                ("Top25% prior-local disagreement", dis >= q75),
                ("Top10% prior-local disagreement", dis >= q90),
            ]
            for subset_name, mask in subset_defs:
                g = dcv[mask].copy()
                if len(g) == 0:
                    continue
                local_pred = g["knn_mean_pred"] if variant == "Mean" else g["knn_gap_pred"]
                bayes_pred = g["bayes_mean_pred"] if variant == "Mean" else g["bayes_gap_pred"]
                y_true = g["y_true"]
                price_true = g["price_true"]
                local_logae = np.abs(y_true - local_pred)
                bayes_logae = np.abs(y_true - bayes_pred)
                local_ape = pct_abs_err(price_true, local_pred)
                bayes_ape = pct_abs_err(price_true, bayes_pred)

                rows.append({
                    "CV Protocol": cv_name,
                    "Variant": variant,
                    "Subset": subset_name,
                    "N": int(len(g)),
                    "Mean Lambda": float(np.mean(g["lambda_mean"] if variant == "Mean" else g["lambda_gap"])),
                    "Local WAPE": float(wape(price_true, np.maximum(MIN_PRICE, np.exp(local_pred)))),
                    "Bayes WAPE": float(wape(price_true, np.maximum(MIN_PRICE, np.exp(bayes_pred)))),
                    "Local P95 LogAE": float(np.percentile(local_logae, 95)),
                    "Bayes P95 LogAE": float(np.percentile(bayes_logae, 95)),
                    "Local P95 APE": float(np.percentile(local_ape, 95)),
                    "Bayes P95 APE": float(np.percentile(bayes_ape, 95)),
                })
    return pd.DataFrame(rows)


def build_decile_gain_table(diag_df):
    rows = []
    for cv in ["KFold", "GroupKFold"]:
        sdf = diag_df[diag_df["CV Protocol"] == cv].copy()
        gap_col = "prior_local_gap_gap"
        sdf = sdf.sort_values(gap_col).copy()
        sdf["Decile"] = pd.qcut(sdf[gap_col].rank(method="first"), 10, labels=False) + 1
        local_ape = np.abs(np.exp(sdf["y_true"]) - np.exp(sdf["knn_gap_pred"])) / np.maximum(1e-8, np.exp(sdf["y_true"]))
        bayes_ape = np.abs(np.exp(sdf["y_true"]) - np.exp(sdf["bayes_gap_pred"])) / np.maximum(1e-8, np.exp(sdf["y_true"]))
        local_logae = np.abs(sdf["y_true"] - sdf["knn_gap_pred"])
        bayes_logae = np.abs(sdf["y_true"] - sdf["bayes_gap_pred"])
        sdf["APE_Gain"] = local_ape - bayes_ape
        sdf["LogAE_Gain"] = local_logae - bayes_logae
        for dec, sub in sdf.groupby("Decile"):
            rows.append({
                "CV Protocol": cv,
                "Decile": int(dec),
                "Median_Gap": float(sub[gap_col].median()),
                "Median_APE_Gain": float(sub["APE_Gain"].median()),
                "Median_LogAE_Gain": float(sub["LogAE_Gain"].median()),
                "P75_APE_Gain": float(sub["APE_Gain"].quantile(0.75)),
                "Mean_Lambda": float(sub["lambda_gap"].mean()),
                "Mean_Neighbors": float(sub["n_gap"].mean()),
            })
    out = pd.DataFrame(rows)
    cv_order = {"KFold": 1, "GroupKFold": 2}
    out["CV_Order"] = out["CV Protocol"].map(cv_order)
    out = out.sort_values(["CV_Order", "Decile"]).drop(columns=["CV_Order"]).reset_index(drop=True)
    return out


# ================= Main Execution =================
def run_main_appendix():
    set_seed(SEED)
    os.makedirs(OUT_MAIN, exist_ok=True)
    df = fetch_from_csv("export_neo4j/api_data_snapshot.csv")

    kf = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(df))
    df_shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    gkf = list(GroupKFold(n_splits=N_SPLITS).split(df_shuf, df_shuf["y"].values, groups=df_shuf[GROUP_COL].values))

    r1, p1, d1 = run_unified_cv(df, kf, "KFold")
    r2, p2, d2 = run_unified_cv(df_shuf, gkf, "GroupKFold")
    res_df = pd.concat([r1, r2], ignore_index=True)
    p_all = pd.concat([p1, p2], ignore_index=True)
    d_all = pd.concat([d1, d2], ignore_index=True)

    p_all.to_csv(os.path.join(OUT_MAIN, "predictions_detail.csv"), index=False)
    d_all.to_csv(os.path.join(OUT_MAIN, "bayes_shrinkage_detail.csv"), index=False)

    agg = res_df.groupby(["CV Protocol", "Method"]).agg(
        {"R2": ["mean", "std"], "RMSE": ["mean", "std"], "WAPE": ["mean", "std"], "Coverage": ["mean"], "Width": ["mean"]}
    ).reset_index()

    def fmt_m(c):
        return lambda r: f"{r[(c, 'mean')]:.3f} ({r[(c, 'std')]:.3f})"

    def fmt_cov(x):
        return f"{x * 100:.2f}%" if pd.notna(x) else "-"

    def fmt_wid(x):
        return f"{x:.3f}" if pd.notna(x) else "-"

    agg["CV_Order"] = agg["CV Protocol"].map({"KFold": 1, "GroupKFold": 2})

    # ---------- Main Table 4 ----------
    t4_methods = {"Prior (Ridge)": 1, "KNN (Mean)": 2, "KNN (GapTrim)": 3, "Bayes (Mean)": 4, "Bayes (GapTrim)": 5}
    t4_df = agg[agg["Method"].isin(t4_methods.keys())].copy()
    t4_df["Order"] = t4_df["Method"].map(t4_methods)
    t4_df = t4_df.sort_values(["CV_Order", "Order"]).reset_index(drop=True)

    out_t4 = pd.DataFrame()
    out_t4["CV Protocol"] = t4_df["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    out_t4["Method"] = t4_df["Method"]
    out_t4["R2"] = t4_df.apply(fmt_m('R2'), axis=1)
    out_t4["RMSE"] = t4_df.apply(fmt_m('RMSE'), axis=1)
    out_t4["WAPE"] = t4_df.apply(fmt_m('WAPE'), axis=1)
    out_t4["PI95 Coverage"] = t4_df[("Coverage", "mean")].apply(fmt_cov)
    out_t4["Avg Width"] = t4_df[("Width", "mean")].apply(fmt_wid)
    out_t4.to_csv(os.path.join(OUT_MAIN, "Table_4_Main_Results.csv"), index=False)

    # ---------- Main Table 5 ----------
    t5_rename = {"Bayes (Mean)": "Base Bayes (Mean)", "Bayes (GapTrim)": "Base Bayes (GapTrim)"}
    t5_methods = {"Base Bayes (Mean)": 1, "Base Bayes (GapTrim)": 2, "Plus Bayes (Robust Cov)": 3, "Stacking (Conformal)": 4}
    t5_df = agg.copy()
    t5_df["Method"] = t5_df["Method"].replace(t5_rename)
    t5_df = t5_df[t5_df["Method"].isin(t5_methods.keys())].copy()
    t5_df["Order"] = t5_df["Method"].map(t5_methods)
    t5_df = t5_df.sort_values(["CV_Order", "Order"]).reset_index(drop=True)

    out_t5 = pd.DataFrame()
    out_t5["CV Protocol"] = t5_df["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    out_t5["Method"] = t5_df["Method"]
    out_t5["R2"] = t5_df.apply(fmt_m('R2'), axis=1)
    out_t5["RMSE"] = t5_df.apply(fmt_m('RMSE'), axis=1)
    out_t5["PI95 Coverage"] = t5_df[("Coverage", "mean")].apply(fmt_cov)
    out_t5["Avg Width"] = t5_df[("Width", "mean")].apply(fmt_wid)
    out_t5.to_csv(os.path.join(OUT_MAIN, "Table_5_Robustness.csv"), index=False)

    # ---------- Appendix B ----------
    b1 = build_shrinkage_diagnostics(d_all).copy()
    b2 = build_tail_risk_table(p_all).copy()
    b3 = build_high_disagreement_table(d_all).copy()
    b4 = build_decile_gain_table(d_all).copy()

    # naming/order for appendix
    b1["CV Protocol"] = b1["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    b2["CV Protocol"] = b2["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    b3["CV Protocol"] = b3["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    b4["CV Protocol"] = b4["CV Protocol"].replace({"GroupKFold": "Group KFold"})

    variant_bucket_order = lambda v: 1 if v == "Mean" else (2 if str(v).startswith("Mean |") else (3 if v == "Gap" else (4 if str(v).startswith("Gap |") else 9)))
    subset_order = {"Top25% prior-local disagreement": 1, "Top10% prior-local disagreement": 2}
    method_order = {
        "Prior (Ridge)": 1,
        "KNN (Mean)": 2,
        "KNN (GapTrim)": 3,
        "Bayes (Mean)": 4,
        "Bayes (GapTrim)": 5,
        "Plus Bayes (Robust Cov)": 6,
        "Stacking (Conformal)": 7,
    }

    b1["CV_Order"] = b1["CV Protocol"].map({"KFold":1, "Group KFold":2})
    b1["Variant_Order"] = b1["Variant"].map(variant_bucket_order)
    b1 = b1.sort_values(["CV_Order", "Variant_Order", "Variant"]).drop(columns=["CV_Order", "Variant_Order"]).reset_index(drop=True)

    b2["CV_Order"] = b2["CV Protocol"].map({"KFold":1, "Group KFold":2})
    b2["Method_Order"] = b2["Method"].map(method_order)
    b2 = b2.sort_values(["CV_Order", "Method_Order"]).drop(columns=["CV_Order", "Method_Order"]).reset_index(drop=True)

    b3["CV_Order"] = b3["CV Protocol"].map({"KFold":1, "Group KFold":2})
    b3["Variant_Order"] = b3["Variant"].map({"Mean":1, "Gap":2})
    b3["Subset_Order"] = b3["Subset"].map(subset_order)
    b3 = b3.sort_values(["CV_Order", "Variant_Order", "Subset_Order"]).drop(columns=["CV_Order", "Variant_Order", "Subset_Order"]).reset_index(drop=True)

    b4["CV_Order"] = b4["CV Protocol"].map({"KFold":1, "Group KFold":2})
    b4 = b4.sort_values(["CV_Order", "Decile"]).drop(columns=["CV_Order"]).reset_index(drop=True)

    b1.to_csv(os.path.join(OUT_MAIN, "Appendix_B1_Shrinkage_Diagnostics.csv"), index=False)
    b2.to_csv(os.path.join(OUT_MAIN, "Appendix_B2_Tail_Risk.csv"), index=False)
    b3.to_csv(os.path.join(OUT_MAIN, "Appendix_B3_High_Disagreement.csv"), index=False)
    b4.to_csv(os.path.join(OUT_MAIN, "Appendix_B4_Decile_Gain.csv"), index=False)

    # ---------- Consolidated workbook ----------
    xlsx_path = os.path.join(OUT_MAIN, "Main_and_Appendix_Tables.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        out_t4.to_excel(writer, sheet_name="Table 4", index=False)
        out_t5.to_excel(writer, sheet_name="Table 5", index=False)
        b1.to_excel(writer, sheet_name="Appendix B1", index=False)
        b2.to_excel(writer, sheet_name="Appendix B2", index=False)
        b3.to_excel(writer, sheet_name="Appendix B3", index=False)
        b4.to_excel(writer, sheet_name="Appendix B4", index=False)

    print("\n[SUCCESS] Generated files in paper_results_main_appendix directory:")
    for fn in [
        "Table_4_Main_Results.csv",
        "Table_5_Robustness.csv",
        "predictions_detail.csv",
        "bayes_shrinkage_detail.csv",
        "Appendix_B1_Shrinkage_Diagnostics.csv",
        "Appendix_B2_Tail_Risk.csv",
        "Appendix_B3_High_Disagreement.csv",
        "Appendix_B4_Decile_Gain.csv",
        "Main_and_Appendix_Tables.xlsx",
    ]:
        print(f"  -> {os.path.join(OUT_MAIN, fn)}")




# Muted journal-style palette
COLORS = {
    "prior": "#C67C83",      # muted rose
    "local": "#8FBF8F",      # muted green
    "bayes": "#4C78A8",      # muted blue
    "line_kf": "#5B8DB8",    # steel blue
    "line_gkf": "#2E5E8C",   # deep blue
    "local_bar": "#B8C7D9",  # cool gray-blue
    "bayes_bar": "#4C78A8",
    "grid": "#D9DDE3",
    "box": "#444444",
}


def load_mechanism_data(base):
    pred = pd.read_csv(os.path.join(base, "predictions_detail.csv"))
    bay = pd.read_csv(os.path.join(base, "bayes_shrinkage_detail.csv"))
    b1 = pd.read_csv(os.path.join(base, "Appendix_B1_Shrinkage_Diagnostics.csv"))
    b3 = pd.read_csv(os.path.join(base, "Appendix_B3_High_Disagreement.csv"))
    b4 = pd.read_csv(os.path.join(base, "Appendix_B4_Decile_Gain.csv"))
    return pred, bay, b1, b3, b4


def _apply_journal_style():
    sns.set_theme(style="white", context="paper", font_scale=1.10)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "legend.title_fontsize": 9.0,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
    })


def build_fig5(pred):
    _apply_journal_style()
    gkf = pred[pred["CV Protocol"].isin(["Group KFold", "GroupKFold"])].copy()
    methods = ["Prior (Ridge)", "KNN (GapTrim)", "Bayes (GapTrim)"]
    palette = {
        "Prior (Ridge)": COLORS["prior"],
        "KNN (GapTrim)": COLORS["local"],
        "Bayes (GapTrim)": COLORS["bayes"],
    }
    plot_data = []
    for m in methods:
        abs_res = np.abs(gkf["y"] - gkf[m])
        plot_data.append(pd.DataFrame({"Method": m, "Absolute log error": abs_res}))
    plot_data = pd.concat(plot_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    sns.violinplot(
        x="Method", y="Absolute log error", data=plot_data,
        order=methods, palette=palette, inner=None, cut=0, linewidth=0.9, ax=ax,
        saturation=0.95
    )
    sns.boxplot(
        x="Method", y="Absolute log error", data=plot_data,
        order=methods, width=0.18, showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": COLORS["box"], "linewidth": 0.9},
        whiskerprops={"color": COLORS["box"], "linewidth": 0.9},
        capprops={"color": COLORS["box"], "linewidth": 0.9},
        medianprops={"color": COLORS["box"], "linewidth": 1.1},
        showfliers=False, ax=ax
    )
    med = plot_data.groupby("Method")["Absolute log error"].median().reindex(methods)
    for i, m in enumerate(methods):
        val = med[m]
        ax.scatter(i, val, color=COLORS["box"], s=16, zorder=4)
        ax.text(i, val + 0.06, f"Median={val:.2f}", ha="center", va="bottom", fontsize=8.3, color="#333333")

    ax.set_xlabel("")
    ax.set_ylabel("Absolute log-price error")
    ax.set_xticklabels(["Prior", "KNN (GapTrim)", "Bayes (GapTrim)"])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    png = os.path.join(OUT_FIG, "Fig5_ColdStart_Error_Distribution.png")
    pdf = os.path.join(OUT_FIG, "Fig5_ColdStart_Error_Distribution.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _pick_subset_rows(b3_gap, cv_value):
    sub = b3_gap[b3_gap["CV Protocol"] == cv_value].copy()
    if sub.empty:
        return sub
    sub["subset_key"] = (
        sub["Subset"].astype(str).str.lower().str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False).str.replace("_", "", regex=False)
        .str.replace("priorlocaldisagreement", "pld", regex=False)
    )
    out_rows = []
    # Robust matching for top-25 and top-10 subset labels
    target_patterns = {
        "Top 25%": ["top25", "25%", "topquart", "q4", "top25pld"],
        "Top 10%": ["top10", "10%", "topdec", "top10pld"],
    }
    for label, patterns in target_patterns.items():
        cur = pd.DataFrame()
        for pat in patterns:
            cur = sub[sub["subset_key"].str.contains(pat, na=False)]
            if not cur.empty:
                break
        if not cur.empty:
            row = cur.iloc[0].copy()
            row["SubsetShort"] = label
            out_rows.append(row)
    if out_rows:
        return pd.DataFrame(out_rows)
    # fallback: take first two rows and infer labels
    out = sub.iloc[: min(2, len(sub))].copy()
    out["SubsetShort"] = [f"Subset {i+1}" for i in range(len(out))]
    return out


def build_fig6(bay, b3, b4):
    _apply_journal_style()
    fig = plt.figure(figsize=(11.8, 7.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.95, 1.25], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.42)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    # (a) GapTrim shrinkage-weight distribution
    gap = bay.copy()
    gap["CV Protocol"] = gap["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    gap_plot = gap[["CV Protocol", "lambda_gap"]].copy()
    sns.violinplot(
        data=gap_plot, x="CV Protocol", y="lambda_gap",
        order=["KFold", "Group KFold"],
        palette=["#AFC6DD", "#5B87B1"], inner=None, cut=0, linewidth=0.9, ax=ax1,
        saturation=0.95
    )
    sns.boxplot(
        data=gap_plot, x="CV Protocol", y="lambda_gap",
        order=["KFold", "Group KFold"], width=0.18, showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": COLORS["box"], "linewidth": 0.9},
        whiskerprops={"color": COLORS["box"], "linewidth": 0.9},
        capprops={"color": COLORS["box"], "linewidth": 0.9},
        medianprops={"color": COLORS["box"], "linewidth": 1.1},
        showfliers=False, ax=ax1
    )
    lam_medians = gap_plot.groupby("CV Protocol")["lambda_gap"].median().reindex(["KFold", "Group KFold"])
    for i, lab in enumerate(["KFold", "Group KFold"]):
        val = lam_medians[lab]
        ax1.scatter(i, val, color=COLORS["box"], s=14, zorder=4)
        ax1.text(i, val + 0.01, f"Median={val:.3f}", ha="center", va="bottom", fontsize=8.1, color="#333333")
    ax1.set_title("(a) GapTrim shrinkage-weight distribution", loc="left", pad=6)
    ax1.set_xlabel("")
    ax1.set_ylabel(r"Shrinkage weight $\lambda_i$")
    ax1.set_ylim(0.62, 0.95)
    ax1.grid(axis="y")
    ax1.grid(axis="x", visible=False)

    # (b) Median APE gain across disagreement deciles
    b4p = b4.copy()
    b4p["CV Protocol"] = b4p["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    sns.lineplot(
        data=b4p, x="Decile", y="Median_APE_Gain", hue="CV Protocol",
        hue_order=["KFold", "Group KFold"], marker="o", dashes=False, linewidth=2.0,
        palette=[COLORS["line_kf"], COLORS["line_gkf"]], ax=ax2
    )
    ax2.axhline(0, color="#777777", linestyle="--", linewidth=0.9)
    ax2.set_title("(b) Median APE gain across disagreement deciles", loc="left", pad=6)
    ax2.set_xlabel("Prior–local disagreement decile")
    ax2.set_ylabel("Median APE gain\n(Local − Bayes)")
    ax2.legend(title="", frameon=False, loc="upper left")
    ax2.grid(True)

    # (c) WAPE in high-disagreement subsets
    b3p = b3.copy()
    b3p["CV Protocol"] = b3p["CV Protocol"].replace({"GroupKFold": "Group KFold"})
    b3_gap = b3p[b3p["Variant"] == "Gap"].copy()
    g_sub = _pick_subset_rows(b3_gap, "Group KFold")
    if g_sub.empty:
        g_sub = _pick_subset_rows(b3_gap, "GroupKFold")
    if g_sub.empty:
        raise ValueError("Could not find Group KFold / Gap rows in Appendix_B3_High_Disagreement.csv")
    x = np.arange(len(g_sub))
    width = 0.34
    ax3.bar(x - width/2, g_sub["Local WAPE"], width=width, color=COLORS["local_bar"], label="Local GapTrim")
    ax3.bar(x + width/2, g_sub["Bayes WAPE"], width=width, color=COLORS["bayes_bar"], label="Bayes (GapTrim)")
    for i, (_, row) in enumerate(g_sub.iterrows()):
        ax3.text(i - width/2, row["Local WAPE"] + 0.04, f"{row['Local WAPE']:.2f}", ha="center", va="bottom", fontsize=8.0)
        ax3.text(i + width/2, row["Bayes WAPE"] + 0.04, f"{row['Bayes WAPE']:.2f}", ha="center", va="bottom", fontsize=8.0)
    ax3.set_xticks(x)
    ax3.set_xticklabels(g_sub["SubsetShort"].tolist())
    ax3.set_title("(c) WAPE in high-disagreement subsets (Group KFold, GapTrim branch)", loc="left", pad=6)
    ax3.set_ylabel("WAPE")
    ax3.set_xlabel("")
    ax3.legend(frameon=False, loc="upper right")
    ax3.grid(axis="y")
    ax3.grid(axis="x", visible=False)

    fig.align_ylabels([ax1, ax2, ax3])
    fig.tight_layout()
    png = os.path.join(OUT_FIG, "Fig6_Shrinkage_Mechanism.png")
    pdf = os.path.join(OUT_FIG, "Fig6_Shrinkage_Mechanism.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run_mechanism_figures():
    base = OUT_MAIN
    pred, bay, b1, b3, b4 = load_mechanism_data(base)
    print(f"[Info] Using input directory: {base}")
    f7 = build_fig5(pred)
    f8 = build_fig6(bay, b3, b4)
    print("[Success] Generated:")
    print(" -", f7[0]); print(" -", f7[1]); print(" -", f8[0]); print(" -", f8[1])




def build_all_tables_workbook() -> str:
    table_files = [
        (Path(OUT_DESC) / 'Table_1_Sample_Composition.csv', 'Table 1'),
        (Path(OUT_DESC) / 'Table_2_Distribution_of_Posted_Quotes.csv', 'Table 2'),
        (Path(OUT_DESC) / 'Table_3_Supplier_Anchoring_Strength.csv', 'Table 3'),
        (Path(OUT_MAIN) / 'Table_4_Main_Results.csv', 'Table 4'),
        (Path(OUT_MAIN) / 'Table_5_Robustness.csv', 'Table 5'),
        (Path(OUT_MAIN) / 'Appendix_B1_Shrinkage_Diagnostics.csv', 'Appendix Table B1'),
        (Path(OUT_MAIN) / 'Appendix_B2_Tail_Risk.csv', 'Appendix Table B2'),
        (Path(OUT_MAIN) / 'Appendix_B3_High_Disagreement.csv', 'Appendix Table B3'),
        (Path(OUT_MAIN) / 'Appendix_B4_Decile_Gain.csv', 'Appendix Table B4'),
    ]
    workbook_path = Path(OUT_MAIN) / 'All_Tables_1_5_and_Appendix_B_1_4.xlsx'
    with pd.ExcelWriter(workbook_path, engine='openpyxl') as writer:
        for csv_path, sheet_name in table_files:
            if not csv_path.exists():
                raise FileNotFoundError(f'Missing table file for workbook export: {csv_path}')
            pd.read_csv(csv_path).to_excel(writer, sheet_name=sheet_name, index=False)
    return str(workbook_path)



def main() -> None:
    ensure_dir(OUT_DESC)
    ensure_dir(OUT_MAIN)
    ensure_dir(OUT_FIG)
    run_descriptive_block()
    run_main_appendix()
    pred, bay, _b1, b3, b4 = load_mechanism_data(OUT_MAIN)
    build_fig5(pred)
    build_fig6(bay, b3, b4)
    workbook_path = build_all_tables_workbook()
    print("[OK] Standalone empirical pipeline finished.")
    print(f"[OK] Descriptive tables: {OUT_DESC}")
    print(f"[OK] Main and appendix tables: {OUT_MAIN}")
    print(f"[OK] Figures: {OUT_FIG}")
    print(f"[OK] Consolidated workbook: {workbook_path}")


if __name__ == "__main__":
    main()
