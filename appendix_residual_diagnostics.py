#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import ast
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import networkx as nx
from gensim.models import Word2Vec

SEED = 42
N_SPLITS = 5
GROUP_COL = "supplier"
VECTOR_SIZE, WALK_LENGTH, NUM_WALKS, WINDOW_SIZE, MIN_COUNT = 32, 15, 20, 3, 1
TEXT_DIM, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE = 64, 50000, 2, (1, 2)
MIN_PRICE = 1e-8
RIDGE_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
OUTPUT_DIR = "paper_result_final/appendix_residual_diagnostics"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def safe_log_price(p):
    return math.log(max(MIN_PRICE, float(p)))


def l2norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def fetch_from_csv(csv_path="export_neo4j/api_data_snapshot.csv"):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["src_list"] = df["src_list"].apply(safe_list)
    df["app_list"] = df["app_list"].apply(safe_list)
    df["supplier"] = df["supplier"].fillna("").astype(str)
    df = df[df["price"].apply(lambda x: pd.notna(x) and float(x) > 0)]
    df = df[df["supplier"] != ""]
    df["y"] = df["price"].apply(safe_log_price)
    df["text"] = (df["name"].fillna("").astype(str) + " " + df["desc"].fillna("").astype(str)).str.strip()
    return df.reset_index(drop=True)


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

    model = Word2Vec(sentences=walks, vector_size=VECTOR_SIZE, window=WINDOW_SIZE,
                     min_count=MIN_COUNT, sg=1, workers=1, seed=SEED)
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


def build_graph_feature(emb, supplier, src_list, app_list):
    def get_vec(k, name):
        return np.asarray(emb.get((k, str(name).strip()), np.zeros(VECTOR_SIZE, dtype=np.float32)), dtype=np.float32)
    def mean_vec(k, lst):
        vs = [get_vec(k, x) for x in lst if np.linalg.norm(get_vec(k, x)) > 1e-12]
        return np.mean(vs, axis=0) if vs else np.zeros(VECTOR_SIZE, dtype=np.float32)
    return np.concatenate([
        get_vec("supplier", supplier),
        mean_vec("src_ind", src_list),
        mean_vec("app_ind", app_list)
    ], axis=0).astype(np.float32)


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


def fit_ridge_prior_with_inner_cv(X, y):
    inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
    best_alpha, best_rmse = None, 1e18
    for a in RIDGE_ALPHAS:
        rmses = []
        for tr, va in inner.split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=a, random_state=SEED))
            ]).fit(X[tr], y[tr])
            pred = model.predict(X[va])
            rmses.append(np.sqrt(np.mean((y[va] - pred) ** 2)))
        cur = float(np.mean(rmses))
        if cur < best_rmse:
            best_rmse, best_alpha = cur, a
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha, random_state=SEED))
    ]).fit(X, y)
    return model, best_alpha


def make_fold_features(tr_df, te_df):
    emb = train_graph_embeddings(tr_df)
    tf = train_text_embedder(tr_df["text"].tolist())
    X_tr = np.concatenate([
        np.stack([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"]) for _, r in tr_df.iterrows()]),
        tf(tr_df["text"].tolist())
    ], axis=1)
    X_te = np.concatenate([
        np.stack([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"]) for _, r in te_df.iterrows()]),
        tf(te_df["text"].tolist())
    ], axis=1)
    return X_tr, X_te


def pooled_prior_residuals(df, cv_name):
    if cv_name == "KFold":
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(df)
    else:
        splitter = GroupKFold(n_splits=N_SPLITS).split(df, groups=df[GROUP_COL].values)

    out = []
    for fold, (tr_idx, te_idx) in enumerate(splitter, start=1):
        tr_df = df.iloc[tr_idx].copy().reset_index(drop=True)
        te_df = df.iloc[te_idx].copy().reset_index(drop=True)
        X_tr, X_te = make_fold_features(tr_df, te_df)
        y_tr = tr_df["y"].values.astype(float)
        y_te = te_df["y"].values.astype(float)
        model, alpha = fit_ridge_prior_with_inner_cv(X_tr, y_tr)
        mu_te = np.asarray(model.predict(X_te), dtype=float)
        resid_te = y_te - mu_te
        out.append(pd.DataFrame({
            "cv_protocol": cv_name,
            "fold": fold,
            "alpha": alpha,
            "y": y_te,
            "mu0": mu_te,
            "residual": resid_te
        }))
    return pd.concat(out, ignore_index=True)


def normality_summary(x):
    x = np.asarray(x, dtype=float)
    jb_stat, jb_p = stats.jarque_bera(x)
    sw_n = min(len(x), 5000)
    sw_stat, sw_p = stats.shapiro(x[:sw_n]) if sw_n >= 3 else (np.nan, np.nan)
    return {
        "Obs.": int(len(x)),
        "Mean": float(np.mean(x)),
        "SD": float(np.std(x, ddof=1)),
        "Skewness": float(stats.skew(x, bias=False)),
        "Excess Kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
        "JB Statistic": float(jb_stat),
        "JB p-value": float(jb_p),
        "Shapiro Statistic": float(sw_stat),
        "Shapiro p-value": float(sw_p),
    }


def plot_hist_qq(resid_df, cv_name, i, outdir):
    x = resid_df["residual"].values.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].hist(x, bins=40, edgecolor="white", alpha=0.85)
    axes[0].set_title(f"(a) {cv_name}: residual histogram")
    axes[0].set_xlabel("Global prior residual (ln price)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    stats.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title(f"(b) {cv_name}: residual Q-Q plot")
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    png = os.path.join(outdir, f"A{i}_Residual_Diagnostics_{cv_name}.png")
    pdf = os.path.join(outdir, f"A{i}_Residual_Diagnostics_{cv_name}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = fetch_from_csv("export_neo4j/api_data_snapshot.csv")

    all_summary = []
    all_resid = []

    i = 0
    for cv_name in ["KFold", "GroupKFold"]:
        resid_df = pooled_prior_residuals(df, cv_name)
        resid_df.to_csv(os.path.join(OUTPUT_DIR, f"A{i+1}_Residuals_{cv_name}.csv"), index=False)
        plot_hist_qq(resid_df, cv_name, i+1, OUTPUT_DIR)
        row = normality_summary(resid_df["residual"].values)
        row["CV Protocol"] = cv_name
        row["Selected Ridge Alphas"] = ", ".join(map(str, sorted(resid_df["alpha"].unique().tolist())))
        all_summary.append(row)
        all_resid.append(resid_df)
        i += 1

    summary_df = pd.DataFrame(all_summary)
    summary_df = summary_df[[
        "CV Protocol", "Obs.", "Mean", "SD", "Skewness", "Excess Kurtosis",
        "JB Statistic", "JB p-value", "Shapiro Statistic", "Shapiro p-value",
        "Selected Ridge Alphas"
    ]]
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "Residual_Normality_Summary.csv"), index=False)

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "Appendix_Table_A1_Residual_Diagnostics.xlsx"), engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for resid_df in all_resid:
            resid_df.to_excel(writer, sheet_name=resid_df["cv_protocol"].iloc[0][:31], index=False)


if __name__ == "__main__":
    main()
