#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""

The program follows the same sequence as the manuscript's empirical argument:
1) data preparation and descriptive market evidence;
2) main empirical evidence, interpreted layer by layer:
   2.1 global benchmark and supplier cold-start fragility;
   2.2 local comparable retrieval and cold-start recovery;
   2.3 Bayesian reconciliation and uncertainty-aware benchmarking;
3) structural/mechanism validation:
   3.1 controlled representation contribution (Text, KG, KG+Text);
   3.2 architecture contribution (Global, Local, non-Bayesian fusion, Bayes);
   3.3 matched-seed supplier-topology transfer diagnostic;
4) statistical robustness and sensitivity:
   4.1 paired bootstrap inference, including architecture contrasts;
   4.2 risk-interval calibration extensions already retained from the primary pipeline;
   4.3 matched-seed, multi-seed graph-embedding dimension sensitivity;
5) manuscript-aligned and unified final exports, including one complete reproducibility workbook.

Performance note (optimized exact version): weighted random-walk transitions are precomputed once per graph instead of rebuilding NetworkX neighbor lists at every walk step. For the same seed, this preserves the exact walk sequence and graph embeddings while reducing graph-embedding runtime substantially. Deterministic text features are also cached once per outer fold during dimension sensitivity.

Figure scope: the pipeline reproduces empirical Fig. 3 and Appendix Figs. A1, A2, B1, and B2.
Manuscript Fig. 1 (conceptual framework) and Fig. 2 (ontology/KG schematic) are intentionally
excluded because they are design/schematic figures rather than empirical outputs.

The primary FINI specification is executed once and is not altered. The script is self-contained
with respect to code (it imports/calls no other project Python file) and uses two frozen data inputs:
(1) the revised anonymized Excel file as the authoritative source for USD/CNY prices and text
(name + description), and (2) api_data_snapshot.csv as the authoritative portable snapshot of
the instantiated KG relations, historical row order, and product node identifiers. The snapshot
is NOT a graph-embedding table. Within every outer fold, graph embeddings and text embeddings
are re-estimated from the outer-training observations only, preventing test-fold leakage.
Auxiliary graph validation uses common fold-specific random seeds so topology or dimension
changes are not confounded with arbitrary seed changes. KFold results are always reported
before Group KFold results.
"""


import argparse
import zlib
import os
import math
import ast
import random
import bisect
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
DEFAULT_EXCEL_INPUT = str(ROOT / "price_files/dataproduct_industry_analysis_list_format_anymous.xlsx")
DEFAULT_KG_SNAPSHOT = str(ROOT / "export_neo4j/api_data_snapshot.csv")
OUTPUT_ROOT = ROOT / "paper_result_final"
OUT_TABLES = str(OUTPUT_ROOT / "paper_results_tables")
OUT_FIG = str(OUTPUT_ROOT / "paper_results_figures")
OUT_INTERMEDIATE = str(OUTPUT_ROOT / "paper_results_intermediate")
OUT_DESC = str(Path(OUT_INTERMEDIATE) / "01_descriptive")
OUT_MAIN = str(Path(OUT_INTERMEDIATE) / "02_primary")

# Manuscript currency convention: the revised Excel input stores the model-facing
# ``price`` directly in USD/call and preserves the original RMB quote in ``price_CNY``.
# The exchange rate is retained only for consistency validation; this script MUST NOT
# convert ``price`` a second time.
CNY_PER_USD = 7.1429

COL_NAME = "name"
COL_PRICE = "price"          # source-of-truth model-facing USD/call price in revised Excel
COL_PRICE_CNY = "price_CNY" # preserved original CNY/call price in revised Excel
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
MIN_PRICE, Z_95, DEFAULT_SIGMA_OBS = 1e-15, 1.96, 0.45
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


def _parse_list_cell(x) -> list[str]:
    """Parse a list-valued Excel cell without changing its substantive content."""
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    text = str(x).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple, set)):
            return [str(v).strip() for v in obj if str(v).strip()]
    except Exception:
        pass
    return [v.strip() for v in text.replace("，", ",").split(",") if v.strip()]


def prepare_modeling_sample(excel_path: str, kg_snapshot_path: str) -> tuple[pd.DataFrame, str]:
    """Merge the revised Excel with the frozen KG snapshot and validate consistency.

    Source roles are deliberately separated:

    * Excel is authoritative for ``price`` (USD/call), ``price_CNY`` (original CNY/call),
      ``name``, and ``desc``. Text embeddings are therefore fitted from Excel-derived
      ``name + desc`` within each outer training fold.
    * ``api_data_snapshot.csv`` is authoritative for the instantiated KG relation snapshot:
      ``pid``, historical row order, ``src_list``, and ``app_list``. It is not a precomputed
      embedding table. Fold-specific graph embeddings are learned later from the training
      rows of this relational snapshot.

    This separation preserves the original KG input/order while ensuring the final paper's
    USD price convention is used exactly. The external snapshot supplied with the final
    reproducibility package must already contain the same exact USD/CNY prices as the Excel;
    if it does not, the script fails loudly rather than silently using rounded/stale prices.
    """
    excel_path = str(Path(excel_path).expanduser().resolve())
    kg_snapshot_path = str(Path(kg_snapshot_path).expanduser().resolve())
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel input not found: {excel_path}")
    if not Path(kg_snapshot_path).exists():
        raise FileNotFoundError(f"KG snapshot not found: {kg_snapshot_path}")

    xls = pd.read_excel(excel_path)
    xls.columns = [str(c).strip() for c in xls.columns]
    required_xls = [
        "name", "price", "price_CNY", "supplier", "desc",
        "src_IndustryCategory", "app_IndustryCategory",
    ]
    missing = [c for c in required_xls if c not in xls.columns]
    if missing:
        raise ValueError(f"Revised Excel is missing required columns: {missing}")

    kg = pd.read_csv(kg_snapshot_path, encoding="utf-8-sig")
    kg.columns = [str(c).strip() for c in kg.columns]
    required_kg = [
        "pid", "name", "desc", "price", "supplier", "src_list", "app_list",
        "price_CNY", "price_USD",
    ]
    missing = [c for c in required_kg if c not in kg.columns]
    if missing:
        raise ValueError(
            "The final KG snapshot must contain pid/name/desc/price/supplier/src_list/app_list/"
            f"price_CNY/price_USD. Missing columns: {missing}"
        )

    if xls["name"].duplicated().any() or kg["name"].duplicated().any():
        raise ValueError("Product name must be unique in both Excel and KG snapshot for exact reconciliation.")
    if set(xls["name"]) != set(kg["name"]):
        only_xls = sorted(set(xls["name"]) - set(kg["name"]))[:10]
        only_kg = sorted(set(kg["name"]) - set(xls["name"]))[:10]
        raise ValueError(
            "Excel and KG snapshot do not contain the same product set. "
            f"Excel-only examples={only_xls}; KG-only examples={only_kg}"
        )

    # Excel price columns are the final paper's exact source of truth.
    xls[COL_PRICE] = pd.to_numeric(xls[COL_PRICE], errors="coerce")
    xls[COL_PRICE_CNY] = pd.to_numeric(xls[COL_PRICE_CNY], errors="coerce")
    if xls[COL_PRICE].isna().any() or (xls[COL_PRICE] <= 0).any():
        raise ValueError("Excel 'price' must contain positive USD/call values for all observations.")
    if xls[COL_PRICE_CNY].isna().any() or (xls[COL_PRICE_CNY] <= 0).any():
        raise ValueError("Excel 'price_CNY' must contain positive CNY/call values for all observations.")
    if (xls[COL_PRICE_CNY] >= 300).any():
        raise ValueError("Final analytical Excel should already exclude price_CNY >= 300 CNY/call.")

    implied_usd = xls[COL_PRICE_CNY].astype(float) / CNY_PER_USD
    if not np.allclose(xls[COL_PRICE].astype(float), implied_usd, rtol=1e-10, atol=1e-12):
        max_diff = float(np.max(np.abs(xls[COL_PRICE].astype(float) - implied_usd)))
        raise ValueError(
            "Excel currency audit failed: price must equal price_CNY / 7.1429. "
            f"Maximum absolute discrepancy={max_diff:.6g}."
        )

    # Align Excel to the HISTORICAL KG snapshot order rather than sorting/recreating it.
    # This preserves the exact ordering used by the original api_data_snapshot.csv.
    xls_idx = xls.set_index("name", drop=False)
    df = kg.copy()
    names = df["name"].astype(str)

    excel_supplier = names.map(xls_idx["supplier"]).astype(str)
    excel_desc = names.map(xls_idx["desc"]).fillna("").astype(str)
    if not np.array_equal(df["supplier"].fillna("").astype(str).to_numpy(), excel_supplier.to_numpy()):
        raise ValueError("Supplier identifiers differ between Excel and KG snapshot.")
    if not np.array_equal(df["desc"].fillna("").astype(str).to_numpy(), excel_desc.to_numpy()):
        raise ValueError("Descriptions differ between Excel and KG snapshot; text source is not reproducibly aligned.")

    # Validate that the KG relation memberships agree with Excel ontology columns.
    kg_src = df["src_list"].apply(_parse_list_cell)
    kg_app = df["app_list"].apply(_parse_list_cell)
    ex_src = names.map(xls_idx["src_IndustryCategory"]).apply(_parse_list_cell)
    ex_app = names.map(xls_idx["app_IndustryCategory"]).apply(_parse_list_cell)
    src_bad = [i for i, (a, b) in enumerate(zip(kg_src, ex_src)) if set(a) != set(b)]
    app_bad = [i for i, (a, b) in enumerate(zip(kg_app, ex_app)) if set(a) != set(b)]
    if src_bad or app_bad:
        raise ValueError(
            "Ontology membership mismatch between Excel and KG snapshot: "
            f"src mismatches={len(src_bad)}, app mismatches={len(app_bad)}."
        )

    # The supplied final KG snapshot must carry the exact revised prices too.
    # This prevents a stale/rounded snapshot from silently becoming the model target.
    snap_price = pd.to_numeric(df["price"], errors="coerce")
    snap_usd = pd.to_numeric(df["price_USD"], errors="coerce")
    snap_cny = pd.to_numeric(df["price_CNY"], errors="coerce")
    exact_usd = names.map(xls_idx[COL_PRICE]).astype(float)
    exact_cny = names.map(xls_idx[COL_PRICE_CNY]).astype(float)
    if not np.allclose(snap_price, exact_usd, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "KG snapshot 'price' does not exactly match the revised Excel USD price. "
            "Use the corrected api_data_snapshot.csv supplied with this program."
        )
    if not np.allclose(snap_usd, exact_usd, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "KG snapshot 'price_USD' does not exactly match the revised Excel USD price."
        )
    if not np.allclose(snap_cny, exact_cny, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "KG snapshot 'price_CNY' does not exactly match the revised Excel CNY price."
        )

    # Build the actual modeling frame in KG-snapshot row order.
    # Prices/text come from Excel; relational memberships/pid/order come from KG snapshot.
    df[COL_PRICE] = exact_usd.to_numpy()
    df[COL_PRICE_CNY] = exact_cny.to_numpy()
    df["price_USD"] = df[COL_PRICE].astype(float)
    df[COL_SUPPLIER] = excel_supplier.to_numpy()
    df["desc"] = excel_desc.to_numpy()
    df["src_list"] = kg_src
    df["app_list"] = kg_app
    df["src_IndustryCategory"] = names.map(xls_idx["src_IndustryCategory"]).to_numpy()
    df["app_IndustryCategory"] = names.map(xls_idx["app_IndustryCategory"]).to_numpy()

    # IMPORTANT: text comes from the Excel content, but is aligned to the KG snapshot order.
    df["text"] = (df[COL_NAME].astype(str) + " " + df["desc"].astype(str)).str.strip()
    df["y"] = np.log(df[COL_PRICE].astype(float))
    df["log_price"] = df["y"]
    df["product_id"] = df[COL_SUPPLIER].astype(str) + "||" + df[COL_NAME].astype(str)
    df = df.reset_index(drop=True)

    ensure_dir(OUT_INTERMEDIATE)
    audit_xlsx = Path(OUT_INTERMEDIATE) / "I00_Validated_Merged_Modeling_Sample_Excel_plus_KG.xlsx"
    prepared_csv = Path(OUT_INTERMEDIATE) / "I01_Modeling_Snapshot_Excel_Prices_Text_KG_Relations.csv"
    source_audit_csv = Path(OUT_INTERMEDIATE) / "I00_Source_Consistency_Audit.csv"

    audit_cols = [
        "pid", "name", "price", "price_CNY", "price_USD", "supplier", "desc",
        "src_IndustryCategory", "app_IndustryCategory", "src_list", "app_list", "product_id",
    ]
    audit_df = df[audit_cols].copy()
    audit_df["src_list"] = audit_df["src_list"].apply(lambda x: repr(list(x)))
    audit_df["app_list"] = audit_df["app_list"].apply(lambda x: repr(list(x)))
    audit_df.to_excel(audit_xlsx, index=False)
    audit_df.to_csv(prepared_csv, index=False, encoding="utf-8-sig", float_format="%.17g")

    source_audit = pd.DataFrame([
        {"Check": "Excel rows", "Value": len(xls)},
        {"Check": "KG snapshot rows", "Value": len(kg)},
        {"Check": "Matched products", "Value": len(df)},
        {"Check": "Suppliers", "Value": int(df[COL_SUPPLIER].nunique())},
        {"Check": "Source-membership mismatches", "Value": len(src_bad)},
        {"Check": "Application-membership mismatches", "Value": len(app_bad)},
        {"Check": "Max |snapshot price - Excel USD|", "Value": float(np.max(np.abs(snap_price - exact_usd)))},
        {"Check": "Max |snapshot price_CNY - Excel CNY|", "Value": float(np.max(np.abs(snap_cny - exact_cny)))},
        {"Check": "Text source", "Value": "Excel name + desc; fitted within each outer training fold"},
        {"Check": "KG source", "Value": "api_data_snapshot.csv pid/order/src_list/app_list; embedded within each outer training fold"},
    ])
    source_audit.to_csv(source_audit_csv, index=False, encoding="utf-8-sig")

    # Full-KG structural audit. This is descriptive/audit evidence only; its embedding is
    # deliberately NOT reused across CV folds because doing so would expose test-fold nodes
    # to representation learning. The same frozen relations are subset to each training fold
    # and embedded there.
    all_src = sorted({v for xs in df["src_list"] for v in xs})
    all_app = sorted({v for xs in df["app_list"] for v in xs})
    n_supplier_edges = int(len(df))
    n_src_edges = int(sum(len(set(xs)) for xs in df["src_list"]))
    n_app_edges = int(sum(len(set(xs)) for xs in df["app_list"]))
    full_kg_audit = pd.DataFrame([
        {"Component": "DataAsset nodes", "Count": int(len(df))},
        {"Component": "Supplier nodes", "Count": int(df[COL_SUPPLIER].nunique())},
        {"Component": "src_IndustryCategory nodes", "Count": int(len(all_src))},
        {"Component": "app_IndustryCategory nodes", "Count": int(len(all_app))},
        {"Component": "Supplier-provide_data-DataAsset edges", "Count": n_supplier_edges},
        {"Component": "src_IndustryCategory-source_industry-DataAsset edges", "Count": n_src_edges},
        {"Component": "DataAsset-applied_to-app_IndustryCategory edges", "Count": n_app_edges},
        {"Component": "Total relation edges before graph simplification", "Count": n_supplier_edges + n_src_edges + n_app_edges},
    ])
    full_kg_audit.to_csv(
        Path(OUT_INTERMEDIATE) / "I02_Full_KG_Structure_Audit_Not_Used_As_Full_Sample_Embedding.csv",
        index=False, encoding="utf-8-sig"
    )

    if len(df) != 2879:
        print(f"[WARN] Expected manuscript N=2,879, but merged input contains N={len(df):,}.")
    if df[COL_SUPPLIER].nunique() != 256:
        print(f"[WARN] Expected 256 suppliers, but merged input contains {df[COL_SUPPLIER].nunique():,}.")

    return df, str(prepared_csv)


def read_descriptive_data(path: str) -> pd.DataFrame:
    """Read the script-generated USD modeling snapshot for descriptive tables/figures."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["src_list"] = df["src_list"].apply(_parse_list_cell)
    df["app_list"] = df["app_list"].apply(_parse_list_cell)
    df[COL_PRICE] = pd.to_numeric(df[COL_PRICE], errors="coerce")
    df[COL_PRICE_CNY] = pd.to_numeric(df[COL_PRICE_CNY], errors="coerce")
    df = df.dropna(subset=[COL_PRICE]).copy()
    df["log_price"] = np.log(df[COL_PRICE].astype(float))
    df["product_id"] = df[COL_SUPPLIER].astype(str) + "||" + df[COL_NAME].astype(str)
    return df.reset_index(drop=True)


def build_sample_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"Statistic": "Number of API listings (posted quotes)", "Value": int(df.shape[0])},
        {"Statistic": "Number of suppliers", "Value": int(df[COL_SUPPLIER].nunique())},
    ])


def build_manuscript_table_1_distribution(df: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame([row(df[COL_PRICE], "price (USD/call)"), row(df["log_price"], "ln(price)")])


# ================= Manuscript display formatting =================
# Keep computational/intermediate objects numeric, but make every manuscript-facing
# table reproduce the paper's visible precision and missing-value symbols exactly.
EM_DASH = "—"


def _paper_fixed(x, digits: int = 3, missing: str = "") -> str:
    if pd.isna(x):
        return missing
    return f"{float(x):.{digits}f}"


def _paper_pct_value(x, digits: int = 2, missing: str = EM_DASH) -> str:
    if pd.isna(x):
        return missing
    return f"{100.0 * float(x):.{digits}f}%"


def _paper_int(x, missing: str = "") -> str:
    if pd.isna(x):
        return missing
    return str(int(round(float(x))))


def _paper_compact3(x, missing: str = "") -> str:
    """Appendix C1 convention: exact integers such as 0/1/5 are shown without decimals;
    otherwise values are shown to three decimals."""
    if pd.isna(x):
        return missing
    z = float(x)
    if abs(z - round(z)) < 1e-12:
        return str(int(round(z)))
    return f"{z:.3f}"


def _paper_sci(x, mantissa_digits: int = 2, missing: str = "") -> str:
    if pd.isna(x):
        return missing
    mant, exp = f"{float(x):.{mantissa_digits}e}".split("e")
    iexp = int(exp)
    sign = "−" if iexp < 0 else ("+" if iexp > 0 else "")
    return f"{mant}e{sign}{abs(iexp)}" if iexp != 0 else mant


def format_table1_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    """Match manuscript Table 1 display precision exactly."""
    out = df.copy().astype(object)
    for i, r in out.iterrows():
        out.at[i, "Obs."] = _paper_int(r["Obs."])
        if str(r["Var."]) == "price (USD/call)":
            for c in ["Mean", "SD", "P25", "Median", "P75"]:
                out.at[i, c] = _paper_fixed(r[c], 4)
            out.at[i, "Min"] = _paper_sci(r["Min"], 2)
            out.at[i, "Max"] = _paper_fixed(r["Max"], 3)
        else:
            for c in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
                out.at[i, c] = _paper_fixed(r[c], 3)
    return out


def format_appendix_a1_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    if "Estimate" in out.columns:
        out["Estimate"] = out["Estimate"].apply(lambda x: _paper_fixed(x, 3))
    return out


def format_appendix_b1_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    if out.empty:
        return out
    if "Obs." in out.columns:
        out["Obs."] = out["Obs."].apply(_paper_int)
    for c in ["Mean", "SD", "Skewness", "Excess Kurtosis", "JB Statistic", "JB p-value",
              "Shapiro Statistic", "Shapiro p-value"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: _paper_fixed(x, 3))
    return out


def format_appendix_c1_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    if out.empty:
        return out
    if "N" in out.columns:
        out["N"] = out["N"].apply(_paper_int)
    for c in [c for c in out.columns if c not in {"CV Protocol", "Variant", "N"}]:
        out[c] = out[c].apply(_paper_compact3)
    return out


def format_appendix_c2_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in [c for c in out.columns if c not in {"CV Protocol", "Method"}]:
        out[c] = out[c].apply(lambda x: _paper_fixed(x, 3))
    return out


def format_appendix_c3_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    if "Obs." in out.columns:
        out["Obs."] = out["Obs."].apply(_paper_int)
    if "Obs" in out.columns:
        out["Obs"] = out["Obs"].apply(_paper_int)
    for c in [c for c in out.columns if c not in {"CV Protocol", "Variant", "Subset", "Obs.", "Obs"}]:
        out[c] = out[c].apply(lambda x: _paper_fixed(x, 3))
    return out


def format_appendix_c4_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    if "Decile" in out.columns:
        out["Decile"] = out["Decile"].apply(_paper_int)
    for c in [c for c in out.columns if c not in {"CV Protocol", "Decile"}]:
        out[c] = out[c].apply(lambda x: _paper_fixed(x, 3))
    return out


def format_main_table3_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in ["KFold R2", "Group KFold R2"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: _paper_fixed(x, 3, EM_DASH))
    if "Group PI95 Coverage" in out.columns:
        out["Group PI95 Coverage"] = out["Group PI95 Coverage"].apply(lambda x: _paper_pct_value(x, 2, EM_DASH))
    if "Group Avg Width" in out.columns:
        out["Group Avg Width"] = out["Group Avg Width"].apply(lambda x: _paper_fixed(x, 3, EM_DASH))
    return out


def build_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
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


def build_appendix_table_a1_supplier_clustering(df: pd.DataFrame) -> pd.DataFrame:
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


def build_appendix_fig_a1_distribution(df: pd.DataFrame, outdir: str) -> tuple[str, str]:
    # Panel (a) plots the raw posted-quote histogram truncated at the 99th percentile for display.
    # Panel (b) plots the natural-log histogram of the same prices on the full positive sample.
    apply_descriptive_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    p99 = df[COL_PRICE].quantile(0.99)
    raw_plot = df.loc[df[COL_PRICE] <= p99, COL_PRICE].dropna().values
    sns.histplot(raw_plot, bins=HIST_BINS, color="#7f8c8d", edgecolor="white", alpha=0.85, kde=False, ax=axes[0])
    axes[0].set_xlabel(f"Posted quote (USD/call; ≤99th percentile = {p99:.2f})", fontweight="bold")
    axes[0].set_ylabel("Frequency", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(ax=axes[0])
    axes[0].text(0.01, 0.99, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontweight="bold")

    sns.histplot(df["log_price"].dropna().values, bins=HIST_BINS, color="#2c3e50", edgecolor="white", alpha=0.85, kde=True, line_kws={"linewidth": 1.8}, ax=axes[1])
    axes[1].set_xlabel("ln(price in USD/call)", fontweight="bold")
    axes[1].set_ylabel("Frequency", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(ax=axes[1])
    axes[1].text(0.01, 0.99, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontweight="bold")

    fig.tight_layout(w_pad=0.8)
    return save_png_pdf(fig, "Appendix_Fig_A1_Distribution_of_Standardized_Posted_Quotes_Before_and_After_Log_Transformation", outdir)


def build_appendix_fig_a2_supplier_clustering(df: pd.DataFrame, outdir: str) -> tuple[str, str]:
    # Panel (a) plots supplier-level boxplots for top suppliers ordered by the median log price.
    # Panel (b) plots the within-supplier deviation histogram after subtracting each supplier median.
    apply_descriptive_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    ss = build_supplier_summary(df)
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
    axes[0].set_ylabel("ln(price in USD/call)", fontweight="bold")
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
    return save_png_pdf(fig, "Appendix_Fig_A2_Descriptive_Evidence_of_Supplier_Level_Clustering_in_Posted_Quotes", outdir)


def run_descriptive_block(input_path: str) -> pd.DataFrame:
    """Generate manuscript Table 1, Appendix Table A1, and Appendix Figs. A1-A2."""
    ensure_dir(OUT_DESC)
    ensure_dir(OUT_FIG)
    df = read_descriptive_data(input_path)
    sample_audit = build_sample_audit_table(df)
    dist_table = build_manuscript_table_1_distribution(df)
    anchor_table = build_appendix_table_a1_supplier_clustering(df)

    sample_audit.to_csv(os.path.join(OUT_DESC, "I02_Sample_Audit.csv"), index=False, encoding="utf-8-sig")
    # Manuscript-facing CSVs reproduce the paper's visible precision exactly.
    format_table1_for_paper(dist_table).to_csv(
        os.path.join(OUT_DESC, "I03_Table_1_Distribution_of_Posted_Quotes_USD.csv"),
        index=False, encoding="utf-8-sig"
    )
    format_appendix_a1_for_paper(anchor_table).to_csv(
        os.path.join(OUT_DESC, "I04_Appendix_Table_A1_Supplier_Level_Clustering.csv"),
        index=False, encoding="utf-8-sig"
    )
    build_supplier_summary(df).to_csv(os.path.join(OUT_DESC, "I05_Supplier_Summary.csv"), index=False, encoding="utf-8-sig")

    build_appendix_fig_a1_distribution(df, OUT_FIG)
    build_appendix_fig_a2_supplier_clustering(df, OUT_FIG)
    return df




# Manuscript Fig. 1 and Fig. 2 are editorial/conceptual graphics and are not
# regenerated by this empirical reproduction pipeline.

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


def load_internal_modeling_snapshot(csv_path):
    """Load the script-generated modeling snapshot; this is not a graph-embedding table."""
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

def _build_weighted_market_graph(df_train, *, include_supplier_edges=True):
    """Build the fold-specific weighted market graph and cache transition weights.

    This function preserves the graph construction used by the manuscript but moves
    the expensive neighbor sorting and cumulative-weight construction OUTSIDE the
    random-walk loop.  The previous implementation repeated ``sorted(G.neighbors())``
    and rebuilt edge-weight lists at every walk step; that was numerically unnecessary
    and dominated runtime.
    """
    total_docs = len(df_train)
    ind_counts = {}
    for _, r in df_train.iterrows():
        for ind in sorted(list(set(r["src_list"] + r["app_list"]))):
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
    ind_weights = {ind: math.log(total_docs / (cnt + 1)) + 1.0 for ind, cnt in ind_counts.items()}

    G = nx.Graph()
    for i, r in df_train.reset_index(drop=True).iterrows():
        prod = f"PROD_{i}"
        if include_supplier_edges and r["supplier"]:
            G.add_edge(f"SUP_{r['supplier']}", prod, weight=3.0)
        for src in r["src_list"]:
            G.add_edge(f"SRC_{src}", prod, weight=ind_weights.get(src, 1.0))
        for app in r["app_list"]:
            G.add_edge(f"APP_{app}", prod, weight=ind_weights.get(app, 1.0))

    # Precompute exactly the same sorted-neighbor / weighted-choice structure that
    # random.choices() previously rebuilt at every single walk step.  Using
    # bisect(cumulative_weights, random()*total) reproduces Python's weighted-choice
    # selection for the same RNG state, while being orders of magnitude faster.
    adjacency = {}
    for node in G.nodes():
        nbrs = sorted(list(G.neighbors(node)))
        weights = [float(G[node][nb]["weight"]) for nb in nbrs]
        cum_weights = list(itertools.accumulate(weights))
        total_weight = float(cum_weights[-1]) if cum_weights else 0.0
        adjacency[node] = (nbrs, cum_weights, total_weight)
    return G, adjacency


def _generate_weighted_walks(adjacency, nodes_sorted, *, rng, reset_sorted_each_round):
    """Generate manuscript random walks using cached transition tables.

    ``reset_sorted_each_round=False`` reproduces the primary pipeline's historical
    in-place shuffle sequence. ``True`` reproduces the auxiliary deterministic
    validation routine, which restarts each walk round from the sorted node order.
    """
    walks = []
    nodes = list(nodes_sorted)
    for _ in range(NUM_WALKS):
        if reset_sorted_each_round:
            nodes = list(nodes_sorted)
        rng.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < WALK_LENGTH:
                nbrs, cum_weights, total_weight = adjacency[walk[-1]]
                if not nbrs:
                    break
                # Equivalent to random.choices(nbrs, weights=weights, k=1)[0]
                # for the same random-number stream, but without repeated list work.
                x = rng.random() * total_weight
                j = bisect.bisect(cum_weights, x, 0, len(nbrs) - 1)
                walk.append(nbrs[j])
            walks.append(walk)
    return walks


def _embedding_dict_from_model(G, model):
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


def train_graph_embeddings(df_train):
    """Primary fold graph embedding, optimized without changing the RNG sequence."""
    G, adjacency = _build_weighted_market_graph(df_train, include_supplier_edges=True)
    nodes_sorted = sorted(list(G.nodes()))
    walks = _generate_weighted_walks(
        adjacency, nodes_sorted, rng=random, reset_sorted_each_round=False
    )
    model = Word2Vec(
        sentences=walks, vector_size=VECTOR_SIZE, window=WINDOW_SIZE,
        min_count=MIN_COUNT, sg=1, workers=1, seed=SEED,
    )
    return _embedding_dict_from_model(G, model)


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
def run_unified_cv(df, splitter, cv_name, capture_primary_cache=False):
    """Run the primary LCMA fold pipeline exactly once.

    When capture_primary_cache=True, this function additionally stores the exact
    graph/text feature blocks and core OOF predictions produced by the primary
    run. The cache is consumed only after the full KFold -> Group KFold sequence has
    completed, so later validation stages cannot perturb the primary RNG sequence. Test rows never enter graph/text fitting.
    """
    print(f"\n--- Executing {cv_name} ---", flush=True)
    results = []
    raw_preds = []
    diag_rows = []
    primary_cache = {}

    for fold, (tr_idx, te_idx) in enumerate(splitter, start=1):
        print(f"  Fold {fold}/{len(splitter)}...", flush=True)
        if len(set(map(int, tr_idx)).intersection(set(map(int, te_idx)))) != 0:
            raise AssertionError(f"{cv_name} fold {fold}: train/test row overlap detected.")

        tr_df = df.iloc[tr_idx].copy().reset_index(drop=True)
        te_df = df.iloc[te_idx].copy().reset_index(drop=True)
        if cv_name == "Group KFold":
            tr_sup = set(tr_df[GROUP_COL].astype(str))
            te_sup = set(te_df[GROUP_COL].astype(str))
            if not tr_sup.isdisjoint(te_sup):
                raise AssertionError(f"Group KFold fold {fold}: supplier leakage detected.")

        # Primary LCMA graph/text fitting: training fold only.
        # Text and graph transforms are each computed once per fold and reused by
        # both the global prior and KNN branches.
        emb = train_graph_embeddings(tr_df)
        tf = train_text_embedder(tr_df["text"].tolist())
        t_tr = tf(tr_df["text"].tolist()).astype(np.float32)
        t_te = tf(te_df["text"].tolist()).astype(np.float32)
        g_tr = np.stack([
            build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False)
            for _, r in tr_df.iterrows()
        ]).astype(np.float32)
        g_te = np.stack([
            build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False)
            for _, r in te_df.iterrows()
        ]).astype(np.float32)

        X_tr = np.concatenate([g_tr, t_tr], axis=1)
        X_te = np.concatenate([g_te, t_te], axis=1)

        y_tr, y_te = tr_df["y"].values.astype(float), te_df["y"].values.astype(float)
        p_te = te_df["price"].values.astype(float)

        prior_model = fit_ridge_prior_with_inner_cv(X_tr, y_tr)
        prior_alpha = float(prior_model.named_steps["ridge"].alpha)
        mu0_tr, mu0_te = np.asarray(prior_model.predict(X_tr)), np.asarray(prior_model.predict(X_te))
        sigma0 = clip(float(np.sqrt(np.mean((y_tr - mu0_tr) ** 2))), 0.10, 1.50)

        g_tr_knn = np.vstack([l2norm(x) for x in g_tr]).astype(np.float32)
        g_te_knn = np.vstack([l2norm(x) for x in g_te]).astype(np.float32)
        t_tr_knn = np.vstack([l2norm(x) for x in t_tr]).astype(np.float32)
        t_te_knn = np.vstack([l2norm(x) for x in t_te]).astype(np.float32)
        X_tr_knn = np.vstack([
            l2norm(np.concatenate([g, t])) for g, t in zip(g_tr_knn, t_tr_knn)
        ]).astype(np.float32)
        X_te_knn = np.vstack([
            l2norm(np.concatenate([g, t])) for g, t in zip(g_te_knn, t_te_knn)
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

        df_out = te_df.copy()
        df_out["CV Protocol"] = cv_name
        df_out["Fold"] = fold
        df_out["Prior Ridge Alpha"] = prior_alpha
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
                "Fold": fold,
                "Method": method,
                "R2": r2_score(y_te, y_pred),
                "RMSE": np.sqrt(np.mean((y_te - y_pred) ** 2)),
                "WAPE": wape(p_te, p_pred),
                "Coverage": np.mean((y_te >= low) & (y_te <= high)) if "KNN" not in method else np.nan,
                "Width": np.mean(high - low) if "KNN" not in method else np.nan
            })
        raw_preds.append(df_out)

        # Cache the fitted primary representation and OOF predictions; no refit or RNG consumption.
        if capture_primary_cache:
            g_tr = np.stack([
                build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False)
                for _, r in tr_df.iterrows()
            ]).astype(np.float32)
            g_te = np.stack([
                build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False)
                for _, r in te_df.iterrows()
            ]).astype(np.float32)
            t_tr = tf(tr_df["text"].tolist()).astype(np.float32)
            t_te = tf(te_df["text"].tolist()).astype(np.float32)

            def arr(method, field="pred"):
                return np.asarray(preds[method][field], dtype=float)

            primary_cache[fold] = {
                "components": {"graph": (g_tr, g_te), "text": (t_tr, t_te)},
                "z": {
                    "y_true": y_te.copy(), "price_true": p_te.copy(),
                    "global": arr("Prior (Ridge)"),
                    "local_mean": arr("KNN (Mean)"),
                    "local_gap": arr("KNN (GapTrim)"),
                    "bayes_mean": arr("Bayes (Mean)"),
                    "bayes_gap": arr("Bayes (GapTrim)"),
                    "bayes_mean_low": arr("Bayes (Mean)", "low"),
                    "bayes_mean_high": arr("Bayes (Mean)", "high"),
                    "bayes_gap_low": arr("Bayes (GapTrim)", "low"),
                    "bayes_gap_high": arr("Bayes (GapTrim)", "high"),
                    "sigma0": sigma0, "sigma_obs": sigma_obs, "rho": rho, "delta": delta,
                    "prior_alpha": prior_alpha,
                },
                "tr_idx": np.asarray(tr_idx, dtype=int),
                "te_idx": np.asarray(te_idx, dtype=int),
            }

    return (
        pd.DataFrame(results),
        pd.concat(raw_preds, ignore_index=True),
        pd.DataFrame(diag_rows),
        primary_cache,
    )


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
    for cv_name in ["KFold", "Group KFold"]:
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
    for cv in ["KFold", "Group KFold"]:
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
    cv_order = {"KFold": 1, "Group KFold": 2}
    out["CV_Order"] = out["CV Protocol"].map(cv_order)
    out = out.sort_values(["CV_Order", "Decile"]).drop(columns=["CV_Order"]).reset_index(drop=True)
    return out


# ================= Main Execution =================
def run_main_appendix(main_input: str | None = None, max_folds=None):
    """Run the primary LCMA main/appendix pipeline once and retain reusable fold caches.

    The published/primary pipeline remains KFold first and Group KFold second. The
    returned cache is additive: it stores exact feature blocks/predictions from this
    single execution so component decomposition does not rerun the fused KG+Text arm.
    """
    set_seed(SEED)
    os.makedirs(OUT_MAIN, exist_ok=True)
    if not main_input:
        raise ValueError("run_main_appendix requires the script-generated modeling snapshot path.")
    df = load_internal_modeling_snapshot(main_input)

    kf = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(df))
    df_shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    gkf = list(GroupKFold(n_splits=N_SPLITS).split(df_shuf, df_shuf["y"].values, groups=df_shuf[GROUP_COL].values))

    if max_folds is not None:
        kf = kf[:max_folds]
        gkf = gkf[:max_folds]

    # Complete the entire primary KFold -> Group KFold sequence before structural validation.
    r1, p1, d1, c1 = run_unified_cv(df, kf, "KFold", capture_primary_cache=True)
    r2, p2, d2, c2 = run_unified_cv(df_shuf, gkf, "Group KFold", capture_primary_cache=True)
    res_df = pd.concat([r1, r2], ignore_index=True)
    p_all = pd.concat([p1, p2], ignore_index=True)
    d_all = pd.concat([d1, d2], ignore_index=True)
    primary_fold_cache = {**{("KFold", k): v for k, v in c1.items()},
                           **{("Group KFold", k): v for k, v in c2.items()}}
    protocol_specs = [("KFold", df, kf), ("Group KFold", df_shuf, gkf)]

    p_all.to_csv(os.path.join(OUT_MAIN, "I06_Primary_OOF_Predictions.csv"), index=False)
    d_all.to_csv(os.path.join(OUT_MAIN, "I07_Bayesian_Reconciliation_Detail.csv"), index=False)

    agg = res_df.groupby(["CV Protocol", "Method"]).agg(
        {"R2": ["mean", "std"], "RMSE": ["mean", "std"], "WAPE": ["mean", "std"], "Coverage": ["mean"], "Width": ["mean"]}
    ).reset_index()

    # Additive numeric intermediate outputs; formatted paper outputs below are preserved.
    cv_order_map = {"KFold": 1, "Group KFold": 2}
    res_df["CV_Order"] = res_df["CV Protocol"].map(cv_order_map)
    res_df = res_df.sort_values(["CV_Order", "Fold", "Method"], kind="stable").drop(columns=["CV_Order"]).reset_index(drop=True)
    res_df.to_csv(os.path.join(OUT_MAIN, "I08_Primary_Fold_Metrics.csv"), index=False, encoding="utf-8-sig")

    primary_numeric_summary = pd.DataFrame({
        "CV Protocol": agg["CV Protocol"],
        "Method": agg["Method"],
        "R2_mean": agg[("R2", "mean")],
        "R2_sd": agg[("R2", "std")],
        "RMSE_mean": agg[("RMSE", "mean")],
        "RMSE_sd": agg[("RMSE", "std")],
        "WAPE_mean": agg[("WAPE", "mean")],
        "WAPE_sd": agg[("WAPE", "std")],
        "Coverage_mean": agg[("Coverage", "mean")],
        "Width_mean": agg[("Width", "mean")],
    })
    primary_numeric_summary["CV_Order"] = primary_numeric_summary["CV Protocol"].map(cv_order_map)
    primary_numeric_summary = primary_numeric_summary.sort_values(["CV_Order", "Method"], kind="stable").drop(columns=["CV_Order"]).reset_index(drop=True)
    primary_numeric_summary.to_csv(os.path.join(OUT_MAIN, "I09_Primary_Numeric_Summary.csv"), index=False, encoding="utf-8-sig")

    def fmt_m(c):
        return lambda r: f"{r[(c, 'mean')]:.3f} ({r[(c, 'std')]:.3f})"

    def fmt_cov(x):
        return f"{x * 100:.2f}%" if pd.notna(x) else EM_DASH

    def fmt_wid(x):
        return f"{x:.3f}" if pd.notna(x) else EM_DASH

    agg["CV_Order"] = agg["CV Protocol"].map({"KFold": 1, "Group KFold": 2})

    # ---------- Main Table 2 ----------
    table2_methods = {"Prior (Ridge)": 1, "KNN (Mean)": 2, "KNN (GapTrim)": 3, "Bayes (Mean)": 4, "Bayes (GapTrim)": 5}
    table2_df = agg[agg["Method"].isin(table2_methods.keys())].copy()
    table2_df["Order"] = table2_df["Method"].map(table2_methods)
    table2_df = table2_df.sort_values(["CV_Order", "Order"]).reset_index(drop=True)

    table2_output = pd.DataFrame()
    table2_output["CV Protocol"] = table2_df["CV Protocol"].replace({"Group KFold": "Group KFold"})
    table2_output["Method"] = table2_df["Method"]
    table2_output["R2"] = table2_df.apply(fmt_m('R2'), axis=1)
    table2_output["RMSE"] = table2_df.apply(fmt_m('RMSE'), axis=1)
    table2_output["WAPE"] = table2_df.apply(fmt_m('WAPE'), axis=1)
    table2_output["PI95 Coverage"] = table2_df[("Coverage", "mean")].apply(fmt_cov)
    table2_output["Avg Width"] = table2_df[("Width", "mean")].apply(fmt_wid)
    table2_output.to_csv(os.path.join(OUT_MAIN, "I10_Table_2_Main_Benchmark_Performance.csv"), index=False, encoding="utf-8-sig")

    # ---------- Appendix Table E1 ----------
    appendix_e1_rename = {"Bayes (Mean)": "Base Bayes (Mean)", "Bayes (GapTrim)": "Base Bayes (GapTrim)"}
    appendix_e1_methods = {"Base Bayes (Mean)": 1, "Base Bayes (GapTrim)": 2, "Plus Bayes (Robust Cov)": 3, "Stacking (Conformal)": 4}
    appendix_e1_df = agg.copy()
    appendix_e1_df["Method"] = appendix_e1_df["Method"].replace(appendix_e1_rename)
    appendix_e1_df = appendix_e1_df[appendix_e1_df["Method"].isin(appendix_e1_methods.keys())].copy()
    appendix_e1_df["Order"] = appendix_e1_df["Method"].map(appendix_e1_methods)
    appendix_e1_df = appendix_e1_df.sort_values(["CV_Order", "Order"]).reset_index(drop=True)

    appendix_e1_output = pd.DataFrame()
    appendix_e1_output["CV Protocol"] = appendix_e1_df["CV Protocol"].replace({"Group KFold": "Group KFold"})
    appendix_e1_output["Method"] = appendix_e1_df["Method"]
    appendix_e1_output["R2"] = appendix_e1_df.apply(fmt_m('R2'), axis=1)
    appendix_e1_output["RMSE"] = appendix_e1_df.apply(fmt_m('RMSE'), axis=1)
    appendix_e1_output["PI95 Coverage"] = appendix_e1_df[("Coverage", "mean")].apply(fmt_cov)
    appendix_e1_output["Avg Width"] = appendix_e1_df[("Width", "mean")].apply(fmt_wid)
    appendix_e1_output.to_csv(os.path.join(OUT_MAIN, "I15_Appendix_Table_E1_Predictive_Uncertainty_Robustness.csv"), index=False, encoding="utf-8-sig")

    # ---------- Appendix C ----------
    b1 = build_shrinkage_diagnostics(d_all).copy()
    b2 = build_tail_risk_table(p_all).copy()
    b3 = build_high_disagreement_table(d_all).copy()
    b4 = build_decile_gain_table(d_all).copy()

    b1["CV Protocol"] = b1["CV Protocol"].replace({"Group KFold": "Group KFold"})
    b2["CV Protocol"] = b2["CV Protocol"].replace({"Group KFold": "Group KFold"})
    b3["CV Protocol"] = b3["CV Protocol"].replace({"Group KFold": "Group KFold"})
    b4["CV Protocol"] = b4["CV Protocol"].replace({"Group KFold": "Group KFold"})

    variant_bucket_order = lambda v: 1 if v == "Mean" else (2 if str(v).startswith("Mean |") else (3 if v == "Gap" else (4 if str(v).startswith("Gap |") else 9)))
    subset_order = {"Top25% prior-local disagreement": 1, "Top10% prior-local disagreement": 2}
    method_order = {
        "Prior (Ridge)": 1, "KNN (Mean)": 2, "KNN (GapTrim)": 3, "Bayes (Mean)": 4,
        "Bayes (GapTrim)": 5, "Plus Bayes (Robust Cov)": 6, "Stacking (Conformal)": 7,
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

    # Freeze the visible precision to the manuscript convention before any paper-facing export.
    b1 = format_appendix_c1_for_paper(b1)
    b2 = format_appendix_c2_for_paper(b2)
    b3 = format_appendix_c3_for_paper(b3)
    b4 = format_appendix_c4_for_paper(b4)

    b1.to_csv(os.path.join(OUT_MAIN, "I11_Appendix_Table_C1_Shrinkage_Diagnostics.csv"), index=False, encoding="utf-8-sig")
    b2.to_csv(os.path.join(OUT_MAIN, "I12_Appendix_Table_C2_Tail_Risk.csv"), index=False, encoding="utf-8-sig")
    b3.to_csv(os.path.join(OUT_MAIN, "I13_Appendix_Table_C3_High_Disagreement.csv"), index=False, encoding="utf-8-sig")
    b4.to_csv(os.path.join(OUT_MAIN, "I14_Appendix_Table_C4_Decile_Gain.csv"), index=False, encoding="utf-8-sig")

    print("\n[SUCCESS] Generated ordered primary intermediate files:")
    for fn in [
        "I10_Table_2_Main_Benchmark_Performance.csv", "I15_Appendix_Table_E1_Predictive_Uncertainty_Robustness.csv", "I06_Primary_OOF_Predictions.csv",
        "I07_Bayesian_Reconciliation_Detail.csv", "I11_Appendix_Table_C1_Shrinkage_Diagnostics.csv",
        "I12_Appendix_Table_C2_Tail_Risk.csv", "I13_Appendix_Table_C3_High_Disagreement.csv",
        "I14_Appendix_Table_C4_Decile_Gain.csv",
        "I08_Primary_Fold_Metrics.csv", "I09_Primary_Numeric_Summary.csv",
    ]:
        print(f"  -> {os.path.join(OUT_MAIN, fn)}")

    return {
        "fold_metrics": res_df,
        "numeric_summary": primary_numeric_summary,
        "predictions": p_all,
        "diagnostics": d_all,
        "table2": table2_output,
        "appendix_e1": appendix_e1_output,
        "appendix_c1": b1,
        "appendix_c2": b2,
        "appendix_c3": b3,
        "appendix_c4": b4,
        "protocols": protocol_specs,
        "primary_fold_cache": primary_fold_cache,
    }



# Muted journal-style palette
COLORS = {
    "prior": "#C67C83",
    "local": "#8FBF8F",
    "bayes": "#4C78A8",
    "line_kf": "#5B8DB8",
    "line_gkf": "#2E5E8C",
    "local_bar": "#B8C7D9",
    "bayes_bar": "#4C78A8",
    "grid": "#D9DDE3",
    "box": "#444444",
}


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


def build_manuscript_fig_3_absolute_log_price_error(pred):
    ensure_dir(OUT_FIG)
    _apply_journal_style()
    gkf = pred[pred["CV Protocol"].isin(["Group KFold", "Group KFold"])].copy()
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
    ax.set_ylabel(r"Absolute log-price error, $|\ln p-\ln \hat{p}|$ (log points)")
    ax.set_xticklabels(["Prior", "KNN (GapTrim)", "Bayes (GapTrim)"])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    png = os.path.join(OUT_FIG, "Fig_3_Absolute_Log_Price_Error_Distributions_Under_Supplier_Cold_Start.png")
    pdf = os.path.join(OUT_FIG, "Fig_3_Absolute_Log_Price_Error_Distributions_Under_Supplier_Cold_Start.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


# ============================================================================
# Integrated empirical validation: representation, architecture, inference, and sensitivity
# ============================================================================
VALIDATION_OUT = Path(OUT_INTERMEDIATE) / "03_validation"
EMPIRICAL_RESULTS_OUT = Path(OUT_TABLES)
MANUSCRIPT_SEQUENCE_OUT = Path(OUT_INTERMEDIATE) / "04_manuscript_sequence"

def configure_output_root(output_root: str | Path) -> None:
    """Configure final tables, figures, and ordered intermediate-output directories."""
    global OUTPUT_ROOT, OUT_TABLES, OUT_FIG, OUT_INTERMEDIATE, OUT_DESC, OUT_MAIN, VALIDATION_OUT, EMPIRICAL_RESULTS_OUT, MANUSCRIPT_SEQUENCE_OUT
    OUTPUT_ROOT = Path(output_root).expanduser().resolve()
    OUT_TABLES = str(OUTPUT_ROOT / "paper_results_tables")
    OUT_FIG = str(OUTPUT_ROOT / "paper_results_figures")
    OUT_INTERMEDIATE = str(OUTPUT_ROOT / "paper_results_intermediate")
    OUT_DESC = str(Path(OUT_INTERMEDIATE) / "01_descriptive")
    OUT_MAIN = str(Path(OUT_INTERMEDIATE) / "02_primary")
    VALIDATION_OUT = Path(OUT_INTERMEDIATE) / "03_validation"
    MANUSCRIPT_SEQUENCE_OUT = Path(OUT_INTERMEDIATE) / "04_manuscript_sequence"
    EMPIRICAL_RESULTS_OUT = Path(OUT_TABLES)


def remove_legacy_workbooks() -> None:
    """Remove obsolete consolidated workbooks that are no longer part of the output contract.

    This prevents stale files from an earlier run from being mistaken for current outputs when
    the same --output-root directory is reused.
    """
    for legacy_name in (
        "Main_and_Appendix_Tables.xlsx",
        "All_Tables_1_5_and_Appendix_B_1_4.xlsx",
    ):
        legacy_path = Path(OUT_MAIN) / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            print(f"[CLEAN] Removed obsolete workbook: {legacy_path}")
VALIDATION_ALPHA_GRID = np.linspace(0.0, 1.0, 21)
VALIDATION_CALIB_FRAC = 0.20
BOOTSTRAP_REPS = 5000
EMBEDDING_DIM_GRID = (16, 32, 64)
GRAPH_VALIDATION_REPEATS = 3

CV_PROTOCOL_ORDER = {"KFold": 1, "Group KFold": 2, "Group KFold": 2}

# Explicit manuscript order. Never rely on alphabetical sorting for method/stage labels.
# This is especially important for Appendix Tables D4, D5, and D7.
METHOD_ORDER = {
    "Global (Ridge)": 1,
    "Local (KNN Mean)": 2,
    "Local (KNN GapTrim)": 3,
    "Global + Local (50/50, non-Bayes)": 4,
    "Global + Local (tuned convex, non-Bayes)": 5,
    "Bayes (Mean)": 6,
    "Bayes (GapTrim)": 7,
}
STAGE_ORDER = {
    "global": 1,
    "Global": 1,
    "local_mean": 2,
    "Local Mean": 2,
    "local_gap": 3,
    "Local GapTrim": 3,
    "bayes_mean": 4,
    "Bayes Mean": 4,
    "bayes_gap": 5,
    "Bayes GapTrim": 5,
}
REPRESENTATION_COMPARISON_ORDER = {
    "KG only vs Text": 1,
    "KG+Text vs Text": 2,
}
ARCHITECTURE_COMPARISON_ORDER = {
    "Local vs Global": 1,
    "Tuned Global+Local vs Local": 2,
    "Bayes vs Tuned non-Bayes": 3,
}


def sort_cv_results(df: pd.DataFrame, extra=None) -> pd.DataFrame:
    if df is None or df.empty or "CV Protocol" not in df.columns:
        return df
    out = df.copy()
    out["__CV_Order"] = out["CV Protocol"].map(CV_PROTOCOL_ORDER).fillna(99)
    cols = ["__CV_Order"] + ([] if extra is None else list(extra))
    out = out.sort_values(cols, kind="stable").drop(columns=["__CV_Order"]).reset_index(drop=True)
    return out


def validation_common_seed(cv_name: str, fold: int, repeat: int = 0, salt: int = 0) -> int:
    """Common-random-number seed for auxiliary graph validation.

    Crucially, the seed does NOT depend on graph dimension or on whether supplier
    edges are included. Within a CV fold and repeat, topology and dimension variants
    therefore receive the same seed. This isolates the intended design change as far
    as possible and avoids the dimension+seed confounding present in earlier drafts.
    """
    cv_code = 100_000 if cv_name == "Group KFold" else 0
    return int(SEED + cv_code + fold * 1009 + repeat * 100_003 + salt)


def train_graph_embeddings_deterministic(
    df_train: pd.DataFrame, *, vector_size: int = VECTOR_SIZE, seed: int = SEED,
    include_supplier_edges: bool = True,
):
    """Train fold-only graph embeddings with deterministic matched seeds.

    Computational optimization only: graph semantics, random-walk sequence and
    Word2Vec settings are unchanged.  Cached transition tables eliminate repeated
    NetworkX neighbor sorting inside millions of walk steps.
    """
    G, adjacency = _build_weighted_market_graph(
        df_train, include_supplier_edges=include_supplier_edges
    )
    rng = random.Random(seed)
    nodes_sorted = sorted(list(G.nodes()))
    walks = _generate_weighted_walks(
        adjacency, nodes_sorted, rng=rng, reset_sorted_each_round=True
    )
    model = Word2Vec(
        sentences=walks, vector_size=vector_size, window=WINDOW_SIZE,
        min_count=MIN_COUNT, sg=1, workers=1, seed=seed,
    )
    return _embedding_dict_from_model(G, model)


def graph_feature_with_dimension(emb, supplier, src_list, app_list, *, vector_size: int, supplier_neutral=False):
    z = np.zeros(vector_size, dtype=np.float32)

    def get_vec(k, name):
        return np.asarray(emb.get((k, str(name).strip()), z), dtype=np.float32)

    def mean_vec(k, values):
        vs = []
        for x in values:
            v = get_vec(k, x)
            if np.linalg.norm(v) > 1e-12:
                vs.append(v)
        return np.mean(vs, axis=0) if vs else z.copy()

    sup = z.copy() if supplier_neutral else get_vec("supplier", supplier)
    return np.concatenate([sup, mean_vec("src_ind", src_list), mean_vec("app_ind", app_list)]).astype(np.float32)


def components_from_embedding(tr_df, te_df, emb, *, vector_size, supplier_neutral=False, text_pair=None):
    g_tr = np.stack([
        graph_feature_with_dimension(emb, r["supplier"], r["src_list"], r["app_list"],
                         vector_size=vector_size, supplier_neutral=supplier_neutral)
        for _, r in tr_df.iterrows()
    ]).astype(np.float32)
    g_te = np.stack([
        graph_feature_with_dimension(emb, r["supplier"], r["src_list"], r["app_list"],
                         vector_size=vector_size, supplier_neutral=supplier_neutral)
        for _, r in te_df.iterrows()
    ]).astype(np.float32)
    out = {"graph": (g_tr, g_te)}
    if text_pair is not None:
        out["text"] = text_pair
    return out


def build_representation_variant(tr_df, te_df, mode, components=None):
    if mode not in {"text", "kg", "fused"}:
        raise ValueError(f"Unknown representation mode: {mode}")
    components = {} if components is None else components
    g_tr = g_te = t_tr = t_te = None
    if mode in {"kg", "fused"}:
        if "graph" in components:
            g_tr, g_te = components["graph"]
        else:
            emb = train_graph_embeddings(tr_df)
            g_tr = np.stack([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False)
                             for _, r in tr_df.iterrows()]).astype(np.float32)
            g_te = np.stack([build_graph_feature(emb, r["supplier"], r["src_list"], r["app_list"], False)
                             for _, r in te_df.iterrows()]).astype(np.float32)
            components["graph"] = (g_tr, g_te)
    if mode in {"text", "fused"}:
        if "text" in components:
            t_tr, t_te = components["text"]
        else:
            tf = train_text_embedder(tr_df["text"].tolist())
            t_tr = tf(tr_df["text"].tolist()).astype(np.float32)
            t_te = tf(te_df["text"].tolist()).astype(np.float32)
            components["text"] = (t_tr, t_te)

    if mode == "kg":
        x_prior_tr, x_prior_te = g_tr, g_te
        x_knn_tr = np.vstack([l2norm(x) for x in g_tr]).astype(np.float32)
        x_knn_te = np.vstack([l2norm(x) for x in g_te]).astype(np.float32)
    elif mode == "text":
        x_prior_tr, x_prior_te = t_tr, t_te
        x_knn_tr = np.vstack([l2norm(x) for x in t_tr]).astype(np.float32)
        x_knn_te = np.vstack([l2norm(x) for x in t_te]).astype(np.float32)
    else:
        x_prior_tr = np.concatenate([g_tr, t_tr], axis=1)
        x_prior_te = np.concatenate([g_te, t_te], axis=1)
        x_knn_tr = np.vstack([l2norm(np.concatenate([l2norm(g), l2norm(t)]))
                              for g, t in zip(g_tr, t_tr)]).astype(np.float32)
        x_knn_te = np.vstack([l2norm(np.concatenate([l2norm(g), l2norm(t)]))
                              for g, t in zip(g_te, t_te)]).astype(np.float32)
    return x_prior_tr, x_prior_te, x_knn_tr, x_knn_te


def fit_stage_models(tr_df, te_df, mode, components=None):
    xptr, xpte, xktr, xkte = build_representation_variant(tr_df, te_df, mode, components=components)
    ytr = tr_df["y"].to_numpy(float)
    yte = te_df["y"].to_numpy(float)
    prior_model = fit_ridge_prior_with_inner_cv(xptr, ytr)
    mu0_tr = np.asarray(prior_model.predict(xptr), float)
    mu0_te = np.asarray(prior_model.predict(xpte), float)
    sigma0 = clip(float(np.sqrt(np.mean((ytr - mu0_tr) ** 2))), 0.10, 1.50)

    nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(tr_df)), metric="cosine").fit(xktr)
    dte, ite = nn.kneighbors(xkte, return_distance=True)
    ste = 1.0 - dte
    sigma_obs = calibrate_sigma_obs(ytr, xktr)
    rho, delta = tune_rho_delta(mu0_tr, sigma0, sigma_obs, ytr, xktr, nn)

    local_mean, local_gap, bayes_mean, bayes_gap = [], [], [], []
    bayes_mean_lo, bayes_mean_hi, bayes_gap_lo, bayes_gap_hi = [], [], [], []
    for i in range(len(te_df)):
        sims, vals = [], []
        for t in range(len(ite[i])):
            if ste[i][t] > 0:
                sims.append(float(ste[i][t]))
                vals.append(float(ytr[int(ite[i][t])]))
            if len(sims) >= EVIDENCE_MAX:
                break
        if not sims:
            lm = lg = bm = bg = float(mu0_te[i])
            sbm = sbg = sigma0
        else:
            order = np.argsort(-np.asarray(sims))
            sims = [sims[k] for k in order]
            vals = [vals[k] for k in order]
            wm = np.asarray([max(1e-6, s) ** KNN_SIM_POW for s in sims])
            lm = float(np.sum(wm * vals) / np.sum(wm))
            bm, sbm = normal_normal_posterior(mu0_te[i], sigma0, np.mean(vals), len(vals), sigma_obs)
            m = gap_trim_count(sims, rho, delta)
            wg = np.asarray([max(1e-6, s) ** KNN_SIM_POW for s in sims[:m]])
            lg = float(np.sum(wg * np.asarray(vals[:m])) / np.sum(wg))
            bg, sbg = normal_normal_posterior(mu0_te[i], sigma0, np.mean(vals[:m]), m, sigma_obs)
        local_mean.append(lm); local_gap.append(lg); bayes_mean.append(bm); bayes_gap.append(bg)
        bayes_mean_lo.append(bm - Z_95 * math.sqrt(sbm**2 + sigma_obs**2))
        bayes_mean_hi.append(bm + Z_95 * math.sqrt(sbm**2 + sigma_obs**2))
        bayes_gap_lo.append(bg - Z_95 * math.sqrt(sbg**2 + sigma_obs**2))
        bayes_gap_hi.append(bg + Z_95 * math.sqrt(sbg**2 + sigma_obs**2))

    return {
        "y_true": yte, "price_true": te_df["price"].to_numpy(float),
        "global": mu0_te, "local_mean": np.asarray(local_mean), "local_gap": np.asarray(local_gap),
        "bayes_mean": np.asarray(bayes_mean), "bayes_gap": np.asarray(bayes_gap),
        "bayes_mean_low": np.asarray(bayes_mean_lo), "bayes_mean_high": np.asarray(bayes_mean_hi),
        "bayes_gap_low": np.asarray(bayes_gap_lo), "bayes_gap_high": np.asarray(bayes_gap_hi),
        "sigma0": sigma0, "sigma_obs": sigma_obs, "rho": rho, "delta": delta,
    }


def validation_metric_row(cv_name, fold, representation, method, y, p, pred, low=None, high=None):
    pred = np.asarray(pred, float)
    p_pred = np.maximum(MIN_PRICE, np.exp(pred))
    row = {
        "CV Protocol": cv_name, "Fold": fold, "Representation": representation, "Method": method,
        "R2": r2_score(y, pred), "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "WAPE": wape(p, p_pred), "PI95 Coverage": np.nan, "Avg Width": np.nan,
    }
    if low is not None and high is not None:
        low, high = np.asarray(low, float), np.asarray(high, float)
        row["PI95 Coverage"] = float(np.mean((y >= low) & (y <= high)))
        row["Avg Width"] = float(np.mean(high - low))
    return row


def validation_metric(cv_name, fold, rep, method, te_df, pred, low=None, high=None):
    return validation_metric_row(cv_name, fold, rep, method, te_df["y"].to_numpy(float),
                         te_df["price"].to_numpy(float), pred, low, high)


def fit_text_pair(tr_df, te_df):
    tf = train_text_embedder(tr_df["text"].tolist())
    return tf(tr_df["text"].tolist()).astype(np.float32), tf(te_df["text"].tolist()).astype(np.float32)


def primary_baseline_from_cache(protocol_specs, primary_fold_cache):
    metrics, preds = [], []
    for cv_name, dcv, splits in protocol_specs:
        for fold, (_tr_idx, te_idx) in enumerate(splits, 1):
            entry = primary_fold_cache[(cv_name, fold)]
            z = entry["z"]
            te_df = dcv.iloc[te_idx].reset_index(drop=True)
            for method, key, lo, hi in [
                ("Global (Ridge)", "global", None, None),
                ("Local (KNN Mean)", "local_mean", None, None),
                ("Local (KNN GapTrim)", "local_gap", None, None),
                ("Bayes (Mean)", "bayes_mean", "bayes_mean_low", "bayes_mean_high"),
                ("Bayes (GapTrim)", "bayes_gap", "bayes_gap_low", "bayes_gap_high"),
            ]:
                metrics.append(validation_metric(
                    cv_name, fold, "KG + Text", method, te_df,
                    z[key], z[lo] if lo else None, z[hi] if hi else None
                ))
            for i in range(len(te_df)):
                preds.append({
                    "CV Protocol": cv_name, "Fold": fold, "row_id": int(te_idx[i]),
                    "supplier": str(te_df.loc[i, GROUP_COL]),
                    "y_true": z["y_true"][i], "price_true": z["price_true"][i],
                    "Representation": "KG + Text",
                    "global": z["global"][i], "local_gap": z["local_gap"][i],
                    "bayes_gap": z["bayes_gap"][i],
                })
    return metrics, preds


def calibration_split(tr_df, cv_name, seed):
    rng = np.random.RandomState(seed)
    n = len(tr_df)
    if cv_name == "Group KFold":
        groups = np.array(sorted(tr_df[GROUP_COL].astype(str).unique()))
        rng.shuffle(groups)
        n_cal_g = max(1, int(round(VALIDATION_CALIB_FRAC * len(groups))))
        cal_groups = set(groups[:n_cal_g])
        cal_mask = tr_df[GROUP_COL].astype(str).isin(cal_groups).to_numpy()
        fit_idx, cal_idx = np.where(~cal_mask)[0], np.where(cal_mask)[0]
    else:
        idx = np.arange(n); rng.shuffle(idx)
        n_cal = max(50, int(round(VALIDATION_CALIB_FRAC * n)))
        n_cal = min(n_cal, max(1, n - 50))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    if len(fit_idx) < 10 or len(cal_idx) < 5:
        raise RuntimeError("Calibration split is too small.")
    return fit_idx, cal_idx


def fit_nonbayes_calibration(tr_df, cv_name, seed):
    fit_idx, cal_idx = calibration_split(tr_df, cv_name, seed)
    fit_df = tr_df.iloc[fit_idx].reset_index(drop=True)
    cal_df = tr_df.iloc[cal_idx].reset_index(drop=True)
    tp = fit_text_pair(fit_df, cal_df)
    emb = train_graph_embeddings_deterministic(fit_df, vector_size=VECTOR_SIZE, seed=seed)
    comp = components_from_embedding(fit_df, cal_df, emb, vector_size=VECTOR_SIZE, text_pair=tp)
    z = fit_stage_models(fit_df, cal_df, "fused", components=comp)
    y, g, l = z["y_true"], z["global"], z["local_gap"]
    best_alpha, best_rmse = .5, np.inf
    for a in VALIDATION_ALPHA_GRID:
        pp = (1-a)*g + a*l
        rr = float(np.sqrt(np.mean((y-pp)**2)))
        if rr < best_rmse:
            best_alpha, best_rmse = float(a), rr
    pe = .5*g + .5*l
    pt = (1-best_alpha)*g + best_alpha*l
    qe = float(np.quantile(np.abs(y-pe), .95, method="higher"))
    qt = float(np.quantile(np.abs(y-pt), .95, method="higher"))
    return best_alpha, qe, qt, best_rmse


def summarize_validation_metrics(fm):
    out = fm.groupby(["CV Protocol", "Representation", "Method"], dropna=False, sort=False).agg(
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        WAPE_mean=("WAPE", "mean"), WAPE_sd=("WAPE", "std"),
        Coverage_mean=("PI95 Coverage", "mean"), Width_mean=("Avg Width", "mean"),
    ).reset_index()
    return sort_cv_results(out, ["Representation", "Method"])


def validation_metric_pair(y, pred_a, pred_b):
    return (
        r2_score(y, pred_a) - r2_score(y, pred_b),
        float(np.sqrt(np.mean((y-pred_a)**2)) - np.sqrt(np.mean((y-pred_b)**2))),
    )


def summarize_repeated_validation_metrics(fm: pd.DataFrame) -> pd.DataFrame:
    """Summarize multi-seed validation without treating seeds as extra independent folds.

    First average repeated seeds within each outer fold, then report the mean and SD
    across the outer folds. The average within-fold seed SD is reported separately as
    a stochastic-sensitivity diagnostic.
    """
    if fm is None or fm.empty:
        return pd.DataFrame()
    group_cols = ["CV Protocol", "Representation", "Method"]
    fold_avg = fm.groupby(group_cols + ["Fold"], sort=False).agg(
        R2=("R2", "mean"), RMSE=("RMSE", "mean"), WAPE=("WAPE", "mean"),
        Coverage=("PI95 Coverage", "mean"), Width=("Avg Width", "mean"),
    ).reset_index()
    seed_sd = fm.groupby(group_cols + ["Fold"], sort=False).agg(
        R2_seed_sd=("R2", "std"), RMSE_seed_sd=("RMSE", "std"),
        WAPE_seed_sd=("WAPE", "std"),
    ).reset_index()
    fold_avg = fold_avg.merge(seed_sd, on=group_cols + ["Fold"], how="left")
    out = fold_avg.groupby(group_cols, sort=False).agg(
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        WAPE_mean=("WAPE", "mean"), WAPE_sd=("WAPE", "std"),
        Coverage_mean=("Coverage", "mean"), Width_mean=("Width", "mean"),
        R2_seed_sd_mean=("R2_seed_sd", "mean"),
        RMSE_seed_sd_mean=("RMSE_seed_sd", "mean"),
        WAPE_seed_sd_mean=("WAPE_seed_sd", "mean"),
    ).reset_index()
    return sort_cv_results(out, ["Representation", "Method"])


def average_repeated_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Average auxiliary predictions across matched random-seed repeats per OOF row."""
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    keys = ["CV Protocol", "Fold", "row_id", "supplier", "Representation"]
    out = pred_df.groupby(keys, sort=False).agg(
        y_true=("y_true", "first"), price_true=("price_true", "first"),
        global_pred=("global", "mean"), local_gap=("local_gap", "mean"),
        bayes_gap=("bayes_gap", "mean"),
    ).reset_index()
    out = out.rename(columns={"global_pred": "global"})
    return sort_cv_results(out, ["Fold", "row_id", "Representation"])


def get_validation_embedding(cache: dict, tr_df: pd.DataFrame, *, cv_name: str, fold: int,
                             vector_size: int, repeat: int, include_supplier_edges: bool):
    """Get or train a deterministic auxiliary embedding using common matched seeds.

    The cache permits the d=32 full-graph embeddings used in topology validation to be
    reused in dimension sensitivity, avoiding duplicate graph training.
    """
    key = (cv_name, int(fold), int(vector_size), int(repeat), bool(include_supplier_edges))
    if key not in cache:
        seed = validation_common_seed(cv_name, fold, repeat=repeat, salt=700000)
        cache[key] = train_graph_embeddings_deterministic(
            tr_df, vector_size=vector_size, seed=seed,
            include_supplier_edges=include_supplier_edges,
        )
    return cache[key], validation_common_seed(cv_name, fold, repeat=repeat, salt=700000)


def run_supplier_topology_validation(protocol_specs, primary_fold_cache, *, repeats=GRAPH_VALIDATION_REPEATS,
                                     embedding_cache=None):
    """Matched-seed, multi-seed decomposition of supplier-specific graph topology.

    Full and supplier-neutral graphs are re-estimated under the same outer fold and the
    same repeat seed. The primary text block is held fixed. Results are summarized by
    first averaging seeds within each fold, then averaging across folds.
    """
    embedding_cache = {} if embedding_cache is None else embedding_cache
    rows, preds = [], []
    reps = [
        ("KG full (matched-seed)", "kg", True, False),
        ("KG supplier-neutral (matched-seed)", "kg", False, True),
        ("Text + KG full (matched-seed)", "fused", True, False),
        ("Text + KG supplier-neutral (matched-seed)", "fused", False, True),
    ]
    for cv_name, dcv, splits in protocol_specs:  # KFold -> Group KFold
        for fold, (tr_idx, te_idx) in enumerate(splits, 1):
            tr_df = dcv.iloc[tr_idx].reset_index(drop=True)
            te_df = dcv.iloc[te_idx].reset_index(drop=True)
            tp = primary_fold_cache[(cv_name, fold)]["components"]["text"]
            for repeat in range(repeats):
                print(f"[supplier-topology] {cv_name} fold {fold}/{len(splits)} seed {repeat+1}/{repeats}", flush=True)
                emb_full, seed = get_validation_embedding(
                    embedding_cache, tr_df, cv_name=cv_name, fold=fold,
                    vector_size=VECTOR_SIZE, repeat=repeat, include_supplier_edges=True,
                )
                emb_neutral, _ = get_validation_embedding(
                    embedding_cache, tr_df, cv_name=cv_name, fold=fold,
                    vector_size=VECTOR_SIZE, repeat=repeat, include_supplier_edges=False,
                )
                comp_full = components_from_embedding(
                    tr_df, te_df, emb_full, vector_size=VECTOR_SIZE,
                    supplier_neutral=False, text_pair=tp,
                )
                comp_neutral = components_from_embedding(
                    tr_df, te_df, emb_neutral, vector_size=VECTOR_SIZE,
                    supplier_neutral=True, text_pair=tp,
                )
                fitted = {
                    "KG full (matched-seed)": fit_stage_models(tr_df, te_df, "kg", components=comp_full),
                    "KG supplier-neutral (matched-seed)": fit_stage_models(tr_df, te_df, "kg", components=comp_neutral),
                    "Text + KG full (matched-seed)": fit_stage_models(tr_df, te_df, "fused", components=comp_full),
                    "Text + KG supplier-neutral (matched-seed)": fit_stage_models(tr_df, te_df, "fused", components=comp_neutral),
                }
                for rep, z in fitted.items():
                    for method, key, lo, hi in [
                        ("Global (Ridge)", "global", None, None),
                        ("Local (KNN Mean)", "local_mean", None, None),
                        ("Local (KNN GapTrim)", "local_gap", None, None),
                        ("Bayes (Mean)", "bayes_mean", "bayes_mean_low", "bayes_mean_high"),
                        ("Bayes (GapTrim)", "bayes_gap", "bayes_gap_low", "bayes_gap_high"),
                    ]:
                        rr = validation_metric(
                            cv_name, fold, rep, method, te_df, z[key],
                            z[lo] if lo else None, z[hi] if hi else None,
                        )
                        rr["Seed Repeat"] = repeat + 1
                        rr["Random Seed"] = seed
                        rows.append(rr)
                    for i in range(len(te_df)):
                        preds.append({
                            "CV Protocol": cv_name, "Fold": fold, "row_id": int(te_idx[i]),
                            "supplier": str(te_df.loc[i, GROUP_COL]),
                            "y_true": z["y_true"][i], "price_true": z["price_true"][i],
                            "Representation": rep, "Seed Repeat": repeat + 1,
                            "Random Seed": seed, "global": z["global"][i],
                            "local_gap": z["local_gap"][i], "bayes_gap": z["bayes_gap"][i],
                        })
    fm = sort_cv_results(pd.DataFrame(rows), ["Fold", "Seed Repeat", "Representation", "Method"])
    sm = summarize_repeated_validation_metrics(fm)
    pred_all = sort_cv_results(pd.DataFrame(preds), ["Fold", "Seed Repeat", "row_id", "Representation"])
    pred_avg = average_repeated_predictions(pred_all)
    return fm, sm, pred_all, pred_avg, embedding_cache


def paired_bootstrap_comparisons(core_pred_long, topology_pred_avg=None, *, B=BOOTSTRAP_REPS, seed=SEED+9090):
    """Paired bootstrap for core representation gains and matched-topology mechanisms.

    KFold resamples OOF observations. Group KFold resamples suppliers as clusters.
    Multi-seed topology predictions are averaged per OOF observation before resampling,
    preventing random-seed repeats from being treated as independent observations.
    """
    comparison_specs = [
        (core_pred_long, "KG + Text", "Text only", "KG+Text vs Text"),
        (core_pred_long, "KG only", "Text only", "KG only vs Text"),
    ]
    if topology_pred_avg is not None and not topology_pred_avg.empty:
        comparison_specs.extend([
            (topology_pred_avg, "KG supplier-neutral (matched-seed)", "KG full (matched-seed)",
             "Supplier-neutral KG vs Full KG (matched-seed)"),
            (topology_pred_avg, "Text + KG supplier-neutral (matched-seed)", "Text + KG full (matched-seed)",
             "Text+Neutral KG vs Text+Full KG (matched-seed)"),
        ])
        # Add Text-only rows from the primary controlled decomposition for the incremental-transfer test.
        text_rows = core_pred_long[core_pred_long["Representation"] == "Text only"].copy()
        bridge = pd.concat([topology_pred_avg, text_rows], ignore_index=True, sort=False)
        comparison_specs.append((
            bridge, "Text + KG supplier-neutral (matched-seed)", "Text only",
            "Text+Neutral KG vs Text",
        ))

    methods = ["global", "local_gap", "bayes_gap"]
    out = []
    for cv in ["KFold", "Group KFold"]:
        for source_df, cand, baseline, label in comparison_specs:
            d = source_df[source_df["CV Protocol"] == cv].copy()
            a = d[d["Representation"] == cand]
            b = d[d["Representation"] == baseline]
            if a.empty or b.empty:
                continue
            m = a.merge(b, on=["Fold", "row_id"], suffixes=("_A", "_B"))
            if m.empty:
                continue
            for method in methods:
                y = m["y_true_A"].to_numpy(float)
                pa = m[f"{method}_A"].to_numpy(float)
                pb = m[f"{method}_B"].to_numpy(float)
                dr2, drmse = validation_metric_pair(y, pa, pb)
                stable = zlib.crc32(f"{cv}|{label}|{method}".encode("utf-8")) % 1_000_000
                rng = np.random.default_rng(seed + stable)
                vals = []
                if cv == "Group KFold":
                    clusters = m["supplier_A"].astype(str).to_numpy()
                    uniq = np.unique(clusters)
                    cluster_to_idx = {g: np.where(clusters == g)[0] for g in uniq}
                    for _ in range(B):
                        gs = rng.choice(uniq, size=len(uniq), replace=True)
                        idx = np.concatenate([cluster_to_idx[g] for g in gs])
                        vals.append(validation_metric_pair(y[idx], pa[idx], pb[idx]))
                    boot_type = "supplier-cluster paired bootstrap"
                else:
                    n = len(m)
                    for _ in range(B):
                        idx = rng.integers(0, n, n)
                        vals.append(validation_metric_pair(y[idx], pa[idx], pb[idx]))
                    boot_type = "observation paired bootstrap"
                arr = np.asarray(vals, float)
                out.append({
                    "CV Protocol": cv, "Comparison": label, "Method stage": method,
                    "Candidate": cand, "Baseline": baseline, "N": len(m), "Bootstrap B": B,
                    "Bootstrap type": boot_type, "Delta R2 (Candidate-Baseline)": dr2,
                    "Delta R2 CI2.5": np.quantile(arr[:,0], .025),
                    "Delta R2 CI97.5": np.quantile(arr[:,0], .975),
                    "Delta RMSE (Candidate-Baseline)": drmse,
                    "Delta RMSE CI2.5": np.quantile(arr[:,1], .025),
                    "Delta RMSE CI97.5": np.quantile(arr[:,1], .975),
                })
    return sort_cv_results(pd.DataFrame(out), ["Comparison", "Method stage"])


def embedding_dimension_sensitivity(protocol_specs, *, repeats=GRAPH_VALIDATION_REPEATS,
                                    max_folds=None, embedding_cache=None):
    """Matched-seed, multi-seed sensitivity to graph embedding dimension.

    Optimization: the deterministic TF-IDF/SVD representation is fit ONCE per outer
    fold and reused across the three graph seeds and dimensions.  This does not alter
    any text feature because the text learner depends only on the outer training fold,
    not on graph seed or graph dimension.
    """
    embedding_cache = {} if embedding_cache is None else embedding_cache
    rows = []
    for cv_name, dcv, splits in protocol_specs:  # KFold first, then Group KFold
        use = splits if max_folds is None else splits[:max_folds]
        for fold, (tr_idx, te_idx) in enumerate(use, 1):
            tr_df = dcv.iloc[tr_idx].reset_index(drop=True)
            te_df = dcv.iloc[te_idx].reset_index(drop=True)
            # Text representation is fold-specific but seed/dimension invariant.
            tp = fit_text_pair(tr_df, te_df)
            for repeat in range(repeats):
                for dim in EMBEDDING_DIM_GRID:
                    print(f"[dimension] {cv_name} fold {fold}/{len(use)} seed {repeat+1}/{repeats} dim={dim}", flush=True)
                    emb, seed = get_validation_embedding(
                        embedding_cache, tr_df, cv_name=cv_name, fold=fold,
                        vector_size=dim, repeat=repeat, include_supplier_edges=True,
                    )
                    comp = components_from_embedding(
                        tr_df, te_df, emb, vector_size=dim, text_pair=tp,
                    )
                    z = fit_stage_models(tr_df, te_df, "fused", components=comp)
                    for method, key in [
                        ("Global (Ridge)", "global"),
                        ("Local (KNN GapTrim)", "local_gap"),
                        ("Bayes (GapTrim)", "bayes_gap"),
                    ]:
                        rr = validation_metric(cv_name, fold, f"KG+Text dim={dim}", method, te_df, z[key])
                        rr["Graph vector size"] = dim
                        rr["Seed Repeat"] = repeat + 1
                        rr["Random Seed"] = seed
                        rows.append(rr)
    fm = sort_cv_results(pd.DataFrame(rows), ["Graph vector size", "Fold", "Seed Repeat", "Method"])
    if fm.empty:
        return fm, pd.DataFrame(), embedding_cache

    group_cols = ["CV Protocol", "Graph vector size", "Method"]
    fold_avg = fm.groupby(group_cols + ["Fold"], sort=False).agg(
        R2=("R2", "mean"), RMSE=("RMSE", "mean")
    ).reset_index()
    seed_sd = fm.groupby(group_cols + ["Fold"], sort=False).agg(
        R2_seed_sd=("R2", "std"), RMSE_seed_sd=("RMSE", "std")
    ).reset_index()
    fold_avg = fold_avg.merge(seed_sd, on=group_cols + ["Fold"], how="left")
    sm = fold_avg.groupby(group_cols, sort=False).agg(
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        R2_seed_sd_mean=("R2_seed_sd", "mean"),
        RMSE_seed_sd_mean=("RMSE_seed_sd", "mean"),
    ).reset_index()
    sm["Validation Seeds"] = repeats
    sm = sort_cv_results(sm, ["Graph vector size", "Method"])
    return fm, sm, embedding_cache


def paired_bootstrap_architecture(architecture_pred_df, *, B=BOOTSTRAP_REPS, seed=SEED+19090):
    """Paired bootstrap for the key architecture contrasts.

    The contrasts follow the manuscript mechanism sequence rather than treating every
    method pair as an independent hypothesis:
      1) Local vs Global: contribution of comparable retrieval;
      2) Tuned Global+Local vs Local: contribution of adding a broad benchmark through
         a non-Bayesian fusion rule;
      3) Bayes vs Tuned non-Bayes: incremental contribution of reliability-sensitive
         Bayesian reconciliation over a trained convex fusion benchmark.

    KFold resamples OOF observations; Group KFold resamples supplier clusters.
    """
    if architecture_pred_df is None or architecture_pred_df.empty:
        return pd.DataFrame()
    specs = [
        ("Local vs Global", "local_gap", "global",
         "Local (KNN GapTrim)", "Global (Ridge)"),
        ("Tuned Global+Local vs Local", "nonbayes_tuned", "local_gap",
         "Global + Local (tuned convex, non-Bayes)", "Local (KNN GapTrim)"),
        ("Bayes vs Tuned non-Bayes", "bayes_gap", "nonbayes_tuned",
         "Bayes (GapTrim)", "Global + Local (tuned convex, non-Bayes)"),
    ]
    out = []
    for cv in ["KFold", "Group KFold"]:
        d = architecture_pred_df[architecture_pred_df["CV Protocol"] == cv].copy()
        if d.empty:
            continue
        y = d["y_true"].to_numpy(float)
        for label, cand_col, base_col, cand_name, base_name in specs:
            pa = d[cand_col].to_numpy(float)
            pb = d[base_col].to_numpy(float)
            dr2, drmse = validation_metric_pair(y, pa, pb)
            stable = zlib.crc32(f"architecture|{cv}|{label}".encode("utf-8")) % 1_000_000
            rng = np.random.default_rng(seed + stable)
            vals = []
            if cv == "Group KFold":
                clusters = d["supplier"].astype(str).to_numpy()
                uniq = np.unique(clusters)
                cluster_to_idx = {g: np.where(clusters == g)[0] for g in uniq}
                for _ in range(B):
                    gs = rng.choice(uniq, size=len(uniq), replace=True)
                    idx = np.concatenate([cluster_to_idx[g] for g in gs])
                    vals.append(validation_metric_pair(y[idx], pa[idx], pb[idx]))
                boot_type = "supplier-cluster paired bootstrap"
            else:
                n = len(d)
                for _ in range(B):
                    idx = rng.integers(0, n, n)
                    vals.append(validation_metric_pair(y[idx], pa[idx], pb[idx]))
                boot_type = "observation paired bootstrap"
            arr = np.asarray(vals, float)
            out.append({
                "CV Protocol": cv,
                "Comparison": label,
                "Method stage": "architecture",
                "Candidate": cand_name,
                "Baseline": base_name,
                "N": len(d),
                "Bootstrap B": B,
                "Bootstrap type": boot_type,
                "Delta R2 (Candidate-Baseline)": dr2,
                "Delta R2 CI2.5": np.quantile(arr[:, 0], .025),
                "Delta R2 CI97.5": np.quantile(arr[:, 0], .975),
                "Delta RMSE (Candidate-Baseline)": drmse,
                "Delta RMSE CI2.5": np.quantile(arr[:, 1], .025),
                "Delta RMSE CI97.5": np.quantile(arr[:, 1], .975),
            })
    return sort_cv_results(pd.DataFrame(out), ["Comparison"])


def run_structural_validation(primary_results, graph_validation_repeats=GRAPH_VALIDATION_REPEATS):
    """Run structural/mechanism validation after the main empirical evidence.

    This stage deliberately excludes bootstrap inference and dimension sensitivity.
    Those are statistical-robustness checks and are run only after the structural
    contribution of the LCMA has been established.

    Structural validation contains three distinct questions:
      1) representation contribution: Text vs KG vs KG+Text;
      2) architecture contribution: Global vs Local vs non-Bayesian fusion vs Bayes;
      3) supplier-topology transfer: matched-seed full vs supplier-neutral KG.
    """
    ensure_dir(str(VALIDATION_OUT))
    protocols = primary_results["protocols"]
    primary_fold_cache = primary_results["primary_fold_cache"]

    # Protocol integrity is enforced as a hard condition rather than exported as a
    # separate audit table. Graph/text learners receive outer-training data only.
    for cv_name, dcv, splits in protocols:
        for fold, (tr_idx, te_idx) in enumerate(splits, 1):
            if set(map(int, tr_idx)).intersection(set(map(int, te_idx))):
                raise AssertionError(f"{cv_name} fold {fold}: train/test row overlap detected.")
            if cv_name == "Group KFold":
                tr_sup = set(dcv.iloc[tr_idx][GROUP_COL].astype(str))
                te_sup = set(dcv.iloc[te_idx][GROUP_COL].astype(str))
                if not tr_sup.isdisjoint(te_sup):
                    raise AssertionError(f"Group KFold fold {fold}: supplier leakage detected.")

    primary_metrics, primary_preds = primary_baseline_from_cache(protocols, primary_fold_cache)
    metrics, pred_long, alpha_rows, architecture_pred_rows = list(primary_metrics), list(primary_preds), [], []

    # ------------------------------------------------------------------
    # 3.1 Controlled representation contribution.
    # ------------------------------------------------------------------
    for cv_name, dcv, splits in protocols:  # KFold -> Group KFold
        for fold, (tr_idx, te_idx) in enumerate(splits, 1):
            print(f"[representation + architecture] {cv_name} fold {fold}/{len(splits)}")
            tr_df = dcv.iloc[tr_idx].reset_index(drop=True)
            te_df = dcv.iloc[te_idx].reset_index(drop=True)
            entry = primary_fold_cache[(cv_name, fold)]
            primary_comp = entry["components"]
            primary_z = entry["z"]

            # Text-only and KG-only reuse the exact T and G blocks of the primary run.
            z_text = fit_stage_models(tr_df, te_df, "text", components={"text": primary_comp["text"]})
            z_kg_only = fit_stage_models(tr_df, te_df, "kg", components={"graph": primary_comp["graph"]})

            for rep, z in [("Text only", z_text), ("KG only", z_kg_only)]:
                for method, key, lo, hi in [
                    ("Global (Ridge)", "global", None, None),
                    ("Local (KNN Mean)", "local_mean", None, None),
                    ("Local (KNN GapTrim)", "local_gap", None, None),
                    ("Bayes (Mean)", "bayes_mean", "bayes_mean_low", "bayes_mean_high"),
                    ("Bayes (GapTrim)", "bayes_gap", "bayes_gap_low", "bayes_gap_high"),
                ]:
                    metrics.append(validation_metric(
                        cv_name, fold, rep, method, te_df, z[key],
                        z[lo] if lo else None, z[hi] if hi else None,
                    ))
                for i in range(len(te_df)):
                    pred_long.append({
                        "CV Protocol": cv_name, "Fold": fold, "row_id": int(te_idx[i]),
                        "supplier": str(te_df.loc[i, GROUP_COL]),
                        "y_true": z["y_true"][i], "price_true": z["price_true"][i],
                        "Representation": rep, "global": z["global"][i],
                        "local_gap": z["local_gap"][i], "bayes_gap": z["bayes_gap"][i],
                    })

            # ------------------------------------------------------------------
            # 3.2 Architecture contribution. Tuning remains inside outer training.
            # ------------------------------------------------------------------
            alpha, qe, qt, crmse = fit_nonbayes_calibration(
                tr_df, cv_name, validation_common_seed(cv_name, fold, repeat=0, salt=900000)
            )
            pe = .5 * primary_z["global"] + .5 * primary_z["local_gap"]
            pt = (1-alpha) * primary_z["global"] + alpha * primary_z["local_gap"]
            metrics.append(validation_metric(
                cv_name, fold, "KG + Text", "Global + Local (50/50, non-Bayes)",
                te_df, pe, pe-qe, pe+qe,
            ))
            metrics.append(validation_metric(
                cv_name, fold, "KG + Text", "Global + Local (tuned convex, non-Bayes)",
                te_df, pt, pt-qt, pt+qt,
            ))
            alpha_rows.append({
                "CV Protocol": cv_name, "Fold": fold, "Local weight alpha": alpha,
                "Calibration RMSE": crmse, "Equal-blend conformal q95": qe,
                "Tuned-blend conformal q95": qt,
            })
            for i in range(len(te_df)):
                architecture_pred_rows.append({
                    "CV Protocol": cv_name, "Fold": fold, "row_id": int(te_idx[i]),
                    "supplier": str(te_df.loc[i, GROUP_COL]),
                    "y_true": float(primary_z["y_true"][i]),
                    "price_true": float(primary_z["price_true"][i]),
                    "global": float(primary_z["global"][i]),
                    "local_gap": float(primary_z["local_gap"][i]),
                    "nonbayes_equal": float(pe[i]),
                    "nonbayes_tuned": float(pt[i]),
                    "bayes_gap": float(primary_z["bayes_gap"][i]),
                    "local_weight_alpha": float(alpha),
                })

    fm = sort_cv_results(pd.DataFrame(metrics), ["Fold", "Representation", "Method"])
    pl = sort_cv_results(pd.DataFrame(pred_long), ["Fold", "row_id", "Representation"])
    al = sort_cv_results(pd.DataFrame(alpha_rows), ["Fold"])
    arch_pred = sort_cv_results(pd.DataFrame(architecture_pred_rows), ["Fold", "row_id"])
    sm = summarize_validation_metrics(fm)

    rep_methods = ["Global (Ridge)", "Local (KNN GapTrim)", "Bayes (GapTrim)"]
    rep_order = ["Text only", "KG only", "KG + Text"]
    rep_table = sm[sm["Method"].isin(rep_methods) & sm["Representation"].isin(rep_order)].copy()
    rep_table["__Rep_Order"] = rep_table["Representation"].map({r:i for i,r in enumerate(rep_order,1)})
    rep_table["__Method_Order"] = rep_table["Method"].map({m:i for i,m in enumerate(rep_methods,1)})
    rep_table = sort_cv_results(rep_table, ["__Rep_Order", "__Method_Order"]).drop(
        columns=["__Rep_Order", "__Method_Order"]
    )

    arch_methods = [
        "Global (Ridge)", "Local (KNN GapTrim)",
        "Global + Local (50/50, non-Bayes)",
        "Global + Local (tuned convex, non-Bayes)", "Bayes (GapTrim)",
    ]
    arch = sm[(sm["Representation"] == "KG + Text") & sm["Method"].isin(arch_methods)].copy()
    arch["__Method_Order"] = arch["Method"].map({m:i for i,m in enumerate(arch_methods,1)})
    arch = sort_cv_results(arch, ["__Method_Order"]).drop(columns=["__Method_Order"])

    # ------------------------------------------------------------------
    # 3.3 Supplier-topology transfer diagnostic under matched repeated seeds.
    # ------------------------------------------------------------------
    graph_cache = {}
    topo_fm, topo_sm, topo_pred_all, topo_pred_avg, graph_cache = run_supplier_topology_validation(
        protocols, primary_fold_cache, repeats=graph_validation_repeats,
        embedding_cache=graph_cache,
    )
    topo_methods = ["Global (Ridge)", "Local (KNN GapTrim)", "Bayes (GapTrim)"]
    topo_order = [
        "KG full (matched-seed)", "KG supplier-neutral (matched-seed)",
        "Text + KG full (matched-seed)", "Text + KG supplier-neutral (matched-seed)",
    ]
    topo_table = topo_sm[topo_sm["Method"].isin(topo_methods) & topo_sm["Representation"].isin(topo_order)].copy()
    topo_table["__Rep_Order"] = topo_table["Representation"].map({r:i for i,r in enumerate(topo_order,1)})
    topo_table["__Method_Order"] = topo_table["Method"].map({m:i for i,m in enumerate(topo_methods,1)})
    topo_table = sort_cv_results(topo_table, ["__Rep_Order", "__Method_Order"]).drop(
        columns=["__Rep_Order", "__Method_Order"]
    )

    # Preserve structural/mechanism intermediate outputs separately from robustness.
    fm.to_csv(VALIDATION_OUT / "I20_Structural_Validation_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    sm.to_csv(VALIDATION_OUT / "I21_Structural_Validation_Summary.csv", index=False, encoding="utf-8-sig")
    rep_table.to_csv(VALIDATION_OUT / "I22_Representation_Ablation.csv", index=False, encoding="utf-8-sig")
    arch.to_csv(VALIDATION_OUT / "I23_Architecture_Ablation.csv", index=False, encoding="utf-8-sig")
    pl.to_csv(VALIDATION_OUT / "I24_Representation_OOF_Predictions.csv", index=False, encoding="utf-8-sig")
    arch_pred.to_csv(VALIDATION_OUT / "I25_Architecture_OOF_Predictions.csv", index=False, encoding="utf-8-sig")
    al.to_csv(VALIDATION_OUT / "I26_NonBayesian_Fusion_Calibration.csv", index=False, encoding="utf-8-sig")
    topo_fm.to_csv(VALIDATION_OUT / "I27_Supplier_Topology_MatchedSeed_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    topo_sm.to_csv(VALIDATION_OUT / "I28_Supplier_Topology_MatchedSeed_Summary.csv", index=False, encoding="utf-8-sig")
    topo_table.to_csv(VALIDATION_OUT / "I29_Supplier_Neutral_Topology_Validation.csv", index=False, encoding="utf-8-sig")
    topo_pred_all.to_csv(VALIDATION_OUT / "I30_Supplier_Topology_MatchedSeed_OOF_AllSeeds.csv", index=False, encoding="utf-8-sig")
    topo_pred_avg.to_csv(VALIDATION_OUT / "I31_Supplier_Topology_MatchedSeed_OOF_SeedAveraged.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(VALIDATION_OUT / "Structural_Validation_Results.xlsx", engine="openpyxl") as w:
        rep_table.to_excel(w, sheet_name="Representation", index=False)
        arch.to_excel(w, sheet_name="Architecture", index=False)
        topo_table.to_excel(w, sheet_name="Supplier Topology", index=False)
        al.to_excel(w, sheet_name="Fusion Calibration", index=False)
        fm.to_excel(w, sheet_name="Fold Metrics", index=False)
        topo_fm.to_excel(w, sheet_name="Topology FoldSeed", index=False)

    return {
        "summary": sm,
        "representation": rep_table,
        "architecture": arch,
        "topology": topo_table,
        "topology_summary": topo_sm,
        "fold_metrics": fm,
        "predictions": pl,
        "architecture_predictions": arch_pred,
        "fusion_calibration": al,
        "topology_pred_all": topo_pred_all,
        "topology_pred_avg": topo_pred_avg,
        "graph_cache": graph_cache,
    }


def run_statistical_robustness(primary_results, structural_results, *, bootstrap_B=BOOTSTRAP_REPS,
                               dimension_sensitivity=True,
                               graph_validation_repeats=GRAPH_VALIDATION_REPEATS):
    """Run statistical inference and sensitivity only after structural validation."""
    ensure_dir(str(VALIDATION_OUT))
    protocols = primary_results["protocols"]
    full_run = (len(protocols[0][2]) == N_SPLITS and len(protocols[1][2]) == N_SPLITS)

    if full_run:
        boot_component = paired_bootstrap_comparisons(
            structural_results["predictions"], structural_results["topology_pred_avg"], B=bootstrap_B
        )
        boot_arch = paired_bootstrap_architecture(
            structural_results["architecture_predictions"], B=bootstrap_B
        )
        # Keep bootstrap outputs in manuscript order rather than alphabetical order.
        if not boot_component.empty:
            cmp_order = boot_component["Comparison"].map(
                REPRESENTATION_COMPARISON_ORDER
            )

            topology_mask = boot_component["Comparison"].eq(
                "Text+Neutral KG vs Text+Full KG (matched-seed)"
            )

            cmp_order = cmp_order.mask(
                cmp_order.isna() & topology_mask,
                3
            ).fillna(99)

            boot_component["__cmp"] = cmp_order.astype(int)

            boot_component["__stage"] = (
                boot_component["Method stage"]
                .map(STAGE_ORDER)
                .fillna(99)
                .astype(int)
            )

            boot_component = sort_cv_results(
                boot_component,
                ["__cmp", "__stage"]
            ).drop(columns=["__cmp", "__stage"])


        if not boot_arch.empty:
            boot_arch["__cmp"] = boot_arch["Comparison"].map(ARCHITECTURE_COMPARISON_ORDER).fillna(99)
            boot_arch = sort_cv_results(boot_arch, ["__cmp"]).drop(columns=["__cmp"])
        boot = pd.concat([boot_component, boot_arch], ignore_index=True, sort=False)
    else:
        boot_component = boot_arch = boot = pd.DataFrame()

    dim_fm = dim_sm = pd.DataFrame()
    graph_cache = structural_results.get("graph_cache", {})
    if dimension_sensitivity:
        dim_fm, dim_sm, graph_cache = embedding_dimension_sensitivity(
            protocols, repeats=graph_validation_repeats,
            embedding_cache=graph_cache,
        )

    if not boot.empty:
        boot.to_csv(VALIDATION_OUT / "I32_Paired_Bootstrap_Representation_Comparisons.csv", index=False, encoding="utf-8-sig")
        boot_arch.to_csv(VALIDATION_OUT / "I33_Paired_Bootstrap_Architecture_Comparisons.csv", index=False, encoding="utf-8-sig")
        neutral_boot = boot_component[
            boot_component["Comparison"].str.contains("Neutral|neutral", regex=True, na=False)
        ].copy()
        neutral_boot.to_csv(VALIDATION_OUT / "I34_Paired_Bootstrap_Supplier_Topology.csv", index=False, encoding="utf-8-sig")
    else:
        neutral_boot = pd.DataFrame()

    if not dim_sm.empty:
        dim_fm.to_csv(VALIDATION_OUT / "I35_Embedding_Dimension_FoldSeed_Metrics.csv", index=False, encoding="utf-8-sig")
        dim_sm.to_csv(VALIDATION_OUT / "I36_Embedding_Dimension_Sensitivity.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(VALIDATION_OUT / "Statistical_Robustness_Results.xlsx", engine="openpyxl") as w:
        pd.DataFrame([{
            "Full 5-fold run": bool(full_run),
            "Paired bootstrap generated": bool(not boot.empty),
            "Dimension sensitivity generated": bool(not dim_sm.empty),
        }]).to_excel(w, sheet_name="Run Info", index=False)
        if not boot.empty:
            boot.to_excel(w, sheet_name="Paired Bootstrap", index=False)
            boot_arch.to_excel(w, sheet_name="Architecture Bootstrap", index=False)
        if not dim_sm.empty:
            dim_sm.to_excel(w, sheet_name="Dimension Sensitivity", index=False)
            dim_fm.to_excel(w, sheet_name="Dimension FoldSeed", index=False)

    # Backward-compatible consolidated validation workbook.
    with pd.ExcelWriter(VALIDATION_OUT / "Intermediate_Validation_Results.xlsx", engine="openpyxl") as w:
        structural_results["fold_metrics"].to_excel(w, sheet_name="Fold Metrics", index=False)
        structural_results["summary"].to_excel(w, sheet_name="Structural Summary", index=False)
        structural_results["representation"].to_excel(w, sheet_name="Representation", index=False)
        structural_results["architecture"].to_excel(w, sheet_name="Architecture", index=False)
        structural_results["fusion_calibration"].to_excel(w, sheet_name="Fusion Calibration", index=False)
        structural_results["topology"].to_excel(w, sheet_name="Supplier Topology", index=False)
        if not boot.empty:
            boot.to_excel(w, sheet_name="Paired Bootstrap", index=False)
        if not dim_sm.empty:
            dim_sm.to_excel(w, sheet_name="Dim Sensitivity", index=False)

    return {
        "bootstrap": boot,
        "bootstrap_component": boot_component,
        "bootstrap_architecture": boot_arch,
        "dimension_summary": dim_sm,
        "dimension_fold_metrics": dim_fm,
    }


def build_final_empirical_results(primary_results, structural_results, robustness_results):
    """Build a manuscript-aligned unified result table, with KFold before Group KFold.

    The table follows the empirical argument rather than the historical order in which
    code modules were developed: main empirical evidence -> structural/mechanism
    validation -> statistical robustness and sensitivity.
    """
    rows = []
    primary = primary_results["numeric_summary"]

    def add_performance_row(r, stage, section, block, representation, order):
        rows.append({
            "CV Protocol": r["CV Protocol"],
            "Empirical Stage": stage,
            "Manuscript Section": section,
            "Result Block": block,
            "Representation / Comparison": representation,
            "Method Stage": r["Method"],
            "R2 Mean": r.get("R2_mean", np.nan), "R2 SD": r.get("R2_sd", np.nan),
            "RMSE Mean": r.get("RMSE_mean", np.nan), "RMSE SD": r.get("RMSE_sd", np.nan),
            "WAPE Mean": r.get("WAPE_mean", np.nan),
            "Coverage Mean": r.get("Coverage_mean", np.nan),
            "Width Mean": r.get("Width_mean", np.nan),
            "Record Order": order,
        })

    # ------------------------------------------------------------------
    # Main empirical evidence: one section per LCMA analytical function.
    # ------------------------------------------------------------------
    main_specs = [
        ("Prior (Ridge)",
         "I. Main empirical evidence", "5.1", "A1. Global benchmark and cold-start fragility", 1),
        ("KNN (Mean)",
         "I. Main empirical evidence", "5.2", "A2. Local comparable retrieval and cold-start recovery", 1),
        ("KNN (GapTrim)",
         "I. Main empirical evidence", "5.2", "A2. Local comparable retrieval and cold-start recovery", 2),
        ("Bayes (Mean)",
         "I. Main empirical evidence", "5.3", "A3. Bayesian reconciliation and uncertainty-aware benchmarking", 1),
        ("Bayes (GapTrim)",
         "I. Main empirical evidence", "5.3", "A3. Bayesian reconciliation and uncertainty-aware benchmarking", 2),
    ]
    for method, stage, section, block, order in main_specs:
        d = primary[primary["Method"] == method]
        for _, r in d.iterrows():
            add_performance_row(r, stage, section, block, "KG + Text", order)

    # ------------------------------------------------------------------
    # Structural/mechanism validation.
    # ------------------------------------------------------------------
    rep = structural_results["representation"]
    rep_order = {"Text only": 1, "KG only": 2, "KG + Text": 3}
    stage_order = {"Global (Ridge)": 1, "Local (KNN GapTrim)": 2, "Bayes (GapTrim)": 3}
    for _, r in rep.iterrows():
        add_performance_row(
            r, "II. Structural/mechanism validation", "5.4.1",
            "B1. Representation contribution", r["Representation"],
            rep_order.get(r["Representation"], 99) * 10 + stage_order.get(r["Method"], 9),
        )

    arch = structural_results["architecture"]
    arch_order = {m: i for i, m in enumerate([
        "Global (Ridge)", "Local (KNN GapTrim)",
        "Global + Local (50/50, non-Bayes)",
        "Global + Local (tuned convex, non-Bayes)",
        "Bayes (GapTrim)",
    ], 1)}
    for _, r in arch.iterrows():
        add_performance_row(
            r, "II. Structural/mechanism validation", "5.4.2",
            "B2. Architecture contribution", "KG + Text",
            arch_order.get(r["Method"], 99),
        )

    topo = structural_results.get("topology", pd.DataFrame())
    topo_order = {
        "KG full (matched-seed)": 1,
        "KG supplier-neutral (matched-seed)": 2,
        "Text + KG full (matched-seed)": 3,
        "Text + KG supplier-neutral (matched-seed)": 4,
    }
    for _, r in topo.iterrows():
        row = {
            "CV Protocol": r["CV Protocol"],
            "Empirical Stage": "II. Structural/mechanism validation",
            "Manuscript Section": "Appendix D3",
            "Result Block": "D3. Supplier-topology transfer diagnostic",
            "Representation / Comparison": r["Representation"],
            "Method Stage": r["Method"],
            "R2 Mean": r["R2_mean"], "R2 SD": r["R2_sd"],
            "RMSE Mean": r["RMSE_mean"], "RMSE SD": r["RMSE_sd"],
            "WAPE Mean": r.get("WAPE_mean", np.nan),
            "Coverage Mean": r.get("Coverage_mean", np.nan),
            "Width Mean": r.get("Width_mean", np.nan),
            "R2 Seed SD Mean": r.get("R2_seed_sd_mean", np.nan),
            "RMSE Seed SD Mean": r.get("RMSE_seed_sd_mean", np.nan),
            "Record Order": topo_order.get(r["Representation"], 99) * 10 + stage_order.get(r["Method"], 9),
        }
        rows.append(row)

    # ------------------------------------------------------------------
    # Statistical robustness and sensitivity.
    # ------------------------------------------------------------------
    boot = robustness_results.get("bootstrap", pd.DataFrame())
    if boot is not None and not boot.empty:
        comparison_order = {name: i for i, name in enumerate(boot["Comparison"].drop_duplicates(), 1)}
        method_order = {"global": 1, "local_gap": 2, "bayes_gap": 3, "architecture": 4}
        for _, r in boot.iterrows():
            rows.append({
                "CV Protocol": r["CV Protocol"],
                "Empirical Stage": "III. Statistical robustness and sensitivity",
                "Manuscript Section": "Appendix D2",
                "Result Block": "D2. Paired-bootstrap inference",
                "Representation / Comparison": r["Comparison"],
                "Method Stage": r["Method stage"],
                "Delta R2": r["Delta R2 (Candidate-Baseline)"],
                "Delta R2 CI Low": r["Delta R2 CI2.5"],
                "Delta R2 CI High": r["Delta R2 CI97.5"],
                "Delta RMSE": r["Delta RMSE (Candidate-Baseline)"],
                "Delta RMSE CI Low": r["Delta RMSE CI2.5"],
                "Delta RMSE CI High": r["Delta RMSE CI97.5"],
                "Bootstrap Type": r["Bootstrap type"], "N": r["N"],
                "Record Order": comparison_order.get(r["Comparison"], 99) * 10 + method_order.get(r["Method stage"], 9),
            })

    # The original Plus-Bayes / conformal extensions are retained, but conceptually
    # reported after the structural validation as risk-interval calibration checks.
    calibration_methods = {
        "Plus Bayes (Robust Cov)": 1,
        "Stacking (Conformal)": 2,
    }
    for method, order in calibration_methods.items():
        d = primary[primary["Method"] == method]
        for _, r in d.iterrows():
            add_performance_row(
                r, "III. Predictive-uncertainty robustness", "5.5 / Appendix E",
                "E1. Predictive-uncertainty robustness and interval calibration", "KG + Text", order,
            )

    dim = robustness_results.get("dimension_summary", pd.DataFrame())
    if dim is not None and not dim.empty:
        dim_method_order = {"Global (Ridge)": 1, "Local (KNN GapTrim)": 2, "Bayes (GapTrim)": 3}
        for _, r in dim.iterrows():
            rows.append({
                "CV Protocol": r["CV Protocol"],
                "Empirical Stage": "III. Statistical robustness and sensitivity",
                "Manuscript Section": "Appendix D4",
                "Result Block": "D4. Graph-embedding dimension sensitivity",
                "Representation / Comparison": f"KG+Text dim={int(r['Graph vector size'])}",
                "Method Stage": r["Method"],
                "Graph Vector Size": int(r["Graph vector size"]),
                "R2 Mean": r["R2_mean"], "R2 SD": r["R2_sd"],
                "RMSE Mean": r["RMSE_mean"], "RMSE SD": r["RMSE_sd"],
                "R2 Seed SD Mean": r.get("R2_seed_sd_mean", np.nan),
                "RMSE Seed SD Mean": r.get("RMSE_seed_sd_mean", np.nan),
                "Validation Seeds": r.get("Validation Seeds", np.nan),
                "Record Order": int(r["Graph vector size"]) * 10 + dim_method_order.get(r["Method"], 9),
            })

    out = pd.DataFrame(rows)
    block_order = {
        "A1. Global benchmark and cold-start fragility": 1,
        "A2. Local comparable retrieval and cold-start recovery": 2,
        "A3. Bayesian reconciliation and uncertainty-aware benchmarking": 3,
        "B1. Representation contribution": 4,
        "B2. Architecture contribution": 5,
        "D3. Supplier-topology transfer diagnostic": 6,
        "D2. Paired-bootstrap inference": 7,
        "E1. Predictive-uncertainty robustness and interval calibration": 8,
        "D4. Graph-embedding dimension sensitivity": 9,
    }
    out["__CV"] = out["CV Protocol"].map(CV_PROTOCOL_ORDER).fillna(99)
    out["__Block"] = out["Result Block"].map(block_order).fillna(99)
    out = out.sort_values(["__CV", "__Block", "Record Order"], kind="stable").drop(
        columns=["__CV", "__Block", "Record Order"]
    ).reset_index(drop=True)
    return out


def export_manuscript_empirical_sequence(primary_results, structural_results, robustness_results, final_table):
    """Export section-aligned views without deleting any original/intermediate output."""
    ensure_dir(str(MANUSCRIPT_SEQUENCE_OUT))
    primary = primary_results["numeric_summary"]

    section_tables = {
        "5.1 Global Benchmark": primary[primary["Method"] == "Prior (Ridge)"].copy(),
        "5.2 Local Retrieval": primary[primary["Method"].isin(["KNN (Mean)", "KNN (GapTrim)"])].copy(),
        "5.3 Bayesian Reconciliation": primary[primary["Method"].isin(["Bayes (Mean)", "Bayes (GapTrim)"])].copy(),
        "5.4.1 Representation": structural_results["representation"].copy(),
        "5.4.2 Architecture": structural_results["architecture"].copy(),
        "Appendix D3 Supplier Topology": structural_results["topology"].copy(),
        "Appendix D2 Paired Bootstrap": robustness_results.get("bootstrap", pd.DataFrame()).copy(),
        "5.5 / Appendix E Interval Calibration": primary[primary["Method"].isin([
            "Plus Bayes (Robust Cov)", "Stacking (Conformal)"
        ])].copy(),
        "Appendix D4 Dimension Sensitivity": robustness_results.get("dimension_summary", pd.DataFrame()).copy(),
    }

    file_names = {
        "5.1 Global Benchmark": "Section_5_1_Global_Benchmark.csv",
        "5.2 Local Retrieval": "Section_5_2_Local_Comparable_Retrieval.csv",
        "5.3 Bayesian Reconciliation": "Section_5_3_Bayesian_Reconciliation.csv",
        "5.4.1 Representation": "Section_5_4_1_Representation_Contribution.csv",
        "5.4.2 Architecture": "Section_5_4_2_Architecture_Contribution.csv",
        "Appendix D3 Supplier Topology": "Appendix_D3_Supplier_Topology_Transfer.csv",
        "Appendix D2 Paired Bootstrap": "Appendix_D2_Paired_Bootstrap_Inference.csv",
        "5.5 / Appendix E Interval Calibration": "Section_5_5_Predictive_Uncertainty_Robustness.csv",
        "Appendix D4 Dimension Sensitivity": "Appendix_D4_Graph_Embedding_Dimension_Sensitivity.csv",
    }
    for name, df in section_tables.items():
        if df is not None and not df.empty:
            df2 = df.copy()
            if name == "Appendix D2 Paired Bootstrap":
                # Combined audit view: representation contrasts first, architecture contrasts second;
                # within representation contrasts enforce Global -> Local -> Bayes.
                df2["__block"] = np.where(df2["Comparison"].isin(REPRESENTATION_COMPARISON_ORDER), 1,
                                    np.where(df2["Comparison"].isin(ARCHITECTURE_COMPARISON_ORDER), 2,
                                    np.where(df2["Comparison"].eq("Text+Neutral KG vs Text+Full KG (matched-seed)"), 3, 99)))
                df2["__cmp"] = df2["Comparison"].map({**REPRESENTATION_COMPARISON_ORDER, **ARCHITECTURE_COMPARISON_ORDER}).fillna(99)
                df2["__stage"] = df2.get("Method stage", pd.Series(index=df2.index, dtype=object)).map(STAGE_ORDER).fillna(99)
                df2 = sort_cv_results(df2, ["__block", "__cmp", "__stage"]).drop(columns=["__block", "__cmp", "__stage"])
            else:
                # Explicit method order where available; otherwise preserve manuscript/numeric order.
                if "Method" in df2.columns:
                    df2["__method"] = df2["Method"].map(METHOD_ORDER).fillna(99)
                    extra = [c for c in ["Representation", "Graph vector size"] if c in df2.columns] + ["__method"]
                    df2 = sort_cv_results(df2, extra).drop(columns=["__method"])
                else:
                    sort_cols = [c for c in ["Representation", "Graph vector size"] if c in df2.columns]
                    df2 = sort_cv_results(df2, sort_cols)
            df2.to_csv(MANUSCRIPT_SEQUENCE_OUT / file_names[name], index=False, encoding="utf-8-sig")
            section_tables[name] = df2

    # Manuscript-ready compact Table 3: LCMA representation and architecture ablation.
    # Detailed bootstrap, topology, and dimension diagnostics belong to Appendix D.
    rep_main = structural_results["representation"].copy()
    rep_main["Panel"] = "Panel A: Representation contribution"
    rep_main["Specification"] = rep_main["Representation"]
    arch_main = structural_results["architecture"].copy()
    arch_main["Panel"] = "Panel B: Architecture contribution"
    arch_main["Specification"] = "KG + Text"
    compact = pd.concat([rep_main, arch_main], ignore_index=True, sort=False)
    keys = ["Panel", "Specification", "Method"]
    k = compact[compact["CV Protocol"] == "KFold"][keys + ["R2_mean", "RMSE_mean"]].copy()
    g = compact[compact["CV Protocol"] == "Group KFold"][keys + ["R2_mean", "RMSE_mean"]].copy()
    k = k.rename(columns={"R2_mean": "KFold R2", "RMSE_mean": "KFold RMSE"})
    g = g.rename(columns={"R2_mean": "Group KFold R2", "RMSE_mean": "Group KFold RMSE"})
    manuscript_table3 = k.merge(g, on=keys, how="outer")
    panel_order = {"Panel A: Representation contribution": 1, "Panel B: Architecture contribution": 2}
    rep_order = {"Text only": 1, "KG only": 2, "KG + Text": 3}
    arch_method_order = {m:i for i,m in enumerate([
        "Global (Ridge)", "Local (KNN GapTrim)",
        "Global + Local (50/50, non-Bayes)",
        "Global + Local (tuned convex, non-Bayes)", "Bayes (GapTrim)"
    ], 1)}
    manuscript_table3["__Panel"] = manuscript_table3["Panel"].map(panel_order).fillna(99)
    manuscript_table3["__Spec"] = manuscript_table3["Specification"].map(rep_order).fillna(99)
    manuscript_table3["__Method"] = manuscript_table3["Method"].map(arch_method_order).fillna(99)
    manuscript_table3 = manuscript_table3.sort_values(
        ["__Panel", "__Spec", "__Method"], kind="stable"
    ).drop(columns=["__Panel", "__Spec", "__Method"]).reset_index(drop=True)
    manuscript_table3.to_csv(
        MANUSCRIPT_SEQUENCE_OUT / "Manuscript_Table_3_LCMA_Ablation.csv",
        index=False, encoding="utf-8-sig"
    )

    # Appendix Table E1: predictive-uncertainty robustness and interval calibration.
    appendix_table_e1_compact = primary[primary["Method"].isin([
        "Bayes (Mean)", "Bayes (GapTrim)",
        "Plus Bayes (Robust Cov)", "Stacking (Conformal)"
    ])].copy()
    calibration_order = {
        "Bayes (Mean)": 1, "Bayes (GapTrim)": 2,
        "Plus Bayes (Robust Cov)": 3, "Stacking (Conformal)": 4,
    }
    appendix_table_e1_compact["__Method"] = appendix_table_e1_compact["Method"].map(calibration_order).fillna(99)
    appendix_table_e1_compact = sort_cv_results(appendix_table_e1_compact, ["__Method"]).drop(columns=["__Method"])
    appendix_table_e1_compact.to_csv(
        MANUSCRIPT_SEQUENCE_OUT / "Appendix_Table_E1_Predictive_Uncertainty_Robustness_and_Interval_Calibration.csv",
        index=False, encoding="utf-8-sig"
    )

    # Appendix-D audit diagnostics corresponding to the current manuscript appendix.
    structural_results["topology"].to_csv(
        MANUSCRIPT_SEQUENCE_OUT / "Appendix_D3_Supplier_Topology_Transfer_Diagnostic.csv",
        index=False, encoding="utf-8-sig"
    )
    if robustness_results.get("bootstrap") is not None and not robustness_results["bootstrap"].empty:
        robustness_results["bootstrap"].to_csv(
            MANUSCRIPT_SEQUENCE_OUT / "Appendix_D2_Paired_Bootstrap_Inference_All_Contrasts.csv",
            index=False, encoding="utf-8-sig"
        )
    if robustness_results.get("dimension_summary") is not None and not robustness_results["dimension_summary"].empty:
        robustness_results["dimension_summary"].to_csv(
            MANUSCRIPT_SEQUENCE_OUT / "Appendix_D4_Graph_Embedding_Dimension_Sensitivity.csv",
            index=False, encoding="utf-8-sig"
        )

    roadmap = pd.DataFrame([
        ["5.1", "Main empirical evidence", "Global benchmark and supplier cold-start fragility",
         "Does a broad market-wide mapping transfer to unseen suppliers?"],
        ["5.2", "Main empirical evidence", "Local comparable retrieval and cold-start recovery",
         "Does local comparable evidence recover viability when the global mapping is fragile?"],
        ["5.3", "Main empirical evidence", "Bayesian reconciliation and uncertainty-aware benchmarking",
         "How are broad and local evidence reconciled, and what uncertainty is reported?"],
        ["5.4.1", "Structural/mechanism validation", "Representation transferability",
         "What do Text, KG, and KG+Text contribute to the comparability representation?"],
        ["5.4.2", "Structural/mechanism validation", "Architecture decomposition",
         "What is contributed by Global, Local, non-Bayesian fusion, and Bayes?"],
        ["Appendix D3", "Appendix D diagnostics", "Supplier-topology transfer diagnostic",
         "Which relational information transfers under supplier shift?"],
        ["Appendix D2", "Appendix D diagnostics", "Paired-bootstrap inference",
         "Are the principal component and architecture differences stable under resampling?"],
        ["5.5 / Appendix E", "Predictive-uncertainty robustness", "Interval calibration",
         "How does interval coverage change under robust/conformal calibration?"],
        ["Appendix D4", "Appendix D diagnostics", "Graph-embedding dimension sensitivity",
         "Are conclusions stable to graph dimension and embedding stochasticity?"],
    ], columns=["Section", "Empirical Stage", "Purpose", "Question"])

    xlsx = MANUSCRIPT_SEQUENCE_OUT / "Manuscript_Empirical_Sequence.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        roadmap.to_excel(w, sheet_name="Roadmap", index=False)
        for name, df in section_tables.items():
            if df is not None and not df.empty:
                sheet = name.replace("/", "-")[:31]
                df.to_excel(w, sheet_name=sheet, index=False)
        manuscript_table3.to_excel(w, sheet_name="Table 3 LCMA Ablation", index=False)
        appendix_table_e1_compact.to_excel(w, sheet_name="Appendix E1 Calibration", index=False)
        final_table.to_excel(w, sheet_name="Unified Results", index=False)
    return xlsx


def build_appendix_b_from_primary_predictions(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Appendix B directly from the exact primary OOF prior predictions.

    This avoids a second independent refit and guarantees that the residual diagnostic
    uses the same global-prior predictions reported in the main results.
    """
    ensure_dir(OUT_MAIN)
    ensure_dir(OUT_FIG)
    rows, detail = [], []
    for cv in ["KFold", "Group KFold"]:
        d = pred_df[pred_df["CV Protocol"] == cv].copy()
        if d.empty:
            continue
        residual = d["y"].to_numpy(float) - d["Prior (Ridge)"].to_numpy(float)
        jb = stats.jarque_bera(residual)
        sw_n = min(len(residual), 5000)
        sw = stats.shapiro(residual[:sw_n]) if sw_n >= 3 else (np.nan, np.nan)
        alphas = sorted(pd.to_numeric(d.get("Prior Ridge Alpha", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist())
        rows.append({
            "CV Protocol": cv,
            "Obs.": int(len(residual)),
            "Mean": float(np.mean(residual)),
            "SD": float(np.std(residual, ddof=1)),
            "Skewness": float(stats.skew(residual, bias=False)),
            "Excess Kurtosis": float(stats.kurtosis(residual, fisher=True, bias=False)),
            "JB Statistic": float(jb.statistic),
            "JB p-value": float(jb.pvalue),
            "Shapiro Statistic": float(sw.statistic if hasattr(sw, "statistic") else sw[0]),
            "Shapiro p-value": float(sw.pvalue if hasattr(sw, "pvalue") else sw[1]),
            "Selected Ridge Alphas": ", ".join(f"{a:g}" for a in alphas),
        })
        dd = d[["CV Protocol", "Fold", "supplier", "y", "Prior (Ridge)"]].copy()
        dd["Residual"] = residual
        detail.append(dd)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].hist(residual, bins=40, edgecolor="white", alpha=0.85)
        axes[0].set_title(f"(a) {cv}: residual histogram")
        axes[0].set_xlabel("Global-prior residual on ln(price) scale")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)
        stats.probplot(residual, dist="norm", plot=axes[1])
        axes[1].set_title(f"(b) {cv}: normal Q–Q plot")
        axes[1].grid(linestyle="--", alpha=0.4)
        fig.tight_layout()
        stem = "Appendix_Fig_B1_Residual_Diagnostics_for_the_Global_Prior_under_KFold" if cv == "KFold" else "Appendix_Fig_B2_Residual_Diagnostics_for_the_Global_Prior_under_Group KFold"
        save_png_pdf(fig, stem, OUT_FIG)

    summary = pd.DataFrame(rows)
    residual_detail = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    summary_paper = format_appendix_b1_for_paper(summary)
    summary_paper.to_csv(Path(OUT_MAIN) / "I16_Appendix_Table_B1_Gaussian_Working_Approximation.csv", index=False, encoding="utf-8-sig")
    residual_detail.to_csv(Path(OUT_MAIN) / "I17_Appendix_B_Residual_Detail.csv", index=False, encoding="utf-8-sig")
    return summary_paper, residual_detail


def build_appendix_d_tables(structural_results, robustness_results) -> dict[str, pd.DataFrame]:
    """Construct Appendix D1–D8 for the manuscript ablation and representation diagnostics."""
    sm = structural_results["summary"].copy()
    rep_order = ["Text only", "KG only", "KG + Text"]
    rep_methods = ["Global (Ridge)", "Local (KNN Mean)", "Local (KNN GapTrim)", "Bayes (Mean)", "Bayes (GapTrim)"]
    c1 = sm[sm["Representation"].isin(rep_order) & sm["Method"].isin(rep_methods)].copy()
    c1["__r"] = c1["Representation"].map({v:i for i,v in enumerate(rep_order)})
    c1["__m"] = c1["Method"].map({v:i for i,v in enumerate(rep_methods)})
    c1 = sort_cv_results(c1, ["__r", "__m"]).drop(columns=["__r", "__m"])

    arch_methods = ["Global (Ridge)", "Local (KNN Mean)", "Local (KNN GapTrim)",
                    "Global + Local (50/50, non-Bayes)", "Global + Local (tuned convex, non-Bayes)",
                    "Bayes (Mean)", "Bayes (GapTrim)"]
    c2 = sm[(sm["Representation"] == "KG + Text") & sm["Method"].isin(arch_methods)].copy()
    c2["__m"] = c2["Method"].map({v:i for i,v in enumerate(arch_methods)})
    c2 = sort_cv_results(c2, ["__m"]).drop(columns=["__m"])

    c3 = structural_results["fusion_calibration"].copy().rename(columns={
        "Local weight alpha": "Tuned-convex local weight alpha",
        "Equal-blend conformal q95": "50/50 conformal q95",
        "Tuned-blend conformal q95": "Tuned conformal q95",
    })

    boot_component = robustness_results.get("bootstrap_component", pd.DataFrame()).copy()
    c4 = boot_component[boot_component["Comparison"].isin(["KG only vs Text", "KG+Text vs Text"])].copy() if not boot_component.empty else pd.DataFrame()
    if not c4.empty:
        c4["__cmp"] = c4["Comparison"].map(REPRESENTATION_COMPARISON_ORDER).fillna(99)
        c4["__stage"] = c4["Method stage"].map(STAGE_ORDER).fillna(99)
        c4 = sort_cv_results(c4, ["__cmp", "__stage"]).drop(columns=["__cmp", "__stage"])

    c5 = robustness_results.get("bootstrap_architecture", pd.DataFrame()).copy()
    if not c5.empty:
        c5["__cmp"] = c5["Comparison"].map(ARCHITECTURE_COMPARISON_ORDER).fillna(99)
        c5 = sort_cv_results(c5, ["__cmp"]).drop(columns=["__cmp"])

    topo_sm = structural_results.get("topology_summary", pd.DataFrame()).copy()
    topo_reps = ["Text + KG full (matched-seed)", "Text + KG supplier-neutral (matched-seed)"]
    topo_methods = ["Global (Ridge)", "Local (KNN GapTrim)", "Bayes (GapTrim)"]
    c6 = topo_sm[topo_sm["Representation"].isin(topo_reps) & topo_sm["Method"].isin(topo_methods)].copy() if not topo_sm.empty else pd.DataFrame()
    if not c6.empty:
        c6["__r"] = c6["Representation"].map({v:i for i,v in enumerate(topo_reps)})
        c6["__m"] = c6["Method"].map({v:i for i,v in enumerate(topo_methods)})
        c6 = sort_cv_results(c6, ["__r", "__m"]).drop(columns=["__r", "__m"])

    c7 = boot_component[boot_component["Comparison"] == "Text+Neutral KG vs Text+Full KG (matched-seed)"].copy() if not boot_component.empty else pd.DataFrame()
    if not c7.empty:
        c7["__stage"] = c7["Method stage"].map(STAGE_ORDER).fillna(99)
        c7 = sort_cv_results(c7, ["__stage"]).drop(columns=["__stage"])

    c8 = robustness_results.get("dimension_summary", pd.DataFrame()).copy()
    if not c8.empty:
        dim_col = "Graph vector size" if "Graph vector size" in c8.columns else "Graph-embedding dimension"
        c8 = c8.rename(columns={dim_col: "Graph-embedding dimension"})
        c8["__m"] = c8["Method"].map({"Global (Ridge)":1, "Local (KNN GapTrim)":2, "Bayes (GapTrim)":3}).fillna(99)
        c8 = sort_cv_results(c8, ["Graph-embedding dimension", "__m"]).drop(columns=["__m"])

    return {"D1": c1, "D2": c2, "D3": c3, "D4": c4, "D5": c5, "D6": c6, "D7": c7, "D8": c8}


def build_main_table3(structural_results: dict) -> pd.DataFrame:
    """Build manuscript Table 3: LCMA ablation, matching the current paper layout."""
    sm = structural_results["summary"].copy()
    rows = []

    def get(cv, rep, method):
        q = sm[(sm["CV Protocol"] == cv) & (sm["Representation"] == rep) & (sm["Method"] == method)]
        return None if q.empty else q.iloc[0]

    # Panel A. Representation ablation
    for rep, method, label in [
        ("Text only", "Global (Ridge)", "Text only — Global"),
        ("KG only", "Global (Ridge)", "KG only — Global"),
        ("KG + Text", "Global (Ridge)", "KG + Text — Global"),
        ("Text only", "Local (KNN GapTrim)", "Text only — Local GapTrim"),
        ("KG only", "Local (KNN GapTrim)", "KG only — Local GapTrim"),
        ("KG + Text", "Local (KNN GapTrim)", "KG + Text — Local GapTrim"),
        ("Text only", "Bayes (GapTrim)", "Text only — Bayes GapTrim"),
        ("KG only", "Bayes (GapTrim)", "KG only — Bayes GapTrim"),
        ("KG + Text", "Bayes (GapTrim)", "KG + Text — Bayes GapTrim"),
    ]:
        k, g = get("KFold", rep, method), get("Group KFold", rep, method)
        if k is not None and g is not None:
            rows.append({
                "Panel": "Panel A. Representation ablation",
                "Specification": label,
                "KFold R2": float(k["R2_mean"]),
                "Group KFold R2": float(g["R2_mean"]),
                "Group PI95 Coverage": np.nan,
                "Group Avg Width": np.nan,
            })

    # Panel B. Architecture ablation (KG + Text)
    arch_labels = [
        ("Global (Ridge)", "Global Ridge"),
        ("Local (KNN GapTrim)", "Local KNN GapTrim"),
        ("Global + Local (50/50, non-Bayes)", "50/50 fusion (non-Bayes)"),
        ("Global + Local (tuned convex, non-Bayes)", "Tuned convex fusion (non-Bayes)"),
        ("Bayes (GapTrim)", "Bayes reconciliation (GapTrim)"),
    ]
    for method, label in arch_labels:
        k, g = get("KFold", "KG + Text", method), get("Group KFold", "KG + Text", method)
        if k is not None and g is not None:
            rows.append({
                "Panel": "Panel B. Architecture ablation (KG + Text)",
                "Specification": label,
                "KFold R2": float(k["R2_mean"]),
                "Group KFold R2": float(g["R2_mean"]),
                "Group PI95 Coverage": float(g["Coverage_mean"]) if pd.notna(g.get("Coverage_mean", np.nan)) else np.nan,
                "Group Avg Width": float(g["Width_mean"]) if pd.notna(g.get("Width_mean", np.nan)) else np.nan,
            })
    return format_main_table3_for_paper(pd.DataFrame(rows))


def _format_mean_sd(df: pd.DataFrame, mean_col: str, sd_col: str, digits: int = 3) -> pd.Series:
    def f(r):
        m, sd = r.get(mean_col, np.nan), r.get(sd_col, np.nan)
        if pd.isna(m):
            return EM_DASH
        if pd.isna(sd):
            return f"{float(m):.{digits}f}"
        return f"{float(m):.{digits}f} ({float(sd):.{digits}f})"
    return df.apply(f, axis=1)


def _format_pct(x, digits=2):
    return EM_DASH if pd.isna(x) else f"{100.0*float(x):.{digits}f}%"


def _format_ci(delta, lo, hi, digits=3):
    if pd.isna(delta):
        return EM_DASH
    return f"{float(delta):+.{digits}f} [{float(lo):+.{digits}f}, {float(hi):+.{digits}f}]"


def build_appendix_d_paper_tables(structural_results, robustness_results) -> dict[str, pd.DataFrame]:
    """Convert the numeric validation outputs into Appendix Tables D1-D8."""
    raw = build_appendix_d_tables(structural_results, robustness_results)

    d1 = raw["D1"].copy()
    if not d1.empty:
        d1 = pd.DataFrame({
            "CV protocol": d1["CV Protocol"].replace({"Group KFold":"Group KFold"}),
            "Representation": d1["Representation"],
            "Method": d1["Method"],
            "R2 mean (SD)": _format_mean_sd(d1, "R2_mean", "R2_sd"),
            "RMSE mean (SD)": _format_mean_sd(d1, "RMSE_mean", "RMSE_sd"),
            "WAPE mean (SD)": _format_mean_sd(d1, "WAPE_mean", "WAPE_sd"),
            "PI95 Coverage": d1["Coverage_mean"].apply(_format_pct),
            "Avg Width": d1["Width_mean"].apply(lambda x: _paper_fixed(x, 3, EM_DASH)),
        })

    d2 = raw["D2"].copy()
    if not d2.empty:
        d2 = pd.DataFrame({
            "CV protocol": d2["CV Protocol"].replace({"Group KFold":"Group KFold"}),
            "Method": d2["Method"],
            "R2 mean (SD)": _format_mean_sd(d2, "R2_mean", "R2_sd"),
            "RMSE mean (SD)": _format_mean_sd(d2, "RMSE_mean", "RMSE_sd"),
            "WAPE mean (SD)": _format_mean_sd(d2, "WAPE_mean", "WAPE_sd"),
            "PI95 Coverage": d2["Coverage_mean"].apply(_format_pct),
            "Avg Width": d2["Width_mean"].apply(lambda x: _paper_fixed(x, 3, EM_DASH)),
        })

    d3 = raw["D3"].copy()
    if not d3.empty:
        d3["CV Protocol"] = d3["CV Protocol"].replace({"Group KFold":"Group KFold"})
        if "Fold" in d3.columns:
            d3["Fold"] = d3["Fold"].apply(_paper_int)
        if "Tuned-convex local weight alpha" in d3.columns:
            d3["Tuned-convex local weight alpha"] = d3["Tuned-convex local weight alpha"].apply(lambda x: _paper_fixed(x, 2))
        for c in ["Calibration RMSE", "50/50 conformal q95", "Tuned conformal q95"]:
            if c in d3.columns:
                d3[c] = d3[c].apply(lambda x: _paper_fixed(x, 3))

    d4r = raw["D4"].copy()
    d4 = pd.DataFrame()
    if not d4r.empty:
        d4 = pd.DataFrame({
            "CV protocol": d4r["CV Protocol"].replace({"Group KFold":"Group KFold"}),
            "Comparison": d4r["Comparison"],
            "Stage": d4r["Method stage"].replace({"global":"Global", "local_gap":"Local GapTrim", "bayes_gap":"Bayes GapTrim"}),
            "Delta R2 [95% CI]": d4r.apply(lambda r: _format_ci(r["Delta R2 (Candidate-Baseline)"], r["Delta R2 CI2.5"], r["Delta R2 CI97.5"]), axis=1),
            "Delta RMSE [95% CI]": d4r.apply(lambda r: _format_ci(r["Delta RMSE (Candidate-Baseline)"], r["Delta RMSE CI2.5"], r["Delta RMSE CI97.5"]), axis=1),
        })

    d5r = raw["D5"].copy()
    d5 = pd.DataFrame()
    if not d5r.empty:
        d5 = pd.DataFrame({
            "CV protocol": d5r["CV Protocol"].replace({"Group KFold":"Group KFold"}),
            "Comparison": d5r["Comparison"],
            "Candidate": d5r["Candidate"],
            "Baseline": d5r["Baseline"],
            "Delta R2 [95% CI]": d5r.apply(lambda r: _format_ci(r["Delta R2 (Candidate-Baseline)"], r["Delta R2 CI2.5"], r["Delta R2 CI97.5"]), axis=1),
            "Delta RMSE [95% CI]": d5r.apply(lambda r: _format_ci(r["Delta RMSE (Candidate-Baseline)"], r["Delta RMSE CI2.5"], r["Delta RMSE CI97.5"]), axis=1),
        })

    d6r = raw["D6"].copy()
    d6 = pd.DataFrame()
    if not d6r.empty:
        d6 = pd.DataFrame({
            "CV protocol": d6r["CV Protocol"].replace({"Group KFold":"Group KFold"}),
            "Representation": d6r["Representation"].replace({
                "Text + KG full (matched-seed)":"Text + full KG",
                "Text + KG supplier-neutral (matched-seed)":"Text + supplier-neutral KG",
            }),
            "Stage": d6r["Method"].replace({
                "Global (Ridge)":"Global",
                "Local (KNN GapTrim)":"Local GapTrim",
                "Bayes (GapTrim)":"Bayes GapTrim",
            }),
            "R2 mean (SD)": _format_mean_sd(d6r, "R2_mean", "R2_sd"),
            "RMSE mean (SD)": _format_mean_sd(d6r, "RMSE_mean", "RMSE_sd"),
            "Mean seed SD of R2": d6r["R2_seed_sd_mean"].apply(lambda x: _paper_fixed(x, 3, EM_DASH)),
        })

    d7r = raw["D7"].copy()
    d7 = pd.DataFrame()
    if not d7r.empty:
        d7 = pd.DataFrame({
            "CV protocol": d7r["CV Protocol"].replace({"Group KFold":"Group KFold"}),
            "Stage": d7r["Method stage"].replace({"global":"Global", "local_gap":"Local GapTrim", "bayes_gap":"Bayes GapTrim"}),
            "Delta R2 neutral-full [95% CI]": d7r.apply(lambda r: _format_ci(r["Delta R2 (Candidate-Baseline)"], r["Delta R2 CI2.5"], r["Delta R2 CI97.5"]), axis=1),
            "Delta RMSE neutral-full [95% CI]": d7r.apply(lambda r: _format_ci(r["Delta RMSE (Candidate-Baseline)"], r["Delta RMSE CI2.5"], r["Delta RMSE CI97.5"]), axis=1),
        })

    d8r = raw["D8"].copy()
    d8 = pd.DataFrame()
    if not d8r.empty:
        rows=[]
        for (cv, dim), g in d8r.groupby(["CV Protocol", "Graph-embedding dimension"], sort=False):
            row={"CV protocol":"Group KFold" if cv=="Group KFold" else cv, "Dimension":int(dim)}
            for method, col in [("Global (Ridge)","Global R2 (fold SD; seed SD)"),
                                ("Local (KNN GapTrim)","Local GapTrim R2 (fold SD; seed SD)"),
                                ("Bayes (GapTrim)","Bayes GapTrim R2 (fold SD; seed SD)")]:
                q=g[g["Method"]==method]
                if q.empty:
                    row[col]=np.nan
                else:
                    rr=q.iloc[0]
                    row[col]=f"{float(rr['R2_mean']):.3f} ({float(rr['R2_sd']):.3f}; {float(rr['R2_seed_sd_mean']):.3f})"
            rows.append(row)
        d8=pd.DataFrame(rows)
        d8["__cv"] = d8["CV protocol"].map({"KFold":1,"Group KFold":2})
        d8=d8.sort_values(["__cv","Dimension"]).drop(columns="__cv").reset_index(drop=True)

    tables = {"D1":d1,"D2":d2,"D3":d3,"D4":d4,"D5":d5,"D6":d6,"D7":d7,"D8":d8}
    validate_appendix_d_table_order(tables)
    return tables


def validate_appendix_d_table_order(tables: dict[str, pd.DataFrame]) -> None:
    """Fail fast if Appendix D rows drift away from manuscript order.

    Required order:
      * method stages: Global -> Local -> Bayes;
      * within Local/Bayes families: Mean -> GapTrim;
      * D5 architecture contrasts follow the LCMA sequence.
    """
    # D1: within each CV/representation, Global, Local Mean, Local GapTrim, Bayes Mean, Bayes GapTrim.
    d1 = tables.get("D1", pd.DataFrame())
    if not d1.empty:
        expected = ["Global (Ridge)", "Local (KNN Mean)", "Local (KNN GapTrim)", "Bayes (Mean)", "Bayes (GapTrim)"]
        for _, g in d1.groupby(["CV protocol", "Representation"], sort=False):
            got = g["Method"].tolist()
            if got != expected:
                raise AssertionError(f"Appendix D1 method order mismatch: {got} != {expected}")

    # D2: same family order, with non-Bayesian fusion between Local and Bayes.
    d2 = tables.get("D2", pd.DataFrame())
    if not d2.empty:
        expected = [
            "Global (Ridge)", "Local (KNN Mean)", "Local (KNN GapTrim)",
            "Global + Local (50/50, non-Bayes)", "Global + Local (tuned convex, non-Bayes)",
            "Bayes (Mean)", "Bayes (GapTrim)",
        ]
        for _, g in d2.groupby(["CV protocol"], sort=False):
            got = g["Method"].tolist()
            if got != expected:
                raise AssertionError(f"Appendix D2 method order mismatch: {got} != {expected}")

    # D4: for each representation contrast, Global -> Local GapTrim -> Bayes GapTrim.
    d4 = tables.get("D4", pd.DataFrame())
    if not d4.empty:
        expected_stage = ["Global", "Local GapTrim", "Bayes GapTrim"]
        expected_cmp = ["KG only vs Text", "KG+Text vs Text"]
        for cv, gcv in d4.groupby("CV protocol", sort=False):
            got_cmp = list(dict.fromkeys(gcv["Comparison"].tolist()))
            if got_cmp != expected_cmp:
                raise AssertionError(f"Appendix D4 comparison order mismatch in {cv}: {got_cmp} != {expected_cmp}")
            for comp, g in gcv.groupby("Comparison", sort=False):
                got = g["Stage"].tolist()
                if got != expected_stage:
                    raise AssertionError(f"Appendix D4 stage order mismatch in {cv}/{comp}: {got} != {expected_stage}")

    # D5: LCMA architecture sequence.
    d5 = tables.get("D5", pd.DataFrame())
    if not d5.empty:
        expected = ["Local vs Global", "Tuned Global+Local vs Local", "Bayes vs Tuned non-Bayes"]
        for cv, g in d5.groupby("CV protocol", sort=False):
            got = g["Comparison"].tolist()
            if got != expected:
                raise AssertionError(f"Appendix D5 comparison order mismatch in {cv}: {got} != {expected}")

    # D6/D7: Global -> Local GapTrim -> Bayes GapTrim.
    d6 = tables.get("D6", pd.DataFrame())
    if not d6.empty:
        expected_stage = ["Global", "Local GapTrim", "Bayes GapTrim"]
        for _, g in d6.groupby(["CV protocol", "Representation"], sort=False):
            got = g["Stage"].tolist()
            if got != expected_stage:
                raise AssertionError(f"Appendix D6 stage order mismatch: {got} != {expected_stage}")

    d7 = tables.get("D7", pd.DataFrame())
    if not d7.empty:
        expected_stage = ["Global", "Local GapTrim", "Bayes GapTrim"]
        for cv, g in d7.groupby("CV protocol", sort=False):
            got = g["Stage"].tolist()
            if got != expected_stage:
                raise AssertionError(f"Appendix D7 stage order mismatch in {cv}: {got} != {expected_stage}")


def export_complete_reproducibility_workbook(primary_results, structural_results, robustness_results,
                                              appendix_b_summary: pd.DataFrame,
                                              appendix_b_detail: pd.DataFrame) -> Path:
    """Export every current manuscript table in exact manuscript order.

    Final location required by the paper package:
      paper_result_final/paper_results_tables/All_Tables_Main_and_Apppendix.xlsx
    """
    ensure_dir(OUT_TABLES)
    d = build_appendix_d_paper_tables(structural_results, robustness_results)

    main1 = pd.read_csv(Path(OUT_DESC) / "I03_Table_1_Distribution_of_Posted_Quotes_USD.csv", dtype=str, keep_default_na=False)
    main2 = primary_results["table2"].copy()
    main3 = build_main_table3(structural_results)
    app_a1 = pd.read_csv(Path(OUT_DESC) / "I04_Appendix_Table_A1_Supplier_Level_Clustering.csv", dtype=str, keep_default_na=False)
    app_b1 = appendix_b_summary.copy()
    app_c1 = primary_results["appendix_c1"].copy()
    app_c2 = primary_results["appendix_c2"].copy()
    app_c3 = primary_results["appendix_c3"].copy()
    app_c4 = primary_results["appendix_c4"].copy()
    app_e1 = primary_results["appendix_e1"].copy()

    # Manuscript-table order.  Numeric prefixes are deliberate: they make both the
    # workbook tabs and the standalone CSV files sortable in paper order.
    ordered = [
        ("01 Table 1", "Table 1. Distribution of posted quotes (USD per call) and ln(price)", main1),
        ("02 Table 2", "Table 2. Main benchmark performance under within-market and supplier cold-start validation", main2),
        ("03 Table 3", "Table 3. LCMA ablation: representation transferability and sources of cold-start recovery", main3),
        ("04 App A1", "Appendix Table A1. Supplier-level clustering in posted quotes", app_a1),
        ("05 App B1", "Appendix Table B1. Gaussian working-approximation diagnostics", app_b1),
        ("06 App C1", "Appendix Table C1. Shrinkage diagnostics", app_c1),
        ("07 App C2", "Appendix Table C2. Tail-risk diagnostics", app_c2),
        ("08 App C3", "Appendix Table C3. High-disagreement subset diagnostics", app_c3),
        ("09 App C4", "Appendix Table C4. Decile-based gain patterns", app_c4),
        ("10 App D1", "Appendix Table D1. Representation ablation", d["D1"]),
        ("11 App D2", "Appendix Table D2. Architecture ablation", d["D2"]),
        ("12 App D3", "Appendix Table D3. Training-fold calibration of non-Bayesian fusion", d["D3"]),
        ("13 App D4", "Appendix Table D4. Paired-bootstrap inference for representation contrasts", d["D4"]),
        ("14 App D5", "Appendix Table D5. Paired-bootstrap inference for architecture contrasts", d["D5"]),
        ("15 App D6", "Appendix Table D6. Matched-seed supplier-topology results", d["D6"]),
        ("16 App D7", "Appendix Table D7. Paired-bootstrap inference for supplier-topology transfer", d["D7"]),
        ("17 App D8", "Appendix Table D8. Graph-embedding dimension sensitivity", d["D8"]),
        ("18 App E1", "Appendix Table E1. Predictive-uncertainty robustness and interval calibration", app_e1),
    ]

    manifest_rows = [
        ["Input: Excel", Path(DEFAULT_EXCEL_INPUT).name, "Authoritative source for price (USD/call), price_CNY, name, and desc/text"],
        ["Input: KG snapshot", Path(DEFAULT_KG_SNAPSHOT).name, "Authoritative portable snapshot for pid, historical row order, src_list, and app_list; not a precomputed embedding table"],
        ["Currency", f"1 USD = {CNY_PER_USD} CNY", "Excel already stores price in USD/call; price_CNY preserves source CNY; the program validates but does not reconvert"],
        ["Sample", "2,879 listings / 256 suppliers", "The supplied Excel file is already the manuscript analytical sample"],
        ["Model validation", "KFold then supplier-held-out Group KFold", "All fold-dependent learning/tuning uses training-fold information only"],
        ["Workbook order", "01-18", "Sheets follow current manuscript Table 1-3, Appendix A-E order"],
    ]
    manifest = pd.DataFrame(manifest_rows, columns=["Item", "Value", "Definition"])

    # Also export one CSV per official table, with the same ordering as the workbook.
    for idx, (sheet, title, df) in enumerate(ordered, start=1):
        safe_title = title.split(". ", 1)[0].replace(" ", "_")
        if title.startswith("Table 1"):
            stem = "01_Table_1_Distribution_of_Posted_Quotes"
        elif title.startswith("Table 2"):
            stem = "02_Table_2_Main_Benchmark_Performance"
        elif title.startswith("Table 3"):
            stem = "03_Table_3_LCMA_Ablation"
        else:
            code = sheet.split(" ", 1)[1].replace(" ", "_")
            stem = f"{idx:02d}_{code}"
        df.to_csv(Path(OUT_TABLES) / f"{stem}.csv", index=False, encoding="utf-8-sig")

    out = Path(OUT_TABLES) / "All_Tables_Main_and_Apppendix.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        manifest.to_excel(writer, sheet_name="00 Manifest", index=False)
        for sheet, _title, df in ordered:
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

        # Lightweight workbook formatting for auditability/readability.
        from openpyxl.styles import Font, Alignment, PatternFill
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="D9EAF7")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for col in ws.columns:
                letter = col[0].column_letter
                max_len = 0
                for cell in col[: min(ws.max_row, 80)]:
                    val = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(val))
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 42)

    # Audit-only details are deliberately kept outside the final paper-table workbook.
    if appendix_b_detail is not None and not appendix_b_detail.empty:
        appendix_b_detail.to_csv(Path(OUT_MAIN) / "I17_Appendix_B_Residual_Detail.csv", index=False, encoding="utf-8-sig")

    return out


def validate_final_result_order(final_table: pd.DataFrame) -> None:
    """Enforce the manuscript-wide reporting order: all KFold rows precede Group KFold rows."""
    if final_table.empty:
        raise AssertionError("Final empirical result table is empty.")
    order = final_table["CV Protocol"].map(CV_PROTOCOL_ORDER).to_numpy()
    if np.any(np.diff(order) < 0):
        raise AssertionError("Final result ordering error: KFold must precede Group KFold.")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Standalone reproduction pipeline for the final FINI manuscript. "
            "Inputs are the revised anonymized Excel and the frozen api_data_snapshot.csv KG relation snapshot."
        )
    )
    ap.add_argument(
        "--input-excel", type=str, default=DEFAULT_EXCEL_INPUT,
        help="Revised analytical Excel containing USD price and preserved price_CNY. Default: dataproduct_industry_analysis_list_format_anymous_USD.xlsx beside this script."
    )
    ap.add_argument(
        "--kg-snapshot", type=str, default=DEFAULT_KG_SNAPSHOT,
        help="Frozen KG relation snapshot (pid/order/src_list/app_list) with exact revised USD/CNY prices. Default: api_data_snapshot.csv beside this script."
    )
    ap.add_argument(
        "--output-root", type=str, default=str(OUTPUT_ROOT),
        help="Root directory for all outputs. Default: paper_result_final beside this script."
    )
    ap.add_argument(
        "--max-folds", type=int, default=None,
        help="Smoke test only: run the first N folds of each protocol. Omit for the full 5+5-fold manuscript analysis."
    )
    ap.add_argument(
        "--bootstrap-B", type=int, default=BOOTSTRAP_REPS,
        help="Paired bootstrap replications for the full analysis. Default: 5000."
    )
    ap.add_argument(
        "--skip-dimension-sensitivity", action="store_true",
        help="Skip the 16/32/64 graph-embedding dimension sensitivity analysis."
    )
    ap.add_argument(
        "--graph-validation-repeats", type=int, default=GRAPH_VALIDATION_REPEATS,
        help="Matched graph-embedding seeds per fold for topology and dimension validation. Default: 3."
    )
    args = ap.parse_args()
    if args.graph_validation_repeats < 1:
        raise ValueError("--graph-validation-repeats must be at least 1.")

    configure_output_root(args.output_root)
    input_excel = str(Path(args.input_excel).expanduser().resolve())
    kg_snapshot = str(Path(args.kg_snapshot).expanduser().resolve())
    if not Path(input_excel).exists():
        raise FileNotFoundError(f"Input Excel not found: {input_excel}")
    if not Path(kg_snapshot).exists():
        raise FileNotFoundError(f"KG snapshot not found: {kg_snapshot}")

    for d in [OUT_TABLES, OUT_FIG, OUT_INTERMEDIATE, OUT_DESC, OUT_MAIN, str(VALIDATION_OUT), str(MANUSCRIPT_SEQUENCE_OUT)]:
        ensure_dir(d)

    pipeline_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 0. Validate/merge revised Excel with the frozen KG relation snapshot.
    # ------------------------------------------------------------------
    phase_t0 = time.perf_counter()
    print("\n=== Phase 0/5: Validate Excel prices/text + KG relational snapshot ===", flush=True)
    prepared_df, generated_snapshot = prepare_modeling_sample(input_excel, kg_snapshot)
    print(f"[OK] Excel is authoritative for USD price, price_CNY, and text (name + desc).")
    print(f"[OK] KG snapshot is authoritative for pid, historical row order, src_list, and app_list.")
    print(f"[OK] Currency relation validated at 1 USD = {CNY_PER_USD} CNY; no second conversion performed.")
    print(f"[OK] Fold-specific graph/text embeddings will be trained from outer-training observations only.")
    print(f"[OK] Merged modeling snapshot: {generated_snapshot}")
    print(f"[TIME] Phase 0: {(time.perf_counter()-phase_t0)/60:.2f} min")

    # Fig. 1 (conceptual LCMA framework) and Fig. 2 (ontology/KG schematic) are
    # manuscript design figures and are intentionally NOT regenerated by this empirical
    # reproduction pipeline. Clear figure artifacts from earlier runs so the directory
    # contains only the figures produced by the current manuscript-aligned pipeline.
    for old_figure in Path(OUT_FIG).glob("*"):
        if old_figure.is_file() and old_figure.suffix.lower() in {".png", ".pdf", ".csv"}:
            old_figure.unlink()

    # ------------------------------------------------------------------
    # Phase 1. Descriptive evidence: Table 1, Appendix A1, Figs. A1-A2.
    # ------------------------------------------------------------------
    phase_t0 = time.perf_counter()
    print("\n=== Phase 1/5: Descriptive market evidence ===", flush=True)
    run_descriptive_block(generated_snapshot)
    print(f"[TIME] Phase 1: {(time.perf_counter()-phase_t0)/60:.2f} min")

    # ------------------------------------------------------------------
    # Phase 2. Main empirical evidence and Appendix C/E inputs.
    # ------------------------------------------------------------------
    phase_t0 = time.perf_counter()
    print("\n=== Phase 2/5: Main empirical evidence (KFold -> Group KFold) ===", flush=True)
    primary_results = run_main_appendix(main_input=generated_snapshot, max_folds=args.max_folds)
    build_manuscript_fig_3_absolute_log_price_error(primary_results["predictions"])
    appendix_b_summary, appendix_b_detail = build_appendix_b_from_primary_predictions(primary_results["predictions"])
    print(f"[TIME] Phase 2: {(time.perf_counter()-phase_t0)/60:.2f} min")

    # ------------------------------------------------------------------
    # Phase 3. Representation/architecture validation for Table 3 and Appendix D.
    # ------------------------------------------------------------------
    phase_t0 = time.perf_counter()
    print("\n=== Phase 3/5: Representation and architecture validation ===", flush=True)
    structural_results = run_structural_validation(
        primary_results,
        graph_validation_repeats=args.graph_validation_repeats,
    )
    print(f"[TIME] Phase 3: {(time.perf_counter()-phase_t0)/60:.2f} min")

    # ------------------------------------------------------------------
    # Phase 4. Bootstrap/topology/dimension robustness for Appendix D.
    # ------------------------------------------------------------------
    phase_t0 = time.perf_counter()
    print("\n=== Phase 4/5: Statistical robustness and sensitivity ===", flush=True)
    robustness_results = run_statistical_robustness(
        primary_results,
        structural_results,
        bootstrap_B=args.bootstrap_B,
        dimension_sensitivity=(not args.skip_dimension_sensitivity),
        graph_validation_repeats=args.graph_validation_repeats,
    )
    print(f"[TIME] Phase 4: {(time.perf_counter()-phase_t0)/60:.2f} min")

    # ------------------------------------------------------------------
    # Phase 5. Final manuscript-ordered exports.
    # ------------------------------------------------------------------
    phase_t0 = time.perf_counter()
    print("\n=== Phase 5/5: Final manuscript-ordered exports ===", flush=True)
    final_table = build_final_empirical_results(primary_results, structural_results, robustness_results)
    validate_final_result_order(final_table)
    final_table.to_csv(Path(OUT_INTERMEDIATE) / "I40_Unified_Empirical_Results.csv", index=False, encoding="utf-8-sig")

    complete_xlsx = export_complete_reproducibility_workbook(
        primary_results, structural_results, robustness_results,
        appendix_b_summary, appendix_b_detail,
    )

    figure_manifest = pd.DataFrame([
        ["Fig. 3", "Fig_3_Absolute_Log_Price_Error_Distributions_Under_Supplier_Cold_Start.png", "Absolute log-price error distributions under supplier cold start"],
        ["Appendix Fig. A1", "Appendix_Fig_A1_Distribution_of_Standardized_Posted_Quotes_Before_and_After_Log_Transformation.png", "Distribution of standardized posted quotes before and after log transformation"],
        ["Appendix Fig. A2", "Appendix_Fig_A2_Descriptive_Evidence_of_Supplier_Level_Clustering_in_Posted_Quotes.png", "Descriptive evidence of supplier-level clustering in posted quotes"],
        ["Appendix Fig. B1", "Appendix_Fig_B1_Residual_Diagnostics_for_the_Global_Prior_under_KFold.png", "Residual diagnostics for the global prior under KFold"],
        ["Appendix Fig. B2", "Appendix_Fig_B2_Residual_Diagnostics_for_the_Global_Prior_under_Group KFold.png", "Residual diagnostics for the global prior under Group KFold"],
    ], columns=["Manuscript figure", "Filename", "Purpose"])
    figure_manifest.to_csv(Path(OUT_FIG) / "00_Figure_Manifest.csv", index=False, encoding="utf-8-sig")

    print(f"[TIME] Phase 5: {(time.perf_counter()-phase_t0)/60:.2f} min")
    print(f"[TIME] Total pipeline: {(time.perf_counter()-pipeline_t0)/60:.2f} min")
    print("\n[OK] Standalone reproduction pipeline finished.")
    print(f"[OK] Final manuscript tables: {OUT_TABLES}")
    print(f"[OK] Complete workbook: {complete_xlsx}")
    print(f"[OK] Paper figures: {OUT_FIG}")
    print(f"[OK] Ordered intermediate/audit files: {OUT_INTERMEDIATE}")
    print("[NOTE] api_data_snapshot.csv is used as the frozen KG relation/order source; embeddings are still re-trained inside each outer fold to prevent leakage.")


if __name__ == "__main__":
    main()
