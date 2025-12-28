#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Social Media Trends — End-to-End Analysis & ML Pipeline
Author: (you)
How to run:
    python social_media_ml_pipeline.py  # defaults to Cleaned_Viral_Social_Media_Trends.csv in repo
    python social_media_ml_pipeline.py --csv path/to/your.csv --datecol Post_Date --savefigs figs/
Notes:
    - Assumes columns:
      ['Post_ID','Post_Date','Platform','Hashtag','Content_Type','Region',
       'Views','Likes','Shares','Comments','Engagement_Level']
    - If Hashtag contains multiple tags, separate them with commas in the CSV (e.g., "#ai,#ml").
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset

# Optional imports guarded at runtime
try:
    import networkx as nx
except Exception as e:
    nx = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import pycountry
except Exception:
    pycountry = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Helpers
# -------------------------

def safe_mkdir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        default="Cleaned_Viral_Social_Media_Trends.csv",
        help="Path to the dataset CSV (defaults to Cleaned_Viral_Social_Media_Trends.csv)",
    )
    p.add_argument("--datecol", type=str, default="Post_Date", help="Name of the datetime column")
    p.add_argument("--top_hashtags", type=int, default=6, help="How many top hashtags to plot in time-series")
    p.add_argument("--resample", type=str, default="W", help="Time resample rule: 'D', 'W', 'M' ...")
    p.add_argument("--figsize", type=str, default="10,6", help="Figure size, e.g., '10,6'")
    p.add_argument("--savefigs", type=str, default="figs", help="Directory to save figures")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

def parse_size(s):
    try:
        w, h = s.split(",")
        return (float(w.strip()), float(h.strip()))
    except Exception:
        return (10, 6)

def infer_numeric(df, candidates):
    out = []
    for c in candidates:
        if c in df.columns:
            out.append(c)
    return out

# -------------------------
# Load & clean
# -------------------------

def load_data(path, datecol):
    if not os.path.exists(path):
        # Try relative to the script directory for convenience
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fallback = os.path.join(script_dir, path)
        if os.path.exists(fallback):
            path = fallback
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if datecol in df.columns:
        df[datecol] = pd.to_datetime(df[datecol], errors="coerce", utc=True).dt.tz_localize(None)
    # Basic dtypes
    for col in ["Views","Likes","Shares","Comments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop obvious empties
    df = df.dropna(subset=[datecol]) if datecol in df.columns else df
    return df

# -------------------------
# Feature Engineering
# -------------------------

def add_time_features(df, datecol="Post_Date"):
    if datecol not in df.columns:
        return df
    dt = df[datecol]
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Week"] = dt.dt.isocalendar().week.astype(int)
    df["Weekday"] = dt.dt.dayofweek
    df["Hour"] = dt.dt.hour
    return df

def add_engagement_rate(df):
    for c in ["Views","Likes","Shares","Comments"]:
        if c not in df.columns:
            df[c] = 0
    # Avoid division by zero
    df["Engagement_Rate"] = (df["Likes"].fillna(0) + df["Shares"].fillna(0) + df["Comments"].fillna(0)) / (df["Views"].replace(0, np.nan))
    df["Engagement_Rate"] = df["Engagement_Rate"].fillna(0.0)
    return df

# -------------------------
# Exploratory Data Analysis
# -------------------------

def exploratory_analysis(df, args, datecol="Post_Date"):
    """Basic global stats and correlation heatmap."""
    save_dir = args.savefigs
    safe_mkdir(save_dir)

    numeric_candidates = ["Views","Likes","Shares","Comments","Engagement_Rate","Hour","Weekday","Month"]
    numeric_cols = infer_numeric(df, numeric_candidates)

    if numeric_cols:
        summary = df[numeric_cols].describe().T
        summary_out = os.path.join(save_dir, "global_numeric_summary.csv")
        summary.to_csv(summary_out)
        print(f"[EDA] Saved numeric summary -> {summary_out}")
    else:
        print("[EDA] No numeric columns available for summary.")

    missing = df.isna().mean().sort_values(ascending=False)
    missing_out = os.path.join(save_dir, "missingness_summary.csv")
    missing.to_csv(missing_out, header=["missing_fraction"])
    print(f"[EDA] Saved missingness summary -> {missing_out}")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=parse_size(args.figsize))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        ax.set_title("Correlation Heatmap (numeric features)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="Correlation")
        fig.tight_layout()
        out = os.path.join(save_dir, "correlation_heatmap.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[EDA] Saved correlation heatmap -> {out}")
    else:
        print("[EDA] Not enough numeric columns for correlation heatmap.")

    if "Platform" in df.columns and numeric_cols:
        platform_stats = df.groupby("Platform")[numeric_cols].mean().round(3)
        platform_out = os.path.join(save_dir, "platform_mean_metrics.csv")
        platform_stats.to_csv(platform_out)
        print(f"[EDA] Saved Platform-level means -> {platform_out}")

# -------------------------
# Time-Series Modeling
# -------------------------

def forecast_top_hashtags(df, args, datecol="Post_Date", forecast_steps=4):
    """Simple exponential smoothing forecasts for top hashtags."""
    if ExponentialSmoothing is None:
        print("[Forecast] statsmodels not available; skipping hashtag forecasting.")
        return

    if ("Hashtag" not in df.columns) or (datecol not in df.columns):
        print("[Forecast] Missing Hashtag or Post_Date; skipping.")
        return

    metric = "Engagement_Rate" if "Engagement_Rate" in df.columns else "Likes"
    df2 = df.dropna(subset=["Hashtag", datecol]).copy()
    if df2.empty:
        print("[Forecast] No valid hashtag records for forecasting.")
        return

    df2["Hashtag"] = df2["Hashtag"].astype(str).str.strip().str.lower()
    top_tags = df2["Hashtag"].value_counts().head(args.top_hashtags).index.tolist()
    if not top_tags:
        print("[Forecast] No hashtags found for forecasting.")
        return

    offset = to_offset(args.resample)
    valid_series = []

    for tag in top_tags:
        ts = (
            df2[df2["Hashtag"] == tag]
            .set_index(datecol)
            .sort_index()[metric]
            .resample(args.resample)
            .mean()
        )
        ts = ts.dropna()
        if len(ts) < max(6, forecast_steps + 1):
            continue
        ts = ts.asfreq(offset, method="ffill")
        valid_series.append((tag, ts))

    if not valid_series:
        print("[Forecast] Not enough data points to build forecasts.")
        return

    rows = len(valid_series)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, (tag, ts) in zip(axes, valid_series):
        try:
            model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None)
            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(forecast_steps)
        except Exception as exc:
            print(f"[Forecast] Failed for #{tag}: {exc}; using trailing mean.")
            forecast_start = ts.index[-1] + offset
            forecast_index = [forecast_start + i * offset for i in range(forecast_steps)]
            fallback_value = ts.tail(3).mean()
            forecast = pd.Series([fallback_value] * forecast_steps, index=forecast_index)

        ax.plot(ts.index, ts.values, label="observed", color="#1f77b4")
        ax.plot(forecast.index, forecast.values, label="forecast", linestyle="--", color="#ff7f0e")
        ax.set_title(f"#{tag} — {metric} ({args.resample} mean)")
        ax.set_ylabel(metric)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out = os.path.join(args.savefigs, f"hashtag_forecasts_{metric}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[Forecast] Saved hashtag forecast plots -> {out}")

# -------------------------
# Time-Series by Hashtag
# -------------------------

def plot_hashtag_timeseries(df, args, datecol="Post_Date"):
    """Aggregate engagement over time per hashtag; plot top-N by volume."""
    figsize = parse_size(args.figsize)
    save_dir = args.savefigs
    safe_mkdir(save_dir)

    if ("Hashtag" not in df.columns) or (datecol not in df.columns):
        print("[TimeSeries] Missing Hashtag or Post_Date; skipping.")
        return

    # Choose an engagement metric to track
    metric = "Engagement_Rate" if "Engagement_Rate" in df.columns else "Likes"
    # Normalize whitespace, handle nulls
    df2 = df.dropna(subset=["Hashtag"]).copy()
    df2["Hashtag"] = df2["Hashtag"].astype(str).str.strip().str.lower()

    # Find top hashtags by count
    top_tags = df2["Hashtag"].value_counts().head(args.top_hashtags).index.tolist()
    fig, ax = plt.subplots(figsize=figsize)

    for tag in top_tags:
        temp = df2[df2["Hashtag"] == tag].set_index(datecol).sort_index()
        ts = temp[metric].resample(args.resample).mean()
        ax.plot(ts.index, ts.values, label=tag)

    ax.set_title(f"{metric} over time — Top {len(top_tags)} hashtags (resample={args.resample})")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.legend(loc="best")
    out = os.path.join(save_dir, f"timeseries_{metric}_top{len(top_tags)}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out}")

# -------------------------
# Hashtag Co-occurrence Graph
# -------------------------

def build_hashtag_cooccur_graph(df, save_dir="figs", min_edge_weight=2):
    """Build a co-occurrence graph assuming comma-separated hashtags in 'Hashtag' column."""
    if "Hashtag" not in df.columns:
        print("[Graph] Missing Hashtag; skipping graph.")
        return
    if nx is None:
        print("[Graph] networkx not available; skipping graph.")
        return

    safe_mkdir(save_dir)
    # Split comma-separated hashtags per post
    tags_series = (
        df["Hashtag"].astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.split(",")
    )

    # Count co-occurrences
    from collections import Counter
    edge_counter = Counter()

    for tags in tags_series:
        tags = [t for t in tags if t]  # remove empties
        unique = sorted(set(tags))
        for i in range(len(unique)):
            for j in range(i+1, len(unique)):
                a, b = unique[i], unique[j]
                edge_counter[(a, b)] += 1

    # Build graph
    G = nx.Graph()
    for (a, b), w in edge_counter.items():
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=w)

    # Simple layout and plot (matplotlib)
    if len(G.nodes) == 0:
        print("[Graph] No edges passed the min_edge_weight; skipping plot.")
        return

    deg = dict(G.degree())
    node_sizes = [50 + 20*deg[n] for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42, k=None)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=[0.5 + 0.3*G[u][v]["weight"] for u, v in G.edges()])
    # Limit labels for readability
    labels = {n: n if deg[n] >= 2 else "" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title("Hashtag Co-occurrence Graph (labels shown for degree ≥ 2)")
    plt.axis("off")
    out = os.path.join(save_dir, "hashtag_cooccurrence_graph.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[Saved] {out}")

# -------------------------
# Region x Platform Heatmap
# -------------------------

def plot_region_platform_heatmap(df, args, agg="Engagement_Rate"):
    """Heatmap of Region x Platform aggregated engagement metric."""
    save_dir = args.savefigs
    safe_mkdir(save_dir)
    if ("Region" not in df.columns) or ("Platform" not in df.columns):
        print("[Heatmap] Missing Region or Platform; skipping heatmap.")
        return

    if agg not in df.columns:
        # fall back to Likes if engagement rate not present
        agg = "Likes" if "Likes" in df.columns else None
    if agg is None:
        print("[Heatmap] No numeric engagement column; skipping heatmap.")
        return

    pivot = (
        df.pivot_table(index="Region", columns="Platform", values=agg, aggfunc="mean")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=parse_size(args.figsize))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{agg} — Region × Platform (mean)")
    fig.colorbar(im, ax=ax, shrink=0.8, label=agg)
    fig.tight_layout()
    out = os.path.join(save_dir, "region_platform_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out}")

# -------------------------
# Geo Mapping
# -------------------------

_REGION_ALIASES = {
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
    "south korea": "Republic of Korea",
    "north korea": "Korea, Democratic People's Republic of",
    "hong kong": "Hong Kong",
    "taiwan": "Taiwan, Province of China",
}

_REGION_ISO_FALLBACK = {
    "united states": "USA",
    "united kingdom": "GBR",
    "great britain": "GBR",
    "england": "GBR",
    "scotland": "GBR",
    "wales": "GBR",
    "uae": "ARE",
    "united arab emirates": "ARE",
    "south korea": "KOR",
    "north korea": "PRK",
    "hong kong": "HKG",
    "taiwan": "TWN",
}

def _normalize_region_name(name):
    if pd.isna(name):
        return None
    norm = str(name).strip()
    if not norm:
        return None
    key = norm.lower()
    return _REGION_ALIASES.get(key, norm)

def _region_to_iso3(name):
    if not name:
        return None
    key = str(name).strip().lower()
    if key in _REGION_ISO_FALLBACK:
        return _REGION_ISO_FALLBACK[key]
    if pycountry is not None:
        try:
            result = pycountry.countries.lookup(name)
            return result.alpha_3
        except Exception:
            pass
    return None

def plot_geo_engagement(df, args, metric="Engagement_Rate"):
    """Geo-level map of engagement intensity saved as HTML."""
    if "Region" not in df.columns:
        print("[Geo] Region column missing; skipping geo mapping.")
        return
    if (metric not in df.columns) and ("Likes" in df.columns):
        metric = "Likes"
    if metric not in df.columns:
        print("[Geo] No engagement metric available; skipping geo mapping.")
        return
    if px is None:
        print("[Geo] plotly not available; skipping geo mapping.")
        return

    save_dir = args.savefigs
    safe_mkdir(save_dir)

    temp = df.copy()
    temp["Region"] = temp["Region"].apply(_normalize_region_name)
    agg = temp.dropna(subset=["Region"]).groupby("Region")[metric].mean().reset_index()
    if agg.empty:
        print("[Geo] No regional aggregates available; skipping.")
        return

    agg["iso_alpha"] = agg["Region"].apply(_region_to_iso3)
    agg = agg.dropna(subset=["iso_alpha"])
    if agg.empty:
        print("[Geo] Unable to map regions to ISO codes; skipping geo plot.")
        return

    fig = px.choropleth(
        agg,
        locations="iso_alpha",
        color=metric,
        hover_name="Region",
        color_continuous_scale="Viridis",
        projection="natural earth",
        title=f"Average {metric} by Region",
    )
    fig.update_layout(coloraxis_colorbar_title=metric)
    out_html = os.path.join(save_dir, "geo_engagement_map.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[Geo] Saved choropleth map -> {out_html}")

# -------------------------
# Mathematical Commentary
# -------------------------

def generate_mathematical_commentary(df, args, datecol="Post_Date"):
    """Derive growth, entropy, and elasticity metrics for interpretation."""
    save_dir = args.savefigs
    safe_mkdir(save_dir)

    lines = []
    lines.append("Analytical Commentary")
    lines.append("----------------------")

    working = df.copy()
    working["Total_Interactions"] = working.get("Likes", 0) + working.get("Shares", 0) + working.get("Comments", 0)

    # Growth rate (CAGR-style over resampled periods)
    if datecol in working.columns:
        ts = (
            working.dropna(subset=[datecol])
            .set_index(datecol)["Total_Interactions"]
            .resample(args.resample)
            .sum()
        )
        ts = ts[ts > 0]
        if len(ts) >= 2 and ts.iloc[0] > 0:
            periods = len(ts) - 1
            growth_rate = (ts.iloc[-1] / ts.iloc[0]) ** (1 / periods) - 1
            lines.append(f"Growth rate (CAGR approximation): {growth_rate:.4f}")
        else:
            lines.append("Growth rate: insufficient positive interaction data across time.")
    else:
        lines.append("Growth rate: date column not present.")

    # Hashtag entropy (diversity of interactions)
    if "Hashtag" in working.columns:
        hash_dist = (
            working.dropna(subset=["Hashtag"])
            .groupby("Hashtag")["Total_Interactions"]
            .sum()
        )
        hash_dist = hash_dist[hash_dist > 0]
        if len(hash_dist) > 1:
            p = hash_dist / hash_dist.sum()
            entropy = -(p * np.log2(p)).sum()
            max_entropy = np.log2(len(p))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else np.nan
            lines.append(f"Hashtag entropy (bits): {entropy:.3f}; normalized: {normalized_entropy:.3f}")
        else:
            lines.append("Hashtag entropy: dominated by a single tag.")
    else:
        lines.append("Hashtag entropy: hashtag column missing.")

    # Elasticity between views and interactions (log-log slope)
    if "Views" in working.columns:
        views = working["Views"].to_numpy(dtype=float)
        interactions = working["Total_Interactions"].to_numpy(dtype=float)
        mask = (views > 0) & (interactions > 0)
        if mask.sum() >= 10:
            log_views = np.log(views[mask])
            log_interactions = np.log(interactions[mask])
            slope, intercept = np.polyfit(log_views, log_interactions, deg=1)
            lines.append(f"Elasticity (d ln interactions / d ln views): {slope:.3f}")
        else:
            lines.append("Elasticity: insufficient positive paired observations.")
    else:
        lines.append("Elasticity: views column missing.")

    out_txt = os.path.join(save_dir, "mathematical_commentary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Commentary] Saved mathematical commentary -> {out_txt}")

# -------------------------
# Supervised ML: Predict Engagement_Level
# -------------------------

def run_ml_prediction(df, args):
    """Train a baseline classifier to predict Engagement_Level."""
    save_dir = args.savefigs
    safe_mkdir(save_dir)

    if "Engagement_Level" not in df.columns:
        print("[ML] Engagement_Level not found; skipping ML classification.")
        return

    # Select features
    cat_cols = [c for c in ["Platform","Content_Type","Region","Hashtag"] if c in df.columns]
    num_cols = [c for c in ["Views","Likes","Shares","Comments","Engagement_Rate","Hour","Weekday","Month"] if c in df.columns]

    # Minimal X, y
    X = df[cat_cols + num_cols].copy()
    y = df["Engagement_Level"].astype(str)

    # Reduce high-cardinality hashtag via frequency encoding
    if "Hashtag" in cat_cols:
        freq = X["Hashtag"].value_counts()
        top = set(freq.head(50).index)
        X["Hashtag"] = X["Hashtag"].where(X["Hashtag"].isin(top), other="_other_")

    # Preprocess
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    clf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=args.random_state, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state, stratify=y)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n[ML] Classification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Engagement_Level")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = os.path.join(save_dir, "ml_confusion_matrix.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[Saved] {out}")
# ============================================================
# 7. Advanced ML Module: Platform-aware Virality Classifier
#    （高级ML模块：考虑平台差异的“能不能火”预测）
# ------------------------------------------------------------
# Business motivation / 业务动机：
# 我们现在的脚本已经能做：
#   - 基础EDA
#   - 时间序列看热门Hashtag
#   - Hashtag共现图
#   - 一个简单的分类demo（如果有 Engagement_Level）
# 但在真实团队里，老板更想听的问题通常是：
#   “如果我今天再发一条这样的内容，它有多大概率进高互动那一档？”
# 而且这个问题要“对平台公平”——TikTok的量级和LinkedIn不一样，
# 所以不能直接用一个全局阈值，我们要在“各自平台内部”去挑top帖子。
#
# 这个模块要做的事情：
#   1. 把每个平台内部的帖子，按 Engagement_Rate（或Likes）排序
#   2. 在每个平台里取例如 90 分位数以上的帖子标记为 1 = viral
#   3. 用表格模型（CatBoost / LightGBM / RF都行）去学
#   4. 给出特征重要性，方便我们解释给老板/市场同事听
#   5. 可以选择把训练好的特征重要性 / 报告存到同一个 figs/ 目录，方便统一查阅
#
# 使用方式：
#   在 main() 里，读完 df、做完特征工程之后，调用：
#       run_platform_aware_viral_classifier(df, args)
#
# 注意：
#   - 这个函数会尽量“只读不改”你的 df（除了临时列），不会破坏你前面逻辑
#   - 你也可以把 percentile_from_args 调高/调低来控制“多严格叫火”
# ============================================================

def run_platform_aware_viral_classifier(
    df,
    args=None,
    datecol: str = "Post_Date",
    engagement_col_candidates=("Engagement_Rate", "Likes", "Views"),
    platform_col: str = "Platform",
    percentile: float = 0.9,
    save_dir: str = "figs",
):
    """
    Train a *platform-aware* virality classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned social media dataframe, after feature engineering
        (i.e. time features + engagement features already added).
    args : argparse.Namespace or None
        We use it only to keep the style consistent with the rest of the file.
        If provided, we will try to read savefigs / datecol from it.
    datecol : str
        Name of the datetime column.
    engagement_col_candidates : tuple[str]
        We will try these in order to decide which column to use
        as the "engagement strength". First one that exists & is numeric wins.
    platform_col : str
        Column that identifies the platform (TikTok, IG, X, YouTube...).
        We will create labels *inside* each platform.
    percentile : float
        Platform-wise percentile to define "viral". 0.9 means
        "top 10% of posts *within each platform*".
    save_dir : str
        Directory to save reports / plots.

    What this function does
    -----------------------
    1. Pick an engagement metric.
    2. For every platform, compute the chosen percentile.
    3. Make a binary label:
           viral = engagement >= platform_percentile
    4. Build feature matrix:
           - categorical: platform, content_type, region, hashtag
           - numeric: views, likes, shares, comments, engagement_rate, hour, weekday, month
    5. Train a classifier, preferring CatBoost (best for mixed tabular).
       If CatBoost is not installed, fall back to XGBoost, then to RandomForest.
    6. Save classification report + confusion matrix + feature importances.

    This is "presentation-grade":
    - Boss can see: what's the definition of viral
    - Team can see: which features actually matter
    - Coworkers can re-use: same df, same args, same folder
    """
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # ------------- 0. sync with upstream args -------------
    if args is not None:
        # prefer args' datecol / savefigs if present
        datecol = getattr(args, "datecol", datecol)
        save_dir = getattr(args, "savefigs", save_dir)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ------------- 1. pick engagement metric ---------------
    metric_col = None
    for c in engagement_col_candidates:
        if c in df.columns:
            metric_col = c
            break
    if metric_col is None:
        print("[viral clf] No engagement-like column found, skipping.")
        return

    # make sure numeric
    df = df.copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # ------------- 2. check platform column ----------------
    if platform_col not in df.columns:
        # if no platform, we can only do global percentile
        print(f"[viral clf] Column '{platform_col}' not found, using GLOBAL percentile.")
        global_thr = df[metric_col].quantile(percentile)
        df["__viral_label__"] = (df[metric_col] >= global_thr).astype(int)
    else:
        # platform-aware threshold
        def _mark_platform(group: pd.DataFrame) -> pd.DataFrame:
            thr = group[metric_col].quantile(percentile)
            group["__viral_label__"] = (group[metric_col] >= thr).astype(int)
            group["__viral_threshold__"] = thr
            return group

        df = df.groupby(platform_col, group_keys=False).apply(_mark_platform)

    # ------------- 3. build feature sets -------------------
    # candidate categorical features
    cat_features = []
    for c in ["Platform", "Content_Type", "Region", "Hashtag"]:
        if c in df.columns:
            cat_features.append(c)

    # candidate numeric features
    num_features = []
    for c in [
        "Views",
        "Likes",
        "Shares",
        "Comments",
        "Engagement_Rate",
        "Hour",
        "Weekday",
        "Month",
    ]:
        if c in df.columns:
            num_features.append(c)

    # drop rows without label
    df = df.dropna(subset=["__viral_label__", metric_col])

    # if literally no positive / no negative, skip
    if df["__viral_label__"].nunique() < 2:
        print("[viral clf] Label collapsed to a single class, nothing to train.")
        return

    # ------------- 4. train-test split ---------------------
    from sklearn.model_selection import train_test_split

    X = df[cat_features + num_features].copy()
    y = df["__viral_label__"].astype(int)

    # split – if we have a date, we could do time-based split;
    # here we'll do random split for simplicity
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ------------- 5. build preprocessing ------------------
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )

    # ------------- 6. choose model (CatBoost -> XGB -> RF) -------------
    model = None
    used_model_name = None

    try:
        from catboost import CatBoostClassifier

        # note: we *could* feed raw cat features to CatBoost directly,
        # but to stay consistent with the rest of the file (which uses
        # sklearn-like flows), we still go through preprocessor.
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    CatBoostClassifier(
                        depth=6,
                        learning_rate=0.08,
                        loss_function="Logloss",
                        verbose=False,
                        random_state=42,
                    ),
                ),
            ]
        )
        used_model_name = "CatBoostClassifier"
    except Exception:
        try:
            from xgboost import XGBClassifier
            from sklearn.pipeline import Pipeline

            model = Pipeline(
                steps=[
                    ("prep", preprocessor),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=42,
                        ),
                    ),
                ]
            )
            used_model_name = "XGBClassifier"
        except Exception:
            # final fallback
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline

            model = Pipeline(
                steps=[
                    ("prep", preprocessor),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=None,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            used_model_name = "RandomForestClassifier"

    # ------------- 7. fit ---------------------
    model.fit(X_train, y_train)

    # ------------- 8. evaluate ----------------
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
    import matplotlib.pyplot as plt

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    print("[viral clf] ===== classification report =====")
    print(report)

    # save report
    report_path = os.path.join(save_dir, "viral_classifier_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model used: {used_model_name}\n")
        f.write(f"Percentile (platform-wise): {percentile}\n")
        f.write(f"Metric column: {metric_col}\n")
        f.write("\n")
        f.write(report)
    print(f"[viral clf] report saved to {report_path}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-viral", "viral"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Viral Classifier – Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "viral_classifier_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[viral clf] confusion matrix saved to {cm_path}")

    # ------------- 9. feature importances (if available) -------------
    # Only possible if our final step has feature_importances_
    # and we can recover the feature names from the preprocessor.
    try:
        final_clf = model.named_steps["clf"]
        # recover feature names
        ohe: OneHotEncoder = model.named_steps["prep"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_features))
        num_feature_names = num_features
        all_feature_names = cat_feature_names + num_feature_names

        importances = getattr(final_clf, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame(
                {"feature": all_feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)
            imp_path = os.path.join(save_dir, "viral_classifier_feature_importances.csv")
            imp_df.to_csv(imp_path, index=False)
            print(f"[viral clf] feature importances saved to {imp_path}")
    except Exception as e:
        print(f"[viral clf] could not extract feature importances: {e}")

    print(f"[viral clf] DONE. Model used: {used_model_name}")
# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    df = load_data(args.csv, datecol=args.datecol)

    # Basic cleaning
    if "Post_ID" in df.columns:
        df = df.drop_duplicates(subset=["Post_ID"])

    # Feature engineering
    df = add_time_features(df, datecol=args.datecol)
    df = add_engagement_rate(df)

    # Analyses
    exploratory_analysis(df, args, datecol=args.datecol)
    plot_hashtag_timeseries(df, args, datecol=args.datecol)
    forecast_top_hashtags(df, args, datecol=args.datecol)
    build_hashtag_cooccur_graph(df, save_dir=args.savefigs, min_edge_weight=2)
    plot_region_platform_heatmap(df, args, agg="Engagement_Rate")
    plot_geo_engagement(df, args, metric="Engagement_Rate")

    # ML
    run_ml_prediction(df, args)
    generate_mathematical_commentary(df, args, datecol=args.datecol)
# 👇 新加的
    run_platform_aware_viral_classifier(df, args, datecol=args.datecol)
if __name__ == "__main__":
    main()
