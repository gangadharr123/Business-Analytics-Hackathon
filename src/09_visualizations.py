"""Step 9: Generate presentation-quality visualizations for the hackathon."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and Streamlit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from config import DELAY_THRESHOLD_MINUTES, ENRICHED_DATA_FILE, REPORTS_DIR

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_PRIMARY   = "#1a3a5c"
_ACCENT    = "#f0a500"
_SAFE      = "#27ae60"
_DANGER    = "#c0392b"
_NEUTRAL   = "#95a5a6"
_LIGHT     = "#ecf0f1"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FIG_DPI = 150

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# 1. Delay probability by hour (line chart)
# ---------------------------------------------------------------------------
def plot_delay_by_hour(df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        df.groupby("hour")["is_delayed"]
        .agg(prob="mean", sem=lambda x: x.sem(), n="count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    # Rush-hour bands
    rush_hours = [7, 8, 9, 16, 17, 18]
    for h in rush_hours:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.12, color=_ACCENT, zorder=0)

    ax.plot(agg["hour"], agg["prob"], color=_PRIMARY, lw=2.5, marker="o", markersize=5, zorder=3)
    ax.fill_between(
        agg["hour"],
        (agg["prob"] - agg["sem"]).clip(lower=0),
        agg["prob"] + agg["sem"],
        alpha=0.15, color=_PRIMARY, zorder=2,
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Delay Probability", fontsize=11)
    ax.set_title("Delay Probability by Hour of Day", fontsize=14, fontweight="bold", pad=12)

    legend_patch = mpatches.Patch(color=_ACCENT, alpha=0.4, label="Rush Hours (7–9, 16–18)")
    ax.legend(handles=[legend_patch], loc="upper left", frameon=True)

    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Delay probability by day of week (bar chart)
# ---------------------------------------------------------------------------
def plot_delay_by_day(df: pd.DataFrame, out_path: Path) -> None:
    agg = df.groupby("weekday")["is_delayed"].mean().reset_index()
    agg.columns = ["weekday", "prob"]
    agg["weekday"] = pd.Categorical(agg["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    agg = agg.sort_values("weekday").dropna(subset=["weekday"])

    colors = [
        _DANGER  if p >= 0.40 else
        _ACCENT  if p >= 0.25 else
        _SAFE
        for p in agg["prob"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(agg["weekday"], agg["prob"], color=colors, width=0.6,
                  edgecolor="white", linewidth=0.8)

    for bar, p in zip(bars, agg["prob"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{p:.1%}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, agg["prob"].max() * 1.25)
    ax.set_xlabel("")
    ax.set_ylabel("Delay Probability", fontsize=11)
    ax.set_title("Delay Probability by Day of Week", fontsize=14, fontweight="bold", pad=12)

    legend_items = [
        mpatches.Patch(color=_DANGER, label="High risk  (≥40%)"),
        mpatches.Patch(color=_ACCENT, label="Medium risk (25–40%)"),
        mpatches.Patch(color=_SAFE,   label="Low risk   (<25%)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True, fontsize=9)

    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Heatmap: hour × day of week
# ---------------------------------------------------------------------------
def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.groupby(["weekday", "hour"])["is_delayed"].mean().unstack(fill_value=np.nan)
    present_days = [d for d in WEEKDAY_ORDER if d in pivot.index]
    pivot = pivot.loc[present_days]

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=min(pivot.max().max(), 1.0),
        annot=True,
        fmt=".0%",
        annot_kws={"size": 7},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Delay Probability", "shrink": 0.75},
    )
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Delay Risk Heatmap: Day of Week × Hour", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Feature importance (horizontal bar)
# ---------------------------------------------------------------------------
def plot_feature_importance(importance_csv: Path, out_path: Path, top_n: int = 12) -> None:
    if not importance_csv.exists():
        print(f"  Skipped feature importance — {importance_csv.name} not found")
        return

    df = pd.read_csv(importance_csv).head(top_n).sort_values("importance_mean")

    # Friendly display names
    name_map = {
        "hour": "Hour of Day",
        "weekday": "Day of Week",
        "is_rush_hour": "Rush Hour",
        "station_id": "Station",
        "direction": "Travel Direction",
        "month": "Month",
        "is_weekend": "Weekend",
        "temp_c": "Temperature (°C)",
        "wind_gusts_kmh": "Wind Gusts",
        "precip_mm": "Precipitation",
        "is_freezing": "Freezing Conditions",
        "has_precipitation": "Has Precipitation",
        "high_winds": "High Winds",
        "has_event": "Major Event",
        "construction_impact": "Construction",
        "strike_impact": "Strike",
        "is_holiday": "Public Holiday",
        "train_type": "Train Type",
    }
    df["label"] = df["feature"].map(lambda f: name_map.get(f, f.replace("_", " ").title()))

    fig, ax = plt.subplots(figsize=(9, 6))
    has_std = "importance_std" in df.columns
    ax.barh(
        df["label"],
        df["importance_mean"],
        xerr=df["importance_std"] if has_std else None,
        color=_PRIMARY,
        edgecolor="white",
        height=0.65,
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": _NEUTRAL},
    )

    ax.set_xlabel("Permutation Importance (Recall Drop)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold", pad=12)
    ax.axvline(0, color=_NEUTRAL, lw=0.8, linestyle="--")

    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Model comparison (grouped bar chart)
# ---------------------------------------------------------------------------
def plot_model_comparison(scan_csv: Path, out_path: Path) -> None:
    if not scan_csv.exists():
        print(f"  Skipped model comparison — {scan_csv.name} not found")
        return

    df = pd.read_csv(scan_csv)
    # Best row per model (highest composite score)
    best = (
        df.sort_values("score", ascending=False)
        .groupby("model", sort=False)
        .first()
        .reset_index()
    )

    metrics = ["accuracy", "recall", "precision", "f1"]
    metric_labels = ["Accuracy", "Recall", "Precision", "F1"]
    colors = [_PRIMARY, _DANGER, _ACCENT, _SAFE]
    x = np.arange(len(best))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, best[metric], width, label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(best["model"], fontsize=11)
    ax.set_ylabel("Score (at Optimal Threshold)", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Model Comparison: Logistic Regression vs Random Forest vs Gradient Boosting",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Delay distribution histogram
# ---------------------------------------------------------------------------
def plot_delay_distribution(df: pd.DataFrame, out_path: Path) -> None:
    # Cap display at 60 min for readability; show percentage of trips beyond cap
    cap = 60
    capped = df[df["delay_in_min"] <= cap]["delay_in_min"]
    pct_beyond = (df["delay_in_min"] > cap).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(capped, bins=range(0, cap + 2, 1), color=_PRIMARY, edgecolor="white", linewidth=0.4)
    ax.axvline(DELAY_THRESHOLD_MINUTES, color=_DANGER, lw=2, linestyle="--",
               label=f"Delay threshold ({DELAY_THRESHOLD_MINUTES} min)")

    ax.set_xlabel("Delay (minutes)", fontsize=11)
    ax.set_ylabel("Number of Trips", fontsize=11)
    ax.set_title("Distribution of Train Delays (0–60 min)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=True)

    if pct_beyond > 0:
        ax.text(0.98, 0.95, f"{pct_beyond:.1%} of trips > {cap} min (not shown)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=_NEUTRAL)

    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate_all(data_file: Path = ENRICHED_DATA_FILE, reports_dir: Path = REPORTS_DIR) -> None:
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        print(f"Data file not found: {data_file}. Run steps 1–2 first.")
        return

    print("Loading data…")
    df = pd.read_csv(data_file)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "delay_in_min"]).copy()
    df["is_delayed"] = (df["delay_in_min"] > DELAY_THRESHOLD_MINUTES).astype(int)

    print("Generating figures…")
    plot_delay_by_hour(df, figures_dir / "delay_by_hour.png")
    plot_delay_by_day(df, figures_dir / "delay_by_day.png")
    plot_heatmap(df, figures_dir / "delay_heatmap.png")

    feat_dir = reports_dir / "feature_analysis"
    plot_feature_importance(
        feat_dir / "top10_features_from_training.csv",
        figures_dir / "feature_importance.png",
    )
    plot_model_comparison(
        feat_dir / "threshold_scan_by_model.csv",
        figures_dir / "model_comparison.png",
    )
    plot_delay_distribution(df, figures_dir / "delay_distribution.png")

    saved = list(figures_dir.glob("*.png"))
    print(f"\nDone — {len(saved)} figures saved to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ENRICHED_DATA_FILE))
    parser.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR))
    args = parser.parse_args()
    generate_all(data_file=Path(args.input), reports_dir=Path(args.reports_dir))
