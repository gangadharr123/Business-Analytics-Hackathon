"""Step 8: Run key studies and produce decision-support outputs for commuters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DELAY_THRESHOLD_MINUTES, ENRICHED_DATA_FILE, REPORTS_DIR


CORRIDOR_KEYWORDS = [
    "frankfurt", "wiesbaden", "eltville", "oestrich", "hattenheim", "geisenheim", "rüdesheim", "rudesheim"
]


def load_data(path: Path, corridor_only: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    required = {"time", "delay_in_min", "weekday", "hour", "train_type", "station_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "delay_in_min"]).copy()

    if corridor_only:
        station_l = df["station_name"].astype(str).str.lower()
        mask = False
        for k in CORRIDOR_KEYWORDS:
            mask = mask | station_l.str.contains(k, regex=False)
        df = df[mask].copy()

    df["is_delayed"] = (df["delay_in_min"] > DELAY_THRESHOLD_MINUTES).astype(int)
    return df


def probability_by_time_day(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["weekday", "hour"], as_index=False)
        .agg(
            delay_probability=("is_delayed", "mean"),
            avg_delay_min=("delay_in_min", "mean"),
            trips=("is_delayed", "size"),
        )
        .sort_values(["weekday", "hour"])
    )


MIN_TRIPS_FOR_RELIABILITY = 30


def probability_for_specific_train(df: pd.DataFrame) -> pd.DataFrame:
    # If a real train identifier exists, use it; else use a practical proxy
    id_candidates = ["train_id", "trip_id", "train_number", "line", "trip_short_name"]
    chosen_id = next((c for c in id_candidates if c in df.columns), None)

    if chosen_id:
        group_cols = [chosen_id, "station_name", "hour"]
        out = (
            df.groupby(group_cols, as_index=False)
            .agg(
                delay_probability=("is_delayed", "mean"),
                avg_delay_min=("delay_in_min", "mean"),
                p90_delay_min=("delay_in_min", lambda x: float(np.percentile(x, 90))),
                trips=("is_delayed", "size"),
            )
            .sort_values("delay_probability", ascending=False)
        )
        out.rename(columns={chosen_id: "train_identifier"}, inplace=True)
        out["identifier_type"] = chosen_id
    else:
        out = (
            df.groupby(["train_type", "station_name", "hour"], as_index=False)
            .agg(
                delay_probability=("is_delayed", "mean"),
                avg_delay_min=("delay_in_min", "mean"),
                p90_delay_min=("delay_in_min", lambda x: float(np.percentile(x, 90))),
                trips=("is_delayed", "size"),
            )
            .sort_values("delay_probability", ascending=False)
        )
        out["identifier_type"] = "proxy(train_type+station+hour)"
        out.rename(columns={"train_type": "train_identifier"}, inplace=True)

    # Filter out statistically unreliable slots (fewer than MIN_TRIPS_FOR_RELIABILITY observations)
    out = out[out["trips"] >= MIN_TRIPS_FOR_RELIABILITY].reset_index(drop=True)
    return out


def reliability_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    delayed = int(df["is_delayed"].sum())
    on_time = total - delayed
    return {
        "delay_threshold_min": DELAY_THRESHOLD_MINUTES,
        "total_trips": int(total),
        "delayed_trips": delayed,
        "on_time_trips": int(on_time),
        "on_time_rate": float(on_time / total) if total else 0.0,
        "delay_rate": float(delayed / total) if total else 0.0,
        "avg_delay_min": float(df["delay_in_min"].mean()) if total else 0.0,
        "p90_delay_min": float(np.percentile(df["delay_in_min"], 90)) if total else 0.0,
    }


def factor_effects(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "is_holiday", "construction_impact", "strike_impact", "rain_mm", "precip_mm", "wind_gusts_kmh",
        "temp_c", "is_peak_hour", "is_weekend", "hour", "month"
    ]
    available = [c for c in candidates if c in df.columns]
    rows = []

    base = float(df["is_delayed"].mean())
    for c in available:
        if pd.api.types.is_numeric_dtype(df[c]):
            high = df[c] > df[c].median()
            low = ~high
            if high.sum() == 0 or low.sum() == 0:
                continue
            rows.append(
                {
                    "factor": c,
                    "delay_prob_high_group": float(df.loc[high, "is_delayed"].mean()),
                    "delay_prob_low_group": float(df.loc[low, "is_delayed"].mean()),
                    "uplift_vs_base": float(df.loc[high, "is_delayed"].mean() - base),
                    "base_delay_probability": base,
                }
            )

    return pd.DataFrame(rows).sort_values("uplift_vs_base", ascending=False)


def buffer_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["weekday", "hour"], as_index=False)
        .agg(
            delay_probability=("is_delayed", "mean"),
            p80_delay=("delay_in_min", lambda x: float(np.percentile(x, 80))),
            p90_delay=("delay_in_min", lambda x: float(np.percentile(x, 90))),
            trips=("is_delayed", "size"),
        )
    )

    def rule(prob: float, p90: float) -> int:
        if prob >= 0.6:
            return int(max(15, round(p90)))
        if prob >= 0.4:
            return int(max(10, round(p90 * 0.8)))
        if prob >= 0.25:
            return int(max(7, round(p90 * 0.6)))
        return 5

    grouped["recommended_buffer_min"] = grouped.apply(
        lambda r: rule(float(r["delay_probability"]), float(r["p90_delay"])), axis=1
    )
    return grouped.sort_values(["weekday", "hour"])


def write_summary_markdown(output_dir: Path, reliability: dict, p_time_day: pd.DataFrame, p_train: pd.DataFrame, effects: pd.DataFrame):
    top_risky_slots = (
        p_time_day[p_time_day["trips"] >= MIN_TRIPS_FOR_RELIABILITY]
        .sort_values("delay_probability", ascending=False)
        .head(5)
    )
    top_train_proxy = p_train.head(5)
    top_factors = effects.head(10)

    lines = [
        "# Key Studies Summary",
        "",
        "## Objective 1: Delay analysis and key drivers",
        f"- Delay threshold used: **>{reliability['delay_threshold_min']} min**",
        f"- Overall delay probability: **{reliability['delay_rate']:.1%}**",
        f"- On-time reliability: **{reliability['on_time_rate']:.1%}**",
        "",
        "### Highest-risk weekday/hour slots (Top 5)",
    ]
    for _, r in top_risky_slots.iterrows():
        lines.append(f"- {r['weekday']} {int(r['hour']):02d}:00 → delay probability {r['delay_probability']:.1%} (n={int(r['trips'])})")

    lines.append("")
    lines.append("### Highest-risk specific-train identifiers/proxies (Top 5)")
    for _, r in top_train_proxy.iterrows():
        lines.append(
            f"- {r['train_identifier']} @ {r['station_name']} {int(r['hour']):02d}:00 → {r['delay_probability']:.1%} (n={int(r['trips'])})"
        )

    lines.append("")
    lines.append("### Key factors associated with higher delay risk (Top 10)")
    for _, r in top_factors.iterrows():
        lines.append(f"- {r['factor']}: uplift vs base = {r['uplift_vs_base']:+.2%}")

    lines.append("")
    lines.append("## Objective 2: Decision support")
    lines.append("- Use `study3_buffer_recommendations.csv` to pick recommended buffer minutes by weekday/hour.")
    lines.append("- Use model inference (`06_smart_commute_tool.py`) for trip-level probability and alert label.")

    (output_dir / "key_studies_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_file: Path, output_dir: Path, corridor_only: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_file, corridor_only=corridor_only)

    p_time_day = probability_by_time_day(df)
    p_train = probability_for_specific_train(df)
    reliability = reliability_summary(df)
    effects = factor_effects(df)
    buffer = buffer_recommendations(df)

    # canonical study outputs answering the key questions
    p_time_day.to_csv(output_dir / "q1_delay_probability_by_time_day.csv", index=False)
    p_train.to_csv(output_dir / "q2_delay_probability_specific_train.csv", index=False)
    (output_dir / "q3_reliability_summary.json").write_text(json.dumps(reliability, indent=2), encoding="utf-8")
    effects.to_csv(output_dir / "q4_factors_affecting_delays.csv", index=False)
    buffer.to_csv(output_dir / "q5_buffer_planning_recommendations.csv", index=False)

    # keep backward-compatible filenames
    p_time_day.to_csv(output_dir / "study1_delay_probability_by_time_day.csv", index=False)
    p_train.to_csv(output_dir / "study1_delay_probability_by_train_proxy.csv", index=False)
    effects.to_csv(output_dir / "study2_factor_effects.csv", index=False)
    buffer.to_csv(output_dir / "study3_buffer_recommendations.csv", index=False)
    (output_dir / "study1_reliability_summary.json").write_text(json.dumps(reliability, indent=2), encoding="utf-8")

    write_summary_markdown(output_dir, reliability, p_time_day, p_train, effects)
    print(f"Saved key studies to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ENRICHED_DATA_FILE))
    parser.add_argument("--output-dir", type=str, default=str(REPORTS_DIR / "key_studies"))
    parser.add_argument("--all-areas", action="store_true", help="Use all areas; default focuses on Frankfurt↔Rheingau corridor")
    args = parser.parse_args()
    run(input_file=Path(args.input), output_dir=Path(args.output_dir), corridor_only=not args.all_areas)
