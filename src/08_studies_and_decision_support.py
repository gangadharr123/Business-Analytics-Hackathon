"""Step 8: Run key studies and produce decision-support outputs for commuters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ENRICHED_DATA_FILE, REPORTS_DIR


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    required = {"time", "delay_in_min", "weekday", "hour", "train_type", "station_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "delay_in_min"]).copy()
    df["is_delayed"] = (df["delay_in_min"] > 3).astype(int)
    return df


def probability_by_time_day(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["weekday", "hour"], as_index=False)
        .agg(
            delay_probability=("is_delayed", "mean"),
            avg_delay_min=("delay_in_min", "mean"),
            trips=("is_delayed", "size"),
        )
        .sort_values(["weekday", "hour"])
    )
    return out


def probability_for_specific_train(df: pd.DataFrame) -> pd.DataFrame:
    # proxy for "specific train": train type + station + hour bucket
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
    return out


def reliability_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    delayed = int(df["is_delayed"].sum())
    on_time = total - delayed
    return {
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
        "is_holiday",
        "construction_impact",
        "strike_impact",
        "rain_mm",
        "precip_mm",
        "wind_gusts_kmh",
        "temp_c",
        "is_peak_hour",
        "is_weekend",
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
                }
            )

    return pd.DataFrame(rows).sort_values("uplift_vs_base", ascending=False)


def buffer_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    # Decision support table by day/hour: how much extra buffer to add
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


def run(input_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_file)

    p_time_day = probability_by_time_day(df)
    p_train = probability_for_specific_train(df)
    reliability = reliability_summary(df)
    effects = factor_effects(df)
    buffer = buffer_recommendations(df)

    p_time_day.to_csv(output_dir / "study1_delay_probability_by_time_day.csv", index=False)
    p_train.to_csv(output_dir / "study1_delay_probability_by_train_proxy.csv", index=False)
    effects.to_csv(output_dir / "study2_factor_effects.csv", index=False)
    buffer.to_csv(output_dir / "study3_buffer_recommendations.csv", index=False)
    (output_dir / "study1_reliability_summary.json").write_text(json.dumps(reliability, indent=2), encoding="utf-8")

    print(f"Saved key studies to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ENRICHED_DATA_FILE))
    parser.add_argument("--output-dir", type=str, default=str(REPORTS_DIR / "key_studies"))
    args = parser.parse_args()
    run(input_file=Path(args.input), output_dir=Path(args.output_dir))
