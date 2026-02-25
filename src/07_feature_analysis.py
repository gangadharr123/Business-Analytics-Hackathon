"""Step 7: Data availability and feature impact analysis for DB delay prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DELAY_THRESHOLD_MINUTES, ENRICHED_DATA_FILE, REPORTS_DIR, STATION_MAP, WEATHER_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_direction(final_dest: str) -> int:
    return 0 if "Frankfurt" in str(final_dest) else 1


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "delay_in_min"])
    df["is_delayed"] = (df["delay_in_min"] > DELAY_THRESHOLD_MINUTES).astype(int)
    df["station_id"] = df["station_name"].map(STATION_MAP)
    df = df.dropna(subset=["station_id"])

    df["direction"] = df["final_destination_station"].apply(get_direction)
    df["month"] = df["time"].dt.month
    df["day_of_month"] = df["time"].dt.day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)

    for c in WEATHER_COLUMNS:
        if c not in df.columns:
            df[c] = 0

    # Weather-centric engineered features
    df["adverse_weather_score"] = (
        1.5 * df["precip_mm"].clip(lower=0)
        + 1.0 * df["rain_mm"].clip(lower=0)
        + 2.0 * df["snow_cm"].clip(lower=0)
        + 0.03 * df["wind_gusts_kmh"].clip(lower=0)
    )
    df["temp_extreme_flag"] = ((df["temp_c"] <= -2) | (df["temp_c"] >= 30)).astype(int)
    df["wind_precip_interaction"] = (df["wind_speed_kmh"].clip(lower=0) * df["precip_mm"].clip(lower=0))
    df["heavy_rain_flag"] = (df["rain_mm"] >= 2.0).astype(int)
    return df


def split_time_based(df: pd.DataFrame):
    df = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_analysis(input_file: Path, output_dir: Path):
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_file)
    data_audit = {
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "missing_ratio": raw.isna().mean().sort_values(ascending=False).to_dict(),
        "dtypes": {k: str(v) for k, v in raw.dtypes.to_dict().items()},
    }
    (output_dir / "data_availability_report.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")

    required = {"time", "station_name", "delay_in_min", "final_destination_station", "hour", "weekday", "train_type"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns for modeling analysis: {missing}")

    df = engineer_features(raw)
    train_df, test_df = split_time_based(df)

    station_hour_risk = train_df.groupby(["station_id", "hour"])["is_delayed"].mean()
    global_delay_rate = float(train_df["is_delayed"].mean())
    train_df["station_hour_risk"] = train_df.set_index(["station_id", "hour"]).index.map(station_hour_risk).astype(float)
    test_df["station_hour_risk"] = test_df.set_index(["station_id", "hour"]).index.map(station_hour_risk).fillna(global_delay_rate).astype(float)

    base_features = [
        "weekday", "hour", "train_type", "station_id", "direction",
        "is_holiday", "construction_impact", "strike_impact", "month", "day_of_month",
        "hour_sin", "hour_cos", "is_weekend", "is_peak_hour", "station_hour_risk",
        "adverse_weather_score", "temp_extreme_flag", "wind_precip_interaction", "heavy_rain_flag",
    ]
    features = base_features + [c for c in WEATHER_COLUMNS if c in df.columns]

    X_train = train_df[features]
    y_train = train_df["is_delayed"]
    X_test = test_df[features]
    y_test = test_df["is_delayed"]

    cat_cols = ["weekday", "train_type", "station_id"]
    num_cols = [c for c in features if c not in cat_cols]

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", StandardScaler(), num_cols)]
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=24,
                    min_samples_leaf=8,
                    class_weight={0: 1, 1: 4},
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    (output_dir / "feature_analysis_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    perm = permutation_importance(pipeline, X_test, y_test, scoring="recall", n_repeats=8, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance_recall.csv", index=False)

    # Also provide grouped weather impact summary
    weather_subset = importance_df[importance_df["feature"].isin(WEATHER_COLUMNS + [
        "adverse_weather_score", "temp_extreme_flag", "wind_precip_interaction", "heavy_rain_flag"
    ])]
    weather_subset.to_csv(output_dir / "weather_feature_importance.csv", index=False)

    logger.info("Saved analysis reports to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ENRICHED_DATA_FILE))
    parser.add_argument("--output-dir", type=str, default=str(REPORTS_DIR / "feature_analysis"))
    args = parser.parse_args()
    run_analysis(input_file=Path(args.input), output_dir=Path(args.output_dir))
