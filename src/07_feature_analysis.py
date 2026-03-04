"""Step 7: Simple data availability and feature impact analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DELAY_THRESHOLD_MINUTES,
    ENRICHED_DATA_FILE,
    REPORTS_DIR,
    STATION_MAP,
    WEATHER_COLUMNS,
    get_parallel_jobs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_direction(final_dest: str) -> int:
    return 0 if "Frankfurt" in str(final_dest) else 1


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "delay_in_min"])
    df["is_delayed"] = (df["delay_in_min"] > DELAY_THRESHOLD_MINUTES).astype(int)
    df["station_id"] = df["station_name"].map(STATION_MAP)
    df = df.dropna(subset=["station_id"])

    df["direction"] = df["final_destination_station"].apply(get_direction)
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    for c in WEATHER_COLUMNS:
        if c not in df.columns:
            df[c] = 0

    df["is_freezing"] = (df["temp_c"] <= 0).astype(int)
    df["has_precipitation"] = ((df["precip_mm"] > 0) | (df["rain_mm"] > 0) | (df["snow_cm"] > 0)).astype(int)
    df["high_winds"] = (df["wind_gusts_kmh"] >= 40).astype(int)
    
    # HARDCODED EVENTS
    event_dates = {
        "2024-10-27", "2025-10-26", # Frankfurt Marathon
        "2024-11-11", "2025-02-27", "2025-03-03", "2025-03-04", "2025-11-11", # Mainz Fastnacht
        "2025-02-23", # German Federal Election
        "2024-10-31", "2025-10-31"  # Halloween
    }
    
    date_strings = df["time"].dt.strftime("%Y-%m-%d")
    is_event_day = date_strings.isin(event_dates).astype(int)
    
    if "has_event" not in df.columns:
        df["has_event"] = is_event_day
    else:
        df["has_event"] = (df["has_event"].fillna(0).astype(int) | is_event_day)

    return df.sort_values("time").reset_index(drop=True)


def split_time_based(df: pd.DataFrame):
    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_analysis(input_file: Path, output_dir: Path):
    safe_jobs = get_parallel_jobs()
    logger.info("Using parallel jobs: %s", safe_jobs)
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

    df = prepare_features(raw)
    train_df, test_df = split_time_based(df)

    base_features = [
        "weekday", "hour", "train_type", "station_id", "direction",
        "is_holiday", "construction_impact", "strike_impact", "month", "is_weekend",
        "is_rush_hour", "is_freezing", "has_precipitation", "high_winds",
        "has_event" 
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
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=safe_jobs,
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

    perm = permutation_importance(pipeline, X_test, y_test, scoring="recall", n_repeats=8, random_state=42, n_jobs=safe_jobs)
    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)
    
    importance_df.to_csv(output_dir / "feature_importance_recall.csv", index=False)
    top10_df = importance_df.head(10).copy()
    top10_df.to_csv(output_dir / "top10_features_recall.csv", index=False)
    logger.info("Top 10 features (analysis): %s", ", ".join(top10_df["feature"].tolist()))

    weather_subset = importance_df[importance_df["feature"].isin(WEATHER_COLUMNS + ["is_freezing", "has_precipitation", "high_winds"])]
    weather_subset.to_csv(output_dir / "weather_feature_importance.csv", index=False)

    logger.info("Saved analysis reports to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ENRICHED_DATA_FILE))
    parser.add_argument("--output-dir", type=str, default=str(REPORTS_DIR / "feature_analysis"))
    args = parser.parse_args()
    run_analysis(input_file=Path(args.input), output_dir=Path(args.output_dir))