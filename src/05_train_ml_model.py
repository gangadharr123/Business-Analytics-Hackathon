"""Step 3: Train and compare models for DB delay risk prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    DELAY_THRESHOLD_MINUTES,
    ENRICHED_DATA_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    REPORTS_DIR,
    STATION_MAP,
    WEATHER_COLUMNS,
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

    # additional feature engineering
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

    # Weather-focused feature engineering
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


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray):
    best = {"threshold": 0.5, "accuracy": 0.0, "recall": 0.0, "score": -1.0}
    for t in np.arange(0.2, 0.81, 0.02):
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        score = rec * 0.7 + acc * 0.3
        if score > best["score"]:
            best = {"threshold": float(t), "accuracy": float(acc), "recall": float(rec), "score": float(score)}
    return best


def train(data_file=ENRICHED_DATA_FILE, model_file=MODEL_FILE, metadata_file=MODEL_METADATA_FILE):
    if not data_file.exists():
        raise FileNotFoundError(f"Data not found at {data_file}. Run Step 2.")

    df = pd.read_csv(data_file)
    required_cols = {"time", "station_name", "delay_in_min", "final_destination_station", "hour", "weekday", "train_type"}
    if missing := sorted(required_cols - set(df.columns)):
        raise ValueError(f"Input schema invalid. Missing columns: {missing}")

    df = prepare_features(df)
    train_df, test_df = split_time_based(df)

    # leakage-safe aggregate prior from train split only
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

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    models = [
        {"name": "Decision Tree", "clf": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42)},
        {"name": "Logistic Regression", "clf": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)},
        {"name": "Random Forest", "clf": RandomForestClassifier(n_estimators=400, max_depth=24, min_samples_leaf=8, class_weight={0: 1, 1: 4}, random_state=42, n_jobs=-1)},
    ]

    results = []
    for m in models:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", m["clf"])])
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        threshold_info = tune_threshold(y_test, y_prob)
        y_pred = (y_prob >= threshold_info["threshold"]).astype(int)
        auc = roc_auc_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        logger.info(
            "%s | accuracy=%.3f recall=%.3f auc=%.3f threshold=%.2f CM=[[%s,%s],[%s,%s]]",
            m["name"], threshold_info["accuracy"], threshold_info["recall"], auc,
            threshold_info["threshold"], tn, fp, fn, tp,
        )

        results.append(
            {
                "name": m["name"],
                "pipeline": pipeline,
                "threshold": threshold_info["threshold"],
                "accuracy": threshold_info["accuracy"],
                "recall": threshold_info["recall"],
                "auc": float(auc),
                "score": threshold_info["score"],
            }
        )

    winner = sorted(results, key=lambda r: r["score"], reverse=True)[0]
    logger.info("Winner: %s", winner["name"])

    model_file.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": winner["pipeline"],
        "station_map": STATION_MAP,
        "feature_columns": features,
        "threshold": winner["threshold"],
        "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES,
    }
    joblib.dump(artifact, model_file, compress=3)

    metadata = {
        "winner": winner["name"],
        "metrics": {"accuracy": winner["accuracy"], "recall": winner["recall"], "auc": winner["auc"], "score": winner["score"]},
        "features": features,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "time_range": {"min": str(df["time"].min()), "max": str(df["time"].max())},
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Post-training feature impact report (recall-focused permutation importance)
    reports_dir = REPORTS_DIR / "feature_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    perm = permutation_importance(
        winner["pipeline"], X_test, y_test, scoring="recall", n_repeats=8, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(reports_dir / "feature_importance_recall_from_training.csv", index=False)

    logger.info("Saved model to %s and metadata to %s", model_file, metadata_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ENRICHED_DATA_FILE))
    parser.add_argument("--model-output", type=str, default=str(MODEL_FILE))
    parser.add_argument("--metadata-output", type=str, default=str(MODEL_METADATA_FILE))
    args = parser.parse_args()
    train(data_file=Path(args.input), model_file=Path(args.model_output), metadata_file=Path(args.metadata_output))
