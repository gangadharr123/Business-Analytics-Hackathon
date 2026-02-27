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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
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

    # Lag features (causal, station-level)
    df = df.sort_values(["station_id", "time"]).reset_index(drop=True)
    df["lag1_delay_min_station"] = df.groupby("station_id")["delay_in_min"].shift(1).fillna(0)
    df["lag1_is_delayed_station"] = df.groupby("station_id")["is_delayed"].shift(1).fillna(0)
    df["rolling_delay_ratio_6_station"] = (
        df.groupby("station_id")["is_delayed"]
        .apply(lambda s: s.shift(1).rolling(window=6, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    return df.sort_values("time").reset_index(drop=True)


def split_time_based_three(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def _metrics_from_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy), "recall": float(recall), "precision": float(precision),
        "specificity": float(specificity), "f1": float(f1), "balanced_accuracy": float(balanced_accuracy),
    }


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray, min_precision: float = 0.35, min_recall: float = 0.55):
    best = {"threshold": 0.5, "score": -1e9}
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= t).astype(int)
        m = _metrics_from_predictions(y_true, y_pred)

        score = (
            0.35 * m["f1"]
            + 0.25 * m["balanced_accuracy"]
            + 0.20 * m["precision"]
            + 0.20 * m["recall"]
        )

        if m["precision"] < min_precision:
            score -= (min_precision - m["precision"]) * 1.5
        if m["recall"] < min_recall:
            score -= (min_recall - m["recall"]) * 1.0

        if score > best["score"]:
            best = {"threshold": float(t), "score": float(score), **m}

    return best


def add_smoothed_station_hour_risk(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float,
):
    global_delay_rate = float(train_df["is_delayed"].mean())
    agg = (
        train_df.groupby(["station_id", "hour"])["is_delayed"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "local_mean", "count": "n"})
    )
    agg["smoothed"] = (agg["local_mean"] * agg["n"] + global_delay_rate * alpha) / (agg["n"] + alpha)
    risk_map = agg["smoothed"]

    train_df["station_hour_risk"] = train_df.set_index(["station_id", "hour"]).index.map(risk_map).fillna(global_delay_rate)
    val_df["station_hour_risk"] = val_df.set_index(["station_id", "hour"]).index.map(risk_map).fillna(global_delay_rate)
    test_df["station_hour_risk"] = test_df.set_index(["station_id", "hour"]).index.map(risk_map).fillna(global_delay_rate)


def train(
    data_file=ENRICHED_DATA_FILE,
    model_file=MODEL_FILE,
    metadata_file=MODEL_METADATA_FILE,
    min_precision: float = 0.35,
    min_recall: float = 0.55,
    target_encoding_alpha: float = 20.0,
):
    safe_jobs = get_parallel_jobs()
    logger.info("Using parallel jobs: %s", safe_jobs)
    if not data_file.exists():
        raise FileNotFoundError(f"Data not found at {data_file}. Run Step 2.")

    df = pd.read_csv(data_file)
    required_cols = {"time", "station_name", "delay_in_min", "final_destination_station", "hour", "weekday", "train_type"}
    if missing := sorted(required_cols - set(df.columns)):
        raise ValueError(f"Input schema invalid. Missing columns: {missing}")

    df = prepare_features(df)
    train_df, val_df, test_df = split_time_based_three(df)
    add_smoothed_station_hour_risk(train_df, val_df, test_df, alpha=target_encoding_alpha)

    base_features = [
        "weekday", "hour", "train_type", "station_id", "direction",
        "is_holiday", "construction_impact", "strike_impact", "month", "day_of_month",
        "hour_sin", "hour_cos", "is_weekend", "is_peak_hour", "station_hour_risk",
        "adverse_weather_score", "temp_extreme_flag", "wind_precip_interaction", "heavy_rain_flag",
        "lag1_delay_min_station", "lag1_is_delayed_station", "rolling_delay_ratio_6_station",
    ]
    features = base_features + [c for c in WEATHER_COLUMNS if c in df.columns]

    X_train = train_df[features].copy()
    y_train = train_df["is_delayed"].astype("int8")
    X_val = val_df[features].copy()
    y_val = val_df["is_delayed"].astype("int8")
    X_test = test_df[features].copy()
    y_test = test_df["is_delayed"].astype("int8")

    # Downcast numeric feature columns to reduce memory pressure on Windows/Python 3.14
    for _df in (X_train, X_val, X_test):
        num_cols_tmp = _df.select_dtypes(include=["float64", "int64"]).columns
        _df[num_cols_tmp] = _df[num_cols_tmp].apply(pd.to_numeric, downcast="float")

    cat_cols = ["weekday", "train_type", "station_id"]
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    models = [
        {"name": "Decision Tree", "clf": DecisionTreeClassifier(max_depth=10, min_samples_leaf=80, class_weight="balanced", random_state=42)},
        {"name": "Logistic Regression", "clf": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)},
        {"name": "Random Forest", "clf": RandomForestClassifier(n_estimators=500, max_depth=18, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=safe_jobs)},
    ]

    results = []
    for m in models:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", m["clf"])])
        try:
            pipeline.fit(X_train, y_train)

            y_val_prob = pipeline.predict_proba(X_val)[:, 1]
            threshold_info = tune_threshold(y_val, y_val_prob, min_precision=min_precision, min_recall=min_recall)

            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_prob >= threshold_info["threshold"]).astype(int)
            test_metrics = _metrics_from_predictions(y_test, y_test_pred)
            auc_test = roc_auc_score(y_test, y_test_prob)

            logger.info(
                "%s | VAL(thr=%.2f, score=%.3f, rec=%.3f, prec=%.3f) | TEST(acc=%.3f rec=%.3f prec=%.3f bal_acc=%.3f auc=%.3f CM=[[%s,%s],[%s,%s]])",
                m["name"], threshold_info["threshold"], threshold_info["score"], threshold_info["recall"], threshold_info["precision"],
                test_metrics["accuracy"], test_metrics["recall"], test_metrics["precision"], test_metrics["balanced_accuracy"], auc_test,
                test_metrics["tn"], test_metrics["fp"], test_metrics["fn"], test_metrics["tp"],
            )

            results.append(
                {
                    "name": m["name"],
                    "pipeline": pipeline,
                    "threshold": threshold_info["threshold"],
                    "val_score": threshold_info["score"],
                    "val_recall": threshold_info["recall"],
                    "val_precision": threshold_info["precision"],
                    "test_metrics": test_metrics,
                    "auc": float(auc_test),
                }
            )
        except MemoryError as exc:
            logger.error("%s failed due to memory error and will be skipped: %s", m["name"], exc)
        except Exception as exc:
            logger.error("%s failed and will be skipped: %s", m["name"], exc)

    if not results:
        raise RuntimeError("All models failed during training. Try increasing memory, reducing dataset size, or setting EBS_PARALLEL_JOBS=1.")

    winner = sorted(results, key=lambda r: (r["val_score"], r["auc"]), reverse=True)[0]
    logger.info("Winner (based on validation score): %s", winner["name"])

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
        "validation": {"score": winner["val_score"], "recall": winner["val_recall"], "precision": winner["val_precision"]},
        "test": {**winner["test_metrics"], "auc": winner["auc"]},
        "features": features,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "time_range": {"min": str(df["time"].min()), "max": str(df["time"].max())},
        "target_encoding_alpha": target_encoding_alpha,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reports_dir = REPORTS_DIR / "feature_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    perm = permutation_importance(
        winner["pipeline"], X_test, y_test, scoring="recall", n_repeats=8, random_state=42, n_jobs=safe_jobs
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
    parser.add_argument("--min-precision", type=float, default=0.35)
    parser.add_argument("--min-recall", type=float, default=0.55)
    parser.add_argument("--target-encoding-alpha", type=float, default=20.0)
    args = parser.parse_args()
    train(
        data_file=Path(args.input),
        model_file=Path(args.model_output),
        metadata_file=Path(args.metadata_output),
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        target_encoding_alpha=args.target_encoding_alpha,
    )
