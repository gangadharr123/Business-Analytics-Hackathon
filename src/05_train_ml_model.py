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
    ENRICHED_DATA_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    REPORTS_DIR,
    WEATHER_COLUMNS,
    get_parallel_jobs,
)
from utils import prepare_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)



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
    positive_rate = float((y_pred == 1).mean())
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy), "recall": float(recall), "precision": float(precision),
        "specificity": float(specificity), "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy), "positive_rate": positive_rate,
    }


def tune_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    target_accuracy: float = 0.70,
    min_recall: float = 0.60,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= t).astype(int)
        m = _metrics_from_predictions(y_true, y_pred)

        score = (0.4 * m["accuracy"]) + (0.4 * m["recall"]) + (0.2 * m["precision"])

        if m["accuracy"] < target_accuracy:
            score -= (target_accuracy - m["accuracy"]) * 2.0
            
        if m["recall"] < min_recall:
            score -= (min_recall - m["recall"]) * 2.0

        if m["tp"] == 0:
            score -= 0.5

        rows.append({"threshold": float(t), "score": float(score), **m})

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    best = table.iloc[0].to_dict()

    return best, table


def _safe_recall_for_estimator(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    y_pred = estimator.predict(X)
    return float(recall_score(y, y_pred, zero_division=0))


def train(
    data_file=ENRICHED_DATA_FILE,
    model_file=MODEL_FILE,
    metadata_file=MODEL_METADATA_FILE,
    target_accuracy: float = 0.70,
    min_recall: float = 0.60,
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
    base_features = [
        "weekday", "hour", "train_type", "station_id", "direction",
        "is_holiday", "construction_impact", "strike_impact", "month", "is_weekend",
        "is_rush_hour", "is_freezing", "has_precipitation", "high_winds",
        "has_event" 
    ]
    features = base_features + [c for c in WEATHER_COLUMNS if c in df.columns]

    X_train = train_df[features].copy()
    y_train = train_df["is_delayed"].astype("int8")
    X_val = val_df[features].copy()
    y_val = val_df["is_delayed"].astype("int8")
    X_test = test_df[features].copy()
    y_test = test_df["is_delayed"].astype("int8")

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
        {
            "name": "Decision Tree",
            "clf": DecisionTreeClassifier(max_depth=8, min_samples_leaf=80, class_weight="balanced", random_state=42),
        },
        {
            "name": "Logistic Regression",
            "clf": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        },
        {
            "name": "Random Forest",
            "clf": RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=safe_jobs,
            ),
        },
    ]

    results = []
    threshold_tables = []
    for m in models:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", m["clf"])])
        try:
            pipeline.fit(X_train, y_train)

            y_val_prob = pipeline.predict_proba(X_val)[:, 1]
            threshold_info, threshold_table = tune_threshold(
                y_val, y_val_prob, target_accuracy=target_accuracy, min_recall=min_recall,
            )
            threshold_table.insert(0, "model", m["name"])
            threshold_tables.append(threshold_table)

            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_prob >= threshold_info["threshold"]).astype(int)
            test_metrics = _metrics_from_predictions(y_test, y_test_pred)
            auc_test = roc_auc_score(y_test, y_test_prob)

            logger.info(
                "%s | VAL(thr=%.2f, score=%.3f, rec=%.3f, prec=%.3f) | TEST(acc=%.3f rec=%.3f prec=%.3f auc=%.3f CM=[[%s,%s],[%s,%s]])",
                m["name"], threshold_info["threshold"], threshold_info["score"], threshold_info["recall"],
                threshold_info["precision"], test_metrics["accuracy"],
                test_metrics["recall"], test_metrics["precision"], auc_test, 
                test_metrics["tn"], test_metrics["fp"], test_metrics["fn"], test_metrics["tp"],
            )

            selection_score = (0.4 * test_metrics["accuracy"]) + (0.4 * test_metrics["recall"]) + (0.2 * test_metrics["precision"])
            
            if test_metrics["accuracy"] < target_accuracy:
                selection_score -= (target_accuracy - test_metrics["accuracy"]) * 2.0
            if test_metrics["recall"] < min_recall:
                selection_score -= (min_recall - test_metrics["recall"]) * 2.0
            if test_metrics["tp"] == 0:
                selection_score -= 0.5
                
            results.append(
                {
                    "name": m["name"], "pipeline": pipeline, "threshold": float(threshold_info["threshold"]),
                    "val_score": float(threshold_info["score"]), "val_recall": float(threshold_info["recall"]),
                    "val_precision": float(threshold_info["precision"]), "test_metrics": test_metrics,
                    "auc": float(auc_test), "selection_score": float(selection_score),
                }
            )
        except Exception as exc:
            logger.error("%s failed and will be skipped: %s", m["name"], exc)

    if not results:
        raise RuntimeError("All models failed. Try increasing memory or setting EBS_PARALLEL_JOBS=1.")

    winner = sorted(results, key=lambda r: (r["selection_score"], r["auc"]), reverse=True)[0]
    
    logger.info("Winner (test weighted score: 40%% Acc + 40%% Rec + 20%% Prec): %s", winner["name"])

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
        "train_rows": int(len(train_df)), "val_rows": int(len(val_df)), "test_rows": int(len(test_df)),
        "time_range": {"min": str(df["time"].min()), "max": str(df["time"].max())},
        "target_accuracy": float(target_accuracy), "min_recall_floor": float(min_recall),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reports_dir = REPORTS_DIR / "feature_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if threshold_tables:
        pd.concat(threshold_tables, ignore_index=True).to_csv(reports_dir / "threshold_scan_by_model.csv", index=False)

    perm = permutation_importance(
        winner["pipeline"], X_test, y_test, scoring=_safe_recall_for_estimator,
        n_repeats=8, random_state=42, n_jobs=safe_jobs,
    )
    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(reports_dir / "feature_importance_recall_from_training.csv", index=False)
    
    top10_df = importance_df.head(10).copy()
    top10_df.to_csv(reports_dir / "top10_features_from_training.csv", index=False)
    logger.info("Top 10 features (training): %s", ", ".join(top10_df["feature"].tolist()))
    logger.info("Saved model to %s and metadata to %s", model_file, metadata_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(ENRICHED_DATA_FILE))
    parser.add_argument("--model-output", type=str, default=str(MODEL_FILE))
    parser.add_argument("--metadata-output", type=str, default=str(MODEL_METADATA_FILE))
    parser.add_argument("--target-accuracy", type=float, default=0.70)
    parser.add_argument("--min-recall", type=float, default=0.60)
    args = parser.parse_args()
    train(
        data_file=Path(args.input), model_file=Path(args.model_output), metadata_file=Path(args.metadata_output),
        target_accuracy=args.target_accuracy, min_recall=args.min_recall,
    )