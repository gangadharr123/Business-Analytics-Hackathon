"""Step 4: Inference tool for commute delay risk."""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import joblib
import pandas as pd

from config import DEFAULT_INFERENCE_PAYLOAD, MODEL_FILE


def recommend_buffer_minutes(risk_probability: float) -> int:
    """Simple decision-support rule for commuter planning buffer."""
    if risk_probability >= 0.60:
        return 15
    if risk_probability >= 0.40:
        return 10
    if risk_probability >= 0.25:
        return 7
    return 5


class SmartCommuteAdvisor:
    def __init__(self, model_filename: str | None = None):
        self.model_path = Path(model_filename) if model_filename else MODEL_FILE
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}. Run training first.")

        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.station_map = data.get("station_map", {})
        self.feature_columns = data.get("feature_columns", [])
        self.threshold = float(data.get("threshold", 0.5))

    def _get_station_id(self, name: str) -> int:
        name_l = name.lower()
        if "frankfurt" in name_l:
            return 0
        if "wiesbaden" in name_l:
            return 2
        if "eltville" in name_l:
            return 4
        if "oestrich" in name_l or "burg" in name_l:
            return 5
        if "hattenheim" in name_l or "schloss" in name_l:
            return 6
        if "geisenheim" in name_l:
            return 7
        if "rudesheim" in name_l or "rüdesheim" in name_l:
            return 8
        return 2

    def get_risk(self, source: str, dest: str, day: str, hour: int, train_type: str = "RB"):
        s_id = self._get_station_id(source)
        d_id = self._get_station_id(dest)
        direction = 0 if s_id > d_id else 1

        base = {
            "weekday": day,
            "hour": hour,
            "train_type": train_type,
            "station_id": s_id,
            "direction": direction,
            "month": datetime.datetime.now().month,
            "day_of_month": datetime.datetime.now().day,
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "is_weekend": 1 if day.lower() in {"saturday", "sunday"} else 0,
            "is_peak_hour": 1 if hour in [6, 7, 8, 9, 16, 17, 18, 19] else 0,
            "station_hour_risk": 0.25,
        }
        base.update(DEFAULT_INFERENCE_PAYLOAD)
        input_data = pd.DataFrame([{k: base.get(k, 0) for k in self.feature_columns}])

        prob = float(self.model.predict_proba(input_data)[0][1])
        label = int(prob >= self.threshold)
        buffer_min = recommend_buffer_minutes(prob)
        return prob, label, buffer_min


if __name__ == "__main__":
    advisor = SmartCommuteAdvisor()
    risk, pred, buffer_min = advisor.get_risk("Frankfurt", "Burg", "Friday", 18)
    print(f"Risk: {risk:.1%} | Predicted delayed={bool(pred)} | threshold={advisor.threshold:.2f} | Suggested buffer={buffer_min} min")
