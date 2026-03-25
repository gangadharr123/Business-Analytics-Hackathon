"""Step 4: Inference tool for commute delay risk."""

from __future__ import annotations

import datetime
from pathlib import Path

import joblib
import pandas as pd

from config import (
    BUFFER_DEFAULT_MINUTES, BUFFER_THRESHOLDS, DEFAULT_INFERENCE_PAYLOAD,
    FREEZING_TEMP_C, HIGH_WINDS_KMH, MODEL_FILE, RUSH_HOURS,
    STATION_ALIASES, STATION_MAP, VALID_TRAIN_TYPES, VALID_WEEKDAYS,
)


def recommend_buffer_minutes(risk_probability: float) -> int:
    """Simple decision-support rule for commuter planning buffer."""
    for threshold, minutes in BUFFER_THRESHOLDS:
        if risk_probability >= threshold:
            return minutes
    return BUFFER_DEFAULT_MINUTES


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
        key = name.strip().lower()
        if key not in STATION_ALIASES:
            raise ValueError(
                f"Unknown station: '{name}'. Valid stations: {sorted(STATION_MAP.keys())}"
            )
        return STATION_ALIASES[key]

    # Include contextual event information so inference mirrors training features.
    def get_risk(self, source: str, dest: str, day: str, hour: int, train_type: str = "RB10", has_event: int = 0):
        if not isinstance(hour, int) or not (0 <= hour <= 23):
            raise ValueError(f"hour must be an integer 0–23, got: {hour!r}")
        if day not in VALID_WEEKDAYS:
            raise ValueError(f"day must be one of {sorted(VALID_WEEKDAYS)}, got: {day!r}")
        if train_type not in VALID_TRAIN_TYPES:
            raise ValueError(f"train_type must be one of {sorted(VALID_TRAIN_TYPES)}, got: {train_type!r}")

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
            "is_weekend": 1 if day.lower() in {"saturday", "sunday"} else 0,
            "is_rush_hour": 1 if hour in RUSH_HOURS else 0,
            "has_event": has_event,
        }
        base.update(DEFAULT_INFERENCE_PAYLOAD)

        # Derive compact weather risk indicators consumed by the trained model.
        base["is_freezing"] = 1 if base.get("temp_c", 0) <= FREEZING_TEMP_C else 0
        base["has_precipitation"] = 1 if (base.get("precip_mm", 0) > 0 or base.get("rain_mm", 0) > 0 or base.get("snow_cm", 0) > 0) else 0
        base["high_winds"] = 1 if base.get("wind_gusts_kmh", 0) >= HIGH_WINDS_KMH else 0

        input_data = pd.DataFrame([{k: base.get(k, 0) for k in self.feature_columns}])

        prob = float(self.model.predict_proba(input_data)[0][1])
        label = int(prob >= self.threshold)
        buffer_min = recommend_buffer_minutes(prob)
        return prob, label, buffer_min, base


if __name__ == "__main__":
    advisor = SmartCommuteAdvisor()
    risk, pred, buffer_min, features = advisor.get_risk("Frankfurt(Main)Hbf", "Oestrich-Winkel", "Friday", 18)
    print(f"Risk: {risk:.1%} | Predicted delayed={bool(pred)} | threshold={advisor.threshold:.2f} | Suggested buffer={buffer_min} min")
