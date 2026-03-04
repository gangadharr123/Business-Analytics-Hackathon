"""Step 4: Inference tool for commute delay risk."""

from __future__ import annotations

import datetime
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
        if "frankfurt" in name_l: return 0
        if "wiesbaden" in name_l: return 2
        if "eltville" in name_l: return 4
        if "oestrich" in name_l or "burg" in name_l: return 5
        if "hattenheim" in name_l or "schloss" in name_l: return 6
        if "geisenheim" in name_l: return 7
        if "rudesheim" in name_l or "rüdesheim" in name_l: return 8
        return 2

    # NEW: Added has_event argument to the signature
    def get_risk(self, source: str, dest: str, day: str, hour: int, train_type: str = "RB", has_event: int = 0):
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
            "is_rush_hour": 1 if hour in [7, 8, 9, 16, 17, 18] else 0,
            "has_event": has_event,  # <--- Added to the payload
        }
        base.update(DEFAULT_INFERENCE_PAYLOAD)
        
        # Dynamic weather flags
        base["is_freezing"] = 1 if base.get("temp_c", 0) <= 0 else 0
        base["has_precipitation"] = 1 if (base.get("precip_mm", 0) > 0 or base.get("rain_mm", 0) > 0 or base.get("snow_cm", 0) > 0) else 0
        base["high_winds"] = 1 if base.get("wind_gusts_kmh", 0) >= 40 else 0

        input_data = pd.DataFrame([{k: base.get(k, 0) for k in self.feature_columns}])

        prob = float(self.model.predict_proba(input_data)[0][1])
        label = int(prob >= self.threshold)
        buffer_min = recommend_buffer_minutes(prob)
        return prob, label, buffer_min, base


if __name__ == "__main__":
    advisor = SmartCommuteAdvisor()
    risk, pred, buffer_min, features = advisor.get_risk("Frankfurt", "Burg", "Friday", 18)
    print(f"Risk: {risk:.1%} | Predicted delayed={bool(pred)} | threshold={advisor.threshold:.2f} | Suggested buffer={buffer_min} min")