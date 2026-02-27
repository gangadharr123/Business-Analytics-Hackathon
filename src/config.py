"""Central configuration for the commute delay pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _dir in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

RAW_DATA_BASENAME = "data-2025-12.parquet"
RAW_DATA_CANDIDATES = [RAW_DIR / RAW_DATA_BASENAME, DATA_DIR / RAW_DATA_BASENAME]
RAW_DATA_FILE = next((p for p in RAW_DATA_CANDIDATES if p.exists()), RAW_DATA_CANDIDATES[0])
FILTERED_DATA_FILE = PROCESSED_DIR / "ebs_commute_data.csv"
ENRICHED_DATA_FILE = PROCESSED_DIR / "ebs_commute_data_enriched.csv"
MODEL_FILE = MODELS_DIR / "delay_model_v3.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "delay_model_v3_metadata.json"

CONSTRUCTION_FILE = DATA_DIR / "construction.csv"
STRIKES_FILE = DATA_DIR / "strikes.csv"
WEATHER_FILE = DATA_DIR / "weather.csv"

DELAY_THRESHOLD_MINUTES = 3

TARGET_STATIONS = [
    "Frankfurt(Main)Hbf", "Frankfurt-Höchst", "Wiesbaden Hbf",
    "Wiesbaden-Biebrich", "Eltville", "Oestrich-Winkel",
    "Hattenheim", "Geisenheim", "Rüdesheim(Rhein)", "Mainz Hbf"
]

REQUIRED_FILTER_COLUMNS = [
    "station_name", "time", "delay_in_min", "train_type", "final_destination_station"
]


STATION_MAP = {
    "Frankfurt(Main)Hbf": 0,
    "Frankfurt-Höchst": 1,
    "Wiesbaden Hbf": 2,
    "Wiesbaden-Biebrich": 3,
    "Eltville": 4,
    "Oestrich-Winkel": 5,
    "Hattenheim": 6,
    "Geisenheim": 7,
    "Rüdesheim(Rhein)": 8,
}

WEATHER_COLUMNS = [
    "temp_c", "humidity_pct", "precip_mm", "rain_mm", "snow_cm",
    "feels_like_c", "wind_gusts_kmh", "wind_speed_kmh", "wind_dir_10m",
]

DEFAULT_INFERENCE_PAYLOAD = {
    "is_holiday": 0,
    "construction_impact": 0,
    "strike_impact": 0,
    "temp_c": 5.0,
    "humidity_pct": 80,
    "precip_mm": 0,
    "rain_mm": 0,
    "snow_cm": 0,
    "feels_like_c": 3.0,
    "wind_gusts_kmh": 15.0,
    "wind_speed_kmh": 10.0,
    "wind_dir_10m": 200,
}


def get_parallel_jobs(default_windows: int = 1, default_other: int = -1) -> int:
    """Return safe parallel jobs; override with EBS_PARALLEL_JOBS env var."""
    raw = os.getenv("EBS_PARALLEL_JOBS")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return default_windows if os.name == "nt" else default_other
