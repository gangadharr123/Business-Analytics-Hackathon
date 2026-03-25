"""Shared utilities for feature engineering."""

from __future__ import annotations

import pandas as pd

from config import (
    DELAY_THRESHOLD_MINUTES, EVENT_DATES, FREEZING_TEMP_C,
    HIGH_WINDS_KMH, RUSH_HOURS, STATION_MAP, WEATHER_COLUMNS,
)


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
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)

    for c in WEATHER_COLUMNS:
        if c not in df.columns:
            df[c] = 0

    df["is_freezing"] = (df["temp_c"] <= FREEZING_TEMP_C).astype(int)
    df["has_precipitation"] = ((df["precip_mm"] > 0) | (df["rain_mm"] > 0) | (df["snow_cm"] > 0)).astype(int)
    df["high_winds"] = (df["wind_gusts_kmh"] >= HIGH_WINDS_KMH).astype(int)

    # Convert Timestamps to "YYYY-MM-DD" strings so isin() comparison works correctly.
    event_dates = {d.strftime("%Y-%m-%d") for dates in EVENT_DATES.values() for d in dates}

    date_strings = df["time"].dt.strftime("%Y-%m-%d")
    is_event_day = date_strings.isin(event_dates).astype(int)

    if "has_event" not in df.columns:
        df["has_event"] = is_event_day
    else:
        df["has_event"] = (df["has_event"].fillna(0).astype(int) | is_event_day)

    return df.sort_values("time").reset_index(drop=True)
