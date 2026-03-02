"""Step 2: Enrich commute data with external factors."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import holidays
import pandas as pd

from config import (
    CONSTRUCTION_FILE,
    ENRICHED_DATA_FILE,
    FILTERED_DATA_FILE,
    STRIKES_FILE,
    WEATHER_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def read_csv_bulletproof(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    separators = [",", ";", "\t"]
    encodings = ["utf-8", "latin1", "cp1252"]

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
                if len(df.columns) > 1:
                    return df
            except (pd.errors.ParserError, UnicodeDecodeError):
                continue

    logger.warning("Failed to read %s with fallback CSV reader.", file_path.name)
    return pd.DataFrame()


def process_weather_data(df_main: pd.DataFrame) -> pd.DataFrame:
    if not WEATHER_FILE.exists():
        logger.warning("Weather file not found; skipping weather enrichment.")
        return df_main

    skip_rows = 0
    with open(WEATHER_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("time") or "time," in line or "time;" in line:
                skip_rows = i
                break

    try:
        w_df = pd.read_csv(WEATHER_FILE, skiprows=skip_rows, sep=None, engine="python")
    except pd.errors.ParserError:
        logger.warning("Weather parsing failed; skipping weather enrichment.")
        return df_main

    w_df.columns = w_df.columns.str.strip().str.lower()
    if "time" not in w_df.columns:
        logger.warning("Weather file has no time column; skipping weather enrichment.")
        return df_main

    rename_map = {}
    for col in w_df.columns:
        if "time" in col:
            rename_map[col] = "time_key"
        elif "temperature_2m" in col:
            rename_map[col] = "temp_c"
        elif "relative_humidity" in col:
            rename_map[col] = "humidity_pct"
        elif "precipitation" in col:
            rename_map[col] = "precip_mm"
        elif "rain" in col:
            rename_map[col] = "rain_mm"
        elif "snowfall" in col:
            rename_map[col] = "snow_cm"
        elif "apparent_temperature" in col:
            rename_map[col] = "feels_like_c"
        elif "gusts" in col:
            rename_map[col] = "wind_gusts_kmh"
        elif "wind_speed_10m" in col:
            rename_map[col] = "wind_speed_kmh"
        elif "direction_10m" in col:
            rename_map[col] = "wind_dir_10m"

    w_df.rename(columns=rename_map, inplace=True)
    w_df["time_key"] = pd.to_datetime(w_df["time_key"], errors="coerce")
    w_df = w_df.dropna(subset=["time_key"])

    w_df["date"] = w_df["time_key"].dt.date
    w_df["hour"] = w_df["time_key"].dt.hour

    keep_cols = ["date", "hour"] + [c for c in rename_map.values() if c != "time_key" and c in w_df.columns]
    w_df_clean = w_df[keep_cols].copy()

    df_main["date"] = df_main["time"].dt.date
    df_merged = pd.merge(df_main, w_df_clean, on=["date", "hour"], how="left")
    fill_cols = [c for c in keep_cols if c not in ["date", "hour"]]
    if fill_cols:
        df_merged[fill_cols] = df_merged[fill_cols].fillna(0)
    return df_merged.drop(columns=["date"])


def process_construction_data(df_main: pd.DataFrame) -> pd.DataFrame:
    const_df = read_csv_bulletproof(CONSTRUCTION_FILE)
    if const_df.empty:
        df_main["construction_impact"] = 0
        return df_main

    const_df.columns = const_df.columns.str.lower().str.strip()
    impact_col = next((c for c in const_df.columns if any(k in c for k in ["impact", "level", "severity"])), None)
    if impact_col is None or "start_date" not in const_df.columns or "end_date" not in const_df.columns:
        df_main["construction_impact"] = 0
        return df_main

    impact_map = {"high": 3, "medium": 2, "low": 1}
    const_df["score"] = const_df[impact_col].astype(str).str.lower().map(impact_map).fillna(1)
    const_df["start_date"] = pd.to_datetime(const_df["start_date"], errors="coerce")
    const_df["end_date"] = pd.to_datetime(const_df["end_date"], errors="coerce")

    date_impact: dict = {}
    for row in const_df.dropna(subset=["start_date", "end_date"]).itertuples(index=False):
        for d in pd.date_range(row.start_date, row.end_date):
            date_impact[d.date()] = max(date_impact.get(d.date(), 0), int(row.score))

    df_main["construction_impact"] = df_main["time"].dt.date.map(date_impact).fillna(0).astype(int)
    return df_main


def process_strike_data(df_main: pd.DataFrame) -> pd.DataFrame:
    strike_df = read_csv_bulletproof(STRIKES_FILE)
    if strike_df.empty:
        df_main["strike_impact"] = 0
        return df_main

    strike_df.columns = strike_df.columns.str.lower().str.strip()
    if "start_date" not in strike_df.columns or "end_date" not in strike_df.columns:
        df_main["strike_impact"] = 0
        return df_main

    strike_df["start_date"] = pd.to_datetime(strike_df["start_date"], errors="coerce")
    strike_df["end_date"] = pd.to_datetime(strike_df["end_date"], errors="coerce")

    strike_dates: set = set()
    for row in strike_df.dropna(subset=["start_date", "end_date"]).itertuples(index=False):
        row_text = " ".join(str(v) for v in row).lower()
        if any(k in row_text for k in ["rail", "train", "gdl", "evg"]):
            for d in pd.date_range(row.start_date, row.end_date):
                strike_dates.add(d.date())

    df_main["strike_impact"] = df_main["time"].dt.date.apply(lambda x: 1 if x in strike_dates else 0)
    return df_main


def enrich_data(input_file: Path = FILTERED_DATA_FILE, output_file: Path = ENRICHED_DATA_FILE) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Input data missing: {input_file}")

    df_train = pd.read_csv(input_file)
    required = {"time", "hour"}
    if missing := sorted(required - set(df_train.columns)):
        raise ValueError(f"Input schema invalid. Missing columns: {missing}")

    df_train["time"] = pd.to_datetime(df_train["time"], errors="coerce")
    df_train = df_train.dropna(subset=["time"])

    unique_years = df_train["time"].dt.year.unique().tolist()
    de_holidays = holidays.Germany(subdiv="HE", years=unique_years)
    df_train["is_holiday"] = df_train["time"].dt.date.apply(lambda x: 1 if x in de_holidays else 0)

    df_train = process_weather_data(df_train)
    df_train = process_construction_data(df_train)
    df_train = process_strike_data(df_train)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_file, index=False)
    logger.info("Saved enriched dataset to %s", output_file)
    return df_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(FILTERED_DATA_FILE))
    parser.add_argument("--output", type=str, default=str(ENRICHED_DATA_FILE))
    args = parser.parse_args()
    enrich_data(input_file=Path(args.input), output_file=Path(args.output))
