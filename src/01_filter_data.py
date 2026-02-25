"""Step 1: Filter raw train data for the EBS commute corridor."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config import FILTERED_DATA_FILE, RAW_DATA_FILE, REQUIRED_FILTER_COLUMNS, TARGET_STATIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _validate_columns(columns: list[str], required: list[str]) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def filter_commuter_data(input_file: Path = RAW_DATA_FILE, output_file: Path = FILTERED_DATA_FILE) -> pd.DataFrame:
    logger.info("Pipeline start: filtering %s", input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Raw data not found at {input_file}")

    schema_columns = pq.read_schema(input_file).names
    _validate_columns(schema_columns, REQUIRED_FILTER_COLUMNS)

    table = pq.read_table(input_file, columns=REQUIRED_FILTER_COLUMNS)
    df = table.to_pandas()
    df_filtered = df[df["station_name"].isin(TARGET_STATIONS)].copy()

    df_filtered["time"] = pd.to_datetime(df_filtered["time"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["time", "delay_in_min"])

    df_filtered["hour"] = df_filtered["time"].dt.hour
    df_filtered["weekday"] = df_filtered["time"].dt.day_name()
    df_filtered["is_weekend"] = (df_filtered["time"].dt.dayofweek >= 5).astype(int)
    df_filtered["is_peak_hour"] = df_filtered["hour"].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_file, index=False)
    logger.info("Saved %s filtered rows to %s", f"{len(df_filtered):,}", output_file)
    return df_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RAW_DATA_FILE))
    parser.add_argument("--output", type=str, default=str(FILTERED_DATA_FILE))
    args = parser.parse_args()
    filter_commuter_data(input_file=Path(args.input), output_file=Path(args.output))
