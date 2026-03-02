"""Step 1: Filter raw train data for the EBS commute corridor."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config import (
    DATA_DIR,
    FILTERED_DATA_FILE,
    RAW_DATA_FILE,
    RAW_DATE_END,
    RAW_DATE_START,
    RAW_DIR,
    RAW_FILE_PATTERN,
    REQUIRED_FILTER_COLUMNS,
    TARGET_STATIONS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


DATE_TOKEN_RE = re.compile(r"data-(\d{4})-(\d{2})\.parquet$")


def _parse_ym(path: Path) -> int | None:
    m = DATE_TOKEN_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)) * 100 + int(m.group(2))


def _resolve_input_file(input_file: Path) -> Path:
    if input_file.exists():
        return input_file

    candidate_paths = [
        RAW_DIR / input_file.name,
        DATA_DIR / input_file.name,
        DATA_DIR / "raw" / input_file.name,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            logger.warning("Input file not found at %s; using detected file at %s", input_file, candidate)
            return candidate

    fallback_files = list(DATA_DIR.glob("*.parquet")) + list(RAW_DIR.glob("*.parquet"))
    if len(fallback_files) == 1:
        logger.warning("Input file not found at %s; using discovered parquet file %s", input_file, fallback_files[0])
        return fallback_files[0]

    raise FileNotFoundError(
        f"Raw data not found at {input_file}.\n"
        f"Checked: {', '.join(str(p) for p in candidate_paths)}.\n"
        "Tip: move parquet to data/raw/data-2025-12.parquet or run with --input <path>."
    )


def _resolve_input_files(input_file: Path | None, input_pattern: str, start_ym: str, end_ym: str) -> list[Path]:
    if input_file is not None:
        return [_resolve_input_file(input_file)]

    start_key = int(start_ym.replace("-", ""))
    end_key = int(end_ym.replace("-", ""))

    candidates = list(RAW_DIR.glob(input_pattern)) + list(DATA_DIR.glob(input_pattern))
    uniq = {p.resolve(): p for p in candidates}.values()
    selected: list[Path] = []
    for p in uniq:
        ym = _parse_ym(p)
        if ym is None:
            continue
        if start_key <= ym <= end_key:
            selected.append(p)

    selected = sorted(selected, key=lambda x: _parse_ym(x) or 0)
    if not selected:
        raise FileNotFoundError(
            f"No parquet files matched pattern '{input_pattern}' in data/raw or data for range {start_ym}..{end_ym}."
        )

    logger.info("Found %d parquet files in range %s..%s", len(selected), start_ym, end_ym)
    return selected


def _validate_columns(columns: list[str], required: list[str]) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _read_and_filter_single(file_path: Path) -> pd.DataFrame:
    schema_columns = pq.read_schema(file_path).names
    _validate_columns(schema_columns, REQUIRED_FILTER_COLUMNS)

    table = pq.read_table(file_path, columns=REQUIRED_FILTER_COLUMNS)
    df = table.to_pandas()
    return df[df["station_name"].isin(TARGET_STATIONS)].copy()


def filter_commuter_data(
    input_file: Path | None = None,
    output_file: Path = FILTERED_DATA_FILE,
    input_pattern: str = RAW_FILE_PATTERN,
    start_ym: str = RAW_DATE_START,
    end_ym: str = RAW_DATE_END,
) -> pd.DataFrame:
    input_files = _resolve_input_files(input_file, input_pattern, start_ym, end_ym)
    logger.info("Pipeline start: filtering %d source file(s)", len(input_files))

    chunks = []
    for fp in input_files:
        logger.info("Reading %s", fp)
        chunks.append(_read_and_filter_single(fp))

    df_filtered = pd.concat(chunks, ignore_index=True)
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
    parser.add_argument("--input", type=str, default=None, help="Single parquet file. If omitted, uses --input-pattern with date range.")
    parser.add_argument("--input-pattern", type=str, default=RAW_FILE_PATTERN, help="Glob pattern for multi-file mode, e.g. data-*.parquet")
    parser.add_argument("--start", type=str, default=RAW_DATE_START, help="Start YYYY-MM for multi-file mode")
    parser.add_argument("--end", type=str, default=RAW_DATE_END, help="End YYYY-MM for multi-file mode")
    parser.add_argument("--output", type=str, default=str(FILTERED_DATA_FILE))
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    filter_commuter_data(
        input_file=input_path,
        output_file=Path(args.output),
        input_pattern=args.input_pattern,
        start_ym=args.start,
        end_ym=args.end,
    )
