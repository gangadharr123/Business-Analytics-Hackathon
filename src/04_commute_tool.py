"""Baseline lookup tool for delay probability by weekday and hour."""

from __future__ import annotations

import os

import pandas as pd

from config import DELAY_THRESHOLD_MINUTES, FILTERED_DATA_FILE


class SimpleCommuteCalculator:
    def __init__(self, data_path: str = str(FILTERED_DATA_FILE)):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"'{data_path}' not found. Run 01_filter_data.py first.")

        self.df = pd.read_csv(data_path)
        self.df["is_delayed"] = (self.df["delay_in_min"] > DELAY_THRESHOLD_MINUTES).astype(int)

        self.risk_matrix = self.df.pivot_table(
            index="weekday", columns="hour", values="is_delayed", aggfunc="mean"
        )

    def get_risk(self, day: str, hour: int):
        if day in self.risk_matrix.index and hour in self.risk_matrix.columns:
            return float(self.risk_matrix.loc[day, hour])
        return None


if __name__ == "__main__":
    tool = SimpleCommuteCalculator()
    while True:
        day = input("Day (e.g. Monday, or 'exit'): ").strip().capitalize()
        if day.lower() in {"exit", "quit"}:
            break
        try:
            hour = int(input("Hour (0-23): "))
        except ValueError:
            print("Invalid hour.")
            continue

        risk = tool.get_risk(day, hour)
        print(f"Historical Delay Probability: {risk:.1%}" if risk is not None else "No data for this time")
