# Business-Analytics-Hackathon
EBS Business Analytics Hackathon

## Commute Reliability Estimator (Pilot)

### One-line goal (simple)
We estimate **"How likely is my train to be delayed?"** for students commuting from Frankfurt to EBS.

### Problem statement (technical)
Build a data-driven model that predicts the **probability of delay** for DB trips using historical rail, weather, disruption, and calendar information.

---

## 1) What data do we have? (very clear)

This project now supports **multi-month historical windows**. Current default run uses **Jul 2024 to Dec 2025** parquet files to improve reliability and seasonality coverage.

### Data sources in the pipeline

| Source | File | Purpose |
|---|---|---|
| Raw DB movement data | `data/raw/data-YYYY-MM.parquet` (Jul 2024..Dec 2025) | Monthly base rail records: station, timestamp, delay, train type, destination |
| Construction disruptions | `data/construction.csv` | Infrastructure work periods and impact level |
| Strike events | `data/strikes.csv` | Strike date ranges and rail-relevant event text |
| Weather observations | `data/weather.csv` | Hourly weather (rain, wind, temperature, etc.) |

Config paths are centralized in `src/config.py`.

### Base rail columns required from raw parquet

The filtering step (`src/01_filter_data.py`) requires:
- `station_name`
- `time`
- `delay_in_min`
- `train_type`
- `final_destination_station`

If these are missing, pipeline stops with a clear schema error.

### Corridor scope (stations considered)
The current corridor covers key Frankfurt ↔ Rheingau/EBS-related stations:
- Frankfurt(Main)Hbf
- Frankfurt-Höchst
- Wiesbaden Hbf
- Wiesbaden-Biebrich
- Eltville
- Oestrich-Winkel
- Hattenheim
- Geisenheim
- Rüdesheim(Rhein)
- Mainz Hbf

---

## 2) Data pipeline in plain English

1. **Filter relevant rail records** (`01_filter_data.py`)
   - Keep only corridor stations.
   - Parse timestamps safely.
   - Create simple time flags (hour, weekday, weekend, peak hour).

2. **Enrich with external context** (`02_enrich_data.py`)
   - Merge hourly weather data.
   - Add holiday flag (Hessen).
   - Add construction impact score by date.
   - Add strike impact indicator by date.

3. **Train and compare models** (`05_train_ml_model.py`)
   - Build features.
   - Use time-based split (train first 80%, test latest 20%).
   - Compare Decision Tree, Logistic Regression, Random Forest.
   - Tune threshold to favor recall while balancing accuracy.

4. **Run smart inference** (`06_smart_commute_tool.py`)
   - Returns delay probability + delayed/not-delayed label.

5. **Analyze feature impact and data availability** (`07_feature_analysis.py`)
   - Generates reports on available data and feature importance.

---

## 3) What does “delayed” mean?

A trip is marked as delayed if:
- `delay_in_min > 3`

This threshold is stored in `DELAY_THRESHOLD_MINUTES` in `src/config.py`.

---

## 4) Features used for feature engineering (core section)

Below is the feature set explained for both non-ML and technical readers.

### A) Time & calendar features

| Feature | Type | Why it can matter |
|---|---|---|
| `hour` | raw numeric | Rush-hour traffic patterns differ from off-peak |
| `weekday` | categorical | Mondays/Fridays often behave differently |
| `is_weekend` | engineered binary | Weekend train operations and demand differ |
| `is_peak_hour` | engineered binary | Captures known high-congestion windows |
| `month` | engineered numeric | Seasonality effect |
| `day_of_month` | engineered numeric | Captures within-month patterns |
| `hour_sin`, `hour_cos` | engineered cyclical | Better representation of time-of-day cycles than raw hour alone |
| `is_holiday` | enriched binary | Public holidays can change traffic and staffing |

### B) Route / rail-operation features

| Feature | Type | Why it can matter |
|---|---|---|
| `station_id` | mapped categorical | Some stations are naturally more delay-prone |
| `train_type` | categorical | RB/RE/etc can have different reliability profiles |
| `direction` | engineered binary | Directional congestion can differ |
| `station_hour_risk` | engineered aggregate prior | Historical delay tendency for each station-hour pair |

### C) Disruption features

| Feature | Type | Why it can matter |
|---|---|---|
| `construction_impact` | enriched ordinal (0-3) | Track works directly affect punctuality |
| `strike_impact` | enriched binary | Strike days increase systemic disruption |

### D) Weather raw features

| Feature |
|---|
| `temp_c` |
| `humidity_pct` |
| `precip_mm` |
| `rain_mm` |
| `snow_cm` |
| `feels_like_c` |
| `wind_gusts_kmh` |
| `wind_speed_kmh` |
| `wind_dir_10m` |

### E) Weather engineered features (focused on delay mechanisms)

| Feature | Formula/logic | Intuition |
|---|---|---|
| `adverse_weather_score` | weighted mix of precipitation + rain + snow + gusts | Single "bad weather stress" signal |
| `temp_extreme_flag` | 1 if temp <= -2°C or >= 30°C else 0 | Extreme heat/cold can increase failures |
| `wind_precip_interaction` | `wind_speed_kmh * precip_mm` | Wind + rain together can be worse than each alone |
| `heavy_rain_flag` | 1 if `rain_mm >= 2.0` | Captures threshold-style operational impact |

---

## 5) How we evaluate model quality

For a commuter safety use case, **missing a likely delay is costly**, so recall is important.

- **Recall**: "Of all truly delayed trains, how many did we catch?"
- **Accuracy**: overall correct predictions
- **ROC-AUC**: ranking quality across thresholds

We tune the classification threshold using a weighted objective:
- `score = 0.7 * recall + 0.3 * accuracy`

This balances practical warning sensitivity with general correctness.

---

## 6) Reports generated for explainability

Run `07_feature_analysis.py` to produce:
- `reports/feature_analysis/data_availability_report.json`
- `reports/feature_analysis/feature_analysis_metrics.json`
- `reports/feature_analysis/feature_importance_recall.csv`
- `reports/feature_analysis/weather_feature_importance.csv`

Training also writes:
- `reports/feature_analysis/feature_importance_recall_from_training.csv`

These files help answer:
1. Which columns are available/missing?
2. Which features most affect recall?
3. Are weather variables actually influencing predictions?

---

## 7) Repository structure

```text
src/
  01_filter_data.py        # Step 1: filter raw parquet + basic time features
  02_enrich_data.py        # Step 2: weather/construction/strikes/holiday enrichment
  04_commute_tool.py       # Baseline lookup tool
  05_train_ml_model.py     # Step 3: model training + threshold tuning + artifact save
  06_smart_commute_tool.py # Step 4: user-facing prediction utility
  07_feature_analysis.py   # Step 5: data audit + feature impact reports
  config.py                # central paths/constants/station map/weather columns

data/
  raw/                     # raw source files
  processed/               # filtered + enriched datasets

models/
  delay_model_v3.joblib
  delay_model_v3_metadata.json

reports/
  feature_analysis/
```

---

## 8) How to run

```bash
pip install -r requirements.txt
python src/01_filter_data.py --start 2024-07 --end 2025-12
python src/02_enrich_data.py
python src/05_train_ml_model.py --min-precision 0.35 --min-recall 0.55
python src/06_smart_commute_tool.py
python src/07_feature_analysis.py --input data/processed/ebs_commute_data_enriched.csv
```

---

## 9) Pilot limitation and scale-up plan

Current model is a pilot trained on **Dec 2025 only**, so it is useful for validating engineering logic but not yet final for production-level reliability.

Next step:
- Expand training window to **Jun 2024 → Dec 2025**
- Re-run analysis reports
- Re-train and compare feature-importance stability across seasons

This will make the model more robust and academically stronger for final evaluation.


## 10) Troubleshooting (Windows path issue)

If you see `FileNotFoundError: Raw data not found ... data\raw\data-2025-12.parquet`, it means your parquet is in `data/` root instead of `data/raw/`.

You can use either option:
1. Move file to `data/raw/data-2025-12.parquet`, or
2. Run with explicit path:

```bash
python src/01_filter_data.py --input data/data-2025-12.parquet
```

The script now also auto-detects common locations (`data/raw` and `data`) and will log which file it selected.



### Training timeout / loky subprocess warning on Windows

If training finishes model scoring but then prints messages like:
- `ERROR: This operation returned because the timeout period expired.`
- `joblib.externals.loky ... Failed to kill subprocesses ... install psutil`

use these fixes:

1. Install/update dependencies (includes `psutil`):
```bash
pip install -r requirements.txt
```

2. Run with safe single-process mode:
```bash
set EBS_PARALLEL_JOBS=1
python src/05_train_ml_model.py --min-precision 0.35 --min-recall 0.55
```

Notes:
- The pipeline now auto-uses safer parallel settings on Windows.
- You can still override parallelism with `EBS_PARALLEL_JOBS` if needed.


## 11) Key objectives and studies (for presentation)

### Objective 1: Analyze delays and identify key drivers
We run descriptive + factor studies to answer:
- Probability of delays by time/day in Frankfurt → Rheingau corridor
- Probability of delays for a specific train proxy (train type + station + hour)
- Overall reliability of public transport in this area
- Factors associated with higher delay probability

### Objective 2: Use insights in a decision support system
We convert those insights into commuter guidance:
- model-based delay risk (`06_smart_commute_tool.py`)
- suggested planning buffer minutes based on predicted risk
- day/hour-level buffer recommendation table from historical data

### Key studies command
```bash
python src/08_studies_and_decision_support.py --input data/processed/ebs_commute_data_enriched.csv
```

### Outputs generated
- `reports/key_studies/study1_delay_probability_by_time_day.csv`
- `reports/key_studies/study1_delay_probability_by_train_proxy.csv`
- `reports/key_studies/study1_reliability_summary.json`
- `reports/key_studies/study2_factor_effects.csv`
- `reports/key_studies/study3_buffer_recommendations.csv`

These files are presentation-ready tables for the project key questions.


### Multi-file mode for raw parquet data

`01_filter_data.py` can automatically read all monthly parquet files in a range.

Example:
```bash
python src/01_filter_data.py --input-pattern "data-*.parquet" --start 2024-07 --end 2025-12
```

If `--input` is provided, it reads that single file; otherwise it uses multi-file mode with the date range.


### Balanced threshold tuning (important for hackathon quality)

`05_train_ml_model.py` now tunes the decision threshold using a balanced objective (not recall-only), combining:
- F1 score
- balanced accuracy
- precision
- recall

You can enforce practical quality floors:
```bash
python src/05_train_ml_model.py --min-precision 0.35 --min-recall 0.55
```

If your model produces too many false alarms, increase `--min-precision` (e.g., 0.40 or 0.45).
If your model misses too many delays, increase `--min-recall`.


### Model reliability upgrades (critical fixes)

- **Train / Validation / Test split** is now used in `05_train_ml_model.py`:
  - Train = fit models
  - Validation = tune threshold
  - Test = final unbiased evaluation
- **Smoothed target encoding** is used for `station_hour_risk` with additive smoothing (`--target-encoding-alpha`) to prevent overfitting on rare station-hour pairs.
- **Lag features** were added (`lag1_is_delayed_station`, `lag1_delay_min_station`, `rolling_delay_ratio_6_station`) to capture delay contagion effects.

Example advanced training command:
```bash
python src/05_train_ml_model.py --min-precision 0.40 --min-recall 0.60 --target-encoding-alpha 20
```
