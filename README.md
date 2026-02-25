# Business-Analytics-Hackathon
EBS Business Analytics Hackathon

## Commute Reliability Estimator (Pilot)

### One-line goal (simple)
We estimate **"How likely is my train to be delayed?"** for students commuting from Frankfurt to EBS.

### Problem statement (technical)
Build a data-driven model that predicts the **probability of delay** for DB trips using historical rail, weather, disruption, and calendar information.

---

## 1) What data do we have? (very clear)

This project currently uses a **pilot window (December 2025)** to validate logic before scaling to a larger period.

### Data sources in the pipeline

| Source | File | Purpose |
|---|---|---|
| Raw DB movement data | `data/raw/data-2025-12.parquet` | Base rail records: station, timestamp, delay, train type, destination |
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
python src/01_filter_data.py
python src/02_enrich_data.py
python src/05_train_ml_model.py
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
