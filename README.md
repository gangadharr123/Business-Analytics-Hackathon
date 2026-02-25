# Business-Analytics-Hackathon
EBS Business Analytics Hackathon

## Commute Reliability Estimator (Pilot)

### Problem statement
Predict the probability of DB train delays for commuters travelling from Frankfurt to EBS.

### Current pilot scope
- Training window: **December 2025 only** (pilot).
- Delay event definition: delayed if `delay_in_min > 3`.
- Objective priority: improve **recall** while maintaining strong **accuracy**.

---

## Project structure

```text
src/
  01_filter_data.py        # Step 1: filter raw parquet to EBS corridor + basic time features
  02_enrich_data.py        # Step 2: enrich with weather / construction / strikes / holidays
  04_commute_tool.py       # Baseline lookup tool (weekday x hour)
  05_train_ml_model.py     # Step 3: time-aware training, model comparison, threshold tuning
  06_smart_commute_tool.py # Step 4: inference tool using saved model + threshold
  07_feature_analysis.py   # Step 5: data availability + feature impact analysis
  config.py                # Central paths, constants, station map

data/
  raw/                     # raw input parquet
  processed/               # filtered + enriched datasets

models/
  delay_model_v3.joblib
  delay_model_v3_metadata.json
```

---

## Feature engineering implemented

The training pipeline now includes:
- Weather-focused engineered features (`adverse_weather_score`, `wind_precip_interaction`, `temp_extreme_flag`, `heavy_rain_flag`).
- Cyclical time encoding (`hour_sin`, `hour_cos`).
- Peak-hour indicator (`is_peak_hour`).
- Weekend indicator (`is_weekend`).
- Calendar features (`month`, `day_of_month`).
- Direction feature (towards Frankfurt vs away).
- Station-hour empirical risk prior (`station_hour_risk`).
- Weather and external disruption features when available.

In addition, model selection now uses a tuned decision threshold to optimize a weighted score favoring recall.

---

## How to run

```bash
pip install -r requirements.txt
python src/01_filter_data.py
python src/02_enrich_data.py
python src/05_train_ml_model.py
python src/06_smart_commute_tool.py
python src/07_feature_analysis.py --input data/processed/ebs_commute_data_enriched.csv
```

---

## Next scale-up step

Once pilot logic is validated, switch input from Dec 2025 only to **Jun 2024 → Dec 2025** with the same pipeline and retrain.

## Notes on review follow-ups

- Added CLI arguments for enrichment and training scripts for reproducible runs in different environments.
- Consolidated external file paths in `config.py` (weather/construction/strikes) to avoid duplicated path logic.
- Improved parquet schema validation in Step 1 before selective reads.
- Added a small smoke test command below to validate training/inference on synthetic data.


## Feature impact outputs

Run `07_feature_analysis.py` to generate:
- `reports/feature_analysis/data_availability_report.json`
- `reports/feature_analysis/feature_analysis_metrics.json`
- `reports/feature_analysis/feature_importance_recall.csv`
- `reports/feature_analysis/weather_feature_importance.csv`
