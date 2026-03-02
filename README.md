# Business-Analytics-Hackathon

## Commute Delay Prediction (Simplified)

This repository predicts whether a train trip is likely to be delayed for the Frankfurt ↔ Rheingau/EBS commute corridor.

The project is intentionally kept **simple and explainable**:
- use the available columns directly,
- avoid heavy feature engineering,
- optimize model behavior for **accuracy and precision**.

---

## Data Sources

The pipeline uses these files:
- Raw rail data: `data/raw/data-YYYY-MM.parquet`
- Weather data: `data/weather.csv`
- Strike data: `data/strikes.csv`
- Construction data: `data/construction.csv`

Core paths and constants are centralized in `src/config.py`.

---

## Pipeline

1. **Filter data** – `src/01_filter_data.py`
   - keeps relevant corridor stations and required rail columns.

2. **Enrich data** – `src/02_enrich_data.py`
   - adds weather, strike, construction, and holiday context.

3. **Train model** – `src/05_train_ml_model.py`
   - uses a time-based train/validation/test split,
   - compares Decision Tree, Logistic Regression, Random Forest,
   - tunes threshold with an objective focused on **accuracy + precision** with a small recall floor to avoid predicting only the majority class.
   - exports `reports/feature_analysis/threshold_scan_by_model.csv` for submission-ready threshold transparency.

4. **Inference tool** – `src/06_smart_commute_tool.py`
   - outputs delay probability, predicted class, and suggested buffer minutes.

5. **Feature analysis** – `src/07_feature_analysis.py`
   - trains a simple analysis model and writes feature-importance reports.

6. **Study reports** – `src/08_studies_and_decision_support.py`
   - produces summary tables for delay patterns and practical buffer planning.

---

## Target Definition

A trip is considered delayed when:
- `delay_in_min > 10`

This threshold is configured via `DELAY_THRESHOLD_MINUTES` in `src/config.py`.

---

## Simplified Feature Set

The training script focuses on direct, low-complexity predictors:
- time/context: `weekday`, `hour`, `month`, `is_weekend`
- route/operation: `station_id`, `direction`, `train_type`
- disruption flags: `is_holiday`, `construction_impact`, `strike_impact`
- weather columns from `WEATHER_COLUMNS` in `src/config.py`

No lag chains, no interaction-heavy composites, and no target-encoding features are required.

---

## Evaluation Focus

Primary focus:
- **Accuracy**
- **Precision**

The threshold tuning objective in training uses a weighted score that prioritizes these metrics, while still enforcing a small minimum recall to avoid degenerate all-negative predictions.

---

## Run Commands

From the repository root:

```bash
python src/01_filter_data.py
python src/02_enrich_data.py
python src/05_train_ml_model.py --min-precision 0.40 --min-recall 0.10
python src/06_smart_commute_tool.py
python src/07_feature_analysis.py
python src/08_studies_and_decision_support.py
```

---

## Outputs

- Model artifact: `models/delay_model_v3.joblib`
- Metadata: `models/delay_model_v3_metadata.json`
- Feature analysis reports: `reports/feature_analysis/`
- Study reports: `reports/key_studies/`


## Submission-ready checklist

- Run the full pipeline in order from the **Run Commands** section.
- Verify `models/delay_model_v3_metadata.json` includes both precision and recall above zero on test data.
- Verify `reports/feature_analysis/threshold_scan_by_model.csv` exists and pick the operating threshold based on your project tradeoff.
- Include `reports/feature_analysis/top10_features_from_training.csv` and `reports/feature_analysis/top10_features_precision.csv` in your appendix/report.
