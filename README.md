# Business-Analytics-Hackathon

## EBS Smart Commute Advisor — RB10 Delay Prediction

This repository predicts whether an RB10 train trip on the **Frankfurt ↔ Rheingau corridor** is likely to be delayed, and surfaces that prediction as a student-facing decision-support tool.

The project is intentionally kept **simple and explainable**:
- use available columns directly with minimal feature engineering,
- optimize model threshold for **balanced recall and accuracy** (not just accuracy),
- present results as actionable commute recommendations, not raw probabilities.

---

## Data Sources

| File | Description |
|------|-------------|
| `data/raw/data-YYYY-MM.parquet` | Raw Deutsche Bahn departure data (Jul 2024 – Dec 2025) |
| `data/weather.csv` | Hourly weather for the Frankfurt region (Open-Meteo) |
| `data/strikes.csv` | Rail strike dates (GDL/EVG actions) |
| `data/construction.csv` | Track construction periods with severity levels |

Core paths and constants are centralized in `src/config.py`.

---

## Pipeline

1. **Filter data** – `src/01_filter_data.py`
   - Keeps only RB10 trains (`train_name` contains "RB10") at valid corridor stations.
   - Normalizes station names and adds `hour`, `weekday`, `is_weekend`, `is_peak_hour`.

2. **Enrich data** – `src/02_enrich_data.py`
   - Joins weather (by date + hour), strike flags, construction impact scores, and public holidays.

3. **Train model** – `src/05_train_ml_model.py`
   - Time-based 70 / 15 / 15 train/validation/test split (no shuffle).
   - Compares **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
   - Tunes decision threshold on validation set using a composite score: `0.4 × accuracy + 0.4 × recall + 0.2 × precision`, with a minimum recall floor of 0.5 to prevent degenerate all-negative predictions.
   - Note: ~10% of trips are delayed — accuracy alone is a misleading metric here. Balanced accuracy and AUC are the primary quality indicators.
   - Exports `reports/feature_analysis/threshold_scan_by_model.csv` for full threshold transparency.

4. **Inference tool** – `src/06_smart_commute_tool.py`
   - Outputs delay probability, predicted class, and recommended buffer minutes for a given departure.

5. **Feature analysis** – `src/07_feature_analysis.py`
   - Trains an analysis-only Random Forest and writes permutation importance reports.

6. **Study reports** – `src/08_studies_and_decision_support.py`
   - Produces delay probability tables by weekday/hour, factor effects, and buffer planning CSV.
   - Filters out slots with fewer than 30 observations to ensure statistical reliability.

7. **Visualizations** – `src/09_visualizations.py`
   - Generates delay-by-hour, delay-by-day, heatmap, feature importance, model comparison, and delay distribution charts.

8. **Streamlit app** – `src/app.py`
   - Student-facing UI: select station, day, and class time → get risk level, delay probability, and recommended buffer.

---

## Target Definition

A trip is labelled **delayed** when:

```
delay_in_min > 10
```

Threshold configured via `DELAY_THRESHOLD_MINUTES` in `src/config.py`.

---

## Feature Set

| Category | Features |
|----------|----------|
| Time | `weekday`, `hour`, `month`, `is_weekend`, `is_rush_hour` |
| Route | `station_id`, `direction`, `train_type` |
| Disruptions | `is_holiday`, `construction_impact`, `strike_impact`, `has_event` |
| Weather | `temp_c`, `humidity_pct`, `precip_mm`, `rain_mm`, `snow_cm`, `feels_like_c`, `wind_gusts_kmh`, `wind_speed_kmh`, `wind_dir_10m` |
| Derived weather | `is_freezing`, `has_precipitation`, `high_winds` |

---

## Evaluation

Primary metrics (in order of importance):
1. **Balanced Accuracy** — corrects for class imbalance (~10% delayed)
2. **AUC** — threshold-independent discrimination quality
3. **Recall** — critical for a safety-oriented use case (missing a delay is costly)
4. **Precision** — reduces unnecessary buffer recommendations

---

## Run Commands

From the repository root:

```bash
python src/01_filter_data.py
python src/02_enrich_data.py
python src/05_train_ml_model.py
python src/06_smart_commute_tool.py
python src/07_feature_analysis.py
python src/08_studies_and_decision_support.py
python src/09_visualizations.py
streamlit run src/app.py
```

---

## Outputs

| Artifact | Location |
|----------|----------|
| Model + threshold | `models/delay_model_v3.joblib` |
| Model metadata | `models/delay_model_v3_metadata.json` |
| Feature importance | `reports/feature_analysis/top10_features_from_training.csv` |
| Threshold scan | `reports/feature_analysis/threshold_scan_by_model.csv` |
| Study reports | `reports/key_studies/` |
| Figures | `reports/figures/` |

---

## Submission Checklist

- [ ] Run the full pipeline in order from the **Run Commands** section.
- [ ] Verify `models/delay_model_v3_metadata.json` — check `balanced_accuracy` and `auc` on test data.
- [ ] Verify `reports/feature_analysis/threshold_scan_by_model.csv` exists.
- [ ] Confirm `streamlit run src/app.py` launches without errors.
- [ ] Include `reports/figures/` charts in the presentation appendix.
