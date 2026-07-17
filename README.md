# German Electricity Demand Time-Series Modelling and Forecasting

This repository contains a modular Python implementation for Student 1's assignment on German electricity demand forecasting.

## Overview

The workflow covers data inspection, preprocessing, exploratory analysis, stationarity analysis, benchmarking, weekly SARIMA and SARIMAX models, feature-based models, and an hourly LSTM prototype.

## Dataset sources

- Electricity demand: Open Power System Data 60-minute single-index dataset, expected locally as `data/raw/time_series_60min_singleindex.csv`.
- Temperature: Berlin daily values from Open-Meteo, expected locally as `data/raw/berlin_temperature_daily.csv`.

Source websites:
- Open Power System Data: https://open-power-system-data.org/
- Open-Meteo: https://open-meteo.com/

## Repository structure

```text
german_electricity_load_forecasting/
├── data/
├── src/
├── tests/
├── outputs/
├── models/
├── report/
├── config.yaml
├── requirements.txt
├── README.md
├── run_pipeline.py
└── .gitignore
```

## Setup

```bash
cd german_electricity_load_forecasting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run each stage

```bash
python run_pipeline.py --stage inspect
python run_pipeline.py --stage data
python run_pipeline.py --stage eda
python run_pipeline.py --stage benchmarks
python run_pipeline.py --stage sarima
python run_pipeline.py --stage sarimax
python run_pipeline.py --stage features
python run_pipeline.py --stage lstm
python run_pipeline.py --stage evaluate
python run_pipeline.py --stage all
```

Run tests and final validation:

```bash
pytest -q
python scripts/final_validation.py
```

## Expected outputs

- Processed CSV files in data/processed
- Figures in outputs/figures
- Tables in outputs/tables
- Logs in outputs/logs
- Models in models/

## Methodology summary

- Electricity is kept from 2015 onward and aligned to UTC.
- Daily and weekly aggregates are average MW values, not summed energy in MWh.
- Weekly temperature is aggregated to match the weekly load series.
- Holiday indicators are generated using the Python holidays package for Germany, with Berlin used as a representative location.

## Leakage prevention

All lag and rolling features are created in a time-aware manner using only past values. The test period is held out chronologically, and future values are never used as predictors.

## Forecast design notes

The feature-based models were evaluated using rolling one-step-ahead predictions. At each test timestamp, lagged load values from earlier observed test periods were available. Therefore, the results should not be interpreted as a fully recursive two-year forecast generated entirely from the training endpoint.

SARIMA and SARIMAX weekly forecasts are generated over the two-year weekly test period. Their interval coverage is reported separately from point accuracy.

## Conditional forecast warning

The SARIMAX forecasts use observed test-period temperature values. This is an explanatory or conditional forecast rather than a true operational forecast because temperature is not known in advance.

## LSTM evaluation warning

The LSTM results are a one-step-ahead test evaluation on hourly data and are not identical to a recursively generated multi-year operational forecast. LSTM hourly metrics must not be directly compared with weekly-model RMSE or MAE because the target resolution is different.

## Reproducibility

The random seed is fixed at 42 for Python, NumPy, scikit-learn, and TensorFlow where possible.

To reproduce the validation pass:

```bash
python run_pipeline.py --stage evaluate
python scripts/final_validation.py
```

## Submission archive

From inside the project folder, create the submission archive with:

```bash
cd ..
zip -r german_electricity_load_forecasting_submission.zip \
german_electricity_load_forecasting \
-x "german_electricity_load_forecasting/.venv/*" \
   "german_electricity_load_forecasting/.git/*" \
   "german_electricity_load_forecasting/**/__pycache__/*" \
   "german_electricity_load_forecasting/.pytest_cache/*"
```

## Known limitations

- Berlin temperature is used as a proxy for German weather.
- The LSTM model is intentionally lightweight to keep runtime manageable.
- SARIMA and SARIMAX use weekly aggregation, which smooths short-term hourly behaviour.
