# Report Evidence Guide

This is a factual evidence guide, not the final academic report.

## Model Results

- Best weekly model: Random Forest
- Random Forest RMSE: 2650.601793 MW
- Random Forest MAE: 1980.456755 MW
- Random Forest MAPE: 3.780390%
- Random Forest sMAPE: 3.672317%
- Random Forest MASE: 0.772349
- Random Forest RMSE improvement over Seasonal Naive: 11.58%
- Random Forest MAE improvement over Seasonal Naive: 14.02%
- Gradient Boosting RMSE improvement over Seasonal Naive: 4.46%
- Gradient Boosting MAE improvement over Seasonal Naive: 2.82%
- Seasonal Naive RMSE: 2997.625258 MW
- Seasonal Naive MAE: 2303.361602 MW

## SARIMA And SARIMAX

- Selected SARIMA: (0,1,6)(0,1,0,52)
- SARIMA AIC: 2409.159134
- SARIMA BIC: 2429.547718
- SARIMA RMSE: 4902.544135 MW
- SARIMAX RMSE: 5130.490603 MW
- SARIMAX point forecast is worse than SARIMA.
- Temperature and holiday covariates did not improve point accuracy in this implementation.
- SARIMA 80% interval coverage: 86/105 = 0.819048
- SARIMA 95% interval coverage: 104/105 = 0.990476
- SARIMAX 80% interval coverage: 103/105 = 0.980952
- SARIMAX 95% interval coverage: 105/105 = 1.000000
- SARIMA 80% coverage is close to nominal.
- SARIMA 95% coverage is conservative.
- SARIMAX intervals are highly conservative and wide.
- High interval coverage does not imply superior point-forecast accuracy.

## Residual Diagnostics

- SARIMA residual initialization burn-in excluded: 60 observations
- Diagnostic residual count: 136
- Residual mean: approximately -0.73
- Ljung-Box p-value: 0.630
- Ljung-Box p-value indicates no strong evidence of remaining autocorrelation at the tested lag.
- Jarque-Bera p-value: 0.000002
- Jarque-Bera p-value rejects normality.
- Forecast intervals should therefore be interpreted cautiously.
- Do not describe residuals as fully ideal or perfectly normal.

## Stationarity Evidence

- Stationarity table contains weekly level, first difference, seasonal difference at lag 52, and first plus seasonal difference.
- Differencing and seasonal period 52 were justified using ADF results, STL decomposition, original ACF/PACF, differenced ACF/PACF, visible annual pattern, and AIC model selection.
- Do not claim differencing was required only because of ADF.

## Feature Models

- Feature model evaluation design: rolling one-step-ahead predictions using observed prior test-period load values.
- The feature-based results should not be interpreted as a fully recursive two-year forecast generated entirely from the training endpoint.
- Top five Random Forest feature importances:
  - load_lag_52: 0.750386
  - temperature_2m_max: 0.039590
  - load_lag_1: 0.032708
  - week_of_year: 0.030487
  - temp_lag_52: 0.022858
- Annual lag is dominant.
- This explains why Seasonal Naive is already a strong benchmark.
- Temperature contributes, but much less than lagged demand.
- Feature importance does not prove causal effect.

## Temperature And Holidays

- Berlin temperature is used as a proxy for Germany.
- Berlin proxy weather is a limitation for national German load.
- Observed test-period weather makes SARIMAX an explanatory or conditional forecast, not an operational forecast using only information known at the training endpoint.

## LSTM

- LSTM hourly one-step RMSE: 1086.449677 MW
- LSTM MAE: 816.904325 MW
- LSTM MAPE: 1.508485%
- LSTM sMAPE: 1.507284%
- LSTM uses hourly resolution.
- LSTM metrics must not be directly compared with weekly-model RMSE or MAE.
- LSTM validation is chronological, uses `shuffle=False`, and does not use the test period for validation.
- LSTM evaluation is one-step-ahead hourly forecasting, not a recursive two-year operational forecast.

## Train Test Dates

- Weekly models train: 2015-01-04 to 2018-09-30
- Weekly models test: 2018-10-07 to 2020-10-04
- Feature models train: 2016-01-03 to 2018-09-30
- Feature models test: 2018-10-07 to 2020-10-04
- LSTM hourly train: 2015-01-01 00:00 to 2018-09-30 23:00
- LSTM hourly test: 2018-10-01 00:00 to 2020-09-30 23:00
