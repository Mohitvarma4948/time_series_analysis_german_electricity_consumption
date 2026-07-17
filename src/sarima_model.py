from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import jarque_bera, probplot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from src.config import FIGURES_DIR, TABLES_DIR
from src.utils import calculate_metrics


def search_sarima(logger: logging.Logger | None = None) -> tuple[pd.DataFrame, dict]:
    if logger is None:
        logger = logging.getLogger('pipeline')
    weekly = pd.read_csv(Path('data/processed/weekly_load_temperature.csv'), parse_dates=['date'])
    weekly = weekly.set_index('date').sort_index()
    weekly = weekly[['load_mw']].dropna()
    weekly = weekly.asfreq('W-SUN')
    train_end = weekly.index.max() - pd.DateOffset(years=2)
    train = weekly[weekly.index <= train_end]
    test = weekly[weekly.index > train_end]

    results = []
    best = None
    seasonal_order = (0, 1, 0, 52)
    for p in range(7):
        for d in range(3):
            for q in range(7):
                order = (p, d, q)
                P, D, Q, _ = seasonal_order
                try:
                    start = time.time()
                    model = SARIMAX(
                        train['load_mw'],
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        simple_differencing=True,
                        concentrate_scale=True,
                    )
                    fit = model.fit(disp=False, maxiter=25)
                    elapsed = time.time() - start
                    converged = fit.mle_retvals.get('converged', False)
                    results.append({
                        'p': p,
                        'd': d,
                        'q': q,
                        'P': P,
                        'D': D,
                        'Q': Q,
                        'AIC': fit.aic,
                        'BIC': fit.bic,
                        'runtime_seconds': elapsed,
                        'converged': converged,
                        'error': '',
                    })
                    if best is None or fit.aic < best['AIC']:
                        best = {'order': order, 'seasonal_order': seasonal_order, 'fit': fit, 'AIC': fit.aic, 'converged': converged}
                except Exception as exc:
                    results.append({
                        'p': p,
                        'd': d,
                        'q': q,
                        'P': P,
                        'D': D,
                        'Q': Q,
                        'AIC': None,
                        'BIC': None,
                        'runtime_seconds': None,
                        'converged': False,
                        'error': str(exc),
                    })
    table = pd.DataFrame(results)
    table.to_csv(TABLES_DIR / 'sarima_grid_search.csv', index=False)
    valid = table.dropna(subset=['AIC'])
    if valid.empty:
        raise RuntimeError('No SARIMA candidate could be fitted.')
    converged = valid[valid['converged'].astype(bool)]
    selected = (converged if not converged.empty else valid).sort_values('AIC').iloc[0]
    final_order = (int(selected['p']), int(selected['d']), int(selected['q']))
    final_seasonal_order = (int(selected['P']), int(selected['D']), int(selected['Q']), 52)
    final_model = SARIMAX(
        train['load_mw'],
        order=final_order,
        seasonal_order=final_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_fit = final_model.fit(disp=False, maxiter=100)
    best = {
        'order': final_order,
        'seasonal_order': final_seasonal_order,
        'fit': final_fit,
        'AIC': final_fit.aic,
        'converged': final_fit.mle_retvals.get('converged', False),
    }
    forecast_result = final_fit.get_forecast(steps=len(test))
    forecast = forecast_result.predicted_mean
    ci_95 = forecast_result.conf_int(alpha=0.05)
    ci_80 = forecast_result.conf_int(alpha=0.20)
    forecast.index = test.index
    ci_95.index = test.index
    ci_80.index = test.index
    metrics = calculate_metrics(test['load_mw'], forecast, seasonal_period=52)
    metrics['Model'] = 'SARIMA'
    metrics.to_csv(TABLES_DIR / 'sarima_metrics.csv', index=False)
    forecast_df = pd.DataFrame({
        'date': test.index,
        'actual': test['load_mw'].values,
        'forecast': forecast.values,
        'lower_80': ci_80.iloc[:, 0].values,
        'upper_80': ci_80.iloc[:, 1].values,
        'lower_95': ci_95.iloc[:, 0].values,
        'upper_95': ci_95.iloc[:, 1].values,
    })
    forecast_df.to_csv(TABLES_DIR / 'sarima_forecast.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.plot(test.index, test['load_mw'], label='Actual')
    ax.plot(test.index, forecast, label='Forecast')
    ax.fill_between(test.index, ci_95.iloc[:, 0], ci_95.iloc[:, 1], color='tab:blue', alpha=0.12, label='95% interval')
    ax.fill_between(test.index, ci_80.iloc[:, 0], ci_80.iloc[:, 1], color='tab:blue', alpha=0.22, label='80% interval')
    ax.set_title('SARIMA weekly forecast')
    ax.set_ylabel('Average load (MW)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sarima_forecast.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    burn = final_fit.loglikelihood_burn
    residuals = final_fit.resid.iloc[burn:].dropna()
    acf_lags = min(52, max(1, len(residuals) // 2 - 1))
    acf_values = acf(residuals, nlags=acf_lags, fft=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=200)
    axes = axes.flatten()
    residuals.plot(ax=axes[0], title='Residuals after initialization burn-in')
    residuals.hist(ax=axes[1], bins=20)
    axes[1].set_title('Residual histogram')
    residuals.plot(kind='kde', ax=axes[2], title='Residual density')
    probplot(residuals, dist='norm', plot=axes[3])
    axes[3].set_title('Residual QQ plot')
    axes[4].bar(range(len(acf_values)), acf_values)
    axes[4].set_title('Residual ACF')
    axes[4].set_xlabel('Lag')
    axes[4].set_ylabel('Autocorrelation')
    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sarima_residual_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    lb_lag = min(52, max(1, len(residuals) - 1))
    lb = acorr_ljungbox(residuals, lags=[lb_lag], return_df=True)
    jb = jarque_bera(residuals)
    diagnostics = pd.DataFrame({
        'metric': [
            'initialization_burn_in',
            'diagnostic_residual_count',
            'residual_mean',
            'residual_std',
            'ljung_box_lag',
            'ljung_box_stat',
            'ljung_box_p_value',
            'jarque_bera_stat',
            'jarque_bera_p_value',
        ],
        'value': [
            burn,
            len(residuals),
            residuals.mean(),
            residuals.std(),
            lb_lag,
            lb['lb_stat'].iloc[0],
            lb['lb_pvalue'].iloc[0],
            jb.statistic,
            jb.pvalue,
        ],
    })
    diagnostics.to_csv(TABLES_DIR / 'sarima_residual_diagnostics.csv', index=False)
    return table, {'best': best, 'forecast': forecast}
