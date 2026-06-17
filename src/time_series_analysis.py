"""
time_series_analysis.py - Time-Based Trend and Seasonal Anomaly Analysis
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Purpose
-------
The anomaly_detection module flags anomalies using cross-sectional statistical
and machine learning methods (Z-Score, IQR, Isolation Forest). This module
complements that work with a strictly time-oriented view of the data:

1. Decompose weekly sales into trend, seasonal, and residual components.
2. Quantify seasonal effects (month-of-year, holiday weeks).
3. Detect anomalies directly from the residual component of the
   decomposition (a value is anomalous if the irregular/residual term is
   unusually large, independent of the absolute sales level).
4. Compare holiday vs. non-holiday anomaly rates to verify whether
   irregular spikes are explained by the calendar.

This separates "is this point unusual relative to its neighbors" (handled
here) from "is this point unusual relative to its peer group" (handled in
anomaly_detection.py). Both views are needed before the data is considered
clean enough for segmentation and forecasting.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    _HAS_SM = True
except ImportError:
    _HAS_SM = False
    print("[WARNING] statsmodels not installed; seasonal decomposition will be skipped.")


# ─────────────────────────────────────────────────────────────────────
# 1. AGGREGATE TO A CHAIN-LEVEL WEEKLY SERIES
# ─────────────────────────────────────────────────────────────────────

def build_weekly_series(df: pd.DataFrame, store: int = None, dept: int = None) -> pd.Series:
    """
    Build a single weekly time series of total Weekly_Sales.

    Parameters
    ----------
    store, dept : optional filters. If both are None, the series is
        aggregated across the entire chain (all stores, all departments).

    Returns
    -------
    pd.Series indexed by Date, weekly frequency (Friday-ending weeks).
    """
    sub = df.copy()
    if store is not None:
        sub = sub[sub['Store'] == store]
    if dept is not None:
        sub = sub[sub['Dept'] == dept]

    series = (
        sub.groupby('Date')['Weekly_Sales']
           .sum()
           .sort_index()
           .asfreq('W-FRI')
    )
    series = series.interpolate(limit_direction='both')
    return series


# ─────────────────────────────────────────────────────────────────────
# 2. SEASONAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────

def decompose_series(series: pd.Series, period: int = 52, model: str = 'additive') -> dict:
    """
    Decompose a weekly series into trend, seasonal, and residual components.

    Parameters
    ----------
    period : int
        Seasonal cycle length in observations (52 for weekly/annual seasonality).
    model : str
        'additive' or 'multiplicative'.

    Returns
    -------
    dict with keys: trend, seasonal, residual, observed
        Values are None if statsmodels is unavailable or the series is
        shorter than two full seasonal cycles.
    """
    if not _HAS_SM or len(series) < 2 * period:
        print(f"[DECOMPOSE] Skipped — need >= {2*period} weeks of data, have {len(series)}.")
        return {'trend': None, 'seasonal': None, 'residual': None, 'observed': series}

    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
    print(f"[DECOMPOSE] Series length={len(series)} weeks, period={period}, model={model}")
    return {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid,
        'observed': result.observed
    }


# ─────────────────────────────────────────────────────────────────────
# 3. RESIDUAL-BASED TIME ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────

def residual_anomalies(decomposition: dict, threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag weeks where the decomposition residual is an outlier — i.e. the
    week's sales deviate from what trend + seasonality predict, regardless
    of the absolute sales level.

    Parameters
    ----------
    threshold : float
        Number of standard deviations of the residual series beyond which
        a week is flagged.

    Returns
    -------
    pd.DataFrame with columns: Date, Observed, Trend, Seasonal, Residual,
    Residual_Z, Is_Time_Anomaly
    """
    resid = decomposition['residual']
    if resid is None:
        print("[RESIDUAL ANOMALY] No residual available (decomposition was skipped).")
        return pd.DataFrame()

    resid_clean = resid.dropna()
    mu, sigma = resid_clean.mean(), resid_clean.std()
    z = (resid_clean - mu) / sigma

    out = pd.DataFrame({
        'Date': resid_clean.index,
        'Observed': decomposition['observed'].reindex(resid_clean.index).values,
        'Trend': decomposition['trend'].reindex(resid_clean.index).values,
        'Seasonal': decomposition['seasonal'].reindex(resid_clean.index).values,
        'Residual': resid_clean.values,
        'Residual_Z': z.values
    })
    out['Is_Time_Anomaly'] = out['Residual_Z'].abs() > threshold

    n = out['Is_Time_Anomaly'].sum()
    print(f"[RESIDUAL ANOMALY] {n} of {len(out)} weeks flagged (|Z| > {threshold})")
    return out


# ─────────────────────────────────────────────────────────────────────
# 4. SEASONAL / HOLIDAY EFFECT QUANTIFICATION
# ─────────────────────────────────────────────────────────────────────

def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Weekly_Sales by calendar month, indexed against the yearly
    average (100 = average month).

    Returns
    -------
    pd.DataFrame with Month, Avg_Sales, Seasonal_Index
    """
    monthly = df.groupby(df['Date'].dt.month)['Weekly_Sales'].mean()
    index = (monthly / monthly.mean() * 100).round(1)
    out = pd.DataFrame({'Avg_Sales': monthly.round(0), 'Seasonal_Index': index})
    out.index.name = 'Month'
    print("\n[SEASONALITY] Monthly seasonal index (100 = yearly average):")
    print(out.to_string())
    return out.reset_index()


def holiday_effect(df: pd.DataFrame) -> dict:
    """
    Compare holiday-week vs. non-holiday-week average sales and quantify
    the lift, both for the full chain and broken down by store type.

    Returns
    -------
    dict with overall lift and a per-store-type breakdown DataFrame.
    """
    hol = df[df['IsHoliday'] == 1]['Weekly_Sales'].mean()
    non_hol = df[df['IsHoliday'] == 0]['Weekly_Sales'].mean()
    overall_lift = (hol / non_hol - 1) * 100

    by_type = df.groupby(['Type', 'IsHoliday'])['Weekly_Sales'].mean().unstack()
    by_type.columns = ['Non_Holiday', 'Holiday']
    by_type['Lift_Pct'] = ((by_type['Holiday'] / by_type['Non_Holiday']) - 1) * 100

    print(f"\n[HOLIDAY EFFECT] Overall holiday sales lift: {overall_lift:.1f}%")
    print(by_type.round(2).to_string())

    return {'overall_lift_pct': overall_lift, 'by_store_type': by_type.reset_index()}


def time_anomaly_holiday_overlap(time_anomalies: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-reference time-based anomalies with the holiday calendar to test
    whether irregular weeks are explained by known calendar events.

    Returns
    -------
    pd.DataFrame summarizing what fraction of flagged anomaly weeks were
    holiday weeks, vs. the base rate of holiday weeks in the full series.
    """
    if time_anomalies.empty:
        return pd.DataFrame()

    holiday_dates = set(df.loc[df['IsHoliday'] == 1, 'Date'].dt.normalize())
    time_anomalies = time_anomalies.copy()
    time_anomalies['Is_Holiday_Week'] = pd.to_datetime(time_anomalies['Date']).dt.normalize().isin(holiday_dates)

    anomaly_holiday_rate = time_anomalies.loc[time_anomalies['Is_Time_Anomaly'], 'Is_Holiday_Week'].mean()
    base_holiday_rate = time_anomalies['Is_Holiday_Week'].mean()

    summary = pd.DataFrame({
        'Metric': ['Holiday rate among flagged anomaly weeks', 'Holiday rate across all weeks'],
        'Value_Pct': [round(anomaly_holiday_rate * 100, 1), round(base_holiday_rate * 100, 1)]
    })
    print("\n[OVERLAP] Time anomalies vs. holiday calendar:")
    print(summary.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_time_series_analysis(df: pd.DataFrame, period: int = 52) -> dict:
    """
    End-to-end time-based trend and seasonality analysis.

    Steps
    -----
    1. Build chain-level weekly series
    2. Seasonal decomposition (trend / seasonal / residual)
    3. Residual-based time anomaly detection
    4. Monthly seasonality index
    5. Holiday effect quantification
    6. Overlap check between time anomalies and holiday calendar

    Returns
    -------
    dict containing all intermediate and final results.
    """
    print("\n" + "=" * 60)
    print("  TIME-BASED TREND & SEASONALITY ANALYSIS")
    print("=" * 60)

    series = build_weekly_series(df)
    decomposition = decompose_series(series, period=period)
    time_anomalies = residual_anomalies(decomposition)
    monthly = monthly_seasonality(df)
    holiday = holiday_effect(df)
    overlap = time_anomaly_holiday_overlap(time_anomalies, df)

    return {
        'weekly_series': series,
        'decomposition': decomposition,
        'time_anomalies': time_anomalies,
        'monthly_seasonality': monthly,
        'holiday_effect': holiday,
        'anomaly_holiday_overlap': overlap
    }


# ─────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os, sys
    BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE)
    from preprocessing import run_preprocessing
    DATA = os.path.join(BASE, '..', 'data')
    df = run_preprocessing(DATA)
    results = run_time_series_analysis(df)
