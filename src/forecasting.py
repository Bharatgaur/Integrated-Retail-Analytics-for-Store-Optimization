"""
forecasting.py - Demand Forecasting
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Models
──────
1. ARIMA / SARIMA   – univariate time-series baseline
2. Random Forest    – ML with rich feature set
3. XGBoost          – gradient boosting (best performer)

Evaluation
──────────
RMSE · MAE · MAPE  (per model, on held-out last 8 weeks)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARNING] XGBoost not installed; XGB model will be skipped.")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_SM = True
except ImportError:
    _HAS_SM = False
    print("[WARNING] statsmodels not installed; SARIMA will be skipped.")


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, MAPE."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE' : mean_absolute_error(y_true, y_pred),
        'MAPE': mape
    }


# ─────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS USED IN ML MODELS
# ─────────────────────────────────────────────────────────────────────

ML_FEATURES = [
    'Store', 'Dept', 'Type_Encoded', 'LogSize',
    'Year', 'Month', 'Week', 'Quarter',
    'IsHoliday', 'IsWeekend',
    'MarkDown_Total', 'HasMarkDown',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'CPI_Change', 'Unemployment_Change', 'Fuel_Change',
    'Sales_Lag1', 'Sales_Lag4', 'Sales_Roll4'
]


def _get_features(df: pd.DataFrame) -> list:
    """Return only columns that exist in df."""
    return [c for c in ML_FEATURES if c in df.columns]


# ─────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT (temporal)
# ─────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame,
                   test_weeks: int = 8) -> tuple:
    """
    Split data into train / test preserving time order.
    Test set = last `test_weeks` unique dates.
    """
    dates     = sorted(df['Date'].unique())
    cutoff    = dates[-test_weeks]
    train     = df[df['Date'] < cutoff].copy()
    test      = df[df['Date'] >= cutoff].copy()
    print(f"[SPLIT] Train: {train['Date'].min().date()} → {train['Date'].max().date()}  ({len(train):,} rows)")
    print(f"[SPLIT] Test : {test['Date'].min().date()}  → {test['Date'].max().date()}   ({len(test):,} rows)")
    return train, test


# ─────────────────────────────────────────────────────────────────────
# 1. SARIMA (per Store-Dept series)
# ─────────────────────────────────────────────────────────────────────

def fit_sarima_series(series: pd.Series,
                      order=(1, 1, 1),
                      seasonal_order=(1, 1, 1, 52),
                      horizon: int = 8) -> np.ndarray:
    """
    Fit SARIMA to a single (Store, Dept) weekly time series.

    Parameters
    ----------
    series         : pd.Series with DatetimeIndex, weekly frequency
    order          : ARIMA (p,d,q)
    seasonal_order : Seasonal (P,D,Q,s) where s=52 for weekly
    horizon        : steps ahead to forecast

    Returns
    -------
    np.ndarray of length `horizon`
    """
    if not _HAS_SM:
        return np.zeros(horizon)

    try:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False, maxiter=50)
        fc = result.forecast(steps=horizon)
        return np.maximum(fc.values, 0)           # clip negatives
    except Exception as e:
        print(f"  [SARIMA WARN] {e}")
        return np.full(horizon, series.mean())    # fallback to mean


def run_sarima_aggregate(df: pd.DataFrame,
                         store: int = 1,
                         dept: int = 1,
                         horizon: int = 8) -> dict:
    """
    Demonstrate SARIMA on a single Store-Dept time series.

    Returns
    -------
    dict with keys: train_series, forecast, metrics_in_sample
    """
    print(f"\n[SARIMA] Store={store} Dept={dept} — fitting …")
    sub = (
        df[(df['Store'] == store) & (df['Dept'] == dept)]
        .set_index('Date')['Weekly_Sales']
        .sort_index()
        .resample('W-FRI').sum()
    )

    n_test     = horizon
    train_ser  = sub.iloc[:-n_test]
    test_ser   = sub.iloc[-n_test:]

    fc = fit_sarima_series(train_ser, horizon=n_test)
    m  = compute_metrics(test_ser.values, fc)
    print(f"  RMSE={m['RMSE']:,.0f}  MAE={m['MAE']:,.0f}  MAPE={m['MAPE']:.2f}%")
    return {'train': train_ser, 'test': test_ser, 'forecast': fc, 'metrics': m}


# ─────────────────────────────────────────────────────────────────────
# 2. RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────

def train_random_forest(train: pd.DataFrame,
                        test: pd.DataFrame) -> tuple:
    """
    Train a Random Forest regressor on the full training set
    and evaluate on the test set.

    Returns
    -------
    (model, metrics, y_pred)
    """
    feats = _get_features(train)
    X_tr, y_tr = train[feats], train['Weekly_Sales']
    X_te, y_te = test[feats],  test['Weekly_Sales']

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_pred = np.maximum(y_pred, 0)

    m = compute_metrics(y_te.values, y_pred)
    print(f"\n[RANDOM FOREST]  RMSE={m['RMSE']:,.0f}  MAE={m['MAE']:,.0f}  MAPE={m['MAPE']:.2f}%")
    return model, m, y_pred


# ─────────────────────────────────────────────────────────────────────
# 3. XGBOOST
# ─────────────────────────────────────────────────────────────────────

def train_xgboost(train: pd.DataFrame,
                  test: pd.DataFrame) -> tuple:
    """
    Train an XGBoost regressor.

    Returns
    -------
    (model, metrics, y_pred)
    """
    if not _HAS_XGB:
        print("[XGB] XGBoost not available.")
        return None, {}, np.array([])

    feats = _get_features(train)
    X_tr, y_tr = train[feats], train['Weekly_Sales']
    X_te, y_te = test[feats],  test['Weekly_Sales']

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False
    )
    y_pred = model.predict(X_te)
    y_pred = np.maximum(y_pred, 0)

    m = compute_metrics(y_te.values, y_pred)
    print(f"[XGBOOST]        RMSE={m['RMSE']:,.0f}  MAE={m['MAE']:,.0f}  MAPE={m['MAPE']:.2f}%")
    return model, m, y_pred


# ─────────────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────

def feature_importance(model, feature_names: list,
                       top_n: int = 15) -> pd.DataFrame:
    """Extract and display top-N feature importances."""
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'Feature'   : feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        print(f"\n[IMPORTANCE] Top {top_n} features:")
        print(imp.to_string(index=False))
        return imp
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_forecasting(df: pd.DataFrame,
                    test_weeks: int = 8) -> dict:
    """
    End-to-end forecasting pipeline.

    Steps
    -----
    1. Temporal train/test split
    2. SARIMA on a sample series
    3. Random Forest (full dataset)
    4. XGBoost       (full dataset)
    5. Compare metrics

    Returns
    -------
    dict containing models, predictions, and metrics.
    """
    print("\n" + "="*60)
    print("  DEMAND FORECASTING PIPELINE")
    print("="*60)

    train, test = temporal_split(df, test_weeks=test_weeks)

    # SARIMA — demonstrate on Store 1, Dept 1
    sarima_res = run_sarima_aggregate(df, store=1, dept=1, horizon=test_weeks)

    # Random Forest
    rf_model, rf_m, rf_pred = train_random_forest(train, test)
    fi_rf = feature_importance(rf_model, _get_features(train))

    # XGBoost
    xgb_model, xgb_m, xgb_pred = train_xgboost(train, test)
    fi_xgb = feature_importance(xgb_model, _get_features(train)) if xgb_model else pd.DataFrame()

    # Summary table
    print("\n" + "-"*50)
    print("  MODEL COMPARISON SUMMARY")
    print("-"*50)
    rows = [
        {'Model': 'SARIMA (Store1-Dept1)', **sarima_res['metrics']},
        {'Model': 'Random Forest',         **rf_m},
    ]
    if xgb_m:
        rows.append({'Model': 'XGBoost', **xgb_m})
    summary = pd.DataFrame(rows).set_index('Model').round(2)
    print(summary.to_string())

    return {
        'train'       : train,
        'test'        : test,
        'sarima'      : sarima_res,
        'rf_model'    : rf_model,
        'rf_metrics'  : rf_m,
        'rf_pred'     : rf_pred,
        'xgb_model'   : xgb_model,
        'xgb_metrics' : xgb_m,
        'xgb_pred'    : xgb_pred,
        'fi_rf'       : fi_rf,
        'fi_xgb'      : fi_xgb,
        'model_summary': summary
    }


# ─────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os, sys
    BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE)
    from preprocessing import run_preprocessing
    from anomaly_detection import run_anomaly_detection
    DATA = os.path.join(BASE, '..', 'data')
    df   = run_preprocessing(DATA)
    df   = run_anomaly_detection(df, handle_strategy='cap')
    res  = run_forecasting(df, test_weeks=8)
