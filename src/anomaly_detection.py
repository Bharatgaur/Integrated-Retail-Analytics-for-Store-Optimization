"""
anomaly_detection.py - Anomaly Detection in Weekly Sales
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Methods implemented
───────────────────
1. Z-Score (statistical)
2. IQR   (statistical)
3. Isolation Forest (ML)

Each method flags rows; results are combined with a consensus flag.
Anomalies are explained and then handled (capped / removed / flagged).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────
# 1. Z-SCORE METHOD
# ─────────────────────────────────────────────────────────────────────

def zscore_anomalies(df: pd.DataFrame,
                     col: str = 'Weekly_Sales',
                     threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag rows where the Z-score of Weekly_Sales exceeds ±threshold.
    Z-score is computed per (Store, Dept) group so each series is
    normalised independently.

    Parameters
    ----------
    threshold : float
        Absolute Z-score cutoff (default 3 → ~0.27 % of normal distribution).

    Returns
    -------
    df with added column 'Anomaly_ZScore' (bool).
    """
    def _z(x):
        mu, sigma = x.mean(), x.std()
        if sigma == 0:
            return pd.Series(False, index=x.index)
        return (np.abs((x - mu) / sigma) > threshold)

    df['Anomaly_ZScore'] = (
        df.groupby(['Store', 'Dept'])[col]
          .transform(_z)
          .astype(bool)
    )
    n = df['Anomaly_ZScore'].sum()
    print(f"[Z-SCORE]  Anomalies detected : {n:,}  ({100*n/len(df):.2f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────
# 2. IQR METHOD
# ─────────────────────────────────────────────────────────────────────

def iqr_anomalies(df: pd.DataFrame,
                  col: str = 'Weekly_Sales',
                  factor: float = 1.5) -> pd.DataFrame:
    """
    Flag rows outside [Q1 - factor*IQR, Q3 + factor*IQR] per (Store, Dept).

    Parameters
    ----------
    factor : float
        IQR multiplier (1.5 = standard; 3.0 = extreme outliers only).
    """
    def _iqr(x):
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        return (x < lo) | (x > hi)

    df['Anomaly_IQR'] = (
        df.groupby(['Store', 'Dept'])[col]
          .transform(_iqr)
          .astype(bool)
    )
    n = df['Anomaly_IQR'].sum()
    print(f"[IQR]      Anomalies detected : {n:,}  ({100*n/len(df):.2f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────
# 3. ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────────

def isolation_forest_anomalies(df: pd.DataFrame,
                                features: list = None,
                                contamination: float = 0.02) -> pd.DataFrame:
    """
    Apply Isolation Forest on multivariate features to detect anomalies.

    Parameters
    ----------
    features      : list of column names to use (default: a curated set)
    contamination : expected fraction of outliers (default 2 %)

    Returns
    -------
    df with 'Anomaly_IF' (bool) column.
    """
    if features is None:
        features = [
            'Weekly_Sales', 'MarkDown_Total', 'Temperature',
            'Fuel_Price', 'CPI', 'Unemployment',
            'Type_Encoded', 'LogSize'
        ]

    # Keep only columns that exist
    features = [f for f in features if f in df.columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].fillna(0))

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    preds = model.fit_predict(X)          # -1 = outlier, 1 = inlier
    df['Anomaly_IF'] = (preds == -1)

    n = df['Anomaly_IF'].sum()
    print(f"[ISO FOREST] Anomalies detected : {n:,}  ({100*n/len(df):.2f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────
# 4. CONSENSUS FLAG
# ─────────────────────────────────────────────────────────────────────

def consensus_anomaly(df: pd.DataFrame, min_votes: int = 2) -> pd.DataFrame:
    """
    Mark a row as a consensus anomaly if ≥ min_votes methods flag it.

    Creates column 'Anomaly_Vote' (int 0-3) and 'Is_Anomaly' (bool).
    """
    df['Anomaly_Vote'] = (
        df['Anomaly_ZScore'].astype(int) +
        df['Anomaly_IQR'].astype(int) +
        df['Anomaly_IF'].astype(int)
    )
    df['Is_Anomaly'] = df['Anomaly_Vote'] >= min_votes

    n = df['Is_Anomaly'].sum()
    print(f"[CONSENSUS] Final anomalies (≥{min_votes} votes) : {n:,}  ({100*n/len(df):.2f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────
# 5. ANOMALY EXPLANATION
# ─────────────────────────────────────────────────────────────────────

def explain_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'Anomaly_Reason' column with a brief human-readable explanation.
    Logic is rule-based and uses domain knowledge.
    """
    reasons = []
    for _, row in df[df['Is_Anomaly']].iterrows():
        r = []
        if row.get('IsHoliday', 0):
            r.append("Holiday week")
        if row.get('HasMarkDown', 0):
            r.append(f"MarkDown event (${row.get('MarkDown_Total',0):,.0f})")
        if row.get('Weekly_Sales', 0) > df['Weekly_Sales'].quantile(0.99):
            r.append("Extreme high sales (>99th pct)")
        if row.get('Weekly_Sales', 0) < df['Weekly_Sales'].quantile(0.01):
            r.append("Extreme low sales (<1st pct)")
        if not r:
            r.append("Statistical anomaly (unexplained)")
        reasons.append("; ".join(r))

    df.loc[df['Is_Anomaly'], 'Anomaly_Reason'] = reasons
    df['Anomaly_Reason'] = df['Anomaly_Reason'].fillna("Normal")
    return df


# ─────────────────────────────────────────────────────────────────────
# 6. ANOMALY HANDLING
# ─────────────────────────────────────────────────────────────────────

def handle_anomalies(df: pd.DataFrame,
                     strategy: str = 'cap') -> pd.DataFrame:
    """
    Handle confirmed anomalies.

    Strategies
    ----------
    'cap'    : Cap Weekly_Sales at store+dept 99th percentile (default)
    'median' : Replace with store+dept rolling median
    'flag'   : Keep values but mark them (useful for downstream models)

    Parameters
    ----------
    strategy : str
        One of {'cap', 'median', 'flag'}.
    """
    if strategy == 'cap':
        upper = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.quantile(0.99)
        )
        n = df['Is_Anomaly'].sum()
        df.loc[df['Is_Anomaly'], 'Weekly_Sales'] = df.loc[
            df['Is_Anomaly'], 'Weekly_Sales'
        ].clip(upper=upper[df['Is_Anomaly']])
        print(f"[HANDLE] '{strategy}': capped {n:,} anomalous rows at 99th percentile")

    elif strategy == 'median':
        rolling_med = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(4, min_periods=1).median()
        )
        n = df['Is_Anomaly'].sum()
        df.loc[df['Is_Anomaly'], 'Weekly_Sales'] = rolling_med[df['Is_Anomaly']]
        print(f"[HANDLE] '{strategy}': replaced {n:,} anomalies with rolling median")

    elif strategy == 'flag':
        print(f"[HANDLE] '{strategy}': anomalies flagged but values kept")

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose cap/median/flag.")

    return df


# ─────────────────────────────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_anomaly_detection(df: pd.DataFrame,
                          handle_strategy: str = 'cap') -> pd.DataFrame:
    """
    End-to-end anomaly detection pipeline.

    Steps
    -----
    1. Z-Score detection
    2. IQR detection
    3. Isolation Forest
    4. Consensus voting
    5. Explanation
    6. Handling
    """
    print("\n" + "="*60)
    print("  ANOMALY DETECTION PIPELINE")
    print("="*60)

    df = zscore_anomalies(df)
    df = iqr_anomalies(df)
    df = isolation_forest_anomalies(df)
    df = consensus_anomaly(df, min_votes=2)
    df = explain_anomalies(df)
    df = handle_anomalies(df, strategy=handle_strategy)

    print("\n[SUMMARY] Anomaly reason breakdown:")
    print(df.loc[df['Is_Anomaly'], 'Anomaly_Reason'].value_counts().head(10))
    return df


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
    df = run_anomaly_detection(df)
    print(df[df['Is_Anomaly']][['Store', 'Dept', 'Date', 'Weekly_Sales', 'Anomaly_Reason']].head(20))
