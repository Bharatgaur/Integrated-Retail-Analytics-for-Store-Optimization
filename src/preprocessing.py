"""
preprocessing.py - Data Preprocessing & Feature Engineering
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Steps:
  1. Load & parse dates
  2. Merge datasets
  3. Handle missing values (MarkDown imputation)
  4. Remove negative sales / outliers
  5. Feature engineering
  6. Encode categoricals
  7. Return clean master dataframe
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD & MERGE
# ─────────────────────────────────────────────────────────────────────

def load_and_merge(data_dir: str) -> pd.DataFrame:
    """
    Load raw CSV files and merge into a single master dataframe.

    Parameters
    ----------
    data_dir : str
        Path to the /data folder containing the three CSV files.

    Returns
    -------
    pd.DataFrame
        Merged raw dataframe with parsed dates.
    """
    sales    = pd.read_csv(f"{data_dir}/sales_data.csv")
    stores   = pd.read_csv(f"{data_dir}/stores_data.csv")
    features = pd.read_csv(f"{data_dir}/features_data.csv")

    # Parse dates (DD/MM/YYYY format)
    for df in [sales, features]:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Merge: sales + stores (on Store), then + features (on Store, Date)
    df = sales.merge(stores, on='Store', how='left')
    df = df.merge(features, on=['Store', 'Date'], how='left', suffixes=('', '_feat'))

    # Drop duplicate IsHoliday from features
    if 'IsHoliday_feat' in df.columns:
        df.drop(columns=['IsHoliday_feat'], inplace=True)

    print(f"[LOAD]  Merged dataframe shape : {df.shape}")
    print(f"[LOAD]  Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────

MARKDOWN_COLS = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with domain-aware strategies:

    - MarkDown columns   → 0 (no markdown event occurred; NaN means no promotion)
    - CPI / Unemployment → forward-fill then backward-fill per store
    - Temperature        → median per store

    Returns the dataframe with no remaining nulls in key columns.
    """
    print("\n[MISSING] Before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # MarkDowns: NaN means no promotional discount → fill with 0
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)

    # CPI & Unemployment: macro indicators change slowly → ffill/bfill per store
    for col in ['CPI', 'Unemployment']:
        df[col] = df.groupby('Store')[col].transform(
            lambda x: x.ffill().bfill()
        )
        # Final safety net: global median
        df[col] = df[col].fillna(df[col].median())

    # Temperature: store-level median
    df['Temperature'] = df.groupby('Store')['Temperature'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fuel_Price: global median
    df['Fuel_Price'] = df['Fuel_Price'].fillna(df['Fuel_Price'].median())

    print("\n[MISSING] After imputation – remaining nulls:")
    remaining = df.isnull().sum()[df.isnull().sum() > 0]
    print(remaining if len(remaining) else "  ✓ None")
    return df


# ─────────────────────────────────────────────────────────────────────
# 3. HANDLE NEGATIVE / ANOMALOUS SALES
# ─────────────────────────────────────────────────────────────────────

def handle_negative_sales(df: pd.DataFrame, clip_floor: float = 0.0) -> pd.DataFrame:
    """
    Negative Weekly_Sales represent returns/corrections.
    Business decision: clip to 0 so forecasting models stay positive.

    Parameters
    ----------
    clip_floor : float
        Lower bound for clipping. Default 0.
    """
    n_neg = (df['Weekly_Sales'] < 0).sum()
    pct   = 100 * n_neg / len(df)
    print(f"\n[NEG SALES] Found {n_neg:,} negative rows ({pct:.2f}%) → clipped to {clip_floor}")
    df['Weekly_Sales'] = df['Weekly_Sales'].clip(lower=clip_floor)
    return df


# ─────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rich temporal, economic, and store-level features.

    New columns
    -----------
    Year, Month, Week, Quarter, DayOfWeek
    IsWeekend
    IsHoliday (bool → int)
    MarkDown_Total   : sum of all 5 markdown columns
    HasMarkDown      : binary flag
    SizeBucket       : Small / Medium / Large / XLarge
    LogSize          : log-transformed store size
    Type_Encoded     : label-encoded store type (A=0, B=1, C=2)
    CPI_Change       : week-over-week change in CPI
    Unemployment_Change
    """
    # ── Temporal features ────────────────────────────────────────────
    df['Year']       = df['Date'].dt.year
    df['Month']      = df['Date'].dt.month
    df['Week']       = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter']    = df['Date'].dt.quarter
    df['DayOfWeek']  = df['Date'].dt.dayofweek      # Monday=0
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)

    # ── Holiday encoding ─────────────────────────────────────────────
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # ── MarkDown aggregations ────────────────────────────────────────
    df['MarkDown_Total'] = df[MARKDOWN_COLS].sum(axis=1)
    df['HasMarkDown']    = (df['MarkDown_Total'] > 0).astype(int)

    # ── Store size features ──────────────────────────────────────────
    df['LogSize'] = np.log1p(df['Size'])
    df['SizeBucket'] = pd.cut(
        df['Size'],
        bins=[0, 50_000, 100_000, 170_000, df['Size'].max() + 1],
        labels=['Small', 'Medium', 'Large', 'XLarge']
    )

    # ── Store type encoding ──────────────────────────────────────────
    type_map = {'A': 2, 'B': 1, 'C': 0}      # A is typically highest-volume
    df['Type_Encoded'] = df['Type'].map(type_map)

    # ── Economic rate-of-change ──────────────────────────────────────
    df = df.sort_values(['Store', 'Date'])
    df['CPI_Change']           = df.groupby('Store')['CPI'].diff().fillna(0)
    df['Unemployment_Change']  = df.groupby('Store')['Unemployment'].diff().fillna(0)
    df['Fuel_Change']          = df.groupby('Store')['Fuel_Price'].diff().fillna(0)

    # ── Lag features (prior week sales) — store+dept level ──────────
    df = df.sort_values(['Store', 'Dept', 'Date'])
    df['Sales_Lag1']  = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).fillna(0)
    df['Sales_Lag4']  = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4).fillna(0)
    df['Sales_Roll4'] = (
        df.groupby(['Store', 'Dept'])['Weekly_Sales']
          .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
          .fillna(0)
    )

    print(f"\n[FEATURES] Engineered {df.shape[1]} total columns")
    return df


# ─────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_preprocessing(data_dir: str) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline and return a clean dataframe.

    Usage
    -----
    from src.preprocessing import run_preprocessing
    df = run_preprocessing('retail_project/data')
    """
    df = load_and_merge(data_dir)
    df = handle_missing_values(df)
    df = handle_negative_sales(df)
    df = engineer_features(df)
    print(f"\n[DONE] Preprocessed dataframe: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(BASE, '..', 'data')
    df = run_preprocessing(DATA)
    print(df.describe().T[['mean', 'std', 'min', 'max']])
