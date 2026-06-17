"""
utils.py - Shared utility functions for the Retail Analytics Project
Author: Senior Data Scientist
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_raw_data():
    """Load all three raw datasets from data directory."""
    sales    = pd.read_csv(os.path.join(DATA_DIR, 'sales_data.csv'))
    stores   = pd.read_csv(os.path.join(DATA_DIR, 'stores_data.csv'))
    features = pd.read_csv(os.path.join(DATA_DIR, 'features_data.csv'))
    return sales, stores, features


def parse_dates(df, col='Date'):
    """Parse date column with mixed formats safely."""
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    return df


def merge_datasets(sales, stores, features):
    """
    Merge all three datasets into one master dataframe.
    Left-join on (Store, Date) to keep all sales rows.
    """
    df = sales.merge(stores, on='Store', how='left')
    df = df.merge(features, on=['Store', 'Date'], how='left', suffixes=('', '_feat'))
    # Drop duplicate IsHoliday column from features
    if 'IsHoliday_feat' in df.columns:
        df.drop(columns=['IsHoliday_feat'], inplace=True)
    return df


# ─────────────────────────────────────────────
# DISPLAY / REPORT HELPERS
# ─────────────────────────────────────────────

def section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subsection(title: str):
    print(f"\n--- {title} ---")


def fmt_num(n, decimals=2):
    """Format a number with commas."""
    return f"{n:,.{decimals}f}"


# ─────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
