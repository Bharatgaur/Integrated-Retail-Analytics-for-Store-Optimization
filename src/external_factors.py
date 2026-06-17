"""
external_factors.py - Impact of External / Macroeconomic Factors on Sales
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Purpose
-------
Quantify how CPI, unemployment, fuel price, and temperature relate to
Weekly_Sales, both at the chain level and broken down by store type and
store segment. This module exists to make the external-factor analysis
auditable as a standalone step, separate from its later use as model
input features in forecasting.py.

Outputs
-------
1. Correlation of each external factor with sales (chain-level).
2. Store-type-level breakdown of the same correlations.
3. A simple linear regression isolating the marginal effect of each
   factor on sales, controlling for store size and type.
4. A ranked summary translating statistical findings into plain-language
   business implications.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

EXTERNAL_FACTORS = ['CPI', 'Unemployment', 'Fuel_Price', 'Temperature', 'MarkDown_Total']


# ─────────────────────────────────────────────────────────────────────
# 1. CHAIN-LEVEL CORRELATION
# ─────────────────────────────────────────────────────────────────────

def chain_level_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between each external factor and
    Weekly_Sales across the full dataset.

    Returns
    -------
    pd.DataFrame with columns: Factor, Correlation, Direction
    """
    factors = [f for f in EXTERNAL_FACTORS if f in df.columns]
    corrs = df[factors + ['Weekly_Sales']].corr()['Weekly_Sales'].drop('Weekly_Sales')

    out = pd.DataFrame({
        'Factor': corrs.index,
        'Correlation': corrs.values.round(3)
    })
    out['Direction'] = np.where(out['Correlation'] > 0, 'Positive', 'Negative')
    out = out.sort_values('Correlation', key=abs, ascending=False).reset_index(drop=True)

    print("\n[CHAIN-LEVEL] External factor correlation with Weekly_Sales:")
    print(out.to_string(index=False))
    return out


# ─────────────────────────────────────────────────────────────────────
# 2. STORE-TYPE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────

def store_type_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repeat the correlation analysis separately for each store type
    (A, B, C) to check whether external sensitivity differs by tier.

    Returns
    -------
    pd.DataFrame indexed by store Type, columns = factors, values = correlation.
    """
    factors = [f for f in EXTERNAL_FACTORS if f in df.columns]
    rows = {}
    for store_type, group in df.groupby('Type'):
        rows[store_type] = group[factors + ['Weekly_Sales']].corr()['Weekly_Sales'].drop('Weekly_Sales')

    out = pd.DataFrame(rows).T.round(3)
    out.index.name = 'Type'

    print("\n[BY STORE TYPE] External factor correlation with Weekly_Sales:")
    print(out.to_string())
    return out.reset_index()


# ─────────────────────────────────────────────────────────────────────
# 3. CONTROLLED LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────

def regression_impact(df: pd.DataFrame, sample_size: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """
    Fit a linear regression of Weekly_Sales on standardized external
    factors plus store size and type, to estimate each factor's marginal
    association with sales while holding the others constant.

    A random sample is used for tractability on the full 400K+ row
    dataset; the relationship being estimated is linear and stable
    enough that sampling does not materially change the coefficients.

    Returns
    -------
    pd.DataFrame with columns: Feature, Standardized_Coefficient
    sorted by absolute magnitude.
    """
    factors = [f for f in EXTERNAL_FACTORS if f in df.columns]
    features = factors + ['LogSize', 'Type_Encoded'] if 'LogSize' in df.columns else factors

    sample = df[features + ['Weekly_Sales']].dropna()
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=random_state)

    scaler = StandardScaler()
    X = scaler.fit_transform(sample[features])
    y = sample['Weekly_Sales'].values

    model = LinearRegression()
    model.fit(X, y)

    out = pd.DataFrame({
        'Feature': features,
        'Standardized_Coefficient': model.coef_.round(2)
    }).sort_values('Standardized_Coefficient', key=abs, ascending=False).reset_index(drop=True)

    r2 = model.score(X, y)
    print(f"\n[REGRESSION] Linear model R-squared = {r2:.3f} (n={len(sample):,})")
    print(out.to_string(index=False))
    return out


# ─────────────────────────────────────────────────────────────────────
# 4. BUSINESS INTERPRETATION
# ─────────────────────────────────────────────────────────────────────

def interpret_factors(chain_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Map each factor's correlation sign and strength to a business
    implication, for direct use in reporting.

    Returns
    -------
    pd.DataFrame with columns: Factor, Correlation, Implication
    """
    implications = {
        'CPI': {
            'positive': "Mild inflation correlates with stable or higher sales; customers may be buying ahead of further price increases.",
            'negative': "Rising prices suppress discretionary spending."
        },
        'Unemployment': {
            'positive': "Unexpected — investigate regional confounds before acting.",
            'negative': "Higher regional unemployment reduces discretionary spending; staff and inventory plans should account for local labor market conditions."
        },
        'Fuel_Price': {
            'positive': "Unexpected — investigate regional confounds before acting.",
            'negative': "Higher fuel prices reduce store visit frequency, particularly relevant for stores with longer average customer travel distance."
        },
        'Temperature': {
            'positive': "Warmer weeks associate with higher footfall and sales.",
            'negative': "Extreme temperatures suppress store visits; relevant for seasonal staffing and inventory timing."
        },
        'MarkDown_Total': {
            'positive': "Promotional spend is associated with higher sales, supporting continued (targeted) markdown investment.",
            'negative': "Unexpected — markdowns may be reactive to already-weak sales weeks rather than driving them; investigate causality before cutting spend."
        }
    }

    rows = []
    for _, r in chain_corr.iterrows():
        factor = r['Factor']
        direction = 'positive' if r['Correlation'] > 0 else 'negative'
        implication = implications.get(factor, {}).get(direction, "No predefined interpretation available.")
        rows.append({'Factor': factor, 'Correlation': r['Correlation'], 'Implication': implication})

    out = pd.DataFrame(rows)
    print("\n[INTERPRETATION] Business implications of external factors:")
    for _, r in out.iterrows():
        print(f"  - {r['Factor']} ({r['Correlation']:+.3f}): {r['Implication']}")
    return out


# ─────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_external_factors_analysis(df: pd.DataFrame) -> dict:
    """
    End-to-end external factors impact analysis.

    Steps
    -----
    1. Chain-level correlation
    2. Store-type breakdown
    3. Controlled linear regression
    4. Business interpretation

    Returns
    -------
    dict containing all intermediate results.
    """
    print("\n" + "=" * 60)
    print("  EXTERNAL FACTORS IMPACT ANALYSIS")
    print("=" * 60)

    chain_corr = chain_level_correlation(df)
    type_corr = store_type_correlation(df)
    regression = regression_impact(df)
    interpretation = interpret_factors(chain_corr)

    return {
        'chain_correlation': chain_corr,
        'store_type_correlation': type_corr,
        'regression_impact': regression,
        'interpretation': interpretation
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
    results = run_external_factors_analysis(df)
