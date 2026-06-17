"""
strategy.py - Personalization, Marketing, and Inventory Strategy Generation
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Purpose
-------
Earlier modules produce segments, forecasts, anomaly flags, and external
factor relationships. This module is the synthesis layer: it converts
those outputs into concrete, segment-specific recommendations for
inventory management, markdown/pricing strategy, and marketing —
the "Personalization Strategies" and "Real-World Application" components
of the project brief.

Each recommendation is generated programmatically from the actual
segment statistics passed in, not hardcoded, so the output reflects
whatever the clustering step produced for a given run.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────
# 1. INVENTORY STRATEGY
# ─────────────────────────────────────────────────────────────────────

def inventory_strategy(profile_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Generate segment-specific inventory and safety-stock recommendations
    based on each segment's sales volatility (Sales_Std relative to mean)
    and holiday lift.

    Returns
    -------
    pd.DataFrame with columns: Segment, Avg_Weekly_Sales, Volatility_Ratio,
    Holiday_Lift, Recommendation
    """
    summary = profile_clustered.groupby('Segment').agg(
        Avg_Weekly_Sales=('Avg_Weekly_Sales', 'mean'),
        Sales_Std=('Sales_Std', 'mean'),
        Holiday_Lift=('Holiday_Lift', 'mean'),
        N_Stores=('Avg_Weekly_Sales', 'count')
    ).round(2)
    summary['Volatility_Ratio'] = (summary['Sales_Std'] / summary['Avg_Weekly_Sales']).round(2)

    recs = []
    for segment, row in summary.iterrows():
        if row['Volatility_Ratio'] < summary['Volatility_Ratio'].median():
            base_rec = "Lower-than-average sales variance; safety stock can be reduced by roughly 15-20% without raising stockout risk."
        else:
            base_rec = "Higher-than-average sales variance; maintain or increase safety stock buffer to avoid stockouts."

        if row['Holiday_Lift'] > 1.10:
            holiday_rec = " Holiday weeks show a strong lift; pre-build inventory 4-6 weeks ahead of major holidays."
        else:
            holiday_rec = " Holiday lift is modest; standard reorder cadence is likely sufficient through the holiday period."

        recs.append({
            'Segment': segment,
            'N_Stores': int(row['N_Stores']),
            'Avg_Weekly_Sales': row['Avg_Weekly_Sales'],
            'Volatility_Ratio': row['Volatility_Ratio'],
            'Holiday_Lift': round(row['Holiday_Lift'], 2),
            'Recommendation': base_rec + holiday_rec
        })

    out = pd.DataFrame(recs)
    print("\n[INVENTORY STRATEGY] Segment-specific recommendations:")
    for _, r in out.iterrows():
        print(f"  - {r['Segment']} ({r['N_Stores']} stores): {r['Recommendation']}")
    return out


# ─────────────────────────────────────────────────────────────────────
# 2. PRICING / MARKDOWN STRATEGY
# ─────────────────────────────────────────────────────────────────────

def pricing_strategy(profile_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Generate segment-specific markdown and pricing recommendations based
    on each segment's markdown frequency and average sales level.

    Returns
    -------
    pd.DataFrame with columns: Segment, Avg_MarkDown, MarkDown_Freq, Recommendation
    """
    summary = profile_clustered.groupby('Segment').agg(
        Avg_Weekly_Sales=('Avg_Weekly_Sales', 'mean'),
        Avg_MarkDown=('Avg_MarkDown', 'mean'),
        MarkDown_Freq=('MarkDown_Freq', 'mean')
    ).round(2)

    sales_median = summary['Avg_Weekly_Sales'].median()

    recs = []
    for segment, row in summary.iterrows():
        if row['Avg_Weekly_Sales'] >= sales_median:
            rec = (
                "Premium / high-volume segment: prioritize loyalty and "
                "experience-based marketing over deep discounting; sustain a "
                "modest 3-5% price premium where local competition allows."
            )
        else:
            rec = (
                "Price-sensitive segment: lean on markdown-driven promotions "
                f"(currently active in {row['MarkDown_Freq']:.0%} of weeks) to "
                "drive volume; ensure markdown timing aligns with regional "
                "economic conditions (CPI, unemployment)."
            )
        recs.append({
            'Segment': segment,
            'Avg_Weekly_Sales': row['Avg_Weekly_Sales'],
            'Avg_MarkDown': row['Avg_MarkDown'],
            'MarkDown_Freq': row['MarkDown_Freq'],
            'Recommendation': rec
        })

    out = pd.DataFrame(recs)
    print("\n[PRICING STRATEGY] Segment-specific recommendations:")
    for _, r in out.iterrows():
        print(f"  - {r['Segment']}: {r['Recommendation']}")
    return out


# ─────────────────────────────────────────────────────────────────────
# 3. MARKETING STRATEGY
# ─────────────────────────────────────────────────────────────────────

def marketing_strategy(profile_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Generate segment-specific marketing recommendations based on
    regional economic sensitivity (unemployment, CPI) and store size.

    Returns
    -------
    pd.DataFrame with columns: Segment, Avg_Unemployment, Avg_CPI, Recommendation
    """
    summary = profile_clustered.groupby('Segment').agg(
        Avg_Unemployment=('Avg_Unemployment', 'mean'),
        Avg_CPI=('Avg_CPI', 'mean'),
        Store_Size=('Store_Size', 'mean')
    ).round(2)

    unemployment_median = summary['Avg_Unemployment'].median()

    recs = []
    for segment, row in summary.iterrows():
        if row['Avg_Unemployment'] > unemployment_median:
            rec = (
                "Operates in higher-unemployment regions: favor value-oriented, "
                "community-focused local marketing over premium brand positioning; "
                "fuel-price sensitivity is also likely to be elevated here."
            )
        else:
            rec = (
                "Operates in lower-unemployment regions: positioned for "
                "loyalty-program investment, premium assortment promotion, "
                "and cross-department bundle marketing."
            )
        recs.append({
            'Segment': segment,
            'Avg_Unemployment': row['Avg_Unemployment'],
            'Avg_CPI': row['Avg_CPI'],
            'Recommendation': rec
        })

    out = pd.DataFrame(recs)
    print("\n[MARKETING STRATEGY] Segment-specific recommendations:")
    for _, r in out.iterrows():
        print(f"  - {r['Segment']}: {r['Recommendation']}")
    return out


# ─────────────────────────────────────────────────────────────────────
# 4. REAL-WORLD CHALLENGES
# ─────────────────────────────────────────────────────────────────────

def real_world_challenges() -> pd.DataFrame:
    """
    Compile the known practical challenges in deploying this system,
    paired with the corresponding mitigation already reflected in the
    project design.

    Returns
    -------
    pd.DataFrame with columns: Challenge, Impact, Mitigation
    """
    rows = [
        {
            'Challenge': 'Data quality',
            'Impact': 'Missing or incorrect MarkDown values bias promotional ROI estimates.',
            'Mitigation': 'Domain-aware imputation (zero-fill for structurally absent MarkDowns) plus explicit missing-value audit before modeling.'
        },
        {
            'Challenge': 'Store heterogeneity',
            'Impact': '45 stores behave differently; a single global model underfits local patterns.',
            'Mitigation': 'Store and department identifiers included as model features; segmentation used to group similar stores before strategy formulation.'
        },
        {
            'Challenge': 'Cold start',
            'Impact': 'New stores or departments have no sales history for lag-based features.',
            'Mitigation': 'Store-type and size-based priors can substitute for missing history until sufficient data accumulates.'
        },
        {
            'Challenge': 'Concept drift',
            'Impact': 'Consumer behavior shifts (economic shocks, demand shocks) degrade model accuracy over time.',
            'Mitigation': 'Time-based train/test split exposes how performance degrades over the holdout period; periodic retraining is recommended in production.'
        },
        {
            'Challenge': 'External data lag',
            'Impact': 'CPI and unemployment figures are published with a multi-week lag.',
            'Mitigation': 'Forward/backward-fill bridges short gaps; leading indicators could substitute for faster signal in a production system.'
        },
        {
            'Challenge': 'No transaction-level data',
            'Impact': 'True market basket analysis is not possible; only department-level co-movement can be inferred.',
            'Mitigation': 'Department correlation and binarized Apriori proxy used, with limitations stated explicitly in reporting.'
        },
        {
            'Challenge': 'Scalability',
            'Impact': 'Real retail chains operate 1,000+ stores; the current scripts run on a single machine.',
            'Mitigation': 'Pipeline functions are modular and stateless per store/department, making them straightforward to parallelize or port to distributed compute if required.'
        }
    ]
    out = pd.DataFrame(rows)
    print("\n[REAL-WORLD CHALLENGES]")
    print(out.to_string(index=False))
    return out


# ─────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_strategy_formulation(profile_clustered: pd.DataFrame) -> dict:
    """
    End-to-end personalization and strategy generation.

    Steps
    -----
    1. Inventory strategy by segment
    2. Pricing / markdown strategy by segment
    3. Marketing strategy by segment
    4. Real-world implementation challenges

    Returns
    -------
    dict containing all strategy outputs.
    """
    print("\n" + "=" * 60)
    print("  PERSONALIZATION & STRATEGY FORMULATION")
    print("=" * 60)

    inventory = inventory_strategy(profile_clustered)
    pricing = pricing_strategy(profile_clustered)
    marketing = marketing_strategy(profile_clustered)
    challenges = real_world_challenges()

    return {
        'inventory_strategy': inventory,
        'pricing_strategy': pricing,
        'marketing_strategy': marketing,
        'real_world_challenges': challenges
    }


# ─────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os, sys
    BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE)
    from preprocessing import run_preprocessing
    from segmentation import run_segmentation

    DATA = os.path.join(BASE, '..', 'data')
    df = run_preprocessing(DATA)
    profile_clustered = run_segmentation(df, n_clusters=4, evaluate=False)
    results = run_strategy_formulation(profile_clustered)
