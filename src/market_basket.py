"""
market_basket.py - Department-Level Market Basket Analysis
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Context
-------
The raw data contains no individual customer transaction or basket-level
records — only aggregated weekly sales per (Store, Dept). True association
rule mining (e.g. Apriori, FP-Growth) requires a transaction × item matrix,
which does not exist here. This module builds a defensible proxy:

1. Pivot weekly sales into a (Store-Week) x Department matrix.
2. Compute pairwise department sales correlation as a co-movement proxy
   for "departments that tend to sell together."
3. Apply the Apriori algorithm (via mlxtend) on a binarized version of the
   same matrix — a department is treated as "purchased" in a given
   store-week if its sales exceed its own median for that store — to
   surface department combinations with high support, confidence, and
   lift, mirroring true basket analysis as closely as the data allows.
4. Translate the strongest associations into cross-selling and
   co-location recommendations.

Limitation is stated explicitly wherever results are reported: these are
department co-occurrence patterns inferred from aggregated sales, not
verified customer-level associations.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    _HAS_MLXTEND = True
except ImportError:
    _HAS_MLXTEND = False
    print("[WARNING] mlxtend not installed; Apriori association rules will be skipped.")


# ─────────────────────────────────────────────────────────────────────
# 1. BUILD STORE-WEEK x DEPARTMENT MATRIX
# ─────────────────────────────────────────────────────────────────────

def build_dept_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the master dataframe into a (Store, Date) x Dept matrix of
    Weekly_Sales. Each row represents one store's sales across all its
    departments in a single week — the closest available analogue to a
    "basket."

    Returns
    -------
    pd.DataFrame, index = (Store, Date), columns = Dept, values = Weekly_Sales
    """
    matrix = df.pivot_table(
        index=['Store', 'Date'],
        columns='Dept',
        values='Weekly_Sales',
        aggfunc='sum',
        fill_value=0
    )
    print(f"[MATRIX] Store-week x Department matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]} departments")
    return matrix


# ─────────────────────────────────────────────────────────────────────
# 2. DEPARTMENT CORRELATION (CO-MOVEMENT PROXY)
# ─────────────────────────────────────────────────────────────────────

def department_correlation(matrix: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation between department sales series
    across all store-weeks. High correlation suggests departments that
    rise and fall together — a candidate signal for bundling or
    co-location, even without basket-level confirmation.

    Returns
    -------
    pd.DataFrame of the top_n most correlated department pairs
    (excluding self-pairs and duplicate symmetric pairs).
    """
    corr = matrix.corr()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    pairs_df = pd.DataFrame(pairs, columns=['Dept_A', 'Dept_B', 'Correlation'])
    pairs_df = pairs_df.dropna().sort_values('Correlation', ascending=False).head(top_n)

    print(f"\n[CORRELATION] Top {top_n} correlated department pairs:")
    print(pairs_df.to_string(index=False))
    return pairs_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# 3. APRIORI ASSOCIATION RULES (BINARIZED PROXY)
# ─────────────────────────────────────────────────────────────────────

def binarize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the sales matrix into a boolean "active department" matrix.
    A department is marked True for a given store-week if its sales that
    week exceed its own median sales for that store (i.e. an above-typical
    selling week for that department), approximating "this department
    contributed meaningfully to the basket this week."
    """
    binarized = matrix.apply(lambda col: col > col.median(), axis=0)
    print(f"[BINARIZE] Converted matrix to boolean activity flags, "
          f"avg activity rate = {binarized.mean().mean():.2%}")
    return binarized


def run_apriori(binarized: pd.DataFrame,
                min_support: float = 0.2,
                min_confidence: float = 0.6,
                max_departments: int = 25,
                top_n: int = 15) -> pd.DataFrame:
    """
    Run the Apriori algorithm on the binarized department-activity matrix
    and extract association rules above the given thresholds.

    Parameters
    ----------
    max_departments : int
        Apriori's itemset search space grows combinatorially with the
        number of columns. With ~80 departments at low support thresholds,
        the candidate itemset matrix becomes too large to hold in memory.
        To keep this tractable, only the `max_departments` highest-activity
        departments (by mean activity rate) are included.

    Returns
    -------
    pd.DataFrame of top_n rules sorted by lift, with columns:
    antecedents, consequents, support, confidence, lift.
    Empty DataFrame if mlxtend is unavailable or no rules clear the
    thresholds.
    """
    if not _HAS_MLXTEND:
        print("[APRIORI] Skipped — mlxtend not available.")
        return pd.DataFrame()

    binarized = binarized.copy()
    if binarized.shape[1] > max_departments:
        top_depts = binarized.mean().sort_values(ascending=False).head(max_departments).index
        binarized = binarized[top_depts]
        print(f"[APRIORI] Restricted to top {max_departments} departments by activity rate "
              f"to keep itemset search tractable.")

    binarized.columns = [f"Dept_{c}" for c in binarized.columns]

    frequent_itemsets = apriori(binarized, min_support=min_support, use_colnames=True, max_len=2)
    if frequent_itemsets.empty:
        print(f"[APRIORI] No itemsets found at min_support={min_support}. Try lowering the threshold.")
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        print(f"[APRIORI] No rules found at min_confidence={min_confidence}. Try lowering the threshold.")
        return pd.DataFrame()

    rules = rules.sort_values('lift', ascending=False).head(top_n)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))

    cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    out = rules[cols].round(3).reset_index(drop=True)

    print(f"\n[APRIORI] Top {len(out)} association rules (by lift):")
    print(out.to_string(index=False))
    return out


# ─────────────────────────────────────────────────────────────────────
# 4. CROSS-SELLING RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────

def generate_cross_sell_recommendations(corr_pairs: pd.DataFrame,
                                        rules: pd.DataFrame,
                                        top_n: int = 5) -> pd.DataFrame:
    """
    Translate the strongest correlation pairs and/or association rules
    into plain-language cross-selling and co-location recommendations.

    Returns
    -------
    pd.DataFrame with columns: Departments, Signal, Recommendation
    """
    recs = []

    for _, row in corr_pairs.head(top_n).iterrows():
        recs.append({
            'Departments': f"Dept {row['Dept_A']} & Dept {row['Dept_B']}",
            'Signal': f"Correlation = {row['Correlation']:.2f}",
            'Recommendation': (
                "Co-locate or run a joint promotional bundle — sales of these "
                "departments rise and fall together, suggesting shared demand drivers."
            )
        })

    if not rules.empty:
        for _, row in rules.head(top_n).iterrows():
            recs.append({
                'Departments': f"{row['antecedents']} -> {row['consequents']}",
                'Signal': f"Lift = {row['lift']:.2f}, Confidence = {row['confidence']:.2f}",
                'Recommendation': (
                    "Strong above-typical co-activity — feature the consequent "
                    "department in promotions targeting the antecedent department."
                )
            })

    out = pd.DataFrame(recs)
    print(f"\n[RECOMMENDATIONS] Generated {len(out)} cross-selling recommendations")
    return out


# ─────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_market_basket_analysis(df: pd.DataFrame,
                               min_support: float = 0.2,
                               min_confidence: float = 0.6) -> dict:
    """
    End-to-end market basket / department association pipeline.

    Steps
    -----
    1. Build store-week x department sales matrix
    2. Compute department correlation pairs
    3. Binarize matrix and run Apriori association rules
    4. Generate cross-selling recommendations

    Returns
    -------
    dict containing matrix, correlation pairs, rules, and recommendations.
    """
    print("\n" + "=" * 60)
    print("  MARKET BASKET ANALYSIS (DEPARTMENT-LEVEL PROXY)")
    print("=" * 60)
    print("NOTE: No individual transaction data is available. Results are")
    print("derived from department-level sales co-movement and an")
    print("above-median activity proxy, not verified customer baskets.\n")

    matrix = build_dept_matrix(df)
    corr_pairs = department_correlation(matrix)
    binarized = binarize_matrix(matrix)
    rules = run_apriori(binarized, min_support=min_support, min_confidence=min_confidence)
    recommendations = generate_cross_sell_recommendations(corr_pairs, rules)

    return {
        'matrix': matrix,
        'correlation_pairs': corr_pairs,
        'rules': rules,
        'recommendations': recommendations
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
    results = run_market_basket_analysis(df)
