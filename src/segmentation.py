"""
segmentation.py - Store & Department Segmentation
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Methods
───────
• K-Means clustering with Elbow + Silhouette evaluation
• Hierarchical clustering (Ward linkage) for validation
• Per-segment business interpretation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────
# 1. BUILD STORE-LEVEL FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────

def build_store_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to one row per store.

    Features used for segmentation
    ──────────────────────────────
    Total_Sales        : total revenue over full period
    Avg_Weekly_Sales   : mean weekly sales
    Sales_Std          : volatility of weekly sales
    Avg_MarkDown       : average total markdown spend
    MarkDown_Freq      : % of weeks with any markdown
    Holiday_Lift       : ratio of holiday to non-holiday sales
    Store_Size         : physical size (sq ft)
    Type_Encoded       : store type (A/B/C → 2/1/0)
    Avg_CPI            : mean CPI in store's region
    Avg_Unemployment   : mean unemployment rate
    """
    grp = df.groupby('Store')

    profile = pd.DataFrame({
        'Total_Sales'      : grp['Weekly_Sales'].sum(),
        'Avg_Weekly_Sales' : grp['Weekly_Sales'].mean(),
        'Sales_Std'        : grp['Weekly_Sales'].std(),
        'Avg_MarkDown'     : grp['MarkDown_Total'].mean(),
        'MarkDown_Freq'    : grp['HasMarkDown'].mean(),   # fraction
        'Store_Size'       : grp['Size'].first(),
        'Type_Encoded'     : grp['Type_Encoded'].first(),
        'Avg_CPI'          : grp['CPI'].mean(),
        'Avg_Unemployment' : grp['Unemployment'].mean(),
    })

    # Holiday lift: avg sales on holiday weeks / avg sales on non-holiday weeks
    hol    = df[df['IsHoliday'] == 1].groupby('Store')['Weekly_Sales'].mean()
    non_hol = df[df['IsHoliday'] == 0].groupby('Store')['Weekly_Sales'].mean()
    profile['Holiday_Lift'] = (hol / non_hol).fillna(1.0)

    print(f"[PROFILE] Store profile matrix : {profile.shape}")
    return profile


# ─────────────────────────────────────────────────────────────────────
# 2. ELBOW METHOD
# ─────────────────────────────────────────────────────────────────────

def elbow_method(X_scaled: np.ndarray,
                 k_range: range = range(2, 11)) -> dict:
    """
    Compute inertia (WCSS) for each k to identify the elbow.

    Returns
    -------
    dict : {k: inertia}
    """
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_

    print("\n[ELBOW] Inertia by k:")
    for k, v in inertias.items():
        print(f"  k={k} → {v:,.0f}")
    return inertias


# ─────────────────────────────────────────────────────────────────────
# 3. SILHOUETTE SCORES
# ─────────────────────────────────────────────────────────────────────

def silhouette_analysis(X_scaled: np.ndarray,
                        k_range: range = range(2, 11)) -> dict:
    """
    Compute silhouette score for each k.

    Returns
    -------
    dict : {k: silhouette_score}
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(scores, key=scores.get)
    print(f"\n[SILHOUETTE] Best k = {best_k}  (score = {scores[best_k]:.4f})")
    for k, s in scores.items():
        flag = " ←" if k == best_k else ""
        print(f"  k={k} → {s:.4f}{flag}")
    return scores


# ─────────────────────────────────────────────────────────────────────
# 4. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────

def kmeans_segmentation(profile: pd.DataFrame,
                        n_clusters: int = 4) -> tuple:
    """
    Fit K-Means on the store profile matrix.

    Returns
    -------
    profile_clustered : pd.DataFrame  (profile + 'Cluster' column)
    scaler            : fitted StandardScaler
    model             : fitted KMeans
    X_scaled          : np.ndarray
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(profile)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    profile_clustered          = profile.copy()
    profile_clustered['Cluster'] = labels

    sil = silhouette_score(X_scaled, labels)
    print(f"\n[K-MEANS] k={n_clusters}  →  Silhouette = {sil:.4f}")
    print(f"[K-MEANS] Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")
    return profile_clustered, scaler, km, X_scaled


# ─────────────────────────────────────────────────────────────────────
# 5. HIERARCHICAL CLUSTERING (validation)
# ─────────────────────────────────────────────────────────────────────

def hierarchical_segmentation(X_scaled: np.ndarray,
                               n_clusters: int = 4) -> np.ndarray:
    """
    Agglomerative hierarchical clustering (Ward linkage).

    Returns
    -------
    np.ndarray of cluster labels.
    """
    model  = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    print(f"[HIERARCHICAL] k={n_clusters}  →  Silhouette = {sil:.4f}")
    return labels


# ─────────────────────────────────────────────────────────────────────
# 6. SEGMENT INTERPRETATION
# ─────────────────────────────────────────────────────────────────────

def interpret_segments(profile_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster mean for each feature and attach a
    human-readable business label.

    Business labels are assigned by ranking clusters on Total_Sales.
    """
    summary = profile_clustered.groupby('Cluster').mean().round(2)

    # Rank clusters by average sales: 0 = lowest, 3 = highest
    rank = summary['Avg_Weekly_Sales'].rank(method='first').astype(int)
    labels_map = {
        1: "C - Low Performers",
        2: "B - Mid Performers",
        3: "A - High Performers",
        4: "S - Super Performers"
    }
    # For arbitrary k, generate generic labels if needed
    if profile_clustered['Cluster'].nunique() != 4:
        labels_map = {i: f"Segment {i}" for i in range(profile_clustered['Cluster'].nunique())}
        rank = summary['Avg_Weekly_Sales'].rank(method='first').astype(int)

    segment_names = {cluster_id: labels_map.get(r, f"Segment {cluster_id}")
                     for cluster_id, r in rank.items()}

    profile_clustered['Segment'] = profile_clustered['Cluster'].map(segment_names)

    print("\n[SEGMENTS] Cluster summary:")
    print(summary[['Avg_Weekly_Sales', 'Store_Size', 'MarkDown_Freq',
                    'Avg_CPI', 'Avg_Unemployment', 'Holiday_Lift']].to_string())
    return profile_clustered


# ─────────────────────────────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_segmentation(df: pd.DataFrame,
                     n_clusters: int = 4,
                     evaluate: bool = True) -> pd.DataFrame:
    """
    Complete store segmentation pipeline.

    Steps
    -----
    1. Build store profile
    2. (Optional) Elbow + Silhouette evaluation
    3. K-Means clustering
    4. Hierarchical validation
    5. Segment interpretation

    Returns
    -------
    pd.DataFrame  : store-level profile with Cluster + Segment columns
    """
    print("\n" + "="*60)
    print("  STORE SEGMENTATION PIPELINE")
    print("="*60)

    profile  = build_store_profile(df)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(profile)

    if evaluate:
        elbow_method(X_scaled, k_range=range(2, 8))
        silhouette_analysis(X_scaled, k_range=range(2, 8))

    profile_clustered, _, _, _ = kmeans_segmentation(profile, n_clusters=n_clusters)
    hier_labels = hierarchical_segmentation(X_scaled, n_clusters=n_clusters)
    profile_clustered['Cluster_Hierarchical'] = hier_labels

    profile_clustered = interpret_segments(profile_clustered)
    return profile_clustered


# ─────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os, sys
    BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE)
    from preprocessing import run_preprocessing
    DATA = os.path.join(BASE, '..', 'data')
    df   = run_preprocessing(DATA)
    seg  = run_segmentation(df, n_clusters=4, evaluate=True)
    print(seg[['Total_Sales', 'Avg_Weekly_Sales', 'Store_Size', 'Segment']].to_string())
