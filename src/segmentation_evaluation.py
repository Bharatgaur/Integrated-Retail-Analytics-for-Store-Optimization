"""
segmentation_evaluation.py - Formal Segmentation Quality Evaluation
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Purpose
-------
segmentation.py reports Silhouette Score inline during clustering. This
module provides a dedicated, multi-metric evaluation layer so that
segmentation quality is assessed independently and thoroughly, as its own
auditable step, using three complementary internal validation metrics:

1. Silhouette Score       — cohesion vs. separation, range [-1, 1], higher is better.
2. Davies-Bouldin Index   — ratio of within-cluster to between-cluster
                            distances, range [0, inf), lower is better.
3. Calinski-Harabasz Score — ratio of between-cluster to within-cluster
                            dispersion, range [0, inf), higher is better.

It also checks external validity by testing whether clusters align with
known business categories (store Type), and produces a stability check by
comparing K-Means labels against Hierarchical clustering labels.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────
# 1. INTERNAL VALIDATION METRICS
# ─────────────────────────────────────────────────────────────────────

def internal_validation_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute the three standard internal cluster validation metrics.

    Returns
    -------
    dict with keys: silhouette, davies_bouldin, calinski_harabasz
    """
    metrics = {
        'silhouette': silhouette_score(X_scaled, labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, labels)
    }

    print("\n[INTERNAL VALIDATION] Cluster quality metrics:")
    print(f"  Silhouette Score      : {metrics['silhouette']:.4f}  (higher is better, range -1 to 1)")
    print(f"  Davies-Bouldin Index   : {metrics['davies_bouldin']:.4f}  (lower is better, range 0 to inf)")
    print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.1f}  (higher is better, range 0 to inf)")
    return metrics


def per_cluster_silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Break the silhouette score down per cluster to identify which
    segments are well-separated and which are weaker / overlapping.

    Returns
    -------
    pd.DataFrame with columns: Cluster, Mean_Silhouette, Min_Silhouette, N_Stores
    """
    sample_scores = silhouette_samples(X_scaled, labels)
    df = pd.DataFrame({'Cluster': labels, 'Silhouette': sample_scores})

    out = df.groupby('Cluster')['Silhouette'].agg(['mean', 'min', 'count']).round(4)
    out.columns = ['Mean_Silhouette', 'Min_Silhouette', 'N_Stores']

    print("\n[PER-CLUSTER] Silhouette breakdown:")
    print(out.to_string())
    return out.reset_index()


# ─────────────────────────────────────────────────────────────────────
# 2. EXTERNAL VALIDATION (BUSINESS ALIGNMENT)
# ─────────────────────────────────────────────────────────────────────

def business_alignment_check(profile_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate cluster assignment against the known business category
    (store Type_Encoded, recovered as A/B/C) to check whether the
    unsupervised segmentation rediscovers a structure that the business
    already recognizes — a useful sanity check when no ground-truth
    labels exist for segmentation itself.

    Returns
    -------
    pd.DataFrame cross-tabulation of Cluster x Store Type.
    """
    type_map_rev = {2: 'A', 1: 'B', 0: 'C'}
    profile = profile_clustered.copy()
    profile['Store_Type'] = profile['Type_Encoded'].map(type_map_rev)

    crosstab = pd.crosstab(profile['Cluster'], profile['Store_Type'])
    print("\n[EXTERNAL VALIDATION] Cluster vs. known store Type:")
    print(crosstab.to_string())

    # Simple alignment score: for each cluster, what fraction belongs to its dominant type
    dominant_frac = crosstab.div(crosstab.sum(axis=1), axis=0).max(axis=1)
    alignment_score = dominant_frac.mean()
    print(f"\n[EXTERNAL VALIDATION] Average dominant-type purity across clusters: {alignment_score:.2%}")
    return crosstab


# ─────────────────────────────────────────────────────────────────────
# 3. STABILITY CHECK (K-MEANS vs. HIERARCHICAL)
# ─────────────────────────────────────────────────────────────────────

def stability_check(kmeans_labels: np.ndarray, hierarchical_labels: np.ndarray) -> float:
    """
    Compare K-Means and Hierarchical clustering label assignments using
    the Adjusted Rand Index (ARI). ARI corrects for chance agreement and
    ranges from -1 (worse than random) to 1 (identical partitions).

    A high ARI indicates the segmentation is not an artifact of one
    specific algorithm but a structure both methods independently find.

    Returns
    -------
    float : Adjusted Rand Index.
    """
    ari = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    print(f"\n[STABILITY] Adjusted Rand Index (K-Means vs. Hierarchical): {ari:.4f}")
    if ari > 0.6:
        print("  -> Strong agreement: segmentation structure is stable across methods.")
    elif ari > 0.3:
        print("  -> Moderate agreement: core structure is consistent, boundaries vary.")
    else:
        print("  -> Weak agreement: segmentation is sensitive to method choice; interpret with caution.")
    return ari


# ─────────────────────────────────────────────────────────────────────
# 4. FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_segmentation_evaluation(profile_clustered: pd.DataFrame,
                                X_scaled: np.ndarray,
                                hierarchical_labels: np.ndarray = None) -> dict:
    """
    End-to-end segmentation quality evaluation.

    Parameters
    ----------
    profile_clustered : pd.DataFrame
        Output of segmentation.run_segmentation(), must contain 'Cluster'
        and 'Type_Encoded' columns.
    X_scaled : np.ndarray
        The scaled feature matrix used to fit the clustering model.
    hierarchical_labels : np.ndarray, optional
        Labels from hierarchical clustering, for the stability check.
        If not provided, 'Cluster_Hierarchical' is read from profile_clustered.

    Returns
    -------
    dict containing all evaluation results.
    """
    print("\n" + "=" * 60)
    print("  SEGMENTATION QUALITY EVALUATION")
    print("=" * 60)

    labels = profile_clustered['Cluster'].values

    metrics = internal_validation_metrics(X_scaled, labels)
    per_cluster = per_cluster_silhouette(X_scaled, labels)
    alignment = business_alignment_check(profile_clustered)

    if hierarchical_labels is None and 'Cluster_Hierarchical' in profile_clustered.columns:
        hierarchical_labels = profile_clustered['Cluster_Hierarchical'].values

    ari = None
    if hierarchical_labels is not None:
        ari = stability_check(labels, hierarchical_labels)

    return {
        'metrics': metrics,
        'per_cluster_silhouette': per_cluster,
        'business_alignment': alignment,
        'adjusted_rand_index': ari
    }


# ─────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os, sys
    BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE)
    from preprocessing import run_preprocessing
    from segmentation import build_store_profile, kmeans_segmentation, hierarchical_segmentation, interpret_segments
    from sklearn.preprocessing import StandardScaler

    DATA = os.path.join(BASE, '..', 'data')
    df = run_preprocessing(DATA)
    profile = build_store_profile(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile)

    profile_clustered, _, _, X_scaled = kmeans_segmentation(profile, n_clusters=4)
    hier_labels = hierarchical_segmentation(X_scaled, n_clusters=4)
    profile_clustered['Cluster_Hierarchical'] = hier_labels
    profile_clustered = interpret_segments(profile_clustered)

    results = run_segmentation_evaluation(profile_clustered, X_scaled)
