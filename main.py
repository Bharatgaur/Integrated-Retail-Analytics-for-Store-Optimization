"""
main.py - End-to-End Pipeline Orchestrator
Project: Integrated Retail Analytics for Store Optimization and Demand Forecasting

Runs every project component in sequence and persists model artifacts and
key result tables to disk. This is the single entry point referenced in
the README's "How to Run" section once setup is complete.

Usage
-----
    python main.py

Outputs
-------
    models/            Serialized model artifacts (.joblib)
    reports/run_summary.txt   Plain-text summary of key results from this run
"""

import os
import sys
import joblib
import pandas as pd
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, 'src')
DATA_DIR = os.path.join(BASE, 'data')
MODELS_DIR = os.path.join(BASE, 'models')
REPORTS_DIR = os.path.join(BASE, 'reports')

sys.path.insert(0, SRC)

from preprocessing import run_preprocessing
from anomaly_detection import run_anomaly_detection
from time_series_analysis import run_time_series_analysis
from segmentation import (
    build_store_profile, kmeans_segmentation,
    hierarchical_segmentation, interpret_segments
)
from segmentation_evaluation import run_segmentation_evaluation
from market_basket import run_market_basket_analysis
from external_factors import run_external_factors_analysis
from forecasting import run_forecasting
from strategy import run_strategy_formulation
from sklearn.preprocessing import StandardScaler


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def save_model(model, name: str):
    if model is None:
        print(f"[SKIP] {name} not available; nothing to save.")
        return
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"[SAVED] {name} -> {path}")


def write_run_summary(results: dict, path: str):
    """Write a compact plain-text summary of this run's key outputs."""
    lines = []
    lines.append("=" * 70)
    lines.append("  PIPELINE RUN SUMMARY")
    lines.append(f"  Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 70)

    lines.append("\n--- Anomaly Detection ---")
    lines.append(f"Consensus anomalies flagged: {results['df']['Is_Anomaly'].sum():,}")

    lines.append("\n--- Segmentation ---")
    lines.append(f"Silhouette Score: {results['seg_eval']['metrics']['silhouette']:.4f}")
    lines.append(f"Davies-Bouldin Index: {results['seg_eval']['metrics']['davies_bouldin']:.4f}")
    lines.append(f"Calinski-Harabasz Score: {results['seg_eval']['metrics']['calinski_harabasz']:.1f}")
    if results['seg_eval']['adjusted_rand_index'] is not None:
        lines.append(f"Adjusted Rand Index (KMeans vs Hierarchical): {results['seg_eval']['adjusted_rand_index']:.4f}")

    lines.append("\n--- Forecasting ---")
    lines.append(results['forecast']['model_summary'].to_string())

    lines.append("\n--- Market Basket (top correlation pairs) ---")
    lines.append(results['basket']['correlation_pairs'].head(5).to_string(index=False))

    lines.append("\n--- External Factors (chain-level correlation) ---")
    lines.append(results['ext_factors']['chain_correlation'].to_string(index=False))

    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n[SAVED] Run summary -> {path}")


def main():
    ensure_dirs()
    print("\n" + "#" * 70)
    print("#  INTEGRATED RETAIL ANALYTICS — FULL PIPELINE RUN")
    print("#" * 70)

    # 1. Preprocessing
    df = run_preprocessing(DATA_DIR)

    # 2. Anomaly Detection
    df = run_anomaly_detection(df, handle_strategy='cap')

    # 3. Time-Based Trend & Seasonality Analysis
    time_series_results = run_time_series_analysis(df)

    # 4. Segmentation
    profile = build_store_profile(df)
    scaler = StandardScaler()
    profile_clustered, scaler, kmeans_model, X_scaled = kmeans_segmentation(profile, n_clusters=4)
    hier_labels = hierarchical_segmentation(X_scaled, n_clusters=4)
    profile_clustered['Cluster_Hierarchical'] = hier_labels
    profile_clustered = interpret_segments(profile_clustered)

    # 5. Segmentation Quality Evaluation
    seg_eval = run_segmentation_evaluation(profile_clustered, X_scaled, hier_labels)

    # 6. Market Basket Analysis
    basket_results = run_market_basket_analysis(df)

    # 7. External Factors Analysis
    ext_factors_results = run_external_factors_analysis(df)

    # 8. Demand Forecasting
    forecast_results = run_forecasting(df, test_weeks=8)

    # 9. Personalization & Strategy Formulation
    strategy_results = run_strategy_formulation(profile_clustered)

    # 10. Persist model artifacts
    save_model(kmeans_model, 'kmeans_segmentation')
    save_model(scaler, 'segmentation_scaler')
    save_model(forecast_results.get('rf_model'), 'random_forest_forecast')
    save_model(forecast_results.get('xgb_model'), 'xgboost_forecast')

    # 11. Write run summary
    results = {
        'df': df,
        'seg_eval': seg_eval,
        'forecast': forecast_results,
        'basket': basket_results,
        'ext_factors': ext_factors_results,
    }
    write_run_summary(results, os.path.join(REPORTS_DIR, 'run_summary.txt'))

    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETE")
    print("#" * 70)
    return {
        'df': df,
        'profile_clustered': profile_clustered,
        'time_series_results': time_series_results,
        'seg_eval': seg_eval,
        'basket_results': basket_results,
        'ext_factors_results': ext_factors_results,
        'forecast_results': forecast_results,
        'strategy_results': strategy_results,
    }


if __name__ == '__main__':
    main()
