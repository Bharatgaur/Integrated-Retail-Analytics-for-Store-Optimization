# Models Directory

Serialized model artifacts are written here when `main.py` is run.

| File | Produced by | Description |
|------|-------------|--------------|
| `kmeans_segmentation.joblib` | `main.py` | Fitted K-Means store segmentation model |
| `segmentation_scaler.joblib` | `main.py` | StandardScaler fitted on the store profile matrix |
| `random_forest_forecast.joblib` | `main.py` | Trained Random Forest demand forecasting model |
| `xgboost_forecast.joblib` | `main.py` | Trained XGBoost demand forecasting model |

Load a saved model with:

```python
import joblib
model = joblib.load('models/xgboost_forecast.joblib')
```

This directory is empty until `main.py` has been run at least once.
