"""
Registers the existing PD/EAD/LGD models into MLflow (local file-based
tracking under mlruns/) so there's a versioned record of what's in
models/*.joblib -- parameters, validated metrics, and lineage -- instead of
just a bare file with no history.

This does NOT change how app.py or api.py load models; they still read
models/*.joblib directly, which is the right call for now (no reason to add
a runtime dependency on the MLflow store for a project this size). This
registry is the "what would you check before trusting this model" record a
real MRM process would want, and the natural next step if this ever needs
promotion/rollback across multiple model versions.

Run: python model_registry.py
View: mlflow ui --backend-store-uri sqlite:///mlflow.db   (then open localhost:5000)
"""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.xgboost

BASE = Path(__file__).parent
DB_PATH = BASE / 'mlflow.db'

# SQLite backend, not the raw filesystem store -- MLflow deprecated plain
# file-store tracking, and the Model Registry (registered_model_name below)
# needs a database backend to work at all. Still fully local, no server.
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
mlflow.set_experiment("credit_risk_models")

print(f"MLflow tracking store: sqlite:///{DB_PATH.as_posix()}")

# ── PD Model ─────────────────────────────────────────────────────────────────

with open(BASE / 'models' / 'pd_metadata.json') as f:
    pd_meta = json.load(f)
pd_model = joblib.load(BASE / 'models' / 'pd_model.joblib')
pd_features = (BASE / 'models' / 'pd_feature_names.csv').read_text().splitlines()[1:]

with mlflow.start_run(run_name="pd_xgboost") as run:
    mlflow.set_tags({'component': 'PD', 'algorithm': 'XGBoost',
                      'notebook': '04_pd_model.ipynb'})
    mlflow.log_params(pd_meta['best_params'])
    mlflow.log_param('n_estimators', pd_meta['n_estimators'])
    mlflow.log_param('n_features', pd_meta['n_features'])
    mlflow.log_metrics({
        'test_auc': pd_meta['test_auc'], 'test_ap': pd_meta['test_ap'],
        'gini': pd_meta['gini'], 'ks_statistic': pd_meta['ks_statistic'],
        'brier_score': pd_meta['brier_score'],
    })
    mlflow.log_dict({'features': pd_features}, "pd_feature_names.json")
    mlflow.xgboost.log_model(pd_model, name="model", registered_model_name="credit_risk_pd")
    print(f"Logged PD model: run_id={run.info.run_id}, AUC={pd_meta['test_auc']:.4f}")

# ── EAD Model (term lookup, not a trained model object) ─────────────────────

with open(BASE / 'models' / 'ead_metadata.json') as f:
    ead_meta = json.load(f)

with mlflow.start_run(run_name="ead_term_lookup") as run:
    mlflow.set_tags({'component': 'EAD', 'algorithm': 'Term-based mean lookup',
                      'notebook': '05_ead_model.ipynb'})
    mlflow.log_metrics({
        'test_mae': ead_meta['test_mae'], 'test_rmse': ead_meta['test_rmse'],
        'test_r2': ead_meta['test_r2'], 'test_wape': ead_meta['test_wape'],
    })
    mlflow.log_dict(ead_meta['lookup'], "ead_lookup.json")
    mlflow.log_param('note', ead_meta['note'][:250])
    print(f"Logged EAD lookup: run_id={run.info.run_id}, WAPE={ead_meta['test_wape']:.4f}")

# ── LGD Model ────────────────────────────────────────────────────────────────

with open(BASE / 'models' / 'lgd_metadata.json') as f:
    lgd_meta = json.load(f)
lgd_model = joblib.load(BASE / 'models' / 'lgd_model.joblib')
lgd_features = (BASE / 'models' / 'lgd_feature_names.csv').read_text().splitlines()[1:]

with mlflow.start_run(run_name="lgd_lightgbm") as run:
    mlflow.set_tags({'component': 'LGD', 'algorithm': 'LightGBM',
                      'notebook': '06_lgd_model.ipynb'})
    mlflow.log_params(lgd_meta['best_params'])
    mlflow.log_param('n_estimators', lgd_meta['n_estimators'])
    mlflow.log_param('n_features', lgd_meta['n_features'])
    mlflow.log_metrics({
        'test_mae': lgd_meta['test_mae'], 'test_rmse': lgd_meta['test_rmse'],
        'test_r2': lgd_meta['test_r2'], 'test_wape': lgd_meta['test_wape'],
    })
    mlflow.log_dict({'features': lgd_features}, "lgd_feature_names.json")
    mlflow.lightgbm.log_model(lgd_model, name="model", registered_model_name="credit_risk_lgd")
    print(f"Logged LGD model: run_id={run.info.run_id}, MAE={lgd_meta['test_mae']:.4f}")

print("\nDone. View the registry with:")
print(f"  mlflow ui --backend-store-uri sqlite:///{DB_PATH.as_posix()}")
