"""
Automated model health check: PSI (population drift) and calibration (ECE),
computed headlessly and written to a timestamped report + an alert file when
a threshold is breached -- the difference between "a human has to open the
Streamlit PSI page and read it" and "this runs on its own and pages someone."

This project has no live production scoring stream to monitor, so this
checks the same reference comparison already validated in the app: train
(2007-2015) vs. test (2016-2017) population/score drift, and out-of-time
calibration error. In a real deployment, "current" would be the last N days
of live scoring requests (e.g. pulled from logs/api_audit.log) instead of
the fixed test set.

Run manually:
    python monitoring.py

Schedule on Windows (no cloud cron available in this environment) via
Task Scheduler, e.g. daily at 6am:
    schtasks /create /tn "CreditRiskMonitoring" /tr "python \"G:\\Projects\\Credit Risk Modeling\\monitoring.py\"" /sc daily /st 06:00
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import credit_core as cc

BASE = Path(__file__).parent
REPORT_DIR = BASE / 'monitoring_reports'
REPORT_DIR.mkdir(exist_ok=True)
ALERT_LOG = BASE / 'logs' / 'monitoring_alerts.log'
ALERT_LOG.parent.mkdir(exist_ok=True)

PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25
ECE_ALERT_THRESHOLD = 0.05


def compute_psi(train_vals, test_vals, max_bins=10):
    eps = 1e-4
    values = np.asarray(train_vals, dtype=float)
    uniq = np.unique(values[~np.isnan(values)])
    if len(uniq) <= max_bins:
        edges = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(values, np.linspace(0, 1, max_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.clip(np.histogram(train_vals, bins=edges)[0] / len(train_vals), eps, None)
    cur_pct = np.clip(np.histogram(test_vals, bins=edges)[0] / len(test_vals), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_ece(pred, actual, n_bins=10):
    d = pd.DataFrame({'pred': pred, 'actual': actual})
    d['decile'] = pd.qcut(d['pred'], n_bins, labels=False, duplicates='drop')
    g = d.groupby('decile').agg(mp=('pred', 'mean'), ar=('actual', 'mean'), n=('pred', 'size'))
    return float((np.abs(g['mp'] - g['ar']) * g['n']).sum() / g['n'].sum())


def run_checks():
    models = cc.load_models(BASE)
    train = pd.read_csv(BASE / 'data' / 'train_fe.csv')
    test = pd.read_csv(BASE / 'data' / 'test_fe.csv')

    train_score = models['pd_model'].predict_proba(train[models['pd_features']])[:, 1]
    test_score = models['pd_model'].predict_proba(test[models['pd_features']])[:, 1]
    score_psi = compute_psi(train_score, test_score)

    monitored_features = [f for f in models['pd_features'] if f != 'issue_year']
    feature_psi = {f: compute_psi(train[f].values, test[f].values) for f in monitored_features}
    worst_feature = max(feature_psi, key=feature_psi.get)
    worst_feature_psi = feature_psi[worst_feature]

    portfolio = cc.load_portfolio(BASE)
    ece = None
    if portfolio is not None and 'issue_year' in portfolio.columns:
        fit_set = portfolio[portfolio['issue_year'] == 2016]
        eval_set = portfolio[portfolio['issue_year'] == 2017]
        eps = 1e-6
        fit_logit = np.log(np.clip(fit_set['pd_pred'], eps, 1 - eps) / (1 - np.clip(fit_set['pd_pred'], eps, 1 - eps)))
        platt = LogisticRegression()
        platt.fit(fit_logit.values.reshape(-1, 1), fit_set['actual_default'])
        eval_logit = np.log(np.clip(eval_set['pd_pred'], eps, 1 - eps) / (1 - np.clip(eval_set['pd_pred'], eps, 1 - eps)))
        recalibrated = platt.predict_proba(eval_logit.values.reshape(-1, 1))[:, 1]
        ece = compute_ece(eval_set['pd_pred'].values, eval_set['actual_default'].values)
        ece_after = compute_ece(recalibrated, eval_set['actual_default'].values)
    else:
        ece_after = None

    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'score_psi': score_psi,
        'score_psi_status': psi_status(score_psi),
        'worst_feature': worst_feature,
        'worst_feature_psi': worst_feature_psi,
        'worst_feature_status': psi_status(worst_feature_psi),
        'calibration_ece_raw': ece,
        'calibration_ece_after_recalibration': ece_after,
        'calibration_status': 'ALERT' if (ece is not None and ece > ECE_ALERT_THRESHOLD) else 'OK',
    }
    return report


def psi_status(value):
    if value >= PSI_ALERT_THRESHOLD:
        return 'ALERT'
    if value >= PSI_WARN_THRESHOLD:
        return 'WARN'
    return 'OK'


def write_report(report):
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = REPORT_DIR / f'report_{ts}.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path


def check_and_alert(report):
    alerts = []
    if report['score_psi_status'] == 'ALERT':
        alerts.append(f"Score PSI {report['score_psi']:.4f} >= {PSI_ALERT_THRESHOLD} -- significant population drift")
    if report['worst_feature_status'] == 'ALERT':
        alerts.append(f"Feature '{report['worst_feature']}' PSI {report['worst_feature_psi']:.4f} >= {PSI_ALERT_THRESHOLD}")
    if report['calibration_status'] == 'ALERT':
        alerts.append(f"Raw calibration ECE {report['calibration_ece_raw']:.4f} >= {ECE_ALERT_THRESHOLD} -- consider refitting the Platt scaling correction")

    for msg in alerts:
        entry = {'timestamp': report['timestamp'], 'alert': msg}
        with open(ALERT_LOG, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        print(f"ALERT: {msg}")

    return alerts


if __name__ == "__main__":
    print("Running model health checks...")
    report = run_checks()
    path = write_report(report)
    print(f"Report written: {path}")
    print(json.dumps(report, indent=2))

    alerts = check_and_alert(report)
    if not alerts:
        print("\nNo alerts. All checks within threshold.")
    else:
        print(f"\n{len(alerts)} alert(s) raised -- see {ALERT_LOG}")
