import pandas as pd
import numpy as np
import joblib

print("=" * 60)
print("Precomputing PSI reference (train 2007-2015 vs test 2016-2017)")
print("=" * 60)

pd_model = joblib.load('models/pd_model.joblib')
pd_features = pd.read_csv('models/pd_feature_names.csv')['feature'].tolist()

print("\n[1/4] Loading train_fe.csv / test_fe.csv...")
train = pd.read_csv('data/train_fe.csv')
test = pd.read_csv('data/test_fe.csv')
print(f"  Train: {train.shape}, Test: {test.shape}")

print("\n[2/4] Scoring PD on both populations...")
train_score = pd_model.predict_proba(train[pd_features])[:, 1]
test_score = pd_model.predict_proba(test[pd_features])[:, 1]

EPS = 1e-4


def make_bins(values, max_bins=10):
    values = np.asarray(values, dtype=float)
    uniq = np.unique(values[~np.isnan(values)])
    if len(uniq) <= max_bins:
        edges = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]])
    else:
        qs = np.linspace(0, 1, max_bins + 1)
        edges = np.unique(np.quantile(values, qs))
        edges[0] = -np.inf
        edges[-1] = np.inf
        if len(edges) < 3:
            edges = np.array([-np.inf, np.median(values), np.inf])
    return edges


def bin_pcts(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    pct = counts / counts.sum()
    return np.clip(pct, EPS, None)


def psi_from_pcts(ref_pct, cur_pct):
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_feature_psi(train_vals, test_vals, max_bins=10):
    edges = make_bins(train_vals, max_bins)
    ref_pct = bin_pcts(train_vals, edges)
    cur_pct = bin_pcts(test_vals, edges)
    psi = psi_from_pcts(ref_pct, cur_pct)
    return psi, edges, ref_pct, cur_pct


print("\n[3/4] Computing PSI for PD score + all PD features...")

# issue_year is excluded: it's the column that DEFINES the train/test split
# (train = 2007-2015, test = 2016-2017, zero overlap by construction), so its
# PSI is a tautological artifact, not real drift, and would swamp every
# genuine signal if left in.
importances = dict(zip(pd_features, pd_model.feature_importances_))
monitored_features = sorted(
    [f for f in pd_features if f != 'issue_year'], key=lambda f: -importances[f])

results = {}

score_psi, score_edges, score_ref, score_cur = compute_feature_psi(train_score, test_score, max_bins=10)
results['__PD_SCORE__'] = {
    'psi': score_psi,
    'edges': score_edges.tolist(),
    'ref_pct': score_ref.tolist(),
    'cur_pct': score_cur.tolist(),
    'importance': None,
}
print(f"  PD Score PSI: {score_psi:.4f}")

for feat in monitored_features:
    psi, edges, ref_pct, cur_pct = compute_feature_psi(
        train[feat].values, test[feat].values, max_bins=10)
    results[feat] = {
        'psi': psi,
        'edges': edges.tolist(),
        'ref_pct': ref_pct.tolist(),
        'cur_pct': cur_pct.tolist(),
        'importance': float(importances[feat]),
    }

print(f"  Computed PSI for {len(monitored_features)} features")
top5 = sorted(monitored_features, key=lambda f: -results[f]['psi'])[:5]
for f in top5:
    print(f"    {f}: PSI = {results[f]['psi']:.4f}")

print("\n[4/4] Saving PSI reference artifact...")
psi_artifact = {
    'results': results,
    'monitored_features': monitored_features,
    'train_n': len(train),
    'test_n': len(test),
    'train_default_rate': float(train['default'].mean()),
    'test_default_rate': float(test['default'].mean()),
}
joblib.dump(psi_artifact, 'models/psi_reference.joblib')
print("  Saved: models/psi_reference.joblib")
print("\nDone!")
