import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

print("=" * 60)
print("Precomputing baseline/challenger comparison")
print("=" * 60)

pd_model = joblib.load('models/pd_model.joblib')
pd_features = pd.read_csv('models/pd_feature_names.csv')['feature'].tolist()
config = joblib.load('models/app_config.joblib')

print("\n[1/4] Loading train_fe.csv / test_fe.csv...")
train = pd.read_csv('data/train_fe.csv')
test = pd.read_csv('data/test_fe.csv')
print(f"  Train: {train.shape}, Test: {test.shape}")


def gini_from_auc(auc):
    return 2 * auc - 1


def ks_from_scores(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


print("\n[2/4] Scoring baselines on the test population...")

results = {}

# --- Baseline 1: sub_grade_encoded only (LendingClub's own assigned grade) ---
auc = roc_auc_score(test['default'], test['sub_grade_encoded'])
results['Sub-Grade Only (LC assigned)'] = {
    'auc': float(auc), 'gini': gini_from_auc(auc),
    'ks': ks_from_scores(test['default'], test['sub_grade_encoded']),
    'n_features': 1,
}

# --- Baseline 2: FICO score only (higher FICO = lower risk, so invert) ---
auc = roc_auc_score(test['default'], -test['fico_score'])
results['FICO Score Only'] = {
    'auc': float(auc), 'gini': gini_from_auc(auc),
    'ks': ks_from_scores(test['default'], -test['fico_score']),
    'n_features': 1,
}

# --- Baseline 3: simple 4-feature logistic regression ---
simple_feats = ['fico_score', 'dti', 'revol_util', 'sub_grade_encoded']
X_train_simple = train[simple_feats].fillna(train[simple_feats].median())
X_test_simple = test[simple_feats].fillna(train[simple_feats].median())
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train_simple, train['default'])
simple_scores = logit.predict_proba(X_test_simple)[:, 1]
auc = roc_auc_score(test['default'], simple_scores)
results['Simple Logistic (4 features)'] = {
    'auc': float(auc), 'gini': gini_from_auc(auc),
    'ks': ks_from_scores(test['default'], simple_scores),
    'n_features': len(simple_feats),
}

# --- Full model: XGBoost (58 features) ---
xgb_scores = pd_model.predict_proba(test[pd_features])[:, 1]
auc = roc_auc_score(test['default'], xgb_scores)
results['XGBoost PD Model (full)'] = {
    'auc': float(auc), 'gini': gini_from_auc(auc),
    'ks': ks_from_scores(test['default'], xgb_scores),
    'n_features': len(pd_features),
}

for name, r in results.items():
    print(f"  {name}: AUC={r['auc']:.4f}, Gini={r['gini']:.4f}, KS={r['ks']:.4f}")

print("\n[3/4] Simplified profit comparison (shared EAD/LGD assumption)...")
# Uses fixed portfolio-average EAD/LGD so only the PD ranking differs across
# methods -- this is an approximation (not each loan's actual realized EAD/LGD),
# but since ALL four methods share the same assumption, the RELATIVE comparison
# (which ranking achieves higher profit) is still a fair, honest comparison.
AVG_EAD = 0.6144
AVG_LGD = 0.9221

sub_grade_int_rates = config['sub_grade_int_rates']
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sub_grades = [f'{g}{n}' for g in grades for n in range(1, 6)]
grade_labels = {i: sg for i, sg in enumerate(sub_grades)}
test_sub_grade = test['sub_grade_encoded'].map(grade_labels)
est_rate = test_sub_grade.map(sub_grade_int_rates).fillna(np.median(list(sub_grade_int_rates.values())))

r = est_rate.values / 100 / 12
term = test['term'].values
installment = np.where(r == 0, test['loan_amnt'].values / term,
                        test['loan_amnt'].values * r / (1 - (1 + r) ** (-term)))
est_revenue = np.clip(installment * term - test['loan_amnt'].values, 0, None)
approx_realized_loss = test['default'].values * AVG_EAD * AVG_LGD * test['loan_amnt'].values

score_map = {
    'Sub-Grade Only (LC assigned)': test['sub_grade_encoded'].values,
    'FICO Score Only': -test['fico_score'].values,
    'Simple Logistic (4 features)': simple_scores,
    'XGBoost PD Model (full)': xgb_scores,
}

for name, scores in score_map.items():
    ranks = pd.Series(scores).rank(pct=True).values
    thresholds = np.linspace(0.5, 0.99, 50)
    best_profit = -np.inf
    for t in thresholds:
        reject = ranks >= t
        tn = ~reject & (test['default'].values == 0)
        fn = ~reject & (test['default'].values == 1)
        profit = est_revenue[tn].sum() - approx_realized_loss[fn].sum()
        best_profit = max(best_profit, profit)
    results[name]['max_approx_profit'] = float(best_profit)
    print(f"  {name}: max approx profit = ${best_profit:,.0f}")

print("\n[4/4] Saving baseline comparison artifact...")
joblib.dump({'results': results, 'simple_feats': simple_feats,
             'avg_ead': AVG_EAD, 'avg_lgd': AVG_LGD},
            'models/baseline_comparison.joblib')
print("  Saved: models/baseline_comparison.joblib")
print("\nDone!")
