import pandas as pd
import numpy as np
import joblib
import json

print("=" * 60)
print("Precomputing app configuration from training data...")
print("=" * 60)

print("\n[1/4] Loading train.csv...")
train = pd.read_csv('data/train.csv')
print(f"  Shape: {train.shape}")

print("\n[2/4] Loading raw data (selected columns for LGD stats)...")
raw_cols = ['addr_state', 'issue_year', 'default', 'recoveries',
            'funded_amnt', 'mths_since_last_delinq']
raw = pd.read_csv('data/df_model_raw.csv', usecols=raw_cols, low_memory=False)
raw_train = raw[raw['issue_year'] <= 2015].copy()
raw_train_defaults = raw_train[raw_train['default'] == 1].copy()
print(f"  Raw train defaults: {len(raw_train_defaults)}")

print("\n[3/4] Computing statistics...")

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
medians = train[num_cols].median().to_dict()
medians['mths_since_last_delinq'] = float(
    raw_train['mths_since_last_delinq'].median()
)
medians['int_rate'] = float(train['int_rate'].median())
medians['installment'] = float(train['installment'].median())

winsor_cols = ['annual_inc', 'dti', 'revol_bal', 'revol_util', 'open_acc', 'total_acc']
winsor_upper = train[winsor_cols].quantile(0.995).to_dict()

smoothing = 100
global_default = float(train['default'].mean())
state_stats = train.groupby('addr_state')['default'].agg(['mean', 'count'])
state_stats['smoothed'] = (
    (state_stats['count'] * state_stats['mean'] + smoothing * global_default) /
    (state_stats['count'] + smoothing)
)
state_default_map = state_stats['smoothed'].to_dict()

raw_train_defaults['lgd'] = (
    1 - (raw_train_defaults['recoveries'] / raw_train_defaults['funded_amnt']).clip(0, 1)
).clip(0, 1)
global_lgd = float(raw_train_defaults['lgd'].mean())
state_lgd_stats = raw_train_defaults.groupby('addr_state')['lgd'].agg(['mean', 'count'])
state_lgd_stats['smoothed'] = (
    (state_lgd_stats['count'] * state_lgd_stats['mean'] + smoothing * global_lgd) /
    (state_lgd_stats['count'] + smoothing)
)
state_lgd_map = state_lgd_stats['smoothed'].to_dict()

def create_features(d):
    d = d.copy()
    d['loan_to_income'] = d['loan_amnt'] / (d['annual_inc'] + 1)
    d['installment_to_income'] = d['installment'] / (d['annual_inc'] / 12 + 1)
    d['revol_bal_to_income'] = d['revol_bal'] / (d['annual_inc'] + 1)
    d['available_credit'] = d['total_rev_hi_lim'] - d['revol_bal']
    d['available_credit_pct'] = d['available_credit'] / (d['total_rev_hi_lim'] + 1)
    d['active_ratio'] = d['open_acc'] / (d['total_acc'] + 1)
    d['delinq_per_account'] = d['delinq_2yrs'] / (d['total_acc'] + 1)
    d['pub_rec_per_account'] = d['pub_rec'] / (d['total_acc'] + 1)
    d['installment_pct_of_loan'] = d['installment'] / (d['loan_amnt'] + 1)
    d['accounts_per_year'] = d['total_acc'] / (d['credit_history_months'] / 12 + 1)
    return d

train_ratios = create_features(train)
ratio_cols = ['loan_to_income', 'installment_to_income', 'revol_bal_to_income',
              'available_credit', 'available_credit_pct', 'active_ratio',
              'delinq_per_account', 'pub_rec_per_account',
              'installment_pct_of_loan', 'accounts_per_year']
ratio_medians = train_ratios[ratio_cols].median().to_dict()
ratio_upper = train_ratios[ratio_cols].quantile(0.995).to_dict()

states = sorted(train['addr_state'].unique().tolist())
purposes = sorted(train['purpose'].unique().tolist())

sub_grade_int_rates = train.groupby('sub_grade')['int_rate'].median().to_dict()

print(f"  States: {len(states)}")
print(f"  Purposes: {len(purposes)}")
print(f"  Global default rate: {global_default:.4f}")
print(f"  Global LGD rate: {global_lgd:.4f}")
print(f"  Median features: {len(medians)}")

config = {
    'medians': medians,
    'winsor_upper': winsor_upper,
    'state_default_map': state_default_map,
    'state_lgd_map': state_lgd_map,
    'global_default_rate': global_default,
    'global_lgd_rate': global_lgd,
    'ratio_medians': ratio_medians,
    'ratio_upper': ratio_upper,
    'states': states,
    'purposes': purposes,
    'sub_grade_int_rates': sub_grade_int_rates,
}

print("\n[4/4] Saving config...")
joblib.dump(config, 'models/app_config.joblib')
print(f"  Saved: models/app_config.joblib")
print("\nDone!")
