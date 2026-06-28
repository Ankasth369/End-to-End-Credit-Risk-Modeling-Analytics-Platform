import pandas as pd
import numpy as np
import joblib

SAMPLE_SIZE = 2000

print("=" * 60)
print("Precomputing SHAP background sample (test vintage 2016-2017)")
print("=" * 60)

config = joblib.load('models/app_config.joblib')
pd_features = pd.read_csv('models/pd_feature_names.csv')['feature'].tolist()
lgd_features = pd.read_csv('models/lgd_feature_names.csv')['feature'].tolist()

print(f"\n[1/3] Scanning df_model_raw.csv in chunks for 2016-2017 rows "
      f"(target {SAMPLE_SIZE} rows)...")

chunks = []
collected = 0
for chunk in pd.read_csv('data/df_model_raw.csv', low_memory=False, chunksize=50000):
    chunk['issue_d_parsed'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y')
    chunk['issue_year'] = chunk['issue_d_parsed'].dt.year
    sub = chunk[chunk['issue_year'].between(2016, 2017)]
    if len(sub) > 0:
        take = sub.sample(n=min(len(sub), 40), random_state=42)
        chunks.append(take)
        collected += len(take)
    if collected >= SAMPLE_SIZE:
        break

df = pd.concat(chunks, ignore_index=True).head(SAMPLE_SIZE)
print(f"  Collected {len(df)} rows from 2016-2017 vintage")

print("\n[2/3] Applying the same cleaning/encoding as notebook 07...")

df['issue_month'] = df['issue_d_parsed'].dt.month

df['ead_pct'] = ((df['funded_amnt'] - df['total_rec_prncp']) / df['funded_amnt']).clip(0, 1)
df['lgd'] = (1 - (df['recoveries'] / df['funded_amnt']).clip(0, 1)).clip(0, 1)

drop_cols = [
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
    'last_fico_range_high', 'last_fico_range_low',
    'debt_settlement_flag', 'debt_settlement_flag_date',
    'settlement_status', 'settlement_date', 'settlement_amount',
    'settlement_percentage', 'settlement_term',
    'hardship_flag', 'hardship_type', 'hardship_reason',
    'hardship_status', 'deferral_term', 'hardship_amount',
    'hardship_start_date', 'hardship_end_date',
    'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
    'hardship_loan_status', 'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    'disbursement_method',
    'id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code',
    'funded_amnt', 'funded_amnt_inv', 'grade', 'loan_status',
    'fico_avg', 'fico_bin', 'issue_date',
]
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

df['term'] = df['term'].str.strip().str.replace(' months', '', regex=False).astype(int)

emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
           '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
           '8 years': 8, '9 years': 9, '10+ years': 10}
df['emp_length'] = df['emp_length'].map(emp_map)

ecl = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
df['credit_history_months'] = (
    (df['issue_d_parsed'].dt.year - ecl.dt.year) * 12 +
    (df['issue_d_parsed'].dt.month - ecl.dt.month)
)
df.drop(columns=['earliest_cr_line', 'issue_d', 'issue_d_parsed'], inplace=True, errors='ignore')

df['home_ownership'] = df['home_ownership'].replace({'NONE': 'OTHER', 'ANY': 'OTHER'})
df.drop(columns=['application_type'], inplace=True, errors='ignore')
df['initial_list_status'] = (df['initial_list_status'] == 'f').astype(int)

df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
df.drop(columns=['fico_range_low', 'fico_range_high'], inplace=True)

medians = config['medians']
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if col in medians and df[col].isnull().any():
        df[col] = df[col].fillna(medians[col])

winsor = config['winsor_upper']
for col, upper in winsor.items():
    if col in df.columns:
        df[col] = df[col].clip(upper=upper)
df['revol_util'] = df['revol_util'].clip(upper=100)
df['dti'] = df['dti'].clip(lower=0)

grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sub_grades = [f'{g}{n}' for g in grades for n in range(1, 6)]
sub_grade_map = {sg: i for i, sg in enumerate(sub_grades)}
df['sub_grade_encoded'] = df['sub_grade'].map(sub_grade_map)

for col, prefix in [('home_ownership', 'home'),
                     ('verification_status', 'verif'),
                     ('purpose', 'purpose')]:
    dummies = pd.get_dummies(df[col], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)

df['state_default_rate'] = df['addr_state'].map(
    config['state_default_map']).fillna(config['global_default_rate'])
df['state_lgd_rate'] = df['addr_state'].map(
    config['state_lgd_map']).fillna(config['global_lgd_rate'])
df.drop(columns=['sub_grade', 'addr_state'], inplace=True, errors='ignore')

ai = df['annual_inc']
df['loan_to_income'] = df['loan_amnt'] / (ai + 1)
df['installment_to_income'] = df['installment'] / (ai / 12 + 1)
df['revol_bal_to_income'] = df['revol_bal'] / (ai + 1)
df['available_credit'] = df['total_rev_hi_lim'] - df['revol_bal']
df['available_credit_pct'] = df['available_credit'] / (df['total_rev_hi_lim'] + 1)
df['active_ratio'] = df['open_acc'] / (df['total_acc'] + 1)
df['delinq_per_account'] = df['delinq_2yrs'] / (df['total_acc'] + 1)
df['pub_rec_per_account'] = df['pub_rec'] / (df['total_acc'] + 1)
df['installment_pct_of_loan'] = df['installment'] / (df['loan_amnt'] + 1)
df['accounts_per_year'] = df['total_acc'] / (df['credit_history_months'] / 12 + 1)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

ratio_medians = config['ratio_medians']
ratio_upper = config['ratio_upper']
for col in ratio_upper:
    if col in df.columns:
        df[col] = df[col].fillna(ratio_medians.get(col, 0))
        df[col] = df[col].clip(upper=ratio_upper[col])

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(medians.get(col, 0))

print("\n[3/3] Reindexing to model feature lists and saving...")
pd_X_sample = df.reindex(columns=pd_features, fill_value=0)
lgd_X_sample = df.reindex(columns=lgd_features, fill_value=0)

joblib.dump({'pd_X_sample': pd_X_sample, 'lgd_X_sample': lgd_X_sample},
            'models/shap_sample.joblib')
print(f"  pd_X_sample: {pd_X_sample.shape}")
print(f"  lgd_X_sample: {lgd_X_sample.shape}")
print("  Saved: models/shap_sample.joblib")
print("\nDone!")
