"""
Shared scoring/preprocessing core for the credit risk model.

Single source of truth for preprocessing, scoring, and calibration logic,
used by both the Streamlit app (app.py) and the FastAPI service (api.py).
Framework-agnostic: no Streamlit imports here. Callers own their own caching.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).parent

# ── Constants ──────────────────────────────────────────────────────────────────

GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
SUB_GRADES = [f'{g}{n}' for g in GRADES for n in range(1, 6)]
SUB_GRADE_MAP = {sg: i for i, sg in enumerate(SUB_GRADES)}

HOME_OPTIONS = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
VERIF_OPTIONS = ['Not Verified', 'Source Verified', 'Verified']

EAD_LOOKUP = {36: 0.5804732715289598, 60: 0.7348155329774531}
AVG_LGD_PORTFOLIO = 0.9221  # portfolio mean LGD prediction, from notebook 07 validation

RISK_LEVELS = [
    (0.10, 'Low',       '#27ae60'),
    (0.20, 'Medium',    '#f39c12'),
    (0.35, 'High',      '#e67e22'),
    (1.00, 'Very High', '#e74c3c'),
]

# ECOA/Reg B-style plain-language descriptions for the top risk-increasing
# features, used to generate adverse-action reason codes from SHAP values.
ADVERSE_ACTION_REASONS = {
    'loan_amnt': lambda v: f"Loan amount requested (${v:,.0f}) is high relative to typical approved loans",
    'term': lambda v: f"Loan term ({int(v)} months) extends repayment risk",
    'emp_length': lambda v: "Length of employment is relatively short",
    'annual_inc': lambda v: f"Annual income (${v:,.0f}) is low relative to loan size",
    'dti': lambda v: f"Debt-to-income ratio ({v:.1f}%) is elevated",
    'inq_last_6mths': lambda v: f"Number of recent credit inquiries ({int(v)}) is high",
    'open_acc': lambda v: "Number of open credit accounts is high",
    'revol_bal': lambda v: f"Revolving credit balance (${v:,.0f}) is high",
    'revol_util': lambda v: f"Revolving credit utilization ({v:.0f}%) is high",
    'total_acc': lambda v: "Total number of credit accounts is high",
    'initial_list_status': lambda v: "Initial listing status is associated with higher risk",
    'tot_coll_amt': lambda v: "Amount sent to collections is elevated",
    'total_rev_hi_lim': lambda v: "Total revolving credit limit is low relative to usage",
    'acc_open_past_24mths': lambda v: f"Accounts opened in the past 24 months ({int(v)}) is high",
    'avg_cur_bal': lambda v: "Average current balance across accounts is high",
    'bc_open_to_buy': lambda v: "Available bankcard credit (open-to-buy) is low",
    'bc_util': lambda v: f"Bankcard utilization ({v:.0f}%) is high",
    'mo_sin_old_il_acct': lambda v: "Oldest installment account is relatively young",
    'mo_sin_old_rev_tl_op': lambda v: "Oldest revolving account is relatively young",
    'mo_sin_rcnt_rev_tl_op': lambda v: "Most recently opened revolving account is very recent",
    'mo_sin_rcnt_tl': lambda v: "Most recently opened credit account is very recent",
    'mort_acc': lambda v: "Number of mortgage accounts is low",
    'mths_since_recent_bc': lambda v: "A bankcard account was opened very recently",
    'mths_since_recent_inq': lambda v: "A credit inquiry occurred very recently",
    'num_accts_ever_120_pd': lambda v: "History of accounts 120+ days past due",
    'num_actv_bc_tl': lambda v: "Number of active bankcard accounts is high",
    'num_bc_sats': lambda v: "Number of satisfactory bankcard accounts is low",
    'num_bc_tl': lambda v: "Total number of bankcard accounts is high",
    'num_il_tl': lambda v: "Number of installment accounts is high",
    'num_op_rev_tl': lambda v: "Number of open revolving accounts is high",
    'num_rev_accts': lambda v: "Total number of revolving accounts is high",
    'num_rev_tl_bal_gt_0': lambda v: "Number of revolving accounts carrying a balance is high",
    'num_tl_op_past_12m': lambda v: f"Accounts opened in the past 12 months ({int(v)}) is high",
    'pct_tl_nvr_dlq': lambda v: f"Percent of accounts never delinquent ({v:.0f}%) is low",
    'percent_bc_gt_75': lambda v: f"Percent of bankcards over 75% utilized ({v:.0f}%) is high",
    'tot_hi_cred_lim': lambda v: "Total high credit limit is low relative to obligations",
    'total_bal_ex_mort': lambda v: "Total balance excluding mortgage is high",
    'total_bc_limit': lambda v: "Total bankcard credit limit is low",
    'total_il_high_credit_limit': lambda v: "Total installment credit limit is low",
    'credit_history_months': lambda v: f"Length of credit history ({v/12:.1f} years) is short",
    'fico_score': lambda v: f"FICO score ({int(v)}) is low",
    'sub_grade_encoded': lambda v: "Internal risk sub-grade is low",
    'home_RENT': lambda v: "Renting rather than owning a home",
    'verif_Verified': lambda v: "Income verification status",
    'purpose_credit_card': lambda v: "Loan purpose is credit card refinancing",
    'purpose_debt_consolidation': lambda v: "Loan purpose is debt consolidation",
    'state_default_rate': lambda v: "Applicant's state has a higher historical default rate",
    'loan_to_income': lambda v: f"Loan amount relative to income ({v:.2f}) is high",
    'revol_bal_to_income': lambda v: "Revolving balance relative to income is high",
    'available_credit': lambda v: "Available (unused) credit is low",
    'available_credit_pct': lambda v: "Percent of credit limit still available is low",
    'active_ratio': lambda v: "Proportion of open-to-total accounts is high",
    'delinq_per_account': lambda v: "Delinquencies relative to number of accounts is elevated",
    'pub_rec_per_account': lambda v: "Public records relative to number of accounts is elevated",
    'installment_pct_of_loan': lambda v: "Monthly installment relative to loan amount is high",
    'accounts_per_year': lambda v: "Rate of new account opening relative to credit history is high",
}

NON_DISCLOSABLE_FEATURES = {'issue_year', 'issue_month'}

ALL_NUMERIC_FIELDS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
    'annual_inc', 'dti', 'fico_score', 'open_acc', 'total_acc',
    'revol_bal', 'revol_util', 'inq_last_6mths', 'delinq_2yrs',
    'pub_rec', 'mort_acc', 'mths_since_last_delinq',
    'credit_history_months', 'issue_year', 'issue_month',
    'initial_list_status', 'tot_coll_amt', 'tot_cur_bal',
    'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal',
    'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths',
    'collections_12_mths_ex_med', 'acc_now_delinq', 'delinq_amnt',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
    'mths_since_recent_bc', 'mths_since_recent_inq',
    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
    'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
    'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
    'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
    'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
]


# ── Artifact Loading (no framework caching — callers own that) ────────────────

def load_models(base=BASE):
    base = Path(base)
    pd_model = joblib.load(base / 'models' / 'pd_model.joblib')
    pd_features = pd.read_csv(base / 'models' / 'pd_feature_names.csv')['feature'].tolist()
    with open(base / 'models' / 'pd_metadata.json') as f:
        pd_meta = json.load(f)

    with open(base / 'models' / 'ead_model.json') as f:
        ead_artifact = json.load(f)
    with open(base / 'models' / 'ead_metadata.json') as f:
        ead_meta = json.load(f)

    lgd_model = joblib.load(base / 'models' / 'lgd_model.joblib')
    lgd_features = pd.read_csv(base / 'models' / 'lgd_feature_names.csv')['feature'].tolist()
    with open(base / 'models' / 'lgd_metadata.json') as f:
        lgd_meta = json.load(f)

    with open(base / 'models' / 'expected_loss_summary.json') as f:
        el_summary = json.load(f)

    return {
        'pd_model': pd_model, 'pd_features': pd_features, 'pd_meta': pd_meta,
        'ead_lookup': {int(k): float(v) for k, v in ead_artifact['lookup'].items()},
        'ead_meta': ead_meta,
        'lgd_model': lgd_model, 'lgd_features': lgd_features, 'lgd_meta': lgd_meta,
        'el_summary': el_summary,
    }


def load_config(base=BASE):
    return joblib.load(Path(base) / 'models' / 'app_config.joblib')


def load_portfolio(base=BASE):
    path = Path(base) / 'data' / 'expected_loss_test_2016_2017.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def fit_calibration_model(portfolio):
    """Platt scaling fit on the FULL 2016-2017 labeled portfolio (not just the
    2016 subset used for the out-of-time validation shown in the app). For
    live/production scoring we want the strongest fit from all available
    labeled data; separate out-of-time evaluation (fit on 2016, tested on
    held-out 2017) is what validates that this approach generalizes."""
    if portfolio is None or 'pd_pred' not in portfolio.columns:
        return None
    eps = 1e-6
    raw = np.clip(portfolio['pd_pred'].values, eps, 1 - eps)
    logit = np.log(raw / (1 - raw))
    platt = LogisticRegression()
    platt.fit(logit.reshape(-1, 1), portfolio['actual_default'].values)
    return platt


def load_calibration_model(base=BASE):
    return fit_calibration_model(load_portfolio(base))


def calibrate_pd(raw_pd, platt):
    if platt is None:
        return np.asarray(raw_pd, dtype=float)
    eps = 1e-6
    raw = np.clip(np.asarray(raw_pd, dtype=float), eps, 1 - eps)
    logit = np.log(raw / (1 - raw))
    return platt.predict_proba(logit.reshape(-1, 1))[:, 1]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def compute_installment(loan_amnt, term, int_rate):
    r = int_rate / 100 / 12
    if r == 0:
        return loan_amnt / term
    return loan_amnt * r / (1 - (1 + r) ** (-term))


def get_risk_level(el_rate):
    for threshold, label, color in RISK_LEVELS:
        if el_rate <= threshold:
            return label, color
    return 'Very High', '#e74c3c'


def preprocess_single(inputs, config, pd_features, lgd_features):
    medians = config['medians']
    row = {}

    for field in ALL_NUMERIC_FIELDS:
        if field in inputs and inputs[field] is not None:
            row[field] = float(inputs[field])
        elif field in medians:
            row[field] = float(medians[field])
        else:
            row[field] = 0.0

    winsor = config['winsor_upper']
    for col, upper in winsor.items():
        if col in row:
            row[col] = min(row[col], upper)
    row['revol_util'] = min(row.get('revol_util', 0), 100)
    row['dti'] = max(row.get('dti', 0), 0)

    row['sub_grade_encoded'] = SUB_GRADE_MAP.get(inputs.get('sub_grade', 'C3'), 12)

    home = inputs.get('home_ownership', 'RENT')
    for opt in ['MORTGAGE', 'OTHER', 'OWN', 'RENT']:
        row[f'home_{opt}'] = 1.0 if home == opt else 0.0

    verif = inputs.get('verification_status', 'Not Verified')
    row['verif_Not Verified'] = 1.0 if verif == 'Not Verified' else 0.0
    row['verif_Source Verified'] = 1.0 if verif == 'Source Verified' else 0.0
    row['verif_Verified'] = 1.0 if verif == 'Verified' else 0.0

    purpose = inputs.get('purpose', 'debt_consolidation')
    for p in config['purposes']:
        row[f'purpose_{p}'] = 1.0 if purpose == p else 0.0

    state = inputs.get('addr_state', 'CA')
    row['state_default_rate'] = config['state_default_map'].get(
        state, config['global_default_rate'])
    row['state_lgd_rate'] = config['state_lgd_map'].get(
        state, config['global_lgd_rate'])

    ai = row['annual_inc']
    row['loan_to_income'] = row['loan_amnt'] / (ai + 1)
    row['installment_to_income'] = row['installment'] / (ai / 12 + 1)
    row['revol_bal_to_income'] = row['revol_bal'] / (ai + 1)
    trhl = row.get('total_rev_hi_lim', medians.get('total_rev_hi_lim', 1))
    row['available_credit'] = trhl - row['revol_bal']
    row['available_credit_pct'] = row['available_credit'] / (trhl + 1)
    row['active_ratio'] = row['open_acc'] / (row['total_acc'] + 1)
    row['delinq_per_account'] = row.get('delinq_2yrs', 0) / (row['total_acc'] + 1)
    row['pub_rec_per_account'] = row.get('pub_rec', 0) / (row['total_acc'] + 1)
    row['installment_pct_of_loan'] = row['installment'] / (row['loan_amnt'] + 1)
    row['accounts_per_year'] = row['total_acc'] / (row['credit_history_months'] / 12 + 1)

    for col, upper in config['ratio_upper'].items():
        if col in row:
            row[col] = min(row[col], upper)

    for col in row:
        if isinstance(row[col], float) and (np.isinf(row[col]) or np.isnan(row[col])):
            row[col] = config['ratio_medians'].get(col, medians.get(col, 0))

    df = pd.DataFrame([row])
    pd_X = df.reindex(columns=pd_features, fill_value=0)
    lgd_X = df.reindex(columns=lgd_features, fill_value=0)
    ead_pct = EAD_LOOKUP.get(int(row['term']), 0.5805)

    return pd_X, lgd_X, ead_pct


def preprocess_batch(df_raw, config, pd_features, lgd_features):
    df = df_raw.copy()
    medians = config['medians']

    if 'term' in df.columns and df['term'].dtype == object:
        df['term'] = df['term'].str.strip().str.replace(' months', '', regex=False).astype(int)

    if 'emp_length' in df.columns and df['emp_length'].dtype == object:
        emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                   '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                   '8 years': 8, '9 years': 9, '10+ years': 10}
        df['emp_length'] = df['emp_length'].map(emp_map)

    if 'home_ownership' in df.columns:
        df['home_ownership'] = df['home_ownership'].replace({'NONE': 'OTHER', 'ANY': 'OTHER'})

    if 'earliest_cr_line' in df.columns and 'credit_history_months' not in df.columns:
        ecl = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
        now = datetime.now()
        df['credit_history_months'] = (now.year - ecl.dt.year) * 12 + (now.month - ecl.dt.month)

    for field in ALL_NUMERIC_FIELDS:
        if field not in df.columns:
            df[field] = medians.get(field, 0)
        elif df[field].isnull().any():
            df[field] = df[field].fillna(medians.get(field, 0))

    if 'issue_year' not in df_raw.columns:
        df['issue_year'] = datetime.now().year
    if 'issue_month' not in df_raw.columns:
        df['issue_month'] = datetime.now().month

    if 'installment' not in df_raw.columns and all(c in df.columns for c in ['loan_amnt', 'term', 'int_rate']):
        df['installment'] = df.apply(
            lambda r: compute_installment(r['loan_amnt'], r['term'], r['int_rate']), axis=1)

    winsor = config['winsor_upper']
    for col, upper in winsor.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=upper)
    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].clip(upper=100)
    if 'dti' in df.columns:
        df['dti'] = df['dti'].clip(lower=0)

    df['sub_grade_encoded'] = df.get('sub_grade', pd.Series(['C3'] * len(df))).map(SUB_GRADE_MAP).fillna(12)

    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    elif 'fico_score' not in df.columns:
        df['fico_score'] = medians.get('fico_score', 700)

    for col, prefix in [('home_ownership', 'home'), ('verification_status', 'verif'), ('purpose', 'purpose')]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefix).astype(float)
            df = pd.concat([df, dummies], axis=1)

    state_col = 'addr_state' if 'addr_state' in df.columns else None
    if state_col:
        df['state_default_rate'] = df[state_col].map(
            config['state_default_map']).fillna(config['global_default_rate'])
        df['state_lgd_rate'] = df[state_col].map(
            config['state_lgd_map']).fillna(config['global_lgd_rate'])
    else:
        df['state_default_rate'] = config['global_default_rate']
        df['state_lgd_rate'] = config['global_lgd_rate']

    df['initial_list_status'] = df.get('initial_list_status', pd.Series([0] * len(df)))
    if df['initial_list_status'].dtype == object:
        df['initial_list_status'] = (df['initial_list_status'] == 'f').astype(int)

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
    ratio_cols = list(config['ratio_upper'].keys())
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].fillna(config['ratio_medians'].get(col, 0))
            df[col] = df[col].clip(upper=config['ratio_upper'][col])

    pd_X = df.reindex(columns=pd_features, fill_value=0)
    lgd_X = df.reindex(columns=lgd_features, fill_value=0)
    ead_pct = df['term'].map(EAD_LOOKUP).fillna(0.5805).values

    return pd_X, lgd_X, ead_pct


# ── Scoring ────────────────────────────────────────────────────────────────────

def score(pd_X, lgd_X, ead_pct, loan_amnt, models):
    pd_pred = models['pd_model'].predict_proba(pd_X)[:, 1]
    lgd_pred = np.clip(models['lgd_model'].predict(lgd_X), 0, 1)
    if isinstance(ead_pct, (int, float)):
        ead_pred = np.array([ead_pct])
    else:
        ead_pred = np.asarray(ead_pct)
    if isinstance(loan_amnt, (int, float)):
        loan_amnt = np.array([loan_amnt])
    else:
        loan_amnt = np.asarray(loan_amnt)
    el = pd_pred * ead_pred * lgd_pred * loan_amnt
    return pd_pred, ead_pred, lgd_pred, el


def get_adverse_action_reasons(shap_values, feature_names, feature_values, top_n=4):
    risk_increasing = [(f, s, feature_values.get(f)) for f, s in zip(feature_names, shap_values)
                        if s > 0 and f not in NON_DISCLOSABLE_FEATURES]
    risk_increasing.sort(key=lambda x: -x[1])
    reasons = []
    for feat, shap_val, val in risk_increasing[:top_n]:
        template = ADVERSE_ACTION_REASONS.get(feat)
        if template is not None:
            try:
                text = template(val)
            except Exception:
                text = feat.replace('_', ' ').title()
        else:
            text = feat.replace('_', ' ').title()
        reasons.append((text, float(shap_val)))
    return reasons


def score_single_loan(inputs, models, config, platt=None):
    """End-to-end: raw loan inputs -> (pd_raw, pd_calibrated, ead, lgd, expected_loss).
    Convenience wrapper used by the API; the Streamlit app calls the lower-level
    functions directly since it needs the intermediate pd_X/lgd_X for SHAP."""
    pd_X, lgd_X, ead_pct = preprocess_single(inputs, config, models['pd_features'], models['lgd_features'])
    pd_pred, ead_pred, lgd_pred, _ = score(pd_X, lgd_X, ead_pct, inputs['loan_amnt'], models)
    pd_raw = float(pd_pred[0])
    pd_cal = float(calibrate_pd([pd_raw], platt)[0]) if platt is not None else pd_raw
    ead_val = float(ead_pred[0])
    lgd_val = float(lgd_pred[0])
    el = pd_cal * ead_val * lgd_val * inputs['loan_amnt']
    return {
        'pd_raw': pd_raw, 'pd_calibrated': pd_cal, 'ead': ead_val, 'lgd': lgd_val,
        'expected_loss': el, 'el_rate': el / inputs['loan_amnt'],
    }
