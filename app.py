import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import shap

import credit_core as cc
from credit_core import (
    GRADES, SUB_GRADES, SUB_GRADE_MAP, HOME_OPTIONS, VERIF_OPTIONS,
    EAD_LOOKUP, AVG_LGD_PORTFOLIO, RISK_LEVELS, ADVERSE_ACTION_REASONS,
    NON_DISCLOSABLE_FEATURES, ALL_NUMERIC_FIELDS,
    compute_installment, get_risk_level, preprocess_single, preprocess_batch,
    score, calibrate_pd, get_adverse_action_reasons,
)

BASE = Path(__file__).parent

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        color: white; text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; font-weight: 400; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; font-weight: 700; }

    .card-blue  { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .card-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .card-amber { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .card-red   { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .card-teal  { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .card-dark  { background: linear-gradient(135deg, #434343 0%, #000000 100%); }

    .risk-badge {
        display: inline-block; padding: 0.4rem 1.2rem;
        border-radius: 20px; font-weight: 700; font-size: 1rem;
        letter-spacing: 0.5px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stRadio label { color: #e0e0e0 !important; }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span { color: #e0e0e0 !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0;
    }

    div.stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    return cc.load_models(BASE)


@st.cache_resource
def load_config():
    return cc.load_config(BASE)


@st.cache_data
def load_portfolio():
    return cc.load_portfolio(BASE)


@st.cache_resource
def load_shap_explainers(_pd_model, _lgd_model):
    pd_explainer = shap.TreeExplainer(_pd_model)
    lgd_explainer = shap.TreeExplainer(_lgd_model)
    return pd_explainer, lgd_explainer


@st.cache_resource
def load_shap_sample():
    path = BASE / 'models' / 'shap_sample.joblib'
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def load_psi_reference():
    path = BASE / 'models' / 'psi_reference.joblib'
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def load_baseline_comparison():
    path = BASE / 'models' / 'baseline_comparison.joblib'
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def get_calibration_model():
    return cc.fit_calibration_model(load_portfolio())


@st.cache_resource
def load_fairlending_sensitivity():
    path = BASE / 'models' / 'fairlending_state_sensitivity.joblib'
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_test_fe():
    path = BASE / 'data' / 'test_fe.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Preprocessing, scoring, and calibration live in credit_core.py ────────────
# (preprocess_single, preprocess_batch, score, calibrate_pd, compute_installment,
# get_risk_level — imported at the top of this file so the API and this app
# never drift apart.)

# ── Gauge Chart ────────────────────────────────────────────────────────────────

def plot_shap_waterfall(feature_names, shap_values, base_value, top_n=10,
                          title="SHAP Explanation", y_label="Contribution"):
    order = np.argsort(-np.abs(shap_values))
    top_idx = order[:top_n]
    rest_idx = order[top_n:]
    rest_sum = float(shap_values[rest_idx].sum()) if len(rest_idx) > 0 else 0.0

    labels = ['Base rate'] + [feature_names[i] for i in top_idx]
    values = [float(base_value)] + [float(shap_values[i]) for i in top_idx]
    if len(rest_idx) > 0:
        labels.append(f'{len(rest_idx)} other features')
        values.append(rest_sum)

    measures = ['absolute'] + ['relative'] * (len(values) - 1)

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=measures,
        x=labels,
        y=values,
        connector={'line': {'color': 'rgba(150,150,150,0.4)'}},
        increasing={'marker': {'color': '#e74c3c'}},
        decreasing={'marker': {'color': '#27ae60'}},
        totals={'marker': {'color': '#667eea'}},
    ))
    fig.update_layout(title=title, height=420, showlegend=False, yaxis_title=y_label,
                       margin=dict(l=20, r=20, t=50, b=120))
    fig.update_xaxes(tickangle=45)
    return fig


def make_gauge(value, title, max_val=1.0, suffix='%', color_ranges=None):
    display_val = value * 100 if suffix == '%' else value
    if color_ranges is None:
        color_ranges = [
            (0, max_val * 0.25 * 100, '#27ae60'),
            (max_val * 0.25 * 100, max_val * 0.5 * 100, '#f39c12'),
            (max_val * 0.5 * 100, max_val * 0.75 * 100, '#e67e22'),
            (max_val * 0.75 * 100, max_val * 100, '#e74c3c'),
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': suffix, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, max_val * 100], 'tickwidth': 1},
            'bar': {'color': '#2c3e50', 'thickness': 0.3},
            'steps': [{'range': [lo, hi], 'color': c} for lo, hi, c in color_ranges],
            'threshold': {
                'line': {'color': '#2c3e50', 'width': 3},
                'thickness': 0.8,
                'value': display_val,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_dashboard(models, config):
    st.title("Credit Risk Dashboard")
    st.markdown("Combined **PD x EAD x LGD** pipeline for Lending Club consumer loans.")

    el_summary = models['el_summary']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card card-blue">
            <h3>Total Exposure</h3><h1>${el_summary['total_exposure']/1e9:.2f}B</h1>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card card-amber">
            <h3>Predicted EL</h3><h1>${el_summary['total_expected_loss']/1e9:.2f}B</h1>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card card-green">
            <h3>EL Rate</h3><h1>{el_summary['el_rate']*100:.1f}%</h1>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card card-red">
            <h3>Realized Loss Rate</h3><h1>{el_summary['realized_loss_rate']*100:.1f}%</h1>
        </div>""", unsafe_allow_html=True)

    st.caption(
        "Figures above are the raw model outputs as originally validated (predicted EL "
        "overshoots realized loss by ~57%). A validated recalibration fix — see Portfolio "
        "Analytics for the corrected portfolio total, or Model Information → PD Model for "
        "the underlying calibration fix — cuts most of that gap without retraining."
    )

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Model Pipeline")
        st.markdown(f"""
| Component | Model | Key Metric |
|-----------|-------|------------|
| **PD** | XGBoost (1,967 trees) | AUC = {models['pd_meta']['test_auc']:.4f} |
| **EAD** | Term-based lookup | 36m: {EAD_LOOKUP[36]:.1%}, 60m: {EAD_LOOKUP[60]:.1%} |
| **LGD** | LightGBM (180 trees) | MAE = {models['lgd_meta']['test_mae']:.4f} |
""")

    with col_r:
        st.subheader("Portfolio Summary")
        fig = go.Figure(go.Bar(
            x=['Predicted EL', 'Realized Loss'],
            y=[el_summary['total_expected_loss'], el_summary['total_realized_loss']],
            marker_color=['#667eea', '#f5576c'],
            text=[f"${el_summary['total_expected_loss']/1e9:.2f}B",
                  f"${el_summary['total_realized_loss']/1e9:.2f}B"],
            textposition='outside',
        ))
        fig.update_layout(
            height=300, yaxis_title='Dollars ($)',
            margin=dict(l=40, r=20, t=20, b=40),
            yaxis=dict(tickformat='$,.0f'),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
**Expected Loss = PD x EAD% x LGD x Loan Amount**

1. **Probability of Default (PD)** — XGBoost classifier predicts the likelihood a borrower
   will default, based on 58 features including credit score, DTI, income, and credit history.
2. **Exposure at Default (EAD)** — Term-based lookup estimates what fraction of the loan
   balance is outstanding at the time of default (58% for 36-month, 73% for 60-month loans).
3. **Loss Given Default (LGD)** — LightGBM regressor predicts the fraction of exposed
   balance not recovered after default, using 90 features.
4. The three components are multiplied by loan amount to produce the dollar expected loss.
""")


def status_badge(label, color):
    return f'<span class="risk-badge" style="background:{color}; color:white; font-size:0.8rem;">{label}</span>'


def page_validation_summary(models):
    st.title("Model Validation Summary")
    st.markdown(
        "One page pulling together every validation check run on this model — discrimination, "
        "calibration, stability, monotonicity, stress resilience, and fair lending. This is what "
        "a model risk management sign-off would look at before approving the model for use."
    )

    st.markdown("---")
    st.markdown(f"""
<div style="padding:1.5rem; background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
     border-radius:16px; color:white;">
<h3 style="margin-top:0; color:white;">Executive Verdict</h3>
<p style="color:white;">Suitable as a <b>portfolio-level risk-ranking tool</b> — it separates
higher- and lower-risk loans meaningfully better than simple heuristics (FICO alone, or
LendingClub's own sub-grade). After the validated recalibration fix, its <b>aggregate</b>
Expected Loss estimate is reasonably close to realized loss. It is <b>not</b> validated for
individual credit-decisioning (the training data only contains already-approved loans — see
"Known Limitations" below), and the geographic sensitivity check below is <b>not</b> a
substitute for genuine fair-lending testing, which this dataset cannot support.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Validation Checks")

    rows_html = []

    # --- Discrimination ---
    auc = models['pd_meta']['test_auc']
    baseline = load_baseline_comparison()
    best_baseline_auc = None
    if baseline is not None:
        others = {k: v['auc'] for k, v in baseline['results'].items() if 'XGBoost' not in k}
        best_baseline_auc = max(others.values()) if others else None
    disc_detail = f"AUC {auc:.4f}"
    if best_baseline_auc:
        disc_detail += f" vs. best simple baseline {best_baseline_auc:.4f} (+{auc-best_baseline_auc:.4f})"
    rows_html.append(("Discrimination", disc_detail, "Good", "#27ae60", "Model Information → PD Model"))

    # --- Calibration ---
    portfolio = load_portfolio()
    if portfolio is not None and 'issue_year' in portfolio.columns:
        fit_set = portfolio[portfolio['issue_year'] == 2016]
        eval_set = portfolio[portfolio['issue_year'] == 2017]
        eps = 1e-6
        fit_logit = np.log(np.clip(fit_set['pd_pred'], eps, 1-eps) / (1 - np.clip(fit_set['pd_pred'], eps, 1-eps)))
        platt_tmp = LogisticRegression()
        platt_tmp.fit(fit_logit.values.reshape(-1, 1), fit_set['actual_default'])
        eval_logit = np.log(np.clip(eval_set['pd_pred'], eps, 1-eps) / (1 - np.clip(eval_set['pd_pred'], eps, 1-eps)))
        recal = platt_tmp.predict_proba(eval_logit.values.reshape(-1, 1))[:, 1]

        def quick_ece(pred, actual, n_bins=10):
            d = pd.DataFrame({'pred': pred, 'actual': actual})
            d['decile'] = pd.qcut(d['pred'], n_bins, labels=False, duplicates='drop')
            g = d.groupby('decile').agg(mp=('pred', 'mean'), ar=('actual', 'mean'), n=('pred', 'size'))
            return float((np.abs(g['mp']-g['ar'])*g['n']).sum()/g['n'].sum())

        ece_before = quick_ece(eval_set['pd_pred'].values, eval_set['actual_default'].values)
        ece_after = quick_ece(recal, eval_set['actual_default'].values)
        rows_html.append(("Calibration", f"ECE {ece_before:.4f} → {ece_after:.4f} after Platt scaling (out-of-time)",
                           "Fixed", "#27ae60", "Model Information → PD Model → Recalibration"))
    else:
        rows_html.append(("Calibration", "Portfolio data unavailable", "Unknown", "#999999", ""))

    # --- Stability (PSI) ---
    psi_ref = load_psi_reference()
    if psi_ref is not None:
        score_psi = psi_ref['results']['__PD_SCORE__']['psi']
        max_feat = max((f for f in psi_ref['monitored_features']),
                       key=lambda f: psi_ref['results'][f]['psi'])
        max_psi = psi_ref['results'][max_feat]['psi']
        label, color = psi_badge(max_psi)
        rows_html.append(("Population Stability", f"Score PSI {score_psi:.4f} (stable); "
                           f"worst feature `{max_feat}` PSI {max_psi:.4f} ({label})",
                           "Mixed" if max_psi >= 0.10 else "Stable",
                           "#f39c12" if max_psi >= 0.10 else "#27ae60", "Model Stability (PSI)"))
    else:
        rows_html.append(("Population Stability", "PSI data unavailable", "Unknown", "#999999", ""))

    # --- Monotonicity ---
    shap_sample = load_shap_sample()
    if shap_sample is not None:
        pd_explainer, _ = load_shap_explainers(models['pd_model'], models['lgd_model'])
        pd_X_sample = shap_sample['pd_X_sample']
        shap_matrix = np.asarray(pd_explainer.shap_values(pd_X_sample))
        expected_direction = {
            'fico_score': 'decreasing', 'dti': 'increasing', 'revol_util': 'increasing',
            'credit_history_months': 'decreasing', 'inq_last_6mths': 'increasing',
            'sub_grade_encoded': 'increasing', 'annual_inc': 'decreasing', 'loan_amnt': 'increasing',
        }
        n_reversed, n_ok = 0, 0
        for feat, expected in expected_direction.items():
            if feat not in pd_X_sample.columns:
                continue
            fidx = models['pd_features'].index(feat)
            d = pd.DataFrame({'val': pd_X_sample[feat].values, 'shap': shap_matrix[:, fidx]})
            try:
                d['bin'] = pd.qcut(d['val'], 8, duplicates='drop')
            except ValueError:
                continue
            binned = d.groupby('bin', observed=True)['shap'].mean().values
            if len(binned) < 3:
                continue
            diffs = np.diff(binned)
            net = 'increasing' if diffs.sum() >= 0 else 'decreasing'
            if net == expected:
                n_ok += 1
            else:
                n_reversed += 1
        status = "Minor Issue" if n_reversed > 0 else "Good"
        color = "#f39c12" if n_reversed > 0 else "#27ae60"
        rows_html.append(("Monotonicity", f"{n_ok}/{n_ok+n_reversed} features directionally sound "
                           f"({n_reversed} reversed)", status, color, "Model Stability (PSI)"))
    else:
        rows_html.append(("Monotonicity", "SHAP sample unavailable", "Unknown", "#999999", ""))

    # --- Lift ---
    if portfolio is not None:
        gains_df = portfolio[['pd_pred', 'actual_default']].sort_values('pd_pred', ascending=False)
        n = len(gains_df)
        top10_idx = int(n * 0.10) - 1
        pct_captured = gains_df['actual_default'].cumsum().iloc[top10_idx] / gains_df['actual_default'].sum()
        rows_html.append(("Lift", f"Top 10% riskiest loans capture {pct_captured:.1%} of actual "
                           f"defaults ({pct_captured/0.10:.2f}x lift)", "Moderate", "#f39c12",
                           "Model Information → PD Model"))

    # --- Stress Resilience ---
    rows_html.append(("Stress Resilience", "Interactive scenario tool available — "
                       "portfolio EL sensitivity to income/DTI/utilization shocks",
                       "Available", "#667eea", "Stress Testing"))

    # --- Fair Lending ---
    fl = load_fairlending_sensitivity()
    if fl is not None:
        rows_html.append(("Fair Lending", f"Not a true disparate impact test (no protected-class/BISG "
                           f"data). Geographic sensitivity: mean |PD shift| {fl['overall_mean_abs_diff']:.4f} "
                           f"from state signal alone", "Caveat", "#e74c3c", "Model Stability (PSI)"))
    else:
        rows_html.append(("Fair Lending", "Fair lending data unavailable", "Unknown", "#999999", ""))

    for name, detail, status, color, page_ref in rows_html:
        c1, c2, c3 = st.columns([1.3, 3.5, 1.2])
        with c1:
            st.markdown(f"**{name}**")
        with c2:
            st.markdown(detail)
        with c3:
            st.markdown(status_badge(status, color), unsafe_allow_html=True)
        if page_ref:
            st.caption(f"→ {page_ref}")
        st.markdown("---")

    st.subheader("Known Limitations")
    st.markdown("""
- **Reject inference bias** — this dataset only contains loans LendingClub actually funded. The
  model has never seen a rejected applicant, so it can't fully generalize to a real underwriting
  decision at the point of application — only to ranking risk among already-approved loans.
- **EAD has no borrower-level signal** — it's a 2-value lookup by loan term (all regression
  attempts had negative R²), not a predictive model.
- **LGD has near-zero R²** (0.0095) — recovery outcomes are driven by the collections process,
  not observable at origination.
- **Single train/test split** — no k-fold or repeated out-of-time validation; all metrics here
  reflect one specific 2007-2015 / 2016-2017 split.
- **Recalibration is scoped to live app scoring** — Single Loan, Batch Scoring, Portfolio
  Analytics, and the Threshold tool all use the corrected PD; the raw `pd_pred` column in the
  saved portfolio CSV and the notebook 07 summary JSON are left as originally validated, for
  traceability back to the original analysis.
- **Fair lending check is a proxy, not a real test** — geographic sensitivity is the closest
  feasible substitute without protected-class or BISG data; it should not be read as evidence
  either for or against disparate impact.
""")


def page_single_loan(models, config):
    st.title("Single Loan Assessment")
    st.markdown("Enter loan and borrower details to estimate credit risk.")

    with st.form("loan_form"):
        st.subheader("Loan Details")
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 15000, step=500)
        with lc2:
            term = st.selectbox("Term (months)", [36, 60], index=0)
        with lc3:
            sub_grade = st.selectbox("Sub-Grade", SUB_GRADES, index=12)
        with lc4:
            default_rate = config['sub_grade_int_rates'].get(sub_grade, 13.0)
            int_rate = st.number_input("Interest Rate (%)", 5.0, 31.0, round(default_rate, 2), step=0.1)

        lc5, lc6 = st.columns(2)
        with lc5:
            purpose = st.selectbox("Purpose", config['purposes'], index=config['purposes'].index('debt_consolidation') if 'debt_consolidation' in config['purposes'] else 0)
        with lc6:
            installment = compute_installment(loan_amnt, term, int_rate)
            st.metric("Monthly Installment (auto)", f"${installment:,.2f}")

        st.subheader("Borrower Profile")
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            annual_inc = st.number_input("Annual Income ($)", 0, 500000, 65000, step=1000)
        with bc2:
            emp_length = st.selectbox("Employment (years)", list(range(0, 11)), index=5,
                                      format_func=lambda x: '< 1' if x == 0 else ('10+' if x == 10 else str(x)))
        with bc3:
            home_ownership = st.selectbox("Home Ownership", HOME_OPTIONS, index=0)
        with bc4:
            addr_state = st.selectbox("State", config['states'],
                                       index=config['states'].index('CA') if 'CA' in config['states'] else 0)

        bc5, bc6 = st.columns(2)
        with bc5:
            verification_status = st.selectbox("Verification Status", VERIF_OPTIONS, index=0)
        with bc6:
            credit_years = st.number_input("Credit History (years)", 0, 50, 15, step=1)

        st.subheader("Credit Profile")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            fico_score = st.number_input("FICO Score", 300, 850, 700, step=5)
        with cc2:
            dti = st.number_input("DTI Ratio", 0.0, 100.0, 18.0, step=0.5)
        with cc3:
            open_acc = st.number_input("Open Accounts", 0, 80, 11, step=1)
        with cc4:
            total_acc = st.number_input("Total Accounts", 0, 150, 25, step=1)

        cc5, cc6, cc7, cc8 = st.columns(4)
        with cc5:
            revol_bal = st.number_input("Revolving Balance ($)", 0, 300000, 15000, step=500)
        with cc6:
            revol_util = st.number_input("Revolving Utilization (%)", 0.0, 100.0, 55.0, step=1.0)
        with cc7:
            inq_last_6mths = st.number_input("Inquiries (6 mo)", 0, 20, 1, step=1)
        with cc8:
            delinq_2yrs = st.number_input("Delinquencies (2 yr)", 0, 30, 0, step=1)

        with st.expander("Advanced Credit Bureau Fields"):
            st.caption("Leave at defaults (training medians) if unknown.")
            m = config['medians']
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                pub_rec = st.number_input("Public Records", 0, 20, int(m.get('pub_rec', 0)))
                mort_acc = st.number_input("Mortgage Accounts", 0, 20, int(m.get('mort_acc', 1)))
                mths_since_last_delinq = st.number_input("Mths Since Last Delinq", 0, 200,
                                                          int(m.get('mths_since_last_delinq', 50)))
            with ac2:
                tot_coll_amt = st.number_input("Total Collection Amt ($)", 0, 100000,
                                                int(m.get('tot_coll_amt', 0)))
                tot_cur_bal = st.number_input("Total Current Balance ($)", 0, 1000000,
                                              int(m.get('tot_cur_bal', 150000)))
                total_rev_hi_lim = st.number_input("Total Revolving Limit ($)", 0, 500000,
                                                    int(m.get('total_rev_hi_lim', 30000)))
            with ac3:
                avg_cur_bal = st.number_input("Avg Current Balance ($)", 0, 500000,
                                              int(m.get('avg_cur_bal', 13000)))
                bc_open_to_buy = st.number_input("BC Open to Buy ($)", 0, 500000,
                                                  int(m.get('bc_open_to_buy', 7000)))
                bc_util = st.number_input("BC Utilization (%)", 0.0, 100.0,
                                          round(float(m.get('bc_util', 60)), 1))
            with ac4:
                pct_tl_nvr_dlq = st.number_input("% Trades Never Delinq", 0.0, 100.0,
                                                   round(float(m.get('pct_tl_nvr_dlq', 94)), 1))
                pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 10,
                                                        int(m.get('pub_rec_bankruptcies', 0)))
                tax_liens = st.number_input("Tax Liens", 0, 10,
                                             int(m.get('tax_liens', 0)))

        submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True,
                                           type="primary")

    if submitted:
        inputs = {
            'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
            'installment': installment, 'sub_grade': sub_grade,
            'emp_length': emp_length, 'home_ownership': home_ownership,
            'verification_status': verification_status, 'purpose': purpose,
            'addr_state': addr_state, 'annual_inc': annual_inc, 'dti': dti,
            'fico_score': fico_score, 'open_acc': open_acc, 'total_acc': total_acc,
            'revol_bal': revol_bal, 'revol_util': revol_util,
            'inq_last_6mths': inq_last_6mths, 'delinq_2yrs': delinq_2yrs,
            'pub_rec': pub_rec, 'mort_acc': mort_acc,
            'mths_since_last_delinq': mths_since_last_delinq,
            'credit_history_months': credit_years * 12,
            'issue_year': datetime.now().year,
            'issue_month': datetime.now().month,
            'tot_coll_amt': tot_coll_amt, 'tot_cur_bal': tot_cur_bal,
            'total_rev_hi_lim': total_rev_hi_lim, 'avg_cur_bal': avg_cur_bal,
            'bc_open_to_buy': bc_open_to_buy, 'bc_util': bc_util,
            'pct_tl_nvr_dlq': pct_tl_nvr_dlq,
            'pub_rec_bankruptcies': pub_rec_bankruptcies, 'tax_liens': tax_liens,
        }

        pd_X, lgd_X, ead_pct = preprocess_single(
            inputs, config, models['pd_features'], models['lgd_features'])
        pd_pred, ead_pred, lgd_pred, el = score(
            pd_X, lgd_X, ead_pct, loan_amnt, models)

        pd_val_raw = float(pd_pred[0])
        platt = get_calibration_model()
        pd_val = float(calibrate_pd([pd_val_raw], platt)[0])
        ead_val = float(ead_pred[0])
        lgd_val = float(lgd_pred[0])
        el_val = pd_val * ead_val * lgd_val * loan_amnt
        el_rate = el_val / loan_amnt

        risk_label, risk_color = get_risk_level(el_rate)

        st.markdown("---")
        st.subheader("Assessment Results")
        st.caption(
            "Expected Loss and the PD gauge use the **recalibrated** PD (see Model Information → "
            "PD Model → Recalibration for validation: fitting on 2016 and testing on held-out 2017 "
            "cut Expected Calibration Error from 0.2135 to 0.0061). The raw XGBoost score is shown "
            "alongside for reference."
        )

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown(f"""
<div style="text-align:center; padding:2rem; background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
     border-radius:16px; color:white;">
    <h3 style="margin:0; opacity:0.8; color:white;">Expected Loss</h3>
    <h1 style="margin:0.5rem 0; font-size:2.5rem; color:white;">${el_val:,.0f}</h1>
    <p style="margin:0.5rem 0; font-size:1.1rem; color:white;">EL Rate: {el_rate:.2%}</p>
    <span class="risk-badge" style="background:{risk_color}; color:white;">{risk_label} Risk</span>
</div>""", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**Component Breakdown**")
            breakdown = pd.DataFrame({
                'Component': ['PD (calibrated)', 'PD (raw model)', 'EAD', 'LGD'],
                'Value': [f'{pd_val:.2%}', f'{pd_val_raw:.2%}', f'{ead_val:.2%}', f'{lgd_val:.2%}'],
                'Dollar Impact': [
                    f'${pd_val * loan_amnt:,.0f}',
                    f'${pd_val_raw * loan_amnt:,.0f}',
                    f'${ead_val * loan_amnt:,.0f}',
                    f'${lgd_val * loan_amnt:,.0f}',
                ],
            })
            st.dataframe(breakdown, hide_index=True, use_container_width=True)

        with rc2:
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                st.plotly_chart(make_gauge(pd_val, "Probability of Default (calibrated)"), use_container_width=True)
            with gc2:
                st.plotly_chart(make_gauge(ead_val, "Exposure at Default"), use_container_width=True)
            with gc3:
                st.plotly_chart(make_gauge(lgd_val, "Loss Given Default"), use_container_width=True)

        st.markdown("---")
        st.subheader("Why This Score? (SHAP Explanation)")

        pd_explainer, lgd_explainer = load_shap_explainers(models['pd_model'], models['lgd_model'])
        pd_shap = np.asarray(pd_explainer.shap_values(pd_X))[0]
        pd_base = float(np.ravel(pd_explainer.expected_value)[0])
        lgd_shap = np.asarray(lgd_explainer.shap_values(lgd_X))[0]
        lgd_base = float(np.ravel(lgd_explainer.expected_value)[0])

        sc1, sc2 = st.columns(2)
        with sc1:
            fig_shap_pd = plot_shap_waterfall(
                models['pd_features'], pd_shap, pd_base, top_n=8,
                title="PD Drivers (log-odds scale)", y_label="Log-odds contribution")
            st.plotly_chart(fig_shap_pd, use_container_width=True)
            pd_margin_final = pd_base + pd_shap.sum()
            st.caption(
                f"Base log-odds {pd_base:.3f} + all feature contributions = {pd_margin_final:.3f} "
                f"→ sigmoid → **{1/(1+np.exp(-pd_margin_final)):.2%} raw model PD** (matches the "
                f"'PD (raw model)' row above, not the calibrated gauge — SHAP explains the "
                f"XGBoost output directly; the calibration step afterward is a fixed rescaling, "
                f"not a feature-based effect, so it isn't part of this breakdown). "
                f"SHAP values for a classifier are additive in log-odds space, not probability — "
                f"a feature's bar height isn't directly 'percentage points of PD'."
            )
        with sc2:
            fig_shap_lgd = plot_shap_waterfall(
                models['lgd_features'], lgd_shap, lgd_base, top_n=8,
                title="LGD Drivers (native scale)", y_label="LGD contribution")
            st.plotly_chart(fig_shap_lgd, use_container_width=True)
            lgd_final = lgd_base + lgd_shap.sum()
            st.caption(
                f"Base LGD {lgd_base:.3f} + all feature contributions = "
                f"**{lgd_final:.3f} predicted LGD** (matches the gauge above; here contributions "
                f"add directly since LGD is a regression target, not a probability)."
            )

        st.markdown("---")
        st.subheader("Adverse Action Reasons (ECOA / Reg B Style)")
        st.caption(
            "If this application were declined, the Equal Credit Opportunity Act requires "
            "disclosing specific reasons. These are the top risk-increasing factors from the "
            "PD SHAP breakdown above (positive log-odds contributions only), converted to "
            "plain language — the same underlying values, just filtered to the unfavorable "
            "direction and worded for a consumer-facing notice."
        )
        pd_row_values = pd_X.iloc[0].to_dict()
        reasons = get_adverse_action_reasons(pd_shap, models['pd_features'], pd_row_values, top_n=4)
        if reasons:
            for i, (text, shap_val) in enumerate(reasons, 1):
                st.markdown(f"**{i}.** {text}")
        else:
            st.markdown("No individual factor increased risk above the base rate for this loan.")

        st.markdown("---")
        st.subheader("What-If Analysis")
        st.caption("Adjust key parameters to see how expected loss changes. Uses calibrated PD, same as the headline Expected Loss above.")

        wi1, wi2, wi3 = st.columns(3)
        with wi1:
            wi_amounts = np.arange(5000, 41000, 1000)
            wi_els = []
            for amt in wi_amounts:
                inp_copy = inputs.copy()
                inp_copy['loan_amnt'] = amt
                inp_copy['installment'] = compute_installment(amt, term, int_rate)
                px_, lx_, ep_ = preprocess_single(inp_copy, config, models['pd_features'], models['lgd_features'])
                p_, e_, l_, _ = score(px_, lx_, ep_, amt, models)
                p_cal = calibrate_pd(p_, platt)
                wi_els.append(float(p_cal[0] * e_[0] * l_[0] * amt))
            fig_wi = go.Figure(go.Scatter(x=wi_amounts, y=wi_els, mode='lines+markers',
                                           line=dict(color='#667eea', width=2), marker=dict(size=4)))
            fig_wi.add_vline(x=loan_amnt, line_dash="dash", line_color="#e74c3c",
                              annotation_text="Current")
            fig_wi.update_layout(title="EL vs Loan Amount", xaxis_title="Loan Amount ($)",
                                  yaxis_title="Expected Loss ($)", height=300,
                                  margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_wi, use_container_width=True)

        with wi2:
            wi_ficos = np.arange(600, 851, 10)
            wi_els2 = []
            for f in wi_ficos:
                inp_copy = inputs.copy()
                inp_copy['fico_score'] = f
                px_, lx_, ep_ = preprocess_single(inp_copy, config, models['pd_features'], models['lgd_features'])
                p_, e_, l_, _ = score(px_, lx_, ep_, loan_amnt, models)
                p_cal = calibrate_pd(p_, platt)
                wi_els2.append(float(p_cal[0] * e_[0] * l_[0] * loan_amnt))
            fig_wi2 = go.Figure(go.Scatter(x=wi_ficos, y=wi_els2, mode='lines+markers',
                                            line=dict(color='#11998e', width=2), marker=dict(size=4)))
            fig_wi2.add_vline(x=fico_score, line_dash="dash", line_color="#e74c3c",
                               annotation_text="Current")
            fig_wi2.update_layout(title="EL vs FICO Score", xaxis_title="FICO Score",
                                   yaxis_title="Expected Loss ($)", height=300,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_wi2, use_container_width=True)

        with wi3:
            wi_dtis = np.arange(0, 51, 2)
            wi_els3 = []
            for d in wi_dtis:
                inp_copy = inputs.copy()
                inp_copy['dti'] = d
                px_, lx_, ep_ = preprocess_single(inp_copy, config, models['pd_features'], models['lgd_features'])
                p_, e_, l_, _ = score(px_, lx_, ep_, loan_amnt, models)
                p_cal = calibrate_pd(p_, platt)
                wi_els3.append(float(p_cal[0] * e_[0] * l_[0] * loan_amnt))
            fig_wi3 = go.Figure(go.Scatter(x=wi_dtis, y=wi_els3, mode='lines+markers',
                                            line=dict(color='#f093fb', width=2), marker=dict(size=4)))
            fig_wi3.add_vline(x=dti, line_dash="dash", line_color="#e74c3c",
                               annotation_text="Current")
            fig_wi3.update_layout(title="EL vs DTI", xaxis_title="DTI Ratio",
                                   yaxis_title="Expected Loss ($)", height=300,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_wi3, use_container_width=True)

        portfolio = load_portfolio()
        if portfolio is not None:
            st.markdown("---")
            st.subheader("Portfolio Comparison")
            st.caption("Portfolio averages below are also calibrated, for a fair comparison against your loan's calibrated numbers.")
            port_pd_calibrated = calibrate_pd(portfolio['pd_pred'].values, platt)
            port_el_calibrated = port_pd_calibrated * portfolio['ead_pred'].values * portfolio['lgd_pred'].values * portfolio['loan_amnt'].values
            pc1, pc2, pc3, pc4 = st.columns(4)
            port_avg_pd = port_pd_calibrated.mean()
            port_avg_lgd = portfolio['lgd_pred'].mean()
            port_avg_el_rate = (port_el_calibrated / portfolio['loan_amnt'].values).mean()
            with pc1:
                delta = pd_val - port_avg_pd
                st.metric("Your PD", f"{pd_val:.2%}", f"{delta:+.2%} vs avg", delta_color="inverse")
            with pc2:
                delta2 = lgd_val - port_avg_lgd
                st.metric("Your LGD", f"{lgd_val:.2%}", f"{delta2:+.2%} vs avg", delta_color="inverse")
            with pc3:
                delta3 = el_rate - port_avg_el_rate
                st.metric("Your EL Rate", f"{el_rate:.2%}", f"{delta3:+.2%} vs avg", delta_color="inverse")
            with pc4:
                pct_rank = (port_el_calibrated < el_val).mean() * 100
                st.metric("Percentile", f"{pct_rank:.0f}th",
                          "Higher risk than most" if pct_rank > 50 else "Lower risk than most")


def page_batch(models, config):
    st.title("Batch Scoring")
    st.markdown("Upload a CSV of loan applications to score them in bulk.")

    with st.expander("Required CSV Columns", expanded=False):
        st.markdown("""
**Minimum required columns:** `loan_amnt`, `term`, `sub_grade`, `purpose`,
`home_ownership`, `verification_status`, `addr_state`, `annual_inc`, `dti`, `fico_score`

**Optional columns (filled with training medians if missing):**
`int_rate`, `installment`, `emp_length`, `open_acc`, `total_acc`,
`revol_bal`, `revol_util`, `inq_last_6mths`, `delinq_2yrs`, `pub_rec`, `mort_acc`,
`mths_since_last_delinq`, `credit_history_months`, and other credit bureau fields.
""")

    sample_data = pd.DataFrame({
        'loan_amnt': [15000, 25000, 8000],
        'term': [36, 60, 36],
        'sub_grade': ['B3', 'D2', 'A4'],
        'purpose': ['debt_consolidation', 'credit_card', 'home_improvement'],
        'home_ownership': ['RENT', 'MORTGAGE', 'OWN'],
        'verification_status': ['Not Verified', 'Verified', 'Source Verified'],
        'addr_state': ['CA', 'NY', 'TX'],
        'annual_inc': [65000, 85000, 45000],
        'dti': [18.5, 22.0, 12.5],
        'fico_score': [700, 680, 740],
        'int_rate': [12.5, 18.0, 8.5],
        'emp_length': [5, 8, 2],
        'open_acc': [11, 15, 7],
        'total_acc': [25, 32, 14],
        'revol_bal': [15000, 22000, 5000],
        'revol_util': [55.0, 72.0, 30.0],
    })

    st.download_button(
        "Download Sample Template",
        sample_data.to_csv(index=False),
        "loan_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload Loan Data (CSV)", type=['csv'])

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded:** {len(df_raw)} loans")
        st.dataframe(df_raw.head(10), use_container_width=True)

        required = ['loan_amnt', 'term', 'sub_grade']
        missing_cols = [c for c in required if c not in df_raw.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return

        if st.button("Score All Loans", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(df_raw)} loans..."):
                pd_X, lgd_X, ead_pct = preprocess_batch(
                    df_raw, config, models['pd_features'], models['lgd_features'])
                pd_pred, ead_pred, lgd_pred, _ = score(
                    pd_X, lgd_X, ead_pct, df_raw['loan_amnt'].values, models)
                platt = get_calibration_model()
                pd_pred_calibrated = calibrate_pd(pd_pred, platt)
                el = pd_pred_calibrated * ead_pred * lgd_pred * df_raw['loan_amnt'].values

            results = df_raw.copy()
            results['pd_score_raw'] = pd_pred
            results['pd_score'] = pd_pred_calibrated
            results['ead_score'] = ead_pred
            results['lgd_score'] = lgd_pred
            results['expected_loss'] = el
            results['el_rate'] = el / df_raw['loan_amnt'].values
            results['risk_level'] = results['el_rate'].apply(lambda x: get_risk_level(x)[0])

            st.success(f"Scored {len(results)} loans!")
            st.caption("`pd_score` is the calibrated PD (used for Expected Loss and risk level); `pd_score_raw` is the uncalibrated XGBoost output.")

            st.markdown("---")
            st.subheader("Summary Statistics")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Total Exposure", f"${results['loan_amnt'].sum():,.0f}")
            with sc2:
                st.metric("Total Expected Loss", f"${results['expected_loss'].sum():,.0f}")
            with sc3:
                st.metric("Avg EL Rate", f"{results['el_rate'].mean():.2%}")
            with sc4:
                st.metric("Avg PD", f"{results['pd_score'].mean():.2%}")

            risk_dist = results['risk_level'].value_counts()
            rc1, rc2 = st.columns(2)
            with rc1:
                fig = px.pie(values=risk_dist.values, names=risk_dist.index,
                             title="Risk Distribution",
                             color=risk_dist.index,
                             color_discrete_map={'Low': '#27ae60', 'Medium': '#f39c12',
                                                  'High': '#e67e22', 'Very High': '#e74c3c'})
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            with rc2:
                fig2 = px.histogram(results, x='expected_loss', nbins=50,
                                     title="Expected Loss Distribution",
                                     color_discrete_sequence=['#667eea'])
                fig2.update_layout(height=350, xaxis_title="Expected Loss ($)",
                                    yaxis_title="Count")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.subheader("Scored Results")
            display_cols = ['loan_amnt', 'term', 'sub_grade', 'purpose',
                           'pd_score', 'pd_score_raw', 'ead_score', 'lgd_score',
                           'expected_loss', 'el_rate', 'risk_level']
            display_cols = [c for c in display_cols if c in results.columns]
            st.dataframe(
                results[display_cols].style.format({
                    'pd_score': '{:.2%}', 'pd_score_raw': '{:.2%}', 'ead_score': '{:.2%}', 'lgd_score': '{:.2%}',
                    'expected_loss': '${:,.0f}', 'el_rate': '{:.2%}',
                    'loan_amnt': '${:,.0f}',
                }),
                use_container_width=True, height=400,
            )

            csv = results.to_csv(index=False)
            st.download_button("Download Scored Results", csv,
                               "scored_loans.csv", "text/csv")


def estimate_loan_revenue(loan_amnt, term, int_rate):
    r = int_rate / 100 / 12
    installment = np.where(r == 0, loan_amnt / term, loan_amnt * r / (1 - (1 + r) ** (-term)))
    return np.clip(installment * term - loan_amnt, 0, None)


@st.cache_data
def compute_threshold_sweep(pd_pred, actual_default, realized_loss, est_revenue,
                              thresholds=np.arange(0.02, 1.0, 0.02)):
    rows = []
    default = actual_default == 1
    for t in thresholds:
        reject = pd_pred >= t
        tp = reject & default
        fn = (~reject) & default
        fp = reject & (~default)
        tn = (~reject) & (~default)
        tp_n, fn_n, fp_n, tn_n = tp.sum(), fn.sum(), fp.sum(), tn.sum()
        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
        recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        net_profit = est_revenue[tn].sum() - realized_loss[fn].sum()
        rows.append({'threshold': t, 'precision': precision, 'recall': recall,
                     'f1': f1, 'net_profit': net_profit})
    return pd.DataFrame(rows)


def compute_threshold_metrics(pd_pred, actual_default, realized_loss, est_revenue, threshold):
    reject = pd_pred >= threshold
    default = actual_default == 1

    tp = reject & default
    fn = (~reject) & default
    fp = reject & (~default)
    tn = (~reject) & (~default)

    tp_n, fn_n, fp_n, tn_n = int(tp.sum()), int(fn.sum()), int(fp.sum()), int(tn.sum())
    tp_val = float(realized_loss[tp].sum())
    fn_val = float(realized_loss[fn].sum())
    fp_val = float(est_revenue[fp].sum())
    tn_val = float(est_revenue[tn].sum())

    precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp_n + tn_n) / len(pd_pred)
    net_profit = tn_val - fn_val

    return dict(tp=tp_n, fn=fn_n, fp=fp_n, tn=tn_n,
                tp_val=tp_val, fn_val=fn_val, fp_val=fp_val, tn_val=tn_val,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                net_profit=net_profit)


def page_portfolio(models, config):
    st.title("Portfolio Analytics")

    portfolio = load_portfolio()
    if portfolio is None:
        st.warning("Portfolio data not found. Run notebook 07 first.")
        return

    grade_labels = {i: sg for i, sg in enumerate(SUB_GRADES)}
    if 'sub_grade' not in portfolio.columns and 'grade_sub' in portfolio.columns:
        portfolio['sub_grade'] = portfolio['grade_sub'].map(grade_labels)
    portfolio['el_rate'] = portfolio['expected_loss'] / portfolio['loan_amnt']
    portfolio['grade'] = portfolio.get('sub_grade', pd.Series(['?'] * len(portfolio))).str[0]

    platt = get_calibration_model()
    portfolio['pd_calibrated'] = calibrate_pd(portfolio['pd_pred'].values, platt)
    portfolio['expected_loss_calibrated'] = (
        portfolio['pd_calibrated'] * portfolio['ead_pred'] * portfolio['lgd_pred'] * portfolio['loan_amnt'])
    portfolio['el_rate_calibrated'] = portfolio['expected_loss_calibrated'] / portfolio['loan_amnt']

    st.subheader("Portfolio Overview")
    st.caption("Figures below are the raw model outputs as originally validated in notebook 07 — see the recalibration comparison right below for the corrected version.")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.metric("Loans", f"{len(portfolio):,}")
    with mc2:
        st.metric("Total Exposure", f"${portfolio['loan_amnt'].sum()/1e9:.2f}B")
    with mc3:
        st.metric("Avg PD", f"{portfolio['pd_pred'].mean():.2%}")
    with mc4:
        st.metric("Avg LGD", f"{portfolio['lgd_pred'].mean():.2%}")
    with mc5:
        st.metric("Avg EL Rate", f"{portfolio['el_rate'].mean():.2%}")

    st.markdown("---")
    st.subheader("Recalibrated vs. Raw: Closing the EL Overshoot")
    st.markdown(
        "Earlier validation found predicted EL overshoots realized loss by ~57% "
        "(156.74% of realized). Applying the Platt-scaling fix validated on Model "
        "Information (out-of-time: fit on 2016, tested on 2017) to the whole portfolio:"
    )
    total_realized = portfolio['realized_loss'].sum() if 'realized_loss' in portfolio.columns else None
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        raw_el = portfolio['expected_loss'].sum()
        cal_el = portfolio['expected_loss_calibrated'].sum()
        st.metric("Total Predicted EL", f"${cal_el/1e9:.2f}B",
                  f"{(cal_el-raw_el)/1e9:+.2f}B vs raw (${raw_el/1e9:.2f}B)", delta_color="inverse")
    with rc2:
        raw_rate = portfolio['el_rate'].mean()
        cal_rate = portfolio['el_rate_calibrated'].mean()
        st.metric("EL Rate", f"{cal_rate:.2%}",
                  f"{cal_rate-raw_rate:+.2%} vs raw ({raw_rate:.2%})", delta_color="inverse")
    with rc3:
        if total_realized:
            raw_pct = raw_el / total_realized * 100
            cal_pct = cal_el / total_realized * 100
            st.metric("% of Realized Loss Captured", f"{cal_pct:.1f}%",
                      f"{cal_pct-raw_pct:+.1f}pp vs raw ({raw_pct:.1f}%)", delta_color="inverse")
    st.caption(
        "Recalibration corrects the average level of PD (and therefore EL) but doesn't "
        "change loan-to-loan ranking — a systematic bias fix, not a better model. All "
        "charts and tables below use the raw `pd_pred`/`expected_loss` columns unless "
        "labeled '(calibrated)', for direct comparability with earlier validation."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distributions", "Risk Segments", "Component Analysis", "Correlations",
        "Threshold & Profitability"
    ])

    with tab1:
        dc1, dc2 = st.columns(2)
        with dc1:
            fig = px.histogram(portfolio, x='pd_pred', nbins=80,
                                title="PD Distribution",
                                color_discrete_sequence=['#667eea'])
            fig.add_vline(x=portfolio['pd_pred'].mean(), line_dash="dash",
                           annotation_text=f"Mean: {portfolio['pd_pred'].mean():.2%}")
            fig.update_layout(height=350, xaxis_title="PD", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        with dc2:
            fig2 = px.histogram(portfolio, x='lgd_pred', nbins=80,
                                 title="LGD Distribution",
                                 color_discrete_sequence=['#f093fb'])
            fig2.add_vline(x=portfolio['lgd_pred'].mean(), line_dash="dash",
                            annotation_text=f"Mean: {portfolio['lgd_pred'].mean():.2%}")
            fig2.update_layout(height=350, xaxis_title="LGD", yaxis_title="Count")
            st.plotly_chart(fig2, use_container_width=True)

        dc3, dc4 = st.columns(2)
        with dc3:
            fig3 = px.histogram(portfolio, x='expected_loss', nbins=80,
                                 title="Expected Loss Distribution ($)",
                                 color_discrete_sequence=['#11998e'])
            fig3.add_vline(x=portfolio['expected_loss'].mean(), line_dash="dash",
                            annotation_text=f"Mean: ${portfolio['expected_loss'].mean():,.0f}")
            fig3.update_layout(height=350, xaxis_title="Expected Loss ($)", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)
        with dc4:
            fig4 = px.histogram(portfolio, x='el_rate', nbins=80,
                                 title="EL Rate Distribution",
                                 color_discrete_sequence=['#f5576c'])
            fig4.update_layout(height=350, xaxis_title="EL Rate", yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        if 'sub_grade' in portfolio.columns:
            seg = portfolio.groupby('sub_grade').agg(
                count=('loan_amnt', 'count'),
                total_exposure=('loan_amnt', 'sum'),
                avg_pd=('pd_pred', 'mean'),
                avg_lgd=('lgd_pred', 'mean'),
                total_el=('expected_loss', 'sum'),
                total_realized=('realized_loss', 'sum'),
            ).reindex(SUB_GRADES).dropna()
            seg['el_rate'] = seg['total_el'] / seg['total_exposure']
            seg['realized_rate'] = seg['total_realized'] / seg['total_exposure']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=seg.index, y=seg['el_rate'], name='Predicted EL Rate',
                                  marker_color='#667eea', opacity=0.8), secondary_y=False)
            fig.add_trace(go.Bar(x=seg.index, y=seg['realized_rate'], name='Realized Loss Rate',
                                  marker_color='#f5576c', opacity=0.8), secondary_y=False)
            fig.add_trace(go.Scatter(x=seg.index, y=seg['count'], name='Loan Count',
                                      mode='lines+markers', line=dict(color='#2ecc71', width=2)),
                           secondary_y=True)
            fig.update_layout(title="Risk Segmentation by Sub-Grade", height=450,
                               barmode='group', xaxis_title="Sub-Grade",
                               xaxis=dict(tickangle=45))
            fig.update_yaxes(title_text="Loss Rate", secondary_y=False)
            fig.update_yaxes(title_text="Loan Count", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                seg.style.format({
                    'count': '{:,.0f}', 'total_exposure': '${:,.0f}',
                    'avg_pd': '{:.2%}', 'avg_lgd': '{:.2%}',
                    'total_el': '${:,.0f}', 'total_realized': '${:,.0f}',
                    'el_rate': '{:.2%}', 'realized_rate': '{:.2%}',
                }),
                use_container_width=True, height=400,
            )

        if 'grade' in portfolio.columns:
            grade_seg = portfolio.groupby('grade').agg(
                count=('loan_amnt', 'count'),
                avg_pd=('pd_pred', 'mean'),
                avg_el_rate=('el_rate', 'mean'),
            ).reindex(GRADES).dropna()

            fig_grade = px.bar(grade_seg.reset_index(), x='grade', y='avg_el_rate',
                                color='avg_pd', title="Average EL Rate by Grade",
                                color_continuous_scale='RdYlGn_r',
                                text=grade_seg['avg_el_rate'].apply(lambda x: f'{x:.1%}'))
            fig_grade.update_layout(height=350, xaxis_title="Grade", yaxis_title="Avg EL Rate",
                                     yaxis_tickformat='.0%')
            st.plotly_chart(fig_grade, use_container_width=True)

    with tab3:
        cc1, cc2 = st.columns(2)
        with cc1:
            fig_ead = px.box(portfolio, x='term', y='ead_pred',
                              title="EAD by Term",
                              color_discrete_sequence=['#667eea'])
            fig_ead.update_layout(height=350, xaxis_title="Term (months)",
                                   yaxis_title="EAD %")
            st.plotly_chart(fig_ead, use_container_width=True)
        with cc2:
            fig_scatter = px.scatter(portfolio.sample(min(5000, len(portfolio)), random_state=42),
                                      x='pd_pred', y='lgd_pred', color='el_rate',
                                      title="PD vs LGD (5K sample)",
                                      color_continuous_scale='Turbo', opacity=0.4,
                                      size_max=5)
            fig_scatter.update_layout(height=350, xaxis_title="PD", yaxis_title="LGD")
            st.plotly_chart(fig_scatter, use_container_width=True)

        fig_comp = make_subplots(rows=1, cols=3,
                                  subplot_titles=["PD x Loan Amt", "EAD x Term", "LGD x Loan Amt"])
        sample = portfolio.sample(min(5000, len(portfolio)), random_state=42)
        fig_comp.add_trace(go.Scatter(x=sample['loan_amnt'], y=sample['pd_pred'],
                                       mode='markers', marker=dict(size=3, color='#667eea', opacity=0.3),
                                       showlegend=False), row=1, col=1)
        fig_comp.add_trace(go.Box(x=sample['term'], y=sample['ead_pred'],
                                   marker_color='#11998e', showlegend=False), row=1, col=2)
        fig_comp.add_trace(go.Scatter(x=sample['loan_amnt'], y=sample['lgd_pred'],
                                       mode='markers', marker=dict(size=3, color='#f093fb', opacity=0.3),
                                       showlegend=False), row=1, col=3)
        fig_comp.update_layout(height=350)
        st.plotly_chart(fig_comp, use_container_width=True)

    with tab4:
        corr_cols = ['pd_pred', 'ead_pred', 'lgd_pred', 'expected_loss', 'loan_amnt', 'term']
        corr_cols = [c for c in corr_cols if c in portfolio.columns]
        corr = portfolio[corr_cols].corr()
        fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                              title="Correlation Matrix", aspect='auto',
                              zmin=-1, zmax=1)
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab5:
        st.markdown("""
Simulates an **accept/reject decision** at a PD cutoff: loans with **calibrated PD** `>= threshold`
are rejected, everything else is approved. Uses calibrated rather than raw PD here because a
threshold that decides real accept/reject outcomes should use the best available probability
estimate — see Model Information → PD Model → Recalibration for the validation
(ECE 0.2135 → 0.0061, out-of-time). Dollar values for rejected/approved loans that
actually defaulted use the **realized loss** already known for this test population.
Dollar values for loans that did **not** default are an *estimate* — this saved dataset
doesn't include each loan's actual interest rate, so revenue is approximated using the
**median interest rate for that loan's sub-grade**, assuming full-term amortization with
no prepayment. Treat revenue/profit figures as directional, not exact.
""")

        if 'sub_grade' not in portfolio.columns or portfolio['sub_grade'].isnull().all():
            st.warning("Sub-grade not available in this dataset — cannot estimate revenue.")
        else:
            sub_grade_rates = config.get('sub_grade_int_rates', {})
            fallback_rate = float(np.median(list(sub_grade_rates.values()))) if sub_grade_rates else 13.0
            est_rate = portfolio['sub_grade'].map(sub_grade_rates).fillna(fallback_rate).values
            est_revenue = estimate_loan_revenue(
                portfolio['loan_amnt'].values, portfolio['term'].values, est_rate)

            pd_arr = portfolio['pd_calibrated'].values
            default_arr = portfolio['actual_default'].values
            loss_arr = portfolio['realized_loss'].values

            model_auc = models['pd_meta']['test_auc']
            youden_threshold_raw = models['pd_meta'].get('optimal_threshold', 0.5)
            # Calibration is a monotonic transform, so it preserves rank order and the
            # underlying TPR/FPR operating point -- just recompute what that same
            # operating point's PD *value* becomes after calibration.
            youden_threshold = float(calibrate_pd([youden_threshold_raw], platt)[0])

            threshold = st.slider(
                "Calibrated PD Rejection Threshold", min_value=0.01, max_value=0.99,
                value=float(round(youden_threshold, 2)), step=0.01,
                help="Loans with calibrated PD at or above this value are 'rejected'. "
                     f"Default is the model's Youden's-J cutoff ({youden_threshold_raw:.4f} raw) "
                     f"mapped through calibration ({youden_threshold:.4f}).",
            )

            m = compute_threshold_metrics(pd_arr, default_arr, loss_arr, est_revenue, threshold)

            st.markdown("**Classification Metrics at this Threshold**")
            cm1, cm2, cm3, cm4, cm5 = st.columns(5)
            with cm1:
                st.metric("Accuracy", f"{m['accuracy']:.2%}")
            with cm2:
                st.metric("Precision", f"{m['precision']:.2%}")
            with cm3:
                st.metric("Recall", f"{m['recall']:.2%}")
            with cm4:
                st.metric("F1 Score", f"{m['f1']:.3f}")
            with cm5:
                st.metric("Model AUC", f"{model_auc:.4f}", help="AUC summarizes the whole ROC curve — it does not change with the threshold slider.")

            st.markdown("**Business Impact at this Threshold**")
            bm1, bm2, bm3, bm4, bm5 = st.columns(5)
            with bm1:
                st.markdown(f"""<div class="metric-card card-green">
                    <h3>Default Saved (TP)</h3><h1>{m['tp']:,}</h1>
                    <p style="margin:0.3rem 0 0 0; font-size:0.9rem;">${m['tp_val']:,.0f} loss avoided</p>
                </div>""", unsafe_allow_html=True)
            with bm2:
                st.markdown(f"""<div class="metric-card card-red">
                    <h3>Value Lost (FN)</h3><h1>{m['fn']:,}</h1>
                    <p style="margin:0.3rem 0 0 0; font-size:0.9rem;">${m['fn_val']:,.0f} realized loss</p>
                </div>""", unsafe_allow_html=True)
            with bm3:
                st.markdown(f"""<div class="metric-card card-amber">
                    <h3>Opportunity Lost (FP)</h3><h1>{m['fp']:,}</h1>
                    <p style="margin:0.3rem 0 0 0; font-size:0.9rem;">${m['fp_val']:,.0f} foregone revenue</p>
                </div>""", unsafe_allow_html=True)
            with bm4:
                st.markdown(f"""<div class="metric-card card-blue">
                    <h3>Revenue Generated (TN)</h3><h1>{m['tn']:,}</h1>
                    <p style="margin:0.3rem 0 0 0; font-size:0.9rem;">${m['tn_val']:,.0f} interest collected</p>
                </div>""", unsafe_allow_html=True)
            with bm5:
                profit_color = 'card-teal' if m['net_profit'] >= 0 else 'card-dark'
                st.markdown(f"""<div class="metric-card {profit_color}">
                    <h3>Net Profit</h3><h1>${m['net_profit']/1e6:.1f}M</h1>
                    <p style="margin:0.3rem 0 0 0; font-size:0.9rem;">Revenue (TN) − Loss (FN)</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            cc1, cc2 = st.columns(2)
            with cc1:
                cm_matrix = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
                fig_cm = px.imshow(cm_matrix, text_auto=',d', color_continuous_scale='Blues',
                                    labels=dict(x="Predicted", y="Actual"),
                                    x=['Approve', 'Reject'], y=['No Default', 'Default'],
                                    title=f"Confusion Matrix @ threshold = {threshold:.2f}")
                fig_cm.update_layout(height=380)
                st.plotly_chart(fig_cm, use_container_width=True)
            with cc2:
                fpr, tpr, roc_thresh = roc_curve(default_arr, pd_arr)
                idx = np.argmin(np.abs(roc_thresh - threshold))
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                              name=f'ROC (AUC={model_auc:.4f})',
                                              line=dict(color='#667eea', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                              line=dict(color='gray', dash='dash'),
                                              showlegend=False))
                fig_roc.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], mode='markers',
                                              marker=dict(size=14, color='#e74c3c', symbol='x'),
                                              name=f'Current threshold ({threshold:.2f})'))
                fig_roc.update_layout(title="ROC Curve — Current Threshold Marked",
                                       xaxis_title="False Positive Rate",
                                       yaxis_title="True Positive Rate", height=380)
                st.plotly_chart(fig_roc, use_container_width=True)

            st.markdown("---")
            st.markdown("**Metrics Across All Thresholds**")

            sweep = compute_threshold_sweep(pd_arr, default_arr, loss_arr, est_revenue)
            best_row = sweep.loc[sweep['net_profit'].idxmax()]

            sc1, sc2 = st.columns(2)
            with sc1:
                fig_profit = go.Figure()
                fig_profit.add_trace(go.Scatter(x=sweep['threshold'], y=sweep['net_profit'],
                                                 mode='lines', line=dict(color='#11998e', width=2),
                                                 name='Net Profit'))
                fig_profit.add_vline(x=threshold, line_dash="dash", line_color="#e74c3c",
                                      annotation_text="Current")
                fig_profit.add_vline(x=best_row['threshold'], line_dash="dot", line_color="#2ecc71",
                                      annotation_text=f"Max @ {best_row['threshold']:.2f}")
                fig_profit.update_layout(title="Net Profit vs. Threshold",
                                          xaxis_title="Threshold", yaxis_title="Net Profit ($)",
                                          height=380)
                st.plotly_chart(fig_profit, use_container_width=True)
            with sc2:
                fig_prf = go.Figure()
                fig_prf.add_trace(go.Scatter(x=sweep['threshold'], y=sweep['precision'],
                                              mode='lines', name='Precision', line=dict(color='#667eea')))
                fig_prf.add_trace(go.Scatter(x=sweep['threshold'], y=sweep['recall'],
                                              mode='lines', name='Recall', line=dict(color='#f5576c')))
                fig_prf.add_trace(go.Scatter(x=sweep['threshold'], y=sweep['f1'],
                                              mode='lines', name='F1', line=dict(color='#f39c12')))
                fig_prf.add_vline(x=threshold, line_dash="dash", line_color="#2c3e50",
                                   annotation_text="Current")
                fig_prf.update_layout(title="Precision / Recall / F1 vs. Threshold",
                                       xaxis_title="Threshold", yaxis_title="Score", height=380)
                st.plotly_chart(fig_prf, use_container_width=True)

            st.caption(
                f"Profit-maximizing threshold in this sweep: **{best_row['threshold']:.2f}** "
                f"(net profit ${best_row['net_profit']:,.0f}) vs. the model's Youden's-J / KS "
                f"threshold of **{youden_threshold:.4f}** — these optimize different things "
                f"(statistical separation vs. this notebook's specific dollar assumptions) "
                f"and won't generally coincide."
            )


def page_model_info(models):
    st.title("Model Information")

    tab1, tab2, tab3 = st.tabs(["PD Model", "EAD Model", "LGD Model"])

    with tab1:
        meta = models['pd_meta']
        st.subheader("Probability of Default — XGBoost Classifier")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("AUC-ROC", f"{meta['test_auc']:.4f}")
        with mc2:
            st.metric("Gini", f"{meta['gini']:.4f}")
        with mc3:
            st.metric("KS Statistic", f"{meta['ks_statistic']:.4f}")
        with mc4:
            st.metric("Brier Score", f"{meta['brier_score']:.4f}")

        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(f"""
**Architecture**
- Estimators: {meta['n_estimators']:,}
- Features: {meta['n_features']}
- Learning Rate: {meta['best_params']['learning_rate']:.4f}
- Max Depth: {meta['best_params']['max_depth']}
- Min Child Weight: {meta['best_params']['min_child_weight']}
- Subsample: {meta['best_params']['subsample']:.4f}
""")
        with ic2:
            try:
                importances = models['pd_model'].feature_importances_
                feat_imp = pd.DataFrame({
                    'feature': models['pd_features'],
                    'importance': importances,
                }).sort_values('importance', ascending=True).tail(20)
                fig = px.bar(feat_imp, x='importance', y='feature', orientation='h',
                              title="Top 20 Features (PD)", color_discrete_sequence=['#667eea'])
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Feature importances unavailable.")

        st.markdown("---")
        st.markdown("**SHAP Feature Importance** — mean |SHAP value| across a 520-loan sample "
                     "from the test vintage (log-odds scale). Unlike the split/gain-based chart "
                     "above, this reflects each feature's actual average impact on individual "
                     "predictions, not just how often the tree splits on it.")
        shap_sample = load_shap_sample()
        if shap_sample is not None:
            pd_explainer, _ = load_shap_explainers(models['pd_model'], models['lgd_model'])
            pd_shap_matrix = np.asarray(pd_explainer.shap_values(shap_sample['pd_X_sample']))
            mean_abs_shap = np.abs(pd_shap_matrix).mean(axis=0)
            shap_imp = pd.DataFrame({
                'feature': models['pd_features'],
                'mean_abs_shap': mean_abs_shap,
            }).sort_values('mean_abs_shap', ascending=True).tail(20)
            fig_shap = px.bar(shap_imp, x='mean_abs_shap', y='feature', orientation='h',
                               title="SHAP Mean |Impact| (PD, log-odds)",
                               color_discrete_sequence=['#11998e'])
            fig_shap.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP sample not found — run precompute_shap_sample.py first.")

        st.markdown("---")
        st.markdown("**Calibration Curve** — for loans grouped into deciles by predicted PD, "
                     "does the average predicted probability match the actual observed default "
                     "rate? AUC/Gini only measure whether the model *ranks* loans correctly; "
                     "this checks whether the actual numbers can be trusted at face value.")
        portfolio_calib = load_portfolio()
        if portfolio_calib is not None and 'pd_pred' in portfolio_calib.columns:
            calib_df = portfolio_calib[['pd_pred', 'actual_default']].copy()
            calib_df['decile'] = pd.qcut(calib_df['pd_pred'], 10, labels=False, duplicates='drop')
            calib = calib_df.groupby('decile').agg(
                mean_predicted=('pd_pred', 'mean'),
                actual_rate=('actual_default', 'mean'),
                count=('pd_pred', 'size'),
            ).reset_index()
            ece = float((np.abs(calib['mean_predicted'] - calib['actual_rate']) * calib['count']).sum()
                        / calib['count'].sum())

            cc1, cc2 = st.columns([3, 1])
            with cc1:
                fig_calib = go.Figure()
                fig_calib.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines', line=dict(color='gray', dash='dash'),
                    name='Perfect calibration', showlegend=True))
                fig_calib.add_trace(go.Scatter(
                    x=calib['mean_predicted'], y=calib['actual_rate'], mode='lines+markers',
                    marker=dict(size=10, color='#667eea'), line=dict(color='#667eea', width=2),
                    name='Observed (by decile)'))
                fig_calib.update_layout(
                    title="Calibration Curve — Predicted PD vs. Actual Default Rate",
                    xaxis_title="Mean Predicted PD (decile)", yaxis_title="Actual Default Rate",
                    height=420, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig_calib, use_container_width=True)
            with cc2:
                st.metric("Expected Calibration Error", f"{ece:.4f}")
                st.caption(
                    "Weighted average gap between predicted and actual rate across deciles. "
                    "Points below the diagonal = model over-predicts risk in that decile; "
                    "above = under-predicts."
                )
        else:
            st.info("Portfolio data not found — run notebook 07 first.")

        st.markdown("---")
        st.markdown("**Recalibration (Platt Scaling)** — fits a 1-parameter logistic correction "
                     "(`corrected_pd = sigmoid(a * logit(pd_pred) + b)`) on 2016 loans, then "
                     "evaluates it on 2017 loans it never saw — an honest out-of-time test, not "
                     "fitting and grading the fix on the same data.")
        if (portfolio_calib is not None and 'issue_year' in portfolio_calib.columns
                and 'pd_pred' in portfolio_calib.columns):
            fit_set = portfolio_calib[portfolio_calib['issue_year'] == 2016]
            eval_set = portfolio_calib[portfolio_calib['issue_year'] == 2017]

            def to_ece(pred, actual, n_bins=10):
                d = pd.DataFrame({'pred': pred, 'actual': actual})
                d['decile'] = pd.qcut(d['pred'], n_bins, labels=False, duplicates='drop')
                g = d.groupby('decile').agg(mean_pred=('pred', 'mean'),
                                             actual_rate=('actual', 'mean'),
                                             count=('pred', 'size')).reset_index()
                ece_val = float((np.abs(g['mean_pred'] - g['actual_rate']) * g['count']).sum()
                                 / g['count'].sum())
                return ece_val, g

            if len(fit_set) > 100 and len(eval_set) > 100:
                eps = 1e-6
                fit_logit = np.log(np.clip(fit_set['pd_pred'], eps, 1 - eps)
                                    / (1 - np.clip(fit_set['pd_pred'], eps, 1 - eps)))
                platt = LogisticRegression()
                platt.fit(fit_logit.values.reshape(-1, 1), fit_set['actual_default'])

                eval_logit = np.log(np.clip(eval_set['pd_pred'], eps, 1 - eps)
                                     / (1 - np.clip(eval_set['pd_pred'], eps, 1 - eps)))
                pd_recalibrated = platt.predict_proba(eval_logit.values.reshape(-1, 1))[:, 1]

                ece_before, g_before = to_ece(eval_set['pd_pred'].values, eval_set['actual_default'].values)
                ece_after, g_after = to_ece(pd_recalibrated, eval_set['actual_default'].values)

                rc1, rc2 = st.columns([3, 1])
                with rc1:
                    fig_recal = go.Figure()
                    fig_recal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                    line=dict(color='gray', dash='dash'),
                                                    name='Perfect calibration'))
                    fig_recal.add_trace(go.Scatter(
                        x=g_before['mean_pred'], y=g_before['actual_rate'], mode='lines+markers',
                        marker=dict(size=9, color='#e74c3c'), line=dict(color='#e74c3c'),
                        name='Before recalibration'))
                    fig_recal.add_trace(go.Scatter(
                        x=g_after['mean_pred'], y=g_after['actual_rate'], mode='lines+markers',
                        marker=dict(size=9, color='#27ae60'), line=dict(color='#27ae60'),
                        name='After recalibration'))
                    fig_recal.update_layout(
                        title="Calibration Before vs. After Platt Scaling (held-out 2017 loans)",
                        xaxis_title="Mean Predicted PD (decile)", yaxis_title="Actual Default Rate",
                        height=420, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig_recal, use_container_width=True)
                with rc2:
                    st.metric("ECE Before", f"{ece_before:.4f}")
                    st.metric("ECE After", f"{ece_after:.4f}",
                              delta=f"{ece_after - ece_before:+.4f}", delta_color="inverse")
                    st.caption(
                        f"Platt scaling: a={platt.coef_[0][0]:.3f}, b={platt.intercept_[0]:.3f}. "
                        f"This fixes calibration only — it doesn't change AUC/ranking, and it "
                        f"isn't wired into the live scoring on other pages (Single Loan, Batch, "
                        f"Portfolio still show the raw model output)."
                    )
            else:
                st.info("Not enough 2016/2017 loans in the saved portfolio to fit/evaluate a split.")
        else:
            st.info("Portfolio data with issue_year not found.")

        st.markdown("---")
        st.subheader("Baseline / Challenger Comparison")
        st.markdown(
            "Is the 58-feature, 1,967-tree XGBoost model actually earning its complexity? "
            "Compared against three simpler alternatives, all evaluated on the same 2016-2017 "
            "test population."
        )
        baseline = load_baseline_comparison()
        if baseline is not None:
            res = baseline['results']
            comp_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Method'})
            comp_df = comp_df[['Method', 'n_features', 'auc', 'gini', 'ks', 'max_approx_profit']]

            bc1, bc2 = st.columns([2, 1])
            with bc1:
                fig_bc = px.bar(comp_df.sort_values('auc'), x='auc', y='Method', orientation='h',
                                 title="AUC: Full Model vs. Simple Baselines",
                                 color='auc', color_continuous_scale='Blues', text='auc')
                fig_bc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_bc.update_layout(height=320, margin=dict(l=20, r=60, t=40, b=20),
                                      xaxis=dict(range=[0.5, 0.8]))
                st.plotly_chart(fig_bc, use_container_width=True)
            with bc2:
                fig_profit = px.bar(comp_df.sort_values('max_approx_profit'),
                                     x='max_approx_profit', y='Method', orientation='h',
                                     title="Max Approx. Profit", color_discrete_sequence=['#11998e'])
                fig_profit.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20),
                                          xaxis_title="$", yaxis_title="")
                st.plotly_chart(fig_profit, use_container_width=True)

            st.dataframe(
                comp_df.style.format({
                    'auc': '{:.4f}', 'gini': '{:.4f}', 'ks': '{:.4f}',
                    'max_approx_profit': '${:,.0f}',
                }),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                f"Profit figures use a shared assumption of EAD={baseline['avg_ead']:.1%}, "
                f"LGD={baseline['avg_lgd']:.1%} across all four methods (not each loan's actual "
                f"realized EAD/LGD) so only the PD ranking differs between them — this makes the "
                f"relative comparison fair, but absolute dollar figures here are approximate, "
                f"and won't match the exact realized-loss figures elsewhere in this app."
            )
        else:
            st.info("Baseline comparison not found — run precompute_baseline.py first.")

        st.markdown("---")
        st.subheader("Cumulative Gains & Lift")
        st.markdown(
            "If you rank all test loans by predicted PD (raw, uncalibrated — ranking is "
            "unaffected by calibration) and reject the riskiest X%, what fraction of actual "
            "defaults do you catch? More intuitive for a business audience than AUC alone."
        )
        gains_df = portfolio_calib[['pd_pred', 'actual_default']].copy() if portfolio_calib is not None else None
        if gains_df is not None:
            gains_df = gains_df.sort_values('pd_pred', ascending=False).reset_index(drop=True)
            n = len(gains_df)
            total_defaults = gains_df['actual_default'].sum()
            gains_df['cum_defaults'] = gains_df['actual_default'].cumsum()
            gains_df['pct_population'] = (np.arange(1, n + 1)) / n
            gains_df['pct_defaults_captured'] = gains_df['cum_defaults'] / total_defaults

            deciles = np.arange(0.1, 1.01, 0.1)
            decile_rows = []
            for d in deciles:
                idx = int(n * d) - 1
                decile_rows.append({
                    'pct_population': d,
                    'pct_defaults_captured': gains_df['pct_defaults_captured'].iloc[idx],
                })
            decile_df = pd.DataFrame(decile_rows)
            decile_df['lift'] = decile_df['pct_defaults_captured'] / decile_df['pct_population']

            gc1, gc2 = st.columns(2)
            with gc1:
                fig_gains = go.Figure()
                fig_gains.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                line=dict(color='gray', dash='dash'),
                                                name='Random baseline'))
                fig_gains.add_trace(go.Scatter(
                    x=decile_df['pct_population'], y=decile_df['pct_defaults_captured'],
                    mode='lines+markers', marker=dict(size=9, color='#667eea'),
                    line=dict(color='#667eea', width=2), name='XGBoost PD model'))
                fig_gains.update_layout(
                    title="Cumulative Gains Chart", xaxis_title="% of Loans Targeted (highest PD first)",
                    yaxis_title="% of Actual Defaults Captured", height=380,
                    xaxis=dict(range=[0, 1], tickformat='.0%'), yaxis=dict(range=[0, 1], tickformat='.0%'))
                st.plotly_chart(fig_gains, use_container_width=True)
            with gc2:
                fig_lift = go.Figure(go.Bar(
                    x=[f"{int(d*100)}%" for d in decile_df['pct_population']],
                    y=decile_df['lift'], marker_color='#11998e'))
                fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                    annotation_text="No better than random")
                fig_lift.update_layout(
                    title="Lift by Decile", xaxis_title="% of Loans Targeted",
                    yaxis_title="Lift (vs. random)", height=380)
                st.plotly_chart(fig_lift, use_container_width=True)

            top10 = decile_df.iloc[0]
            top20 = decile_df.iloc[1]
            st.caption(
                f"Targeting the riskiest 10% of loans by predicted PD catches "
                f"**{top10['pct_defaults_captured']:.1%}** of actual defaults ({top10['lift']:.2f}x lift); "
                f"the riskiest 20% catches **{top20['pct_defaults_captured']:.1%}** ({top20['lift']:.2f}x lift)."
            )
        else:
            st.info("Portfolio data not found for gains chart.")

    with tab2:
        meta = models['ead_meta']
        st.subheader("Exposure at Default — Term-Based Lookup")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("36-Month EAD", f"{EAD_LOOKUP[36]:.2%}")
        with mc2:
            st.metric("60-Month EAD", f"{EAD_LOOKUP[60]:.2%}")
        with mc3:
            st.metric("Test WAPE", f"{meta.get('test_wape', 0):.4f}")

        st.markdown(f"""
**Why a lookup table?**

All regression models (Linear Regression, Random Forest, XGBoost, LightGBM) produced negative R²
on the EAD prediction task. The fundamental issue is that origination-time features cannot predict
*when* a borrower will default during their loan term — and the timing of default is the primary
driver of how much balance is outstanding (EAD).

The term-based mean is an industry-standard approach for unsecured consumer EAD estimation.
Longer-term loans have higher EAD because defaults tend to occur in the first half of the loan
life, when the outstanding balance is still high.

- **MAE:** {meta.get('test_mae', 0):.4f}
- **RMSE:** {meta.get('test_rmse', 0):.4f}
- **R²:** {meta.get('test_r2', 0):.4f} (negative indicates no better than baseline)
""")

    with tab3:
        meta = models['lgd_meta']
        st.subheader("Loss Given Default — LightGBM Regressor")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("MAE", f"{meta['test_mae']:.4f}")
        with mc2:
            st.metric("RMSE", f"{meta['test_rmse']:.4f}")
        with mc3:
            st.metric("R²", f"{meta['test_r2']:.4f}")
        with mc4:
            st.metric("WAPE", f"{meta.get('test_wape', 0):.4f}")

        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(f"""
**Architecture**
- Estimators: {meta['n_estimators']}
- Features: {meta['n_features']}
- Learning Rate: {meta['best_params']['learning_rate']:.4f}
- Num Leaves: {meta['best_params']['num_leaves']}
- Max Depth: {meta['best_params']['max_depth']}
- Min Child Samples: {meta['best_params']['min_child_samples']}

**Interpretation**

The near-zero R² is expected for LGD modeling in consumer lending. Recovery outcomes are driven
largely by the collections process, legal proceedings, and borrower circumstances at the time of
default — factors not observable at origination. The model's value is in identifying the small
but meaningful variation it can capture (MAE = {meta['test_mae']:.4f} on a [0,1] scale).
""")
        with ic2:
            try:
                importances = models['lgd_model'].feature_importances_
                feat_imp = pd.DataFrame({
                    'feature': models['lgd_features'],
                    'importance': importances,
                }).sort_values('importance', ascending=True).tail(20)
                fig = px.bar(feat_imp, x='importance', y='feature', orientation='h',
                              title="Top 20 Features (LGD)", color_discrete_sequence=['#f093fb'])
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Feature importances unavailable.")

        st.markdown("---")
        st.markdown("**SHAP Feature Importance** — mean |SHAP value| across a 520-loan sample "
                     "from the test vintage (native LGD scale).")
        shap_sample = load_shap_sample()
        if shap_sample is not None:
            _, lgd_explainer = load_shap_explainers(models['pd_model'], models['lgd_model'])
            lgd_shap_matrix = np.asarray(lgd_explainer.shap_values(shap_sample['lgd_X_sample']))
            mean_abs_shap = np.abs(lgd_shap_matrix).mean(axis=0)
            shap_imp = pd.DataFrame({
                'feature': models['lgd_features'],
                'mean_abs_shap': mean_abs_shap,
            }).sort_values('mean_abs_shap', ascending=True).tail(20)
            fig_shap = px.bar(shap_imp, x='mean_abs_shap', y='feature', orientation='h',
                               title="SHAP Mean |Impact| (LGD)",
                               color_discrete_sequence=['#e74c3c'])
            fig_shap.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP sample not found — run precompute_shap_sample.py first.")


def psi_badge(psi):
    if psi < 0.10:
        return "Stable", "#27ae60"
    elif psi < 0.25:
        return "Moderate Shift", "#f39c12"
    else:
        return "Significant Shift", "#e74c3c"


def format_bin_labels(edges):
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lo_s = "-inf" if not np.isfinite(lo) else f"{lo:.2f}"
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.2f}"
        labels.append(f"[{lo_s}, {hi_s})")
    return labels


STRESS_SAMPLE_SIZE = 50000


@st.cache_data
def load_stress_sample():
    """A fixed random sample, not the full 462,426-loan population: scoring
    the full population through the 1,967-tree PD model takes ~15-18s per
    call (measured directly, not assumed), which makes a slider genuinely
    unusable. A 50,000-row sample gives mean PD within 0.0001 of the full
    population (verified) at roughly 13x the speed. Dollar totals below are
    scaled back up to estimate the full-portfolio total."""
    test_fe = load_test_fe()
    sample = test_fe.sample(n=min(STRESS_SAMPLE_SIZE, len(test_fe)), random_state=42)
    scale_factor = len(test_fe) / len(sample)
    return sample, scale_factor, len(test_fe)


@st.cache_data
def compute_baseline_pd_scores(_pd_model, pd_features):
    """Cached separately from the stressed scenario: the baseline (zero-shift)
    scoring never changes between slider moves, so there's no reason to pay
    for it on every interaction."""
    sample, _, _ = load_stress_sample()
    return _pd_model.predict_proba(sample[pd_features])[:, 1]


def page_stress_testing(models):
    st.title("Stress Testing")
    st.markdown(
        "Shifts key inputs across a sample of the 2016-2017 test population to simulate an "
        "adverse economic scenario, rescores PD through the actual XGBoost model, and shows how "
        "total portfolio Expected Loss moves. This is the same kind of scenario-sensitivity check "
        "used in CCAR/DFAST-style bank stress testing."
    )
    _, scale_factor, full_n = load_stress_sample()
    st.caption(
        f"Scope: uses a random {STRESS_SAMPLE_SIZE:,}-loan sample of the {full_n:,}-loan test "
        f"population (scoring the full population takes ~15-18s per interaction through this "
        f"1,967-tree model — measured directly — which isn't workable for a slider; a sample "
        f"this size gives mean PD within 0.0001 of the full population). Dollar totals are scaled "
        f"by {scale_factor:.2f}x to estimate the full-portfolio total. Shifts DTI, income, "
        "revolving utilization, and recent inquiries directly, and recomputes the two "
        "income-dependent ratio features (`loan_to_income`, `revol_bal_to_income`) that feed into "
        "the model. Other engineered ratios (e.g. `active_ratio`, `installment_pct_of_loan`) are "
        "held at their original values for tractability. EAD uses the real term-based lookup; LGD "
        f"uses the fixed portfolio average ({AVG_LGD_PORTFOLIO:.1%}), same simplification as the "
        "baseline comparison on Model Information."
    )

    test_fe, _, _ = load_stress_sample()
    if test_fe is None:
        st.warning("test_fe.csv not found.")
        return

    pd_features = models['pd_features']
    missing = [f for f in pd_features if f not in test_fe.columns]
    if missing:
        st.error(f"test_fe.csv is missing features the PD model needs: {missing}")
        return

    platt = get_calibration_model()

    st.markdown("---")
    st.subheader("Scenario Inputs")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        dti_shift = st.slider("DTI shift (+ points)", 0.0, 20.0, 0.0, step=1.0)
    with sc2:
        income_shift_pct = st.slider("Income shift (− %)", 0.0, 40.0, 0.0, step=5.0)
    with sc3:
        revol_util_shift = st.slider("Revolving utilization shift (+ points)", 0.0, 30.0, 0.0, step=5.0)
    with sc4:
        inq_shift = st.slider("Recent inquiries shift (+ count)", 0, 5, 0, step=1)

    is_baseline_scenario = (dti_shift == 0 and income_shift_pct == 0
                             and revol_util_shift == 0 and inq_shift == 0)

    base_pd_raw = compute_baseline_pd_scores(models['pd_model'], pd_features)
    base_pd_cal = calibrate_pd(base_pd_raw, platt)

    if is_baseline_scenario:
        # Nothing shifted -- stressed is identical to baseline, skip rescoring
        # 462K rows a second time for a no-op scenario.
        stressed_pd_cal = base_pd_cal
    else:
        stressed_X = test_fe[pd_features].copy()
        stressed_income = test_fe['annual_inc'] * (1 - income_shift_pct / 100)
        if 'dti' in stressed_X.columns:
            stressed_X['dti'] = np.clip(test_fe['dti'] + dti_shift, 0, 100)
        if 'revol_util' in stressed_X.columns:
            stressed_X['revol_util'] = np.clip(test_fe['revol_util'] + revol_util_shift, 0, 100)
        if 'inq_last_6mths' in stressed_X.columns:
            stressed_X['inq_last_6mths'] = test_fe['inq_last_6mths'] + inq_shift
        if 'annual_inc' in stressed_X.columns:
            stressed_X['annual_inc'] = stressed_income
        if 'loan_to_income' in stressed_X.columns:
            stressed_X['loan_to_income'] = test_fe['loan_amnt'] / (stressed_income + 1)
        if 'revol_bal_to_income' in stressed_X.columns:
            stressed_X['revol_bal_to_income'] = test_fe['revol_bal'] / (stressed_income + 1)

        with st.spinner(f"Rescoring {len(test_fe):,} sampled loans under this scenario..."):
            stressed_pd_raw = models['pd_model'].predict_proba(stressed_X)[:, 1]
            stressed_pd_cal = calibrate_pd(stressed_pd_raw, platt)

    ead = test_fe['term'].map(EAD_LOOKUP).fillna(0.5805).values
    loan_amnt = test_fe['loan_amnt'].values
    base_el = base_pd_cal * ead * AVG_LGD_PORTFOLIO * loan_amnt
    stressed_el = stressed_pd_cal * ead * AVG_LGD_PORTFOLIO * loan_amnt
    base_el_total_est = base_el.sum() * scale_factor
    stressed_el_total_est = stressed_el.sum() * scale_factor

    st.markdown("---")
    st.subheader("Portfolio Impact")
    st.caption(f"Total EL figures are estimated for the full {full_n:,}-loan portfolio "
               f"(sample total x {scale_factor:.2f}); Avg PD is the sample mean directly.")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Baseline Avg PD", f"{base_pd_cal.mean():.2%}")
    with m2:
        st.metric("Stressed Avg PD", f"{stressed_pd_cal.mean():.2%}",
                   f"{stressed_pd_cal.mean()-base_pd_cal.mean():+.2%}", delta_color="inverse")
    with m3:
        st.metric("Baseline Total EL (est.)", f"${base_el_total_est/1e9:.2f}B")
    with m4:
        pct_change = (stressed_el_total_est - base_el_total_est) / base_el_total_est * 100
        st.metric("Stressed Total EL (est.)", f"${stressed_el_total_est/1e9:.2f}B",
                   f"{pct_change:+.1f}%", delta_color="inverse")

    dc1, dc2 = st.columns(2)
    with dc1:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=base_pd_cal, nbinsx=60, name='Baseline',
                                         marker_color='#667eea', opacity=0.6))
        fig_dist.add_trace(go.Histogram(x=stressed_pd_cal, nbinsx=60, name='Stressed',
                                         marker_color='#e74c3c', opacity=0.6))
        fig_dist.update_layout(title="Calibrated PD Distribution: Baseline vs. Stressed",
                                barmode='overlay', xaxis_title="Calibrated PD",
                                yaxis_title="Count", height=380)
        st.plotly_chart(fig_dist, use_container_width=True)
    with dc2:
        fig_bar = go.Figure(go.Bar(
            x=['Baseline', 'Stressed'], y=[base_el_total_est, stressed_el_total_est],
            marker_color=['#667eea', '#e74c3c'],
            text=[f"${base_el_total_est/1e9:.2f}B", f"${stressed_el_total_est/1e9:.2f}B"],
            textposition='outside'))
        fig_bar.update_layout(title="Total Portfolio Expected Loss (estimated)",
                               yaxis_title="Dollars ($)", height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    if is_baseline_scenario:
        st.info("All sliders at zero — this shows the baseline scenario against itself. "
                "Move a slider above to see the stressed comparison.")


def page_model_stability(models):
    st.title("Model Stability (PSI)")
    st.markdown(
        "**Population Stability Index** compares the PD model's score and input-feature "
        "distributions between the training population (2007-2015) and the test population "
        "(2016-2017) it was never trained on. A low PSI means the population the model sees "
        "hasn't drifted from what it learned on; a high PSI is an early warning that the model "
        "may need retraining — independent of whether its AUC still looks fine."
    )
    st.caption(
        "Convention: PSI < 0.10 = stable · 0.10-0.25 = moderate shift, investigate · "
        "> 0.25 = significant shift, consider retraining."
    )

    psi_ref = load_psi_reference()
    if psi_ref is None:
        st.warning("PSI reference not found — run precompute_psi.py first.")
        return

    results = psi_ref['results']
    monitored = psi_ref['monitored_features']

    score_info = results['__PD_SCORE__']
    score_psi = score_info['psi']
    label, color = psi_badge(score_psi)

    st.markdown("---")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.markdown(f"""<div style="text-align:center; padding:1.5rem; background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
     border-radius:16px; color:white;">
    <h3 style="margin:0; opacity:0.8; color:white;">PD Score PSI</h3>
    <h1 style="margin:0.5rem 0; font-size:2.5rem; color:white;">{score_psi:.4f}</h1>
    <span class="risk-badge" style="background:{color}; color:white;">{label}</span>
</div>""", unsafe_allow_html=True)
        st.caption(
            f"Train default rate: {psi_ref['train_default_rate']:.2%} "
            f"({psi_ref['train_n']:,} loans) · Test default rate: "
            f"{psi_ref['test_default_rate']:.2%} ({psi_ref['test_n']:,} loans). "
            f"Note PSI measures whether the *shape* of the score distribution shifted — "
            f"it can stay low even while the realized default rate moves, since that's a "
            f"calibration question, not a distribution-drift one."
        )
    with sc2:
        edges = score_info['edges']
        bin_labels = format_bin_labels(edges)
        fig_score = go.Figure()
        fig_score.add_trace(go.Bar(x=bin_labels, y=score_info['ref_pct'], name='Train (2007-2015)',
                                    marker_color='#667eea', opacity=0.8))
        fig_score.add_trace(go.Bar(x=bin_labels, y=score_info['cur_pct'], name='Test (2016-2017)',
                                    marker_color='#e74c3c', opacity=0.8))
        fig_score.update_layout(title="PD Score Distribution: Train vs. Test", barmode='group',
                                 xaxis_title="PD Score Bin", yaxis_title="% of Population",
                                 height=380, xaxis=dict(tickangle=30))
        st.plotly_chart(fig_score, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature-Level Drift")
    st.caption(
        "`issue_year` is excluded — it's the column that defines the train/test split "
        "(zero overlap by construction), so its PSI would be a tautological artifact, not real drift."
    )

    top_n = st.slider("Number of features to show", 5, len(monitored), 15)
    feat_psi = pd.DataFrame({
        'feature': monitored,
        'psi': [results[f]['psi'] for f in monitored],
        'importance': [results[f]['importance'] for f in monitored],
    }).sort_values('psi', ascending=False).head(top_n)

    fig_bar = px.bar(
        feat_psi.sort_values('psi'), x='psi', y='feature', orientation='h',
        title=f"Top {top_n} Features by PSI",
        color='psi', color_continuous_scale=['#27ae60', '#f39c12', '#e74c3c'],
        range_color=[0, 0.30],
    )
    fig_bar.add_vline(x=0.10, line_dash="dash", line_color="gray")
    fig_bar.add_vline(x=0.25, line_dash="dash", line_color="gray")
    fig_bar.update_layout(height=max(350, 25 * top_n), margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Bin-Level Detail")
    selected_feature = st.selectbox("Select a feature to inspect", monitored,
                                     index=monitored.index(feat_psi.iloc[0]['feature'])
                                     if len(feat_psi) > 0 else 0)
    fi = results[selected_feature]
    fbin_labels = format_bin_labels(fi['edges'])

    dc1, dc2 = st.columns([2, 1])
    with dc1:
        fig_detail = go.Figure()
        fig_detail.add_trace(go.Bar(x=fbin_labels, y=fi['ref_pct'], name='Train',
                                     marker_color='#667eea', opacity=0.8))
        fig_detail.add_trace(go.Bar(x=fbin_labels, y=fi['cur_pct'], name='Test',
                                     marker_color='#e74c3c', opacity=0.8))
        fig_detail.update_layout(title=f"{selected_feature} — Train vs. Test (PSI = {fi['psi']:.4f})",
                                  barmode='group', xaxis_title="Bin", yaxis_title="% of Population",
                                  height=350, xaxis=dict(tickangle=30))
        st.plotly_chart(fig_detail, use_container_width=True)
    with dc2:
        detail_df = pd.DataFrame({
            'Bin': fbin_labels, 'Train %': fi['ref_pct'], 'Test %': fi['cur_pct'],
        })
        st.dataframe(
            detail_df.style.format({'Train %': '{:.2%}', 'Test %': '{:.2%}'}),
            hide_index=True, use_container_width=True, height=350,
        )

    st.markdown("---")
    st.subheader("Monotonicity Check")
    st.markdown(
        "Does PD actually behave sensibly across the full range of each feature — e.g. does "
        "risk consistently fall as FICO rises, with no weird kinks? XGBoost doesn't enforce "
        "monotonic constraints by default, so this isn't guaranteed just because the model "
        "performs well overall. Checked using SHAP values (log-odds contribution) on the "
        "520-loan sample, binned into deciles."
    )

    shap_sample = load_shap_sample()
    if shap_sample is not None:
        pd_explainer, _ = load_shap_explainers(models['pd_model'], models['lgd_model'])
        pd_X_sample = shap_sample['pd_X_sample']
        shap_matrix = np.asarray(pd_explainer.shap_values(pd_X_sample))

        expected_direction = {
            'fico_score': 'decreasing', 'dti': 'increasing', 'revol_util': 'increasing',
            'credit_history_months': 'decreasing', 'inq_last_6mths': 'increasing',
            'sub_grade_encoded': 'increasing', 'annual_inc': 'decreasing',
            'loan_amnt': 'increasing',
        }

        def bin_direction(feature_vals, shap_vals, n_bins=8):
            d = pd.DataFrame({'val': feature_vals, 'shap': shap_vals})
            try:
                d['bin'] = pd.qcut(d['val'], n_bins, duplicates='drop')
            except ValueError:
                return None
            binned = d.groupby('bin', observed=True).agg(
                bin_mid=('val', 'mean'), mean_shap=('shap', 'mean')).reset_index(drop=True)
            if len(binned) < 3:
                return None
            diffs = np.diff(binned['mean_shap'].values)
            net_direction = 'increasing' if diffs.sum() >= 0 else 'decreasing'
            violations = int((diffs < 0).sum() if net_direction == 'increasing' else (diffs > 0).sum())
            return binned, net_direction, violations, len(diffs)

        rows = []
        binned_results = {}
        for feat, expected in expected_direction.items():
            if feat not in pd_X_sample.columns:
                continue
            fidx = models['pd_features'].index(feat)
            result = bin_direction(pd_X_sample[feat].values, shap_matrix[:, fidx])
            if result is None:
                continue
            binned, observed, violations, total = result
            binned_results[feat] = binned
            status = "OK" if observed == expected and violations == 0 else (
                "Minor bumps" if observed == expected else "REVERSED")
            rows.append({'Feature': feat, 'Expected': expected, 'Observed (net)': observed,
                         'Bin-to-bin violations': f"{violations}/{total}", 'Status': status})

        check_df = pd.DataFrame(rows)
        st.dataframe(check_df, hide_index=True, use_container_width=True)

        n_reversed = (check_df['Status'] == 'REVERSED').sum()
        n_bumpy = (check_df['Status'] == 'Minor bumps').sum()
        if n_reversed > 0:
            st.error(f"{n_reversed} feature(s) show a net direction opposite to business "
                     f"expectation — worth investigating before trusting this model's behavior "
                     f"at the extremes of that feature's range.")
        elif n_bumpy > 0:
            st.warning(f"All features point the expected direction on net, but {n_bumpy} show "
                       f"some non-monotonic bumps bin-to-bin — likely noise in a 520-loan sample "
                       f"rather than a real problem, but worth re-checking on a larger sample "
                       f"before relying on this for regulatory monotonicity sign-off.")
        else:
            st.success("All checked features are monotonic in the expected direction, with no "
                       "bin-to-bin reversals in this sample.")

        feat_to_plot = st.selectbox("Inspect a feature's dependence plot", list(binned_results.keys()))
        if feat_to_plot:
            b = binned_results[feat_to_plot]
            fig_dep = go.Figure(go.Scatter(x=b['bin_mid'], y=b['mean_shap'], mode='lines+markers',
                                            marker=dict(size=10, color='#667eea'),
                                            line=dict(color='#667eea', width=2)))
            fig_dep.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_dep.update_layout(
                title=f"SHAP Dependence: {feat_to_plot} (expected: {expected_direction[feat_to_plot]})",
                xaxis_title=feat_to_plot, yaxis_title="Mean SHAP value (log-odds) per decile bin",
                height=380)
            st.plotly_chart(fig_dep, use_container_width=True)
    else:
        st.info("SHAP sample not found — run precompute_shap_sample.py first.")

    st.markdown("---")
    st.subheader("Fair Lending / Geographic Sensitivity")
    st.markdown(
        "**This is not a true disparate impact test.** Genuine fair lending analysis needs "
        "protected-class data (race, ethnicity, sex) or a BISG (Bayesian Improved Surname "
        "Geocoding) proxy built from census surname/geography race distributions — neither "
        "exists in this dataset. Pretending otherwise would give false confidence, so this "
        "isn't attempted here."
    )
    st.markdown(
        "What *is* checked: this is US consumer credit, and `addr_state` is exactly the kind "
        "of geographic feature regulators scrutinize as a potential proxy for protected classes. "
        "Below, a 10,400-loan sample (2016-2017 vintage, all 50 states) is scored twice through "
        "the actual PD model — once with `state_default_rate` as normally computed, once with "
        "it forced to the population-wide average (neutralizing state-specific information) — "
        "to see how much predicted PD shifts per state when the state signal is removed."
    )

    fl = load_fairlending_sensitivity()
    if fl is not None:
        by_state = fl['by_state'].copy()
        fl1, fl2, fl3 = st.columns(3)
        with fl1:
            st.metric("Sample Size", f"{fl['sample_n']:,}")
        with fl2:
            st.metric("Mean |PD shift| from state signal", f"{fl['overall_mean_abs_diff']:.4f}")
        with fl3:
            reliable = by_state[by_state['n'] >= 50]
            max_reliable = reliable.iloc[0] if len(reliable) > 0 else None
            if max_reliable is not None:
                st.metric(f"Largest shift (n≥50): {max_reliable['addr_state']}",
                          f"{max_reliable['avg_diff']:+.4f}", f"n={int(max_reliable['n'])}")

        by_state['reliable'] = by_state['n'] >= 50
        fig_state = px.bar(
            by_state.sort_values('avg_diff'), x='avg_diff', y='addr_state', orientation='h',
            color='reliable', color_discrete_map={True: '#667eea', False: '#cccccc'},
            title="Avg PD Shift When State Signal Is Removed, by State",
            labels={'avg_diff': 'PD (with state) − PD (state neutralized)', 'addr_state': 'State',
                    'reliable': 'n ≥ 50'},
            hover_data=['n'],
        )
        fig_state.update_layout(height=900, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_state, use_container_width=True)

        st.caption(
            "Gray bars have fewer than 50 loans in this sample — their large-looking shifts are "
            "likely sampling noise, not a real state effect, and shouldn't be read as findings. "
            "Blue bars (n≥50) are the ones worth taking seriously. A shift here means the model's "
            "risk assessment for that state depends materially on the state signal itself, beyond "
            "what the applicant's own financial profile explains — worth investigating further "
            "with a real BISG-based race/ethnicity proxy before concluding anything about "
            "disparate impact one way or the other."
        )
    else:
        st.info("Fair lending sensitivity data not found — run precompute_fairlending.py first.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    models = load_models()
    config = load_config()

    st.sidebar.markdown("## 🏦 Credit Risk")
    st.sidebar.markdown("**Analyzer**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Validation Summary", "Single Loan", "Batch Scoring",
         "Portfolio Analytics", "Model Stability (PSI)", "Stress Testing",
         "Model Information"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.75rem; opacity:0.7; color:#a0a0a0;">
    PD: XGBoost | EAD: Term Lookup<br>
    LGD: LightGBM | EL = PD x EAD x LGD<br><br>
    Built with Lending Club data<br>
    Train: 2007-2015 | Test: 2016-2017
    </div>
    """, unsafe_allow_html=True)

    if page == "Dashboard":
        page_dashboard(models, config)
    elif page == "Validation Summary":
        page_validation_summary(models)
    elif page == "Single Loan":
        page_single_loan(models, config)
    elif page == "Batch Scoring":
        page_batch(models, config)
    elif page == "Portfolio Analytics":
        page_portfolio(models, config)
    elif page == "Model Stability (PSI)":
        page_model_stability(models)
    elif page == "Stress Testing":
        page_stress_testing(models)
    elif page == "Model Information":
        page_model_info(models)


if __name__ == "__main__":
    main()
