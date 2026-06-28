import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import credit_core as cc

BASE = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def models():
    return cc.load_models(BASE)


@pytest.fixture(scope="session")
def config():
    return cc.load_config(BASE)


@pytest.fixture(scope="session")
def portfolio():
    return cc.load_portfolio(BASE)


@pytest.fixture(scope="session")
def platt(portfolio):
    return cc.fit_calibration_model(portfolio)


@pytest.fixture
def sample_inputs():
    """A representative, fully-specified loan application."""
    return {
        'loan_amnt': 15000, 'term': 36, 'int_rate': 12.5, 'installment': 501.85,
        'sub_grade': 'B3', 'emp_length': 5, 'home_ownership': 'RENT',
        'verification_status': 'Not Verified', 'purpose': 'debt_consolidation',
        'addr_state': 'CA', 'annual_inc': 65000, 'dti': 18.5,
        'fico_score': 700, 'open_acc': 11, 'total_acc': 25,
        'revol_bal': 15000, 'revol_util': 55.0, 'inq_last_6mths': 1,
        'delinq_2yrs': 0, 'pub_rec': 0, 'mort_acc': 1,
        'mths_since_last_delinq': 50, 'credit_history_months': 180,
        'issue_year': 2026, 'issue_month': 8,
        'tot_coll_amt': 0, 'tot_cur_bal': 150000, 'total_rev_hi_lim': 30000,
        'avg_cur_bal': 13000, 'bc_open_to_buy': 7000, 'bc_util': 60,
        'pct_tl_nvr_dlq': 94, 'pub_rec_bankruptcies': 0, 'tax_liens': 0,
    }
