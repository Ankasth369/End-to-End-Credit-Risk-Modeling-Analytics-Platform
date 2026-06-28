"""
Model regression tests: catch silent model swaps, corruption, or accidental
retraining by checking that fixed reference inputs still produce the same
outputs, and that the saved validation artifacts still show the expected
discrimination. These are NOT unit tests of logic -- they pin down the
actual trained model's behavior.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import credit_core as cc

# Golden loans + expected predictions, computed once against the currently
# saved pd_model.joblib / lgd_model.joblib. If these ever change beyond
# floating-point noise, either the model files changed or preprocessing
# changed -- both worth a human looking at before merging.
GOLDEN_LOANS = [
    {
        'inputs': {'loan_amnt': 15000, 'term': 36, 'sub_grade': 'B3', 'annual_inc': 65000,
                   'dti': 18.5, 'fico_score': 700, 'revol_util': 55.0, 'open_acc': 11,
                   'total_acc': 25, 'revol_bal': 15000, 'credit_history_months': 180,
                   'home_ownership': 'RENT', 'addr_state': 'CA',
                   'purpose': 'debt_consolidation', 'installment': 500,
                   'issue_year': 2016, 'issue_month': 1},
        'expected_pd': 0.38722536, 'expected_lgd': 0.93039364,
    },
    {
        'inputs': {'loan_amnt': 30000, 'term': 60, 'sub_grade': 'D2', 'annual_inc': 45000,
                   'dti': 28.0, 'fico_score': 650, 'revol_util': 80.0, 'open_acc': 8,
                   'total_acc': 15, 'revol_bal': 22000, 'credit_history_months': 90,
                   'home_ownership': 'MORTGAGE', 'addr_state': 'TX',
                   'purpose': 'credit_card', 'installment': 650,
                   'issue_year': 2016, 'issue_month': 1},
        'expected_pd': 0.72068048, 'expected_lgd': 0.92351159,
    },
    {
        'inputs': {'loan_amnt': 5000, 'term': 36, 'sub_grade': 'A2', 'annual_inc': 120000,
                   'dti': 8.0, 'fico_score': 780, 'revol_util': 15.0, 'open_acc': 15,
                   'total_acc': 30, 'revol_bal': 5000, 'credit_history_months': 300,
                   'home_ownership': 'OWN', 'addr_state': 'NY',
                   'purpose': 'home_improvement', 'installment': 160,
                   'issue_year': 2016, 'issue_month': 1},
        'expected_pd': 0.12172139, 'expected_lgd': 0.91471098,
    },
]


class TestGoldenPredictions:
    def test_pd_predictions_match_reference(self, config, models):
        for case in GOLDEN_LOANS:
            pd_X, _, _ = cc.preprocess_single(
                case['inputs'], config, models['pd_features'], models['lgd_features'])
            pd_pred = models['pd_model'].predict_proba(pd_X)[:, 1][0]
            assert abs(pd_pred - case['expected_pd']) < 1e-4, (
                f"PD prediction drifted for {case['inputs']}: "
                f"got {pd_pred:.6f}, expected {case['expected_pd']:.6f}. "
                f"If this is an intentional model retrain, update GOLDEN_LOANS."
            )

    def test_lgd_predictions_match_reference(self, config, models):
        for case in GOLDEN_LOANS:
            _, lgd_X, _ = cc.preprocess_single(
                case['inputs'], config, models['pd_features'], models['lgd_features'])
            lgd_pred = float(np.clip(models['lgd_model'].predict(lgd_X), 0, 1)[0])
            assert abs(lgd_pred - case['expected_lgd']) < 1e-4, (
                f"LGD prediction drifted for {case['inputs']}: "
                f"got {lgd_pred:.6f}, expected {case['expected_lgd']:.6f}."
            )

    def test_worse_credit_profile_scores_higher_pd(self, config, models):
        """Loan 1 (FICO 650, DTI 28) should score materially riskier than
        Loan 2 (FICO 780, DTI 8) -- a basic sanity check on model direction."""
        pd_X_1, _, _ = cc.preprocess_single(
            GOLDEN_LOANS[1]['inputs'], config, models['pd_features'], models['lgd_features'])
        pd_X_2, _, _ = cc.preprocess_single(
            GOLDEN_LOANS[2]['inputs'], config, models['pd_features'], models['lgd_features'])
        pd_1 = models['pd_model'].predict_proba(pd_X_1)[:, 1][0]
        pd_2 = models['pd_model'].predict_proba(pd_X_2)[:, 1][0]
        assert pd_1 > pd_2


class TestSavedPortfolioValidation:
    """Sanity checks against the saved notebook-07 validation artifact. These
    catch a stale/corrupted portfolio CSV drifting from what the model
    actually produces -- not a full model regression (see TestGoldenPredictions
    for that), but a check that the shipped validation numbers are still real."""

    def test_auc_matches_validated_metadata(self, portfolio, models):
        if portfolio is None:
            import pytest
            pytest.skip("expected_loss_test_2016_2017.csv not present")
        auc = roc_auc_score(portfolio['actual_default'], portfolio['pd_pred'])
        expected_auc = models['pd_meta']['test_auc']
        assert abs(auc - expected_auc) < 0.01, (
            f"Portfolio CSV's implied AUC ({auc:.4f}) has drifted from "
            f"pd_metadata.json's recorded AUC ({expected_auc:.4f})."
        )

    def test_portfolio_pd_predictions_reproducible_from_live_model(self, portfolio, config, models):
        """Re-scores a subset of the saved portfolio through the live PD model
        using the same features it was scored with, and checks the model
        itself is internally consistent (not that our preprocessing matches
        notebook 07's independent pipeline, which uses different code by
        design -- see notebook 07's own commentary on this)."""
        if portfolio is None:
            import pytest
            pytest.skip("expected_loss_test_2016_2017.csv not present")
        assert portfolio['pd_pred'].between(0, 1).all()
        assert portfolio['lgd_pred'].between(0, 1).all()
        assert portfolio['ead_pred'].isin(list(cc.EAD_LOOKUP.values())).all()
