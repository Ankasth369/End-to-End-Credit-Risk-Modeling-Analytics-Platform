import numpy as np
import pandas as pd

import credit_core as cc


class TestScore:
    def test_expected_loss_formula(self, sample_inputs, config, models):
        pd_X, lgd_X, ead_pct = cc.preprocess_single(
            sample_inputs, config, models['pd_features'], models['lgd_features'])
        pd_pred, ead_pred, lgd_pred, el = cc.score(
            pd_X, lgd_X, ead_pct, sample_inputs['loan_amnt'], models)
        expected_el = pd_pred[0] * ead_pred[0] * lgd_pred[0] * sample_inputs['loan_amnt']
        assert abs(el[0] - expected_el) < 1e-6

    def test_pd_and_lgd_in_valid_range(self, sample_inputs, config, models):
        pd_X, lgd_X, ead_pct = cc.preprocess_single(
            sample_inputs, config, models['pd_features'], models['lgd_features'])
        pd_pred, ead_pred, lgd_pred, _ = cc.score(
            pd_X, lgd_X, ead_pct, sample_inputs['loan_amnt'], models)
        assert 0.0 <= pd_pred[0] <= 1.0
        assert 0.0 <= lgd_pred[0] <= 1.0
        assert ead_pred[0] in (0.5804732715289598, 0.7348155329774531)

    def test_scalar_and_array_inputs_agree(self, sample_inputs, config, models):
        pd_X, lgd_X, ead_pct = cc.preprocess_single(
            sample_inputs, config, models['pd_features'], models['lgd_features'])
        scalar_result = cc.score(pd_X, lgd_X, ead_pct, sample_inputs['loan_amnt'], models)
        array_result = cc.score(pd_X, lgd_X, np.array([ead_pct]),
                                 np.array([sample_inputs['loan_amnt']]), models)
        for a, b in zip(scalar_result, array_result):
            assert np.allclose(a, b)


class TestCalibratePd:
    def test_none_platt_returns_input_unchanged(self):
        raw = np.array([0.1, 0.5, 0.9])
        result = cc.calibrate_pd(raw, None)
        assert np.allclose(result, raw)

    def test_preserves_rank_order(self, platt):
        """Platt scaling is a monotonic transform of the logit -- calibrated PD
        must preserve the same ordering as raw PD. Several pages in the app
        (e.g. the Threshold & Profitability ROC curve) rely on this holding."""
        raw = np.array([0.05, 0.15, 0.30, 0.45, 0.60, 0.80, 0.95])
        calibrated = cc.calibrate_pd(raw, platt)
        assert np.all(np.diff(calibrated) > 0), "calibration must preserve rank order"

    def test_output_in_valid_probability_range(self, platt):
        raw = np.linspace(0.001, 0.999, 50)
        calibrated = cc.calibrate_pd(raw, platt)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_handles_extreme_values_without_nan(self, platt):
        raw = np.array([0.0, 1.0, 1e-10, 1 - 1e-10])
        calibrated = cc.calibrate_pd(raw, platt)
        assert not np.any(np.isnan(calibrated))


class TestGetRiskLevel:
    def test_boundaries(self):
        assert cc.get_risk_level(0.05)[0] == 'Low'
        assert cc.get_risk_level(0.10)[0] == 'Low'
        assert cc.get_risk_level(0.15)[0] == 'Medium'
        assert cc.get_risk_level(0.20)[0] == 'Medium'
        assert cc.get_risk_level(0.30)[0] == 'High'
        assert cc.get_risk_level(0.35)[0] == 'High'
        assert cc.get_risk_level(0.50)[0] == 'Very High'


class TestComputeInstallment:
    def test_zero_rate_is_simple_division(self):
        assert cc.compute_installment(12000, 12, 0) == 1000.0

    def test_matches_standard_amortization_formula(self):
        loan_amnt, term, int_rate = 15000, 36, 12.5
        result = cc.compute_installment(loan_amnt, term, int_rate)
        r = int_rate / 100 / 12
        expected = loan_amnt * r / (1 - (1 + r) ** (-term))
        assert abs(result - expected) < 1e-9
