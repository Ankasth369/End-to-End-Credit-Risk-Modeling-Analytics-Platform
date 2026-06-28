import numpy as np

import credit_core as cc


class TestGetAdverseActionReasons:
    def test_excludes_non_disclosable_features_even_with_high_positive_shap(self):
        """Regression test for a real bug: issue_year (today's scoring date, not
        a credit factor) surfaced as reason #2 in the app because it can carry a
        large positive SHAP value. It and issue_month must never be disclosable
        adverse-action reasons regardless of their SHAP contribution."""
        feature_names = ['issue_year', 'fico_score', 'dti', 'issue_month', 'revol_util']
        shap_values = np.array([0.50, 0.30, 0.20, 0.45, 0.10])  # issue_year/month rank highest
        feature_values = {'issue_year': 2026, 'fico_score': 620, 'dti': 25.0,
                           'issue_month': 8, 'revol_util': 70.0}

        reasons = cc.get_adverse_action_reasons(shap_values, feature_names, feature_values, top_n=4)

        reason_texts = ' '.join(r[0].lower() for r in reasons)
        assert 'issue' not in reason_texts
        assert len(reasons) == 3  # only the 3 legitimate features remain

    def test_only_positive_shap_values_included(self):
        feature_names = ['fico_score', 'dti', 'revol_util']
        shap_values = np.array([-0.30, 0.20, -0.10])  # fico_score and revol_util protective
        feature_values = {'fico_score': 780, 'dti': 25.0, 'revol_util': 10.0}

        reasons = cc.get_adverse_action_reasons(shap_values, feature_names, feature_values)

        assert len(reasons) == 1
        assert 'debt-to-income' in reasons[0][0].lower()

    def test_sorted_by_shap_magnitude_descending(self):
        feature_names = ['fico_score', 'dti', 'revol_util']
        shap_values = np.array([0.10, 0.40, 0.25])
        feature_values = {'fico_score': 620, 'dti': 30.0, 'revol_util': 80.0}

        reasons = cc.get_adverse_action_reasons(shap_values, feature_names, feature_values)

        assert [round(r[1], 2) for r in reasons] == [0.40, 0.25, 0.10]

    def test_respects_top_n_limit(self):
        feature_names = [f'feat_{i}' for i in range(10)]
        shap_values = np.arange(10, dtype=float) + 1  # all positive, increasing
        feature_values = {f: 1.0 for f in feature_names}

        reasons = cc.get_adverse_action_reasons(shap_values, feature_names, feature_values, top_n=4)

        assert len(reasons) == 4

    def test_no_risk_increasing_features_returns_empty(self):
        feature_names = ['fico_score', 'dti']
        shap_values = np.array([-0.1, -0.2])
        feature_values = {'fico_score': 780, 'dti': 5.0}

        reasons = cc.get_adverse_action_reasons(shap_values, feature_names, feature_values)

        assert reasons == []

    def test_unmapped_feature_falls_back_to_title_case(self):
        feature_names = ['some_unmapped_feature_xyz']
        shap_values = np.array([0.5])
        feature_values = {'some_unmapped_feature_xyz': 1.0}

        reasons = cc.get_adverse_action_reasons(shap_values, feature_names, feature_values)

        assert reasons[0][0] == 'Some Unmapped Feature Xyz'
