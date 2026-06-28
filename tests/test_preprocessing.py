import numpy as np
import pandas as pd

import credit_core as cc


class TestPreprocessSingle:
    def test_output_columns_match_feature_lists(self, sample_inputs, config, models):
        pd_X, lgd_X, ead_pct = cc.preprocess_single(
            sample_inputs, config, models['pd_features'], models['lgd_features'])
        assert list(pd_X.columns) == models['pd_features']
        assert list(lgd_X.columns) == models['lgd_features']
        assert len(pd_X) == 1
        assert len(lgd_X) == 1

    def test_no_nulls_in_output(self, sample_inputs, config, models):
        pd_X, lgd_X, _ = cc.preprocess_single(
            sample_inputs, config, models['pd_features'], models['lgd_features'])
        assert not pd_X.isnull().any().any()
        assert not lgd_X.isnull().any().any()
        assert np.isfinite(pd_X.values).all()
        assert np.isfinite(lgd_X.values).all()

    def test_ead_lookup_36_and_60_month(self, sample_inputs, config, models):
        for term, expected in [(36, 0.5804732715289598), (60, 0.7348155329774531)]:
            inp = dict(sample_inputs, term=term)
            _, _, ead_pct = cc.preprocess_single(
                inp, config, models['pd_features'], models['lgd_features'])
            assert ead_pct == expected

    def test_ead_lookup_unknown_term_falls_back(self, sample_inputs, config, models):
        inp = dict(sample_inputs, term=45)
        _, _, ead_pct = cc.preprocess_single(
            inp, config, models['pd_features'], models['lgd_features'])
        assert ead_pct == 0.5805

    def test_sub_grade_encoding(self, sample_inputs, config, models):
        for sg, expected_rank in [('A1', 0), ('C3', 12), ('G5', 34)]:
            inp = dict(sample_inputs, sub_grade=sg)
            pd_X, _, _ = cc.preprocess_single(
                inp, config, models['pd_features'], models['lgd_features'])
            assert pd_X['sub_grade_encoded'].iloc[0] == expected_rank

    def test_missing_sub_grade_defaults_to_C3(self, sample_inputs, config, models):
        inp = dict(sample_inputs)
        del inp['sub_grade']
        pd_X, _, _ = cc.preprocess_single(
            inp, config, models['pd_features'], models['lgd_features'])
        assert pd_X['sub_grade_encoded'].iloc[0] == 12

    def test_revol_util_clipped_to_winsor_and_100_cap(self, sample_inputs, config, models):
        """revol_util is clipped twice: first to the training-fit 99.5th
        percentile winsor bound, then to a hard 100 cap. The winsor bound is
        typically tighter, so the effective cap is whichever is smaller."""
        inp = dict(sample_inputs, revol_util=150)
        pd_X, _, _ = cc.preprocess_single(
            inp, config, models['pd_features'], models['lgd_features'])
        expected_cap = min(config['winsor_upper']['revol_util'], 100)
        assert pd_X['revol_util'].iloc[0] == expected_cap
        assert pd_X['revol_util'].iloc[0] <= 100

    def test_dti_floored_at_zero(self, sample_inputs, config, models):
        inp = dict(sample_inputs, dti=-5)
        pd_X, _, _ = cc.preprocess_single(
            inp, config, models['pd_features'], models['lgd_features'])
        assert pd_X['dti'].iloc[0] == 0

    def test_home_ownership_one_hot(self, sample_inputs, config, models):
        inp = dict(sample_inputs, home_ownership='OWN')
        _, lgd_X, _ = cc.preprocess_single(
            inp, config, models['pd_features'], models['lgd_features'])
        assert lgd_X['home_OWN'].iloc[0] == 1.0
        assert lgd_X['home_OTHER'].iloc[0] == 0.0
        assert lgd_X['home_RENT'].iloc[0] == 0.0

    def test_loan_to_income_derived_correctly(self, sample_inputs, config, models):
        inp = dict(sample_inputs, loan_amnt=10000, annual_inc=50000)
        pd_X, _, _ = cc.preprocess_single(
            inp, config, models['pd_features'], models['lgd_features'])
        expected = 10000 / (50000 + 1)
        assert abs(pd_X['loan_to_income'].iloc[0] - expected) < 1e-9


class TestPreprocessBatch:
    def _raw_df(self, n=3):
        return pd.DataFrame({
            'loan_amnt': [15000, 8000, 25000],
            'term': ['36 months', '60 months', '36 months'],
            'int_rate': [12.5, 8.5, 18.0],
            'sub_grade': ['B3', 'A4', 'D2'],
            'emp_length': ['5 years', '10+ years', '< 1 year'],
            'home_ownership': ['RENT', 'OWN', 'MORTGAGE'],
            'verification_status': ['Not Verified', 'Verified', 'Source Verified'],
            'purpose': ['debt_consolidation', 'home_improvement', 'credit_card'],
            'addr_state': ['CA', 'TX', 'NY'],
            'annual_inc': [65000, 45000, 85000],
            'dti': [18.5, 12.5, 22.0],
            'fico_range_low': [695, 735, 675],
            'fico_range_high': [699, 739, 679],
            'open_acc': [11, 7, 15],
            'total_acc': [25, 14, 32],
            'revol_bal': [15000, 5000, 22000],
            'revol_util': [55.0, 30.0, 72.0],
            'inq_last_6mths': [1, 0, 2],
            'delinq_2yrs': [0, 0, 1],
            'pub_rec': [0, 0, 0],
        })[:n]

    def test_output_shape_and_columns(self, config, models):
        df_raw = self._raw_df()
        pd_X, lgd_X, ead_pct = cc.preprocess_batch(
            df_raw, config, models['pd_features'], models['lgd_features'])
        assert len(pd_X) == 3
        assert list(pd_X.columns) == models['pd_features']
        assert list(lgd_X.columns) == models['lgd_features']
        assert len(ead_pct) == 3

    def test_term_string_parsed(self, config, models):
        df_raw = self._raw_df()
        pd_X, _, ead_pct = cc.preprocess_batch(
            df_raw, config, models['pd_features'], models['lgd_features'])
        assert list(pd_X['term']) == [36, 60, 36]
        assert ead_pct[0] == 0.5804732715289598
        assert ead_pct[1] == 0.7348155329774531

    def test_emp_length_mapped(self, config, models):
        df_raw = self._raw_df()
        pd_X, _, _ = cc.preprocess_batch(
            df_raw, config, models['pd_features'], models['lgd_features'])
        assert list(pd_X['emp_length']) == [5, 10, 0]

    def test_fico_score_averaged(self, config, models):
        df_raw = self._raw_df()
        pd_X, _, _ = cc.preprocess_batch(
            df_raw, config, models['pd_features'], models['lgd_features'])
        assert pd_X['fico_score'].iloc[0] == 697.0

    def test_missing_optional_columns_filled_with_medians(self, config, models):
        df_raw = self._raw_df()[['loan_amnt', 'term', 'sub_grade', 'annual_inc', 'dti',
                                   'fico_range_low', 'fico_range_high', 'home_ownership',
                                   'verification_status', 'purpose', 'addr_state',
                                   'open_acc', 'total_acc', 'revol_bal', 'revol_util',
                                   'inq_last_6mths', 'delinq_2yrs', 'pub_rec']]
        pd_X, lgd_X, _ = cc.preprocess_batch(
            df_raw, config, models['pd_features'], models['lgd_features'])
        assert not pd_X.isnull().any().any()
        assert not lgd_X.isnull().any().any()

    def test_no_nulls_or_infs_in_output(self, config, models):
        df_raw = self._raw_df()
        pd_X, lgd_X, ead_pct = cc.preprocess_batch(
            df_raw, config, models['pd_features'], models['lgd_features'])
        assert np.isfinite(pd_X.values).all()
        assert np.isfinite(lgd_X.values).all()
        assert np.isfinite(ead_pct).all()
