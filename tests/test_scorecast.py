"""
tests/test_scorecast.py
=======================
Unit tests for ScorecastML pipeline.

Run: python -m pytest tests/ -v

Author: Ahad Hashmani
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from data.loader import (
    generate_synthetic_season,
    TEAM_PROFILES,
    TARGET_SEASONS,
)
from features.engineer import (
    compute_team_rolling_stats,
    build_matchup_features,
    get_feature_columns,
    ROLL_WINDOWS,
)
from models.train import (
    time_split,
    train_logistic_regression,
    evaluate_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw_season():
    return generate_synthetic_season("2023-24", n_games_per_team=20, seed=99)


@pytest.fixture(scope="module")
def team_stats(raw_season):
    return compute_team_rolling_stats(raw_season)


@pytest.fixture(scope="module")
def matchup_df(team_stats):
    return build_matchup_features(team_stats)


# ── Data loader tests ─────────────────────────────────────────────────
class TestDataLoader:
    def test_synthetic_has_all_teams(self, raw_season):
        teams = set(raw_season["TEAM_ABBREVIATION"].unique())
        assert teams == set(TEAM_PROFILES.keys())

    def test_synthetic_has_expected_columns(self, raw_season):
        for col in ["GAME_ID", "PTS", "WL", "IS_HOME", "GAME_DATE"]:
            assert col in raw_season.columns

    def test_synthetic_pts_positive(self, raw_season):
        assert (raw_season["PTS"] > 0).all()

    def test_win_loss_binary(self, raw_season):
        assert set(raw_season["WL"].unique()).issubset({"W", "L"})

    def test_home_away_balanced(self, raw_season):
        counts = raw_season["IS_HOME"].value_counts()
        ratio = counts[0] / counts[1]
        assert 0.85 < ratio < 1.15, f"Home/away imbalance: {ratio:.2f}"

    def test_season_label_present(self, raw_season):
        assert (raw_season["SEASON"] == "2023-24").all()

    def test_force_synthetic_does_not_call_nba_api(self, monkeypatch, tmp_path):
        import data.loader as loader

        def boom(season):
            raise AssertionError(f"nba_api was called for {season}")

        monkeypatch.setattr(loader, "_load_from_nba_api", boom)
        monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)
        df = loader.load_season("2023-24", force_synthetic=True)
        assert (df["SOURCE"] == "synthetic").all()
        assert list(tmp_path.glob("*.parquet"))


# ── Feature engineering tests ─────────────────────────────────────────
class TestFeatureEngineering:
    def test_rolling_stats_has_zscore_columns(self, team_stats):
        for w in ROLL_WINDOWS:
            assert f"WIN_RATE_{w}" in team_stats.columns

    def test_days_rest_non_negative(self, team_stats):
        assert (team_stats["DAYS_REST"] >= 0).all()

    def test_b2b_is_binary(self, team_stats):
        assert set(team_stats["IS_B2B"].unique()).issubset({0, 1})

    def test_matchup_df_has_target(self, matchup_df):
        assert "HOME_WIN" in matchup_df.columns

    def test_home_win_is_binary(self, matchup_df):
        assert set(matchup_df["HOME_WIN"].unique()).issubset({0, 1})

    def test_home_win_rate_plausible(self, matchup_df):
        rate = matchup_df["HOME_WIN"].mean()
        assert 0.50 < rate < 0.65, f"Home win rate {rate:.3f} outside expected range"

    def test_diff_features_present(self, matchup_df):
        diff_cols = get_feature_columns(matchup_df)
        assert len(diff_cols) > 0
        assert all(c.startswith("DIFF_") for c in diff_cols)

    def test_matchup_df_no_nan_in_features(self, matchup_df):
        diff_cols = get_feature_columns(matchup_df)
        assert matchup_df[diff_cols].isna().sum().sum() == 0

    def test_both_teams_in_matchup(self, matchup_df):
        assert "H_TEAM_ABBREVIATION" in matchup_df.columns
        assert "A_TEAM_ABBREVIATION" in matchup_df.columns


# ── Model tests ───────────────────────────────────────────────────────
class TestModels:
    def test_time_split_no_leakage(self, matchup_df):
        """Test set must be strictly after train set."""
        if matchup_df["SEASON"].nunique() < 2:
            pytest.skip("Need multiple seasons for split test")
        train_df, test_df = time_split(matchup_df, test_seasons=1)
        assert train_df["GAME_DATE"].max() <= test_df["GAME_DATE"].min()

    def test_logistic_regression_trains(self, matchup_df):
        diff_cols = get_feature_columns(matchup_df)
        X = matchup_df[diff_cols].values.astype(np.float32)
        y = matchup_df["HOME_WIN"].values
        split = int(len(X) * 0.8)
        model, imp = train_logistic_regression(X[:split], y[:split], diff_cols)
        assert hasattr(model, "predict_proba")
        assert len(imp) == len(diff_cols)

    def test_predict_proba_sums_to_one(self, matchup_df):
        diff_cols = get_feature_columns(matchup_df)
        X = matchup_df[diff_cols].values.astype(np.float32)
        y = matchup_df["HOME_WIN"].values
        split = int(len(X) * 0.8)
        model, _ = train_logistic_regression(X[:split], y[:split], diff_cols)
        proba = model.predict_proba(X[split:])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_accuracy_above_random(self, matchup_df):
        """Model should beat a random 50% baseline."""
        diff_cols = get_feature_columns(matchup_df)
        X = matchup_df[diff_cols].values.astype(np.float32)
        y = matchup_df["HOME_WIN"].values
        split = int(len(X) * 0.8)
        model, _ = train_logistic_regression(X[:split], y[:split], diff_cols)
        result = evaluate_model(
            model, X[split:], y[split:],
            "LR", diff_cols, split, {}
        )
        assert result.accuracy > 0.50, f"Accuracy {result.accuracy:.3f} not above random"

    def test_feature_importance_sums_positive(self, matchup_df):
        diff_cols = get_feature_columns(matchup_df)
        X = matchup_df[diff_cols].values.astype(np.float32)
        y = matchup_df["HOME_WIN"].values
        model, imp = train_logistic_regression(X, y, diff_cols)
        assert sum(imp.values()) > 0


# ── Prediction tests ──────────────────────────────────────────────────
class TestPrediction:
    def test_predict_returns_valid_probabilities(self, matchup_df, team_stats):
        diff_cols = get_feature_columns(matchup_df)
        X = matchup_df[diff_cols].values.astype(np.float32)
        y = matchup_df["HOME_WIN"].values
        model, _ = train_logistic_regression(X, y, diff_cols)

        from predict.matchup import predict_matchup
        pred = predict_matchup("BOS", "MIA", team_stats, model, diff_cols)
        assert 0 <= pred.home_win_prob <= 1
        assert abs(pred.home_win_prob + pred.away_win_prob - 1.0) < 1e-5

    def test_predict_winner_consistent_with_prob(self, matchup_df, team_stats):
        diff_cols = get_feature_columns(matchup_df)
        X = matchup_df[diff_cols].values.astype(np.float32)
        y = matchup_df["HOME_WIN"].values
        model, _ = train_logistic_regression(X, y, diff_cols)

        from predict.matchup import predict_matchup
        pred = predict_matchup("BOS", "MIA", team_stats, model, diff_cols)
        if pred.home_win_prob >= 0.5:
            assert pred.predicted_winner == "BOS"
        else:
            assert pred.predicted_winner == "MIA"
