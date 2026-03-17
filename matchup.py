"""
predict/matchup.py
==================
Predict the outcome of any NBA matchup given two team abbreviations.

Usage:
    python main.py predict --home BOS --away MIA

How it works:
    1. Load the most recent rolling stats for both teams
    2. Compute differential features (home - away)
    3. Pass through the trained GBT model
    4. Return win probability + confidence band

Author: Ahad Hashmani
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data.loader import TEAM_PROFILES

log = logging.getLogger(__name__)


@dataclass
class MatchupPrediction:
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_winner: str
    confidence: str          # "low" / "medium" / "high"
    key_factors: Dict[str, float]

    def display(self) -> str:
        home_name = TEAM_PROFILES.get(self.home_team, {}).get("name", self.home_team)
        away_name = TEAM_PROFILES.get(self.away_team, {}).get("name", self.away_team)
        winner_name = TEAM_PROFILES.get(self.predicted_winner, {}).get("name", self.predicted_winner)

        bars_home = "█" * int(self.home_win_prob * 30)
        bars_away = "█" * int(self.away_win_prob * 30)

        lines = [
            "",
            "  ┌─────────────────────────────────────────────┐",
            f"  │  ScorecastML Matchup Prediction              │",
            "  ├─────────────────────────────────────────────┤",
            f"  │  {home_name:<20} (HOME)               │",
            f"  │  vs. {away_name:<20} (AWAY)          │",
            "  ├─────────────────────────────────────────────┤",
            f"  │  Home win probability : {self.home_win_prob:.1%}               │",
            f"  │  {bars_home:<30}              │",
            f"  │  Away win probability : {self.away_win_prob:.1%}               │",
            f"  │  {bars_away:<30}              │",
            "  ├─────────────────────────────────────────────┤",
            f"  │  Predicted winner: {winner_name:<26}│",
            f"  │  Confidence: {self.confidence:<33}│",
            "  ├─────────────────────────────────────────────┤",
            "  │  Key differentials (home advantage):         │",
        ]
        for feat, val in list(self.key_factors.items())[:5]:
            clean = feat.replace("DIFF_", "").replace("_", " ").lower()
            direction = "▲" if val > 0 else "▼"
            lines.append(f"  │    {direction} {clean:<28} {val:+.3f}    │")
        lines.append("  └─────────────────────────────────────────────┘")
        return "\n".join(lines)


def _get_confidence(prob: float) -> str:
    if prob >= 0.72 or prob <= 0.28:
        return "high"
    elif prob >= 0.60 or prob <= 0.40:
        return "medium"
    else:
        return "low"


def predict_matchup(
    home_team: str,
    away_team: str,
    team_stats: pd.DataFrame,
    model,
    feature_cols,
) -> MatchupPrediction:
    """
    Build a feature vector for the given matchup using the most
    recent rolling stats for each team, then run inference.
    """
    home_team = home_team.upper()
    away_team = away_team.upper()

    if home_team not in TEAM_PROFILES:
        raise ValueError(f"Unknown team: {home_team}. Use abbreviation e.g. 'BOS'")
    if away_team not in TEAM_PROFILES:
        raise ValueError(f"Unknown team: {away_team}. Use abbreviation e.g. 'MIA'")

    # Get latest stats for each team
    home_stats = (
        team_stats[team_stats["TEAM_ABBREVIATION"] == home_team]
        .sort_values("GAME_DATE")
        .tail(1)
    )
    away_stats = (
        team_stats[team_stats["TEAM_ABBREVIATION"] == away_team]
        .sort_values("GAME_DATE")
        .tail(1)
    )

    if home_stats.empty or away_stats.empty:
        raise ValueError("Insufficient data for one or both teams.")

    home_row = home_stats.iloc[0]
    away_row = away_stats.iloc[0]

    # Build differential feature vector matching training feature names
    feat_vec = {}
    for col in feature_cols:
        raw = col.replace("DIFF_", "")
        h_col = raw
        a_col = raw
        h_val = home_row.get(h_col, 0.0) or 0.0
        a_val = away_row.get(a_col, 0.0) or 0.0
        feat_vec[col] = float(h_val) - float(a_val)

    feat_vec["DIFF_HOME_ADVANTAGE"] = 1.0

    X = np.array([[feat_vec.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    prob = float(model.predict_proba(X)[0, 1])
    winner = home_team if prob >= 0.5 else away_team

    # Key factors: top differentials by absolute magnitude
    diff_importance = {k: v for k, v in feat_vec.items()
                       if k != "DIFF_HOME_ADVANTAGE"}
    top_factors = dict(
        sorted(diff_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    return MatchupPrediction(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=round(prob, 4),
        away_win_prob=round(1 - prob, 4),
        predicted_winner=winner,
        confidence=_get_confidence(prob),
        key_factors=top_factors,
    )


def list_teams() -> None:
    """Print all available team abbreviations."""
    print("\n  Available Teams:")
    print("  " + "-" * 40)
    tiers = {1: "Elite", 2: "Contender", 3: "Fringe", 4: "Rebuilding"}
    for tier_num, tier_label in tiers.items():
        teams = [(abbr, p["name"]) for abbr, p in TEAM_PROFILES.items()
                 if p["tier"] == tier_num]
        print(f"\n  {tier_label}:")
        for abbr, name in sorted(teams):
            print(f"    {abbr:<5} {name}")
    print()
