"""
features/engineer.py
====================
Transforms raw game logs into a matchup feature matrix.

For every game, we compute rolling statistics for BOTH teams
over the preceding N games, then take differentials. This gives
the model a view of current form rather than season averages.

Features generated per team (then differenced home - away):
    - Rolling win rate (5, 10, 20 games)
    - Rolling points scored / allowed
    - Rolling net rating (pts - pts_opp per 100 poss proxy)
    - Rolling shooting efficiency (FG%, 3P%, TS%)
    - Rolling turnover rate
    - Rolling rebound margin
    - Rest days since last game
    - Back-to-back flag
    - Home/away record split

Author: Ahad Hashmani
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROLL_WINDOWS = [5, 10, 20]


def _true_shooting(pts: pd.Series, fga: pd.Series, fta: pd.Series) -> pd.Series:
    """TS% = PTS / (2 * (FGA + 0.44 * FTA))"""
    denom = 2 * (fga + 0.44 * fta)
    return (pts / denom.replace(0, np.nan)).fillna(0)


def compute_team_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, sort by date and compute rolling features.
    Returns a DataFrame with one row per (team, game) with all
    rolling statistics attached.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN"] = (df["WL"] == "W").astype(int)

    all_teams = []

    for team, group in df.groupby("TEAM_ABBREVIATION"):
        g = group.sort_values("GAME_DATE").copy()
        g = g.reset_index(drop=True)

        # Rest days
        g["DAYS_REST"] = g["GAME_DATE"].diff().dt.days.fillna(3).clip(upper=7)
        g["IS_B2B"] = (g["DAYS_REST"] <= 1).astype(int)

        # Approximate FGA from FG_PCT (pts * 0.42 as rough proxy since FGA not always in game logs)
        g["FGA_PROXY"] = (g["PTS"] * 0.42).round()
        g["FTA_PROXY"] = (g["PTS"] * 0.22).round()
        g["TS_PCT"] = _true_shooting(g["PTS"], g["FGA_PROXY"], g["FTA_PROXY"])

        for w in ROLL_WINDOWS:
            shift = g.shift(1)   # never include current game in its own rolling window
            roll = shift.rolling(w, min_periods=max(1, w // 3))

            g[f"WIN_RATE_{w}"]     = roll["WIN"].mean().round(4)
            g[f"PTS_AVG_{w}"]      = roll["PTS"].mean().round(2)
            g[f"PTS_OPP_AVG_{w}"]  = roll["PTS_OPP"].mean().round(2)
            g[f"NET_PTS_{w}"]      = (g[f"PTS_AVG_{w}"] - g[f"PTS_OPP_AVG_{w}"]).round(2)
            g[f"FG_PCT_AVG_{w}"]   = roll["FG_PCT"].mean().round(4)
            g[f"FG3_PCT_AVG_{w}"]  = roll["FG3_PCT"].mean().round(4)
            g[f"TS_PCT_AVG_{w}"]   = roll["TS_PCT"].mean().round(4)
            g[f"TOV_AVG_{w}"]      = roll["TOV"].mean().round(2)
            g[f"REB_AVG_{w}"]      = roll["REB"].mean().round(2)
            g[f"AST_AVG_{w}"]      = roll["AST"].mean().round(2)
            g[f"PLUS_MINUS_{w}"]   = roll["PLUS_MINUS"].mean().round(2)

        # Home/away win rate (last 20 home or away games)
        home_games = g[g["IS_HOME"] == 1].copy()
        away_games = g[g["IS_HOME"] == 0].copy()

        home_wr = home_games["WIN"].shift(1).rolling(20, min_periods=1).mean()
        away_wr = away_games["WIN"].shift(1).rolling(20, min_periods=1).mean()

        g["HOME_WIN_RATE"] = np.nan
        g["AWAY_WIN_RATE"] = np.nan
        g.loc[home_games.index, "HOME_WIN_RATE"] = home_wr.values
        g.loc[away_games.index, "AWAY_WIN_RATE"] = away_wr.values

        g["HOME_WIN_RATE"] = g["HOME_WIN_RATE"].fillna(g["WIN_RATE_10"])
        g["AWAY_WIN_RATE"] = g["AWAY_WIN_RATE"].fillna(g["WIN_RATE_10"])

        all_teams.append(g)

    return pd.concat(all_teams, ignore_index=True)


def build_matchup_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, pair the home and away team stats and compute
    differential features (home - away).

    Returns one row per game with:
        - GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM
        - Differential features (prefix DIFF_)
        - Raw home/away features (prefix H_ and A_)
        - Target: HOME_WIN (1 = home team won)
    """
    ts = team_stats.copy()

    home_df = ts[ts["IS_HOME"] == 1].copy()
    away_df = ts[ts["IS_HOME"] == 0].copy()

    # Merge home and away on GAME_ID
    home_df = home_df.rename(columns=lambda c: f"H_{c}" if c not in ["GAME_ID", "GAME_DATE", "SEASON"] else c)
    away_df = away_df.rename(columns=lambda c: f"A_{c}" if c not in ["GAME_ID", "GAME_DATE", "SEASON"] else c)

    matchups = pd.merge(
        home_df, away_df,
        on=["GAME_ID", "GAME_DATE", "SEASON"],
        how="inner",
        suffixes=("", "_dup"),
    )

    # Drop any duplicate columns from merge
    matchups = matchups[[c for c in matchups.columns if not c.endswith("_dup")]]

    matchups["HOME_WIN"] = (matchups["H_WL"] == "W").astype(int)

    # Differential features
    diff_features = [
        f"WIN_RATE_{w}" for w in ROLL_WINDOWS
    ] + [
        f"PTS_AVG_{w}" for w in ROLL_WINDOWS
    ] + [
        f"NET_PTS_{w}" for w in ROLL_WINDOWS
    ] + [
        f"FG_PCT_AVG_{w}" for w in ROLL_WINDOWS
    ] + [
        f"FG3_PCT_AVG_{w}" for w in ROLL_WINDOWS
    ] + [
        f"TS_PCT_AVG_{w}" for w in ROLL_WINDOWS
    ] + [
        f"TOV_AVG_{w}" for w in ROLL_WINDOWS
    ] + [
        f"PLUS_MINUS_{w}" for w in ROLL_WINDOWS
    ] + ["DAYS_REST", "IS_B2B"]

    for feat in diff_features:
        h_col = f"H_{feat}"
        a_col = f"A_{feat}"
        if h_col in matchups.columns and a_col in matchups.columns:
            matchups[f"DIFF_{feat}"] = (matchups[h_col] - matchups[a_col]).round(4)

    matchups["DIFF_HOME_ADVANTAGE"] = 1.0   # constant home court flag

    diff_cols = [c for c in matchups.columns if c.startswith("DIFF_")]
    meta_cols = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "H_TEAM_ABBREVIATION", "A_TEAM_ABBREVIATION",
        "H_IS_B2B", "A_IS_B2B",
        "HOME_WIN",
    ]

    return matchups[meta_cols + diff_cols].dropna(subset=diff_cols).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all DIFF_ columns for use as model features."""
    return [c for c in df.columns if c.startswith("DIFF_")]
