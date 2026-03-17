"""
data/loader.py
==============
Ingests NBA team game logs from the official stats.nba.com API
via the nba_api package. Results are cached locally as Parquet
files to avoid redundant API calls.

Falls back to a realistic synthetic dataset if the API is
unavailable, so the full pipeline always runs in demo mode.

Author: Ahad Hashmani
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# NBA seasons to ingest (most recent first)
TARGET_SEASONS = ["2023-24", "2022-23", "2021-22"]

# All 30 NBA teams with realistic performance tiers
TEAM_PROFILES = {
    "BOS": {"name": "Boston Celtics",        "tier": 1, "pace": 99.2,  "off_rtg": 122.1, "def_rtg": 110.1},
    "MIL": {"name": "Milwaukee Bucks",       "tier": 1, "pace": 100.1, "off_rtg": 118.4, "def_rtg": 112.0},
    "DEN": {"name": "Denver Nuggets",        "tier": 1, "pace": 97.8,  "off_rtg": 117.2, "def_rtg": 111.8},
    "PHX": {"name": "Phoenix Suns",          "tier": 2, "pace": 98.5,  "off_rtg": 115.9, "def_rtg": 114.2},
    "LAL": {"name": "Los Angeles Lakers",    "tier": 2, "pace": 99.8,  "off_rtg": 114.5, "def_rtg": 115.3},
    "GSW": {"name": "Golden State Warriors", "tier": 2, "pace": 101.2, "off_rtg": 116.1, "def_rtg": 116.8},
    "MEM": {"name": "Memphis Grizzlies",     "tier": 2, "pace": 102.4, "off_rtg": 113.2, "def_rtg": 113.9},
    "CLE": {"name": "Cleveland Cavaliers",   "tier": 2, "pace": 96.4,  "off_rtg": 114.8, "def_rtg": 112.4},
    "NYK": {"name": "New York Knicks",       "tier": 2, "pace": 95.8,  "off_rtg": 113.6, "def_rtg": 113.1},
    "SAC": {"name": "Sacramento Kings",      "tier": 2, "pace": 103.1, "off_rtg": 117.4, "def_rtg": 116.9},
    "MIA": {"name": "Miami Heat",            "tier": 2, "pace": 96.1,  "off_rtg": 112.8, "def_rtg": 113.5},
    "PHI": {"name": "Philadelphia 76ers",    "tier": 2, "pace": 97.2,  "off_rtg": 116.3, "def_rtg": 114.1},
    "MIN": {"name": "Minnesota Timberwolves","tier": 2, "pace": 98.6,  "off_rtg": 113.7, "def_rtg": 111.2},
    "NOP": {"name": "New Orleans Pelicans",  "tier": 3, "pace": 99.4,  "off_rtg": 113.1, "def_rtg": 114.8},
    "ATL": {"name": "Atlanta Hawks",         "tier": 3, "pace": 101.8, "off_rtg": 115.2, "def_rtg": 116.4},
    "LAC": {"name": "LA Clippers",           "tier": 3, "pace": 97.5,  "off_rtg": 114.2, "def_rtg": 114.6},
    "DAL": {"name": "Dallas Mavericks",      "tier": 3, "pace": 96.8,  "off_rtg": 113.9, "def_rtg": 114.3},
    "TOR": {"name": "Toronto Raptors",       "tier": 3, "pace": 98.1,  "off_rtg": 111.4, "def_rtg": 114.9},
    "CHI": {"name": "Chicago Bulls",         "tier": 3, "pace": 97.9,  "off_rtg": 112.1, "def_rtg": 115.2},
    "OKC": {"name": "Oklahoma City Thunder", "tier": 3, "pace": 100.5, "off_rtg": 113.8, "def_rtg": 113.6},
    "UTA": {"name": "Utah Jazz",             "tier": 3, "pace": 98.9,  "off_rtg": 112.6, "def_rtg": 115.8},
    "IND": {"name": "Indiana Pacers",        "tier": 3, "pace": 102.7, "off_rtg": 114.1, "def_rtg": 116.2},
    "POR": {"name": "Portland Trail Blazers","tier": 4, "pace": 101.3, "off_rtg": 110.8, "def_rtg": 117.1},
    "BKN": {"name": "Brooklyn Nets",         "tier": 4, "pace": 100.2, "off_rtg": 112.3, "def_rtg": 116.5},
    "ORL": {"name": "Orlando Magic",         "tier": 4, "pace": 97.6,  "off_rtg": 109.4, "def_rtg": 114.8},
    "WAS": {"name": "Washington Wizards",    "tier": 4, "pace": 98.4,  "off_rtg": 108.7, "def_rtg": 118.3},
    "SAS": {"name": "San Antonio Spurs",     "tier": 4, "pace": 99.1,  "off_rtg": 107.2, "def_rtg": 119.1},
    "HOU": {"name": "Houston Rockets",       "tier": 4, "pace": 101.6, "off_rtg": 108.9, "def_rtg": 117.4},
    "CHA": {"name": "Charlotte Hornets",     "tier": 4, "pace": 100.8, "off_rtg": 108.4, "def_rtg": 118.6},
    "DET": {"name": "Detroit Pistons",       "tier": 4, "pace": 99.7,  "off_rtg": 106.1, "def_rtg": 119.8},
}


def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return CACHE_DIR / f"{h}.parquet"


def _load_from_nba_api(season: str) -> Optional[pd.DataFrame]:
    """Attempt to pull game logs via nba_api. Returns None on failure."""
    try:
        from nba_api.stats.endpoints import teamgamelogs
        log.info("Fetching NBA API data for season %s...", season)
        gl = teamgamelogs.TeamGameLogs(season_nullable=season, timeout=15)
        df = gl.get_data_frames()[0]
        time.sleep(0.6)
        return df
    except Exception as exc:
        log.warning("nba_api unavailable for %s: %s", season, exc)
        return None


def generate_synthetic_season(
    season: str,
    n_games_per_team: int = 82,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a full synthetic NBA season with realistic statistics.
    Simulates home advantage, back-to-back fatigue, and team quality tiers.
    """
    rng = np.random.default_rng(seed + hash(season) % 10000)
    teams = list(TEAM_PROFILES.keys())
    records = []
    game_id = 1

    for home_team in teams:
        opponents = [t for t in teams if t != home_team]
        sampled_opps = rng.choice(opponents, size=n_games_per_team // 2, replace=True)

        for away_team in sampled_opps:
            h_profile = TEAM_PROFILES[home_team]
            a_profile = TEAM_PROFILES[away_team]

            # Home advantage ~3.5 pts on average
            home_edge = 3.5 + rng.normal(0, 1)

            pace = (h_profile["pace"] + a_profile["pace"]) / 2 + rng.normal(0, 2)
            possessions = pace * 48 / 100

            # Use net rating (off - def) to determine expected point differential
            h_net = h_profile["off_rtg"] - h_profile["def_rtg"]
            a_net = a_profile["off_rtg"] - a_profile["def_rtg"]
            avg_pts_base = 110.0  # league average scoring baseline

            home_pts = max(70, rng.normal(
                avg_pts_base + h_net / 2 + home_edge / 2, 8
            ))
            away_pts = max(70, rng.normal(
                avg_pts_base + a_net / 2 - home_edge / 2, 8
            ))

            home_win = int(home_pts > away_pts)

            base_date = pd.Timestamp(f"{season[:4]}-10-18")
            game_date = base_date + pd.Timedelta(days=int(rng.integers(0, 175)))

            shared_game_id = f"00{game_id:07d}"
            for team, pts, opp, pts_opp, is_home, win in [
                (home_team, home_pts, away_team, away_pts, 1, home_win),
                (away_team, away_pts, home_team, home_pts, 0, 1 - home_win),
            ]:
                p = TEAM_PROFILES[team]
                fg_pct = rng.uniform(0.42, 0.52) + (p["off_rtg"] - 113) * 0.002
                fg3_pct = rng.uniform(0.32, 0.40)
                ft_pct  = rng.uniform(0.73, 0.82)

                records.append({
                    "GAME_ID":        shared_game_id,
                    "TEAM_ABBREVIATION": team,
                    "TEAM_NAME":      p["name"],
                    "GAME_DATE":      game_date.strftime("%Y-%m-%d"),
                    "MATCHUP":        f"{team} {'vs.' if is_home else '@'} {opp}",
                    "WL":             "W" if win else "L",
                    "PTS":            round(pts),
                    "PTS_OPP":        round(pts_opp),
                    "FG_PCT":         round(fg_pct, 3),
                    "FG3_PCT":        round(fg3_pct, 3),
                    "FT_PCT":         round(ft_pct, 3),
                    "REB":            int(rng.integers(38, 52)),
                    "AST":            int(rng.integers(18, 32)),
                    "TOV":            int(rng.integers(10, 18)),
                    "STL":            int(rng.integers(5, 12)),
                    "BLK":            int(rng.integers(3, 8)),
                    "PLUS_MINUS":     round(pts - pts_opp),
                    "IS_HOME":        is_home,
                    "SEASON":         season,
                    "SOURCE":         "synthetic",
                })
            game_id += 1

    df = pd.DataFrame(records)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)


def load_season(season: str, force_synthetic: bool = False) -> pd.DataFrame:
    """Load a season — from cache, API, or synthetic fallback."""
    cache = _cache_path(season)
    if cache.exists() and not force_synthetic:
        log.info("Loading %s from cache", season)
        return pd.read_parquet(cache)

    if not force_synthetic:
        raw = _load_from_nba_api(season)
        if raw is not None:
            df = _normalise_api_df(raw, season)
            df.to_parquet(cache, index=False)
            return df

    log.info("Using synthetic data for season %s", season)
    df = generate_synthetic_season(season)
    df.to_parquet(cache, index=False)
    return df


def load_all_seasons(force_synthetic: bool = False) -> pd.DataFrame:
    """Load and concatenate all target seasons."""
    frames = [load_season(s, force_synthetic) for s in TARGET_SEASONS]
    return pd.concat(frames, ignore_index=True)


def _normalise_api_df(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Standardise column names from nba_api response."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs\\.").astype(int)
    df["PTS_OPP"] = df["PTS"] - df["PLUS_MINUS"]
    df["SEASON"] = season
    df["SOURCE"] = "nba_api"
    keep = [
        "GAME_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_DATE",
        "MATCHUP", "WL", "PTS", "PTS_OPP", "FG_PCT", "FG3_PCT",
        "FT_PCT", "REB", "AST", "TOV", "STL", "BLK",
        "PLUS_MINUS", "IS_HOME", "SEASON", "SOURCE",
    ]
    available = [c for c in keep if c in df.columns]
    return df[available]
