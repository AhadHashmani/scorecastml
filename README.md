# ScorecastML — NBA Game Outcome Prediction Engine

A full machine learning pipeline that predicts NBA game outcomes using rolling team performance features, time-series cross-validation, and two-model comparison (Logistic Regression vs. Gradient Boosted Trees).

Built as a data science portfolio project demonstrating end-to-end ML engineering: data ingestion, feature engineering, model training, evaluation with calibration curves, and a CLI prediction tool.

---

## Architecture

```
scorecastml/
├── data/
│   └── loader.py          NBA API ingestion (nba_api) with Parquet caching + synthetic fallback
├── features/
│   └── engineer.py        Rolling stats (5/10/20 game windows), matchup differentials
├── models/
│   ├── train.py           Logistic Regression + Gradient Boosting, time-based split, model serialisation
│   └── saved/             Serialised .pkl model files (created at runtime)
├── evaluation/
│   └── metrics.py         ROC curves, calibration curves, confusion matrix, feature importance
├── predict/
│   └── matchup.py         CLI matchup predictor with confidence bands
├── tests/
│   └── test_scorecast.py  Unit tests (pytest)
├── cache/                 Parquet cache for API responses
└── main.py                Entry point
```

---

## Quickstart

```bash
git clone https://github.com/ahadhashmani/scorecastml
cd scorecastml
pip install -r requirements.txt

# Train both models and generate evaluation plots
python main.py train

# Force synthetic data (no NBA API needed)
python main.py train --synthetic

# Predict a specific matchup
python main.py predict --home BOS --away MIA

# List all 30 team abbreviations
python main.py teams

# Run tests
python -m pytest tests/ -v
```

---

## Features

For every game, rolling statistics are computed over the **preceding** 5, 10, and 20 games (never including the current game — no data leakage). Home and away stats are then differenced to produce the model's input features.

| Feature Group | Windows | Description |
|---|---|---|
| Win rate | 5, 10, 20 | Rolling win percentage |
| Net points | 5, 10, 20 | Rolling pts scored minus pts allowed |
| Plus/minus | 5, 10, 20 | Rolling game point differential |
| FG%, 3P%, TS% | 5, 10, 20 | Shooting efficiency |
| Turnover rate | 5, 10, 20 | Avg turnovers per game |
| Rest days | — | Days since last game |
| Back-to-back flag | — | Whether team played previous night |
| Home advantage | — | Constant home court indicator |

All features are **differentials** (home team value − away team value), so the model learns what separates winning teams from losing teams rather than memorising team identities.

---

## Models

**Logistic Regression** — interpretable baseline with L2 regularisation and StandardScaler preprocessing. Coefficient magnitudes indicate feature importance.

**Gradient Boosted Trees** — 300 estimators, learning rate 0.05, max depth 4, with early stopping. Captures non-linear interactions between features.

Both models are evaluated with:
- Accuracy
- ROC-AUC
- Log-Loss
- Brier Score (calibration quality)
- Calibration curve (reliability diagram)

---

## Train/Test Split

Time-based split — the most recent season is always held out as the test set. Training uses only games that occurred before the test games, eliminating any possibility of data leakage.

```
Train: 2021-22, 2022-23  →  Test: 2023-24
```

---

## Demo Mode

If `nba_api` is unavailable or rate-limited, the system automatically falls back to a synthetic dataset that replicates realistic NBA statistics: team quality tiers, home court advantage (~62% home win rate), seasonal variation, and back-to-back fatigue. All pipeline logic runs identically.

---

## Sample Output

```
  ┌─────────────────────────────────────────────┐
  │  ScorecastML Matchup Prediction              │
  ├─────────────────────────────────────────────┤
  │  Boston Celtics       (HOME)                │
  │  vs. Miami Heat       (AWAY)                │
  ├─────────────────────────────────────────────┤
  │  Home win probability : 68.6%               │
  │  ████████████████████                        │
  │  Away win probability : 31.4%               │
  │  █████████                                   │
  ├─────────────────────────────────────────────┤
  │  Predicted winner: Boston Celtics            │
  │  Confidence: medium                          │
  ├─────────────────────────────────────────────┤
  │  Key differentials (home advantage):         │
  │    ▲ net pts 5                    +17.8      │
  │    ▲ net pts 10                   +17.7      │
  │    ▲ plus minus 5                 +17.6      │
  └─────────────────────────────────────────────┘
```

---

## Stack

Python 3.10+ · scikit-learn · pandas · NumPy · matplotlib · nba_api · pyarrow · pytest

---

## Author

Ahad Hashmani · [linkedin.com/in/ahadhashmani](https://linkedin.com/in/ahadhashmani) · UT Austin
