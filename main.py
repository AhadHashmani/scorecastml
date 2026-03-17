"""
main.py
=======
ScorecastML — NBA Game Outcome Prediction Engine

Commands
--------
Train models and run full evaluation:
    python main.py train

Predict a specific matchup:
    python main.py predict --home BOS --away MIA

List available team abbreviations:
    python main.py teams

Author: Ahad Hashmani
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scorecastml")


def cmd_train(args) -> None:
    from data.loader import load_all_seasons
    from features.engineer import compute_team_rolling_stats, build_matchup_features, get_feature_columns
    from models.train import train_and_evaluate
    from evaluation.metrics import run_full_evaluation

    log.info("Loading game data...")
    raw = load_all_seasons(force_synthetic=args.synthetic)
    log.info("Loaded %d game records across %d teams", len(raw), raw["TEAM_ABBREVIATION"].nunique())

    log.info("Computing rolling features...")
    team_stats = compute_team_rolling_stats(raw)

    log.info("Building matchup feature matrix...")
    matchup_df = build_matchup_features(team_stats)
    feature_cols = get_feature_columns(matchup_df)
    log.info("Feature matrix: %d matchups × %d features", len(matchup_df), len(feature_cols))

    log.info("Training models...")
    lr_result, gbt_result = train_and_evaluate(matchup_df, feature_cols)

    run_full_evaluation([lr_result, gbt_result], matchup_df, feature_cols)


def cmd_predict(args) -> None:
    from data.loader import load_all_seasons
    from features.engineer import compute_team_rolling_stats, build_matchup_features, get_feature_columns
    from models.train import load_model
    from predict.matchup import predict_matchup

    try:
        model, feature_cols = load_model("gbt")
        log.info("Loaded saved GBT model")
    except FileNotFoundError:
        log.warning("No saved model found — training now...")
        raw = load_all_seasons(force_synthetic=True)
        team_stats = compute_team_rolling_stats(raw)
        matchup_df = build_matchup_features(team_stats)
        feature_cols = get_feature_columns(matchup_df)
        from models.train import train_and_evaluate
        _, gbt_result = train_and_evaluate(matchup_df, feature_cols)
        model = gbt_result.model

    raw = load_all_seasons(force_synthetic=True)
    team_stats = compute_team_rolling_stats(raw)

    pred = predict_matchup(args.home, args.away, team_stats, model, feature_cols)
    print(pred.display())


def cmd_teams(args) -> None:
    from predict.matchup import list_teams
    list_teams()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScorecastML — NBA Game Outcome Prediction Engine"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train models and run evaluation")
    p_train.add_argument("--synthetic", action="store_true",
                         help="Force use of synthetic data (skip NBA API)")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = sub.add_parser("predict", help="Predict a matchup")
    p_pred.add_argument("--home", required=True, help="Home team abbreviation (e.g. BOS)")
    p_pred.add_argument("--away", required=True, help="Away team abbreviation (e.g. MIA)")
    p_pred.set_defaults(func=cmd_predict)

    # teams
    p_teams = sub.add_parser("teams", help="List all team abbreviations")
    p_teams.set_defaults(func=cmd_teams)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
