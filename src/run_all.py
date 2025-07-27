import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.train_cat import cat_main
from src.models.train_lgb import lgb_main
from src.models.train_xgb import xgb_main


def ensemble_predictions(lgb_pred, xgb_pred, cat_pred, weights=None):
    """Ensemble predictions from multiple models"""
    if weights is None:
        weights = [1 / 3, 1 / 3, 1 / 3]  # Equal weights

    ensemble_pred = (
        weights[0] * lgb_pred + weights[1] * xgb_pred + weights[2] * cat_pred
    )
    return ensemble_pred


def evaluate_ensemble(lgb_oof, xgb_oof, cat_oof, weights=None):
    """Evaluate ensemble performance on OOF predictions"""
    target_col = "target"

    if weights is None:
        weights = [1 / 3, 1 / 3, 1 / 3]

    # Ensemble OOF predictions
    ensemble_oof = (
        weights[0] * lgb_oof["oof_pred"]
        + weights[1] * xgb_oof["oof_pred"]
        + weights[2] * cat_oof["oof_pred"]
    )

    # Calculate ensemble score
    ensemble_score = roc_auc_score(lgb_oof[target_col], ensemble_oof)

    return ensemble_score, ensemble_oof


def main():
    parser = argparse.ArgumentParser(description="Run ATMA20 training")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgb", "xgb", "cat", "all"],
        default=["all"],
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--ensemble", action="store_true", help="Create ensemble of all models"
    )

    args = parser.parse_args()

    results = {}

    if "all" in args.models:
        models_to_run = ["lgb", "xgb", "cat"]
    else:
        models_to_run = args.models

    print("=" * 60)
    print("ATMA20 Training Pipeline")
    print("=" * 60)
    print(f"Running models: {models_to_run}")
    print()

    # Run individual models
    if "lgb" in models_to_run:
        print("Running LightGBM...")
        lgb_models, lgb_oof, lgb_pred, lgb_sub = lgb_main()
        results["lgb"] = {
            "models": lgb_models,
            "oof": lgb_oof,
            "pred": lgb_pred,
            "sub": lgb_sub,
        }
        print()

    if "xgb" in models_to_run:
        print("Running XGBoost...")
        xgb_models, xgb_oof, xgb_pred, xgb_sub = xgb_main()
        results["xgb"] = {
            "models": xgb_models,
            "oof": xgb_oof,
            "pred": xgb_pred,
            "sub": xgb_sub,
        }
        print()

    if "cat" in models_to_run:
        print("Running CatBoost...")
        cat_models, cat_oof, cat_pred, cat_sub = cat_main()
        results["cat"] = {
            "models": cat_models,
            "oof": cat_oof,
            "pred": cat_pred,
            "sub": cat_sub,
        }
        print()

    # Create ensemble if requested and multiple models were run
    if args.ensemble and len(results) > 1:
        print("=" * 60)
        print("Creating Ensemble...")
        print("=" * 60)

        # Get individual scores
        individual_scores = {}
        for model_name, result in results.items():
            target_col = "Recommended IND"
            score = roc_auc_score(result["oof"][target_col], result["oof"]["oof_pred"])
            individual_scores[model_name] = score
            print(f"{model_name.upper()} OOF Score: {score:.4f}")

        # Create ensemble predictions
        if len(results) == 3:  # All three models
            lgb_pred = results["lgb"]["pred"]
            xgb_pred = results["xgb"]["pred"]
            cat_pred = results["cat"]["pred"]

            lgb_oof = results["lgb"]["oof"]
            xgb_oof = results["xgb"]["oof"]
            cat_oof = results["cat"]["oof"]

            # Simple average ensemble
            ensemble_pred = ensemble_predictions(lgb_pred, xgb_pred, cat_pred)
            ensemble_score, ensemble_oof = evaluate_ensemble(lgb_oof, xgb_oof, cat_oof)

            print(f"Ensemble OOF Score: {ensemble_score:.4f}")

            # Save ensemble submission
            ensemble_sub = pd.DataFrame({"target": ensemble_pred})
            ensemble_sub.to_csv("ensemble_submission.csv", index=False)

            # Save ensemble OOF
            ensemble_oof_df = lgb_oof.copy()
            ensemble_oof_df["oof_pred"] = ensemble_oof
            ensemble_oof_df.to_csv("ensemble_oof.csv", index=False)

            print("Ensemble results saved: ensemble_submission.csv, ensemble_oof.csv")

    print("=" * 60)
    print("Training Pipeline Completed!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
