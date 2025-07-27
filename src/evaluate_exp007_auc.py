#!/usr/bin/env python3
"""
Calculate AUC scores for exp007 experiments and ensemble
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_auc_for_experiment(exp_dir, model_type):
    """Calculate AUC score for a single experiment"""
    oof_file = exp_dir / f"oof_{model_type}.csv"
    print(f"Calculating AUC for {oof_file}")
    if not oof_file.exists():
        return None

    try:
        oof_df = pd.read_csv(oof_file)

        # Load training data to get true labels
        train_df = pd.read_csv("/kaggle/input/atma20/train.csv")

        # Merge with train data to get true target values
        merged = oof_df.merge(
            train_df[["id", "target"]], left_index=True, right_on="id", how="left"
        )

        if "target_y" in merged.columns and "oof_pred" in merged.columns:
            # Use true target and oof predictions
            y_true = merged["target_y"]
            y_pred = merged["oof_pred"]
        elif "target_y" in merged.columns and "target_x" in merged.columns:
            # target_x is prediction, target_y is true
            y_true = merged["target_y"]
            y_pred = merged["target_x"]
        else:
            print(f"Warning: Could not find appropriate columns in {oof_file}")
            return None

        # Remove any NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return None

        auc_score = roc_auc_score(y_true, y_pred)
        return auc_score

    except Exception as e:
        print(f"Error calculating AUC for {exp_dir}: {e}")
        return None


def main():
    print("=== exp007 AUC Score Evaluation ===")
    print("")

    working_dir = Path("/kaggle/working")
    seeds = [42, 510, 111]
    models = ["lgb", "cat", "xgb"]

    scores = {}

    # Calculate individual experiment scores
    print("Individual experiment AUC scores:")
    for seed in seeds:
        for model in models:
            exp_dir = working_dir / f"exp007_seed{seed}_{model}_{model}"
            if exp_dir.exists():
                auc = calculate_auc_for_experiment(exp_dir, model)
                if auc is not None:
                    exp_name = f"seed{seed}_{model}"
                    scores[exp_name] = auc
                    print(f"  {exp_name}: {auc:.6f}")
                else:
                    print(f"  {exp_name}: Could not calculate AUC")
            else:
                print(f"  seed{seed}_{model}: Directory not found")

    # Calculate ensemble AUC
    print(f"\nEnsemble AUC score:")
    ensemble_dir = working_dir / "exp007_ensemble_weighted"
    if ensemble_dir.exists():
        try:
            # Load ensemble OOF
            oof_ensemble = pd.read_csv(ensemble_dir / "oof_ensemble.csv")

            # Load training data
            train_df = pd.read_csv("/kaggle/input/atma20/train.csv")

            # Merge to get true labels
            merged = oof_ensemble.merge(
                train_df[["id", "target"]], left_index=True, right_on="id", how="left"
            )

            if "target_y" in merged.columns and "target_x" in merged.columns:
                y_true = merged["target_y"]
                y_pred = merged["target_x"]

                # Remove NaN values
                mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                y_true = y_true[mask]
                y_pred = y_pred[mask]

                if len(y_true) > 0:
                    ensemble_auc = roc_auc_score(y_true, y_pred)
                    print(f"  Ensemble (LGB:0.7, CAT:0.2, XGB:0.1): {ensemble_auc:.6f}")
                else:
                    print("  Ensemble: No valid data for AUC calculation")
            else:
                print("  Ensemble: Could not find appropriate columns")
        except Exception as e:
            print(f"  Ensemble: Error calculating AUC - {e}")
    else:
        print("  Ensemble: Directory not found")

    # Summary statistics
    if scores:
        print(f"\nSummary by model type:")
        for model in models:
            model_scores = [score for name, score in scores.items() if model in name]
            if model_scores:
                print(
                    f"  {model.upper()}: mean={np.mean(model_scores):.6f}, std={np.std(model_scores):.6f}, range=[{min(model_scores):.6f}, {max(model_scores):.6f}]"
                )

        print(f"\nOverall statistics:")
        all_scores = list(scores.values())
        print(
            f"  All experiments: mean={np.mean(all_scores):.6f}, std={np.std(all_scores):.6f}"
        )
        print(f"  Best individual: {max(all_scores):.6f}")
        print(f"  Worst individual: {min(all_scores):.6f}")


if __name__ == "__main__":
    main()
