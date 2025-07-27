#!/usr/bin/env python3
"""
exp006 Multi-Seed Ensemble Script
Combines all exp006 experiments with specified weights:
- LightGBM: 0.7
- CatBoost: 0.2
- XGBoost: 0.1
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    print("=== exp006 Multi-Seed Ensemble ===")
    print("Combining all exp006 experiments with weighted average")
    print("Weights: LGB=0.7, CAT=0.2, XGB=0.1")
    print("")

    # Configuration
    working_dir = Path("/kaggle/working")
    output_dir = working_dir / "exp007_ensemble_weighted"
    output_dir.mkdir(exist_ok=True)

    # Model weights
    weights = {"lgb": 0.7, "cat": 0.2, "xgb": 0.1}

    seeds = [42, 510, 111]

    # Find all exp006 experiments
    exp_dirs = []
    for seed in seeds:
        for model in ["lgb", "cat", "xgb"]:
            exp_dir = working_dir / f"exp007_seed{seed}_{model}"
            if exp_dir.exists():
                exp_dirs.append(
                    {
                        "path": exp_dir,
                        "seed": seed,
                        "model": model,
                        "weight": weights[model],
                    }
                )
                print(f"✓ Found: {exp_dir.name}")
            else:
                print(f"✗ Missing: exp006_seed{seed}_{model}_{model}")

    print(f"\nFound {len(exp_dirs)} experiments")

    # Load and combine submissions
    print("\nLoading submissions and OOF predictions...")

    # Simple lists to track experiments
    experiments = []
    for exp in exp_dirs:
        experiments.append(exp)
        print(f"  ✓ Will process: {exp['path'].name}")

    if not experiments:
        print("❌ No experiments found!")
        return

    # Combine submissions with weights
    print(f"\nCombining {len(experiments)} submission files...")

    # Initialize the final submission with proper index
    final_submission = pd.DataFrame()
    final_submission["target"] = 0.0

    # Add weighted predictions
    for exp in exp_dirs:
        sub_file = exp["path"] / "submission.csv"
        if sub_file.exists():
            sub_df = pd.read_csv(sub_file)
            if final_submission.empty:
                final_submission = sub_df.copy()
                final_submission["target"] = sub_df["target"] * exp["weight"]
            else:
                final_submission["target"] += sub_df["target"] * exp["weight"]
            print(f"  ✓ Added {exp['path'].name} with weight {exp['weight']}")

    print(f"Final submission shape: {final_submission.shape}")
    print(
        f"Target range: {final_submission['target'].min():.6f} - {final_submission['target'].max():.6f}"
    )

    # Combine OOF predictions
    print(f"\nCombining {len(experiments)} OOF files...")

    # Initialize the final OOF
    final_oof = pd.DataFrame()

    # Add weighted OOF predictions
    for exp in exp_dirs:
        oof_file = exp["path"] / f"oof_{exp['model']}.csv"
        if oof_file.exists():
            oof_df = pd.read_csv(oof_file)
            if final_oof.empty:
                final_oof = oof_df.copy()
                # Use oof_pred column if available, otherwise use target
                pred_col = "oof_pred" if "oof_pred" in oof_df.columns else "target"
                final_oof["target"] = oof_df[pred_col] * exp["weight"]
            else:
                pred_col = "oof_pred" if "oof_pred" in oof_df.columns else "target"
                final_oof["target"] += oof_df[pred_col] * exp["weight"]
            print(f"  ✓ Added {exp['path'].name} with weight {exp['weight']}")

    print(f"Final OOF shape: {final_oof.shape}")
    print(
        f"OOF target range: {final_oof['target'].min():.6f} - {final_oof['target'].max():.6f}"
    )

    # Save results
    submission_file = output_dir / "submission.csv"
    oof_file = output_dir / "oof_ensemble.csv"

    final_submission.to_csv(submission_file, index=False)
    final_oof.to_csv(oof_file, index=False)

    print(f"\n✅ Results saved:")
    print(f"  📁 Output directory: {output_dir}")
    print(f"  📄 Submission: {submission_file}")
    print(f"  📄 OOF predictions: {oof_file}")

    # Create experiment metadata
    metadata_file = output_dir / "ensemble_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"exp006 Multi-Seed Weighted Ensemble\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(exp_dirs)}\n")
        f.write(f"Weights: LGB=0.6, CAT=0.3, XGB=0.1\n\n")
        f.write("Experiments included:\n")
        for exp in exp_dirs:
            f.write(f"  - {exp['path'].name}: weight={exp['weight']}\n")
        f.write(f"\nFinal submission statistics:\n")
        f.write(f"  Shape: {final_submission.shape}\n")
        f.write(
            f"  Target range: {final_submission['target'].min():.6f} - {final_submission['target'].max():.6f}\n"
        )
        f.write(f"  Target mean: {final_submission['target'].mean():.6f}\n")
        f.write(f"  Target std: {final_submission['target'].std():.6f}\n")

    print(f"  📄 Metadata: {metadata_file}")

    # Show weight verification
    print(f"\n📊 Weight verification:")
    total_weight = sum(weights.values())
    print(f"  Total weight: {total_weight}")
    for model, weight in weights.items():
        count = len([e for e in exp_dirs if e["model"] == model])
        print(f"  {model.upper()}: {weight} (from {count} experiments)")

    print(f"\n🎉 exp006 Multi-Seed Ensemble completed successfully!")


if __name__ == "__main__":
    main()
