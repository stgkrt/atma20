import argparse
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    CatParams,
    KaggleConfig,
    LGBParams,
    XGBParams,
    get_cat_parser,
    get_lgb_parser,
    get_xgb_parser,
)
from src.data_utils import load_and_prepare_data, save_results
from src.model_utils import make_prediction
from src.models.train_cat import fit_cat
from src.models.train_lgb import fit_lgbm
from src.models.train_xgb import fit_xgb


def run_lgb_model(args):
    """Run LightGBM model"""
    print("ATMA20 LightGBM Model")
    print("=" * 50)

    # Use Kaggle config with experiment name
    config = KaggleConfig.from_args(args)
    config.EXP_NAME = f"{args.exp_name}_lgb"
    config.OUTPUT_DIR = config.OUTPUT_DIR.parent / config.EXP_NAME

    print(f"Experiment: {config.EXP_NAME}")
    print(f"Output directory: {config.OUTPUT_DIR}")

    # Load and prepare data
    train_df, test_df = load_and_prepare_data(config)

    print(f"Final train shape: {train_df.shape}")
    print(f"Final test shape: {test_df.shape}")

    # Get LightGBM parameters
    params = LGBParams.get_params(args)

    # Train model
    oof_df, target, models, evals_results, feat_cols = fit_lgbm(
        train_df, config, params
    )

    # Make predictions
    pred = make_prediction(models, test_df, feat_cols, model_type="lgb")

    # Save results
    save_results(pred, oof_df, config, "lgb")

    print(f"\nLightGBM training completed! Results saved to {config.OUTPUT_DIR}")


def run_xgb_model(args):
    """Run XGBoost model"""
    print("ATMA20 XGBoost Model")
    print("=" * 50)

    # Use Kaggle config with experiment name
    config = KaggleConfig.from_args(args)
    config.EXP_NAME = f"{args.exp_name}_xgb"
    config.OUTPUT_DIR = config.OUTPUT_DIR.parent / config.EXP_NAME

    print(f"Experiment: {config.EXP_NAME}")
    print(f"Output directory: {config.OUTPUT_DIR}")

    # Load and prepare data
    train_df, test_df = load_and_prepare_data(config)

    print(f"Final train shape: {train_df.shape}")
    print(f"Final test shape: {test_df.shape}")

    # Get XGBoost parameters
    params = XGBParams.get_params(args)

    # Train model
    oof_df, target, models, evals_results, feat_cols = fit_xgb(train_df, config, params)

    # Make predictions
    pred = make_prediction(models, test_df, feat_cols, model_type="xgb")

    # Save results
    save_results(pred, oof_df, config, "xgb")

    print(f"\nXGBoost training completed! Results saved to {config.OUTPUT_DIR}")


def run_cat_model(args):
    """Run CatBoost model"""
    print("ATMA20 CatBoost Model")
    print("=" * 50)

    # Use Kaggle config with experiment name
    config = KaggleConfig.from_args(args)
    config.EXP_NAME = f"{args.exp_name}_cat"
    config.OUTPUT_DIR = config.OUTPUT_DIR.parent / config.EXP_NAME

    print(f"Experiment: {config.EXP_NAME}")
    print(f"Output directory: {config.OUTPUT_DIR}")

    # Load and prepare data
    train_df, test_df = load_and_prepare_data(config)

    print(f"Final train shape: {train_df.shape}")
    print(f"Final test shape: {test_df.shape}")

    # Get CatBoost parameters
    params = CatParams.get_params(args)

    # Train model
    oof_df, target, models, evals_results, feat_cols = fit_cat(train_df, config, params)

    # Make predictions
    pred = make_prediction(models, test_df, feat_cols, model_type="cat")

    # Save results
    save_results(pred, oof_df, config, "cat")

    print(f"\nCatBoost training completed! Results saved to {config.OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Run individual ATMA20 models")
    parser.add_argument("model", choices=["lgb", "xgb", "cat"], help="Model to run")

    # Add common arguments
    parser.add_argument(
        "--exp-name",
        type=str,
        default="experiment",
        help="Experiment name (will be suffixed with model name)",
    )

    # Parse known args to get the model name, then get the appropriate parser
    temp_args, _ = parser.parse_known_args()

    # Get the appropriate parser based on model type
    if temp_args.model == "lgb":
        parser = get_lgb_parser()
    elif temp_args.model == "xgb":
        parser = get_xgb_parser()
    elif temp_args.model == "cat":
        parser = get_cat_parser()

    # Parse all arguments
    args = parser.parse_args()

    # Run the appropriate model
    if args.model == "lgb":
        run_lgb_model(args)
    elif args.model == "xgb":
        run_xgb_model(args)
    elif args.model == "cat":
        run_cat_model(args)


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
