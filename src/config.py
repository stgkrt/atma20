"""
Configuration file for ATMA20 project
Supports argparse for parameter modification
"""

import argparse
from pathlib import Path


def get_base_parser():
    """Get base argument parser with common arguments"""
    parser = argparse.ArgumentParser(description="ATMA20 Training Script")

    # Data paths
    parser.add_argument("--train-csv", type=str, help="Path to training CSV file")
    parser.add_argument("--test-csv", type=str, help="Path to test CSV file")
    parser.add_argument(
        "--sample-sub-csv", type=str, help="Path to sample submission CSV file"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    # Training parameters
    parser.add_argument(
        "--n-estimators", type=int, default=10000, help="Number of estimators"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--random-state", type=int, default=510, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=50, help="Early stopping rounds"
    )
    parser.add_argument(
        "--verbose-eval", type=int, default=30, help="Verbose evaluation frequency"
    )

    # Cross-validation
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of cross-validation folds"
    )

    # Experiment settings
    parser.add_argument(
        "--exp-name", type=str, default="experiment", help="Experiment name"
    )
    parser.add_argument(
        "--exp-category", type=str, default="baseline", help="Experiment category"
    )

    return parser


# Base configuration
class BaseConfig:
    # Project settings
    EXP_CATEGORY = "baseline"

    # Local development paths
    BASE_DIR = Path("/kaggle")
    INPUT_DIR = BASE_DIR / "input" / "atma20"
    TRAIN_CSV = INPUT_DIR / "train.csv"
    TEST_CSV = INPUT_DIR / "test.csv"
    SAMPLE_SUB_CSV = INPUT_DIR / "sample_submission.csv"

    # Output directory
    OUTPUT_DIR = BASE_DIR / "working"

    # Data columns
    target_col = "target"
    unused_col = [target_col, "社員番号", "category", "fold"]
    threshold = 0.5

    # Cross-validation settings
    FOLDS = [0, 1, 2, 3, 4]

    # Training settings
    num_boost_round = 1000
    early_stopping_rounds = 50
    verbose_eval = 30

    @classmethod
    def from_args(cls, args):
        """Create config from parsed arguments"""
        config = cls()

        # Update with command line arguments if provided
        if args.train_csv:
            config.TRAIN_CSV = args.train_csv
        if args.test_csv:
            config.TEST_CSV = args.test_csv
        if args.sample_sub_csv:
            config.SAMPLE_SUB_CSV = args.sample_sub_csv
        if args.output_dir:
            config.OUTPUT_DIR = Path(args.output_dir)

        # Set experiment name and create experiment-specific output directory
        config.EXP_NAME = args.exp_name
        config.EXP_CATEGORY = args.exp_category
        config.OUTPUT_DIR = config.OUTPUT_DIR / args.exp_name

        config.early_stopping_rounds = args.early_stopping_rounds
        config.verbose_eval = args.verbose_eval
        config.FOLDS = list(range(args.n_folds))

        return config


class KaggleConfig(BaseConfig):
    """Configuration for Kaggle environment"""

    BASE_DIR = Path("/kaggle")
    INPUT_DIR = Path("/kaggle/input/atma20")
    TRAIN_CSV = Path("/kaggle/input/atma20/train.csv")
    TEST_CSV = Path("/kaggle/input/atma20/test.csv")
    SAMPLE_SUB_CSV = Path("/kaggle/input/atma20/sample_submission.csv")
    OUTPUT_DIR = Path("/kaggle/working")

    @classmethod
    def from_args(cls, args):
        """Create config from arguments for Kaggle environment"""
        config = super().from_args(args)
        # Override paths for Kaggle environment
        config.BASE_DIR = Path("/kaggle")
        config.INPUT_DIR = Path("/kaggle/input/atma20")
        config.TRAIN_CSV = Path("/kaggle/input/atma20/train.csv")
        config.TEST_CSV = Path("/kaggle/input/atma20/test.csv")
        config.SAMPLE_SUB_CSV = Path("/kaggle/input/atma20/sample_submission.csv")
        # Keep the experiment-specific output directory
        config.OUTPUT_DIR = Path("/kaggle/working") / args.exp_name
        return config


class LocalConfig(BaseConfig):
    """Configuration for local development"""

    pass  # Uses BaseConfig settings


# Model-specific parameters
def get_lgb_parser():
    """Get LightGBM specific argument parser"""
    parser = get_base_parser()

    # Model selection
    parser.add_argument(
        "model",
        nargs="?",
        default="lgb",
        choices=["lgb", "xgb", "cat"],
        help="Model to run",
    )

    # LightGBM specific parameters
    parser.add_argument(
        "--lgb-max-depth", type=int, default=6, help="LightGBM max depth"
    )
    parser.add_argument(
        "--lgb-num-leaves", type=int, default=128, help="LightGBM number of leaves"
    )
    parser.add_argument(
        "--lgb-feature-fraction",
        type=float,
        default=0.8,
        help="LightGBM feature fraction",
    )
    parser.add_argument(
        "--lgb-bagging-fraction",
        type=float,
        default=0.8,
        help="LightGBM bagging fraction",
    )
    parser.add_argument(
        "--lgb-lambda-l1", type=float, default=0.3, help="LightGBM L1 regularization"
    )
    parser.add_argument(
        "--lgb-lambda-l2", type=float, default=0.3, help="LightGBM L2 regularization"
    )
    parser.add_argument(
        "--lgb-min-child-samples",
        type=int,
        default=20,
        help="LightGBM min child samples",
    )

    return parser


def get_xgb_parser():
    """Get XGBoost specific argument parser"""
    parser = get_base_parser()

    # Model selection
    parser.add_argument(
        "model",
        nargs="?",
        default="xgb",
        choices=["lgb", "xgb", "cat"],
        help="Model to run",
    )

    # XGBoost specific parameters
    parser.add_argument(
        "--xgb-tree-method",
        type=str,
        default="gpu_hist",
        choices=["gpu_hist", "hist", "exact"],
        help="XGBoost tree method",
    )
    parser.add_argument(
        "--xgb-max-depth", type=int, default=6, help="XGBoost max depth"
    )
    parser.add_argument(
        "--xgb-subsample", type=float, default=0.8, help="XGBoost subsample ratio"
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.8,
        help="XGBoost column sample by tree",
    )

    return parser


def get_cat_parser():
    """Get CatBoost specific argument parser"""
    parser = get_base_parser()

    # Model selection
    parser.add_argument(
        "model",
        nargs="?",
        default="cat",
        choices=["lgb", "xgb", "cat"],
        help="Model to run",
    )

    # CatBoost specific parameters
    parser.add_argument("--cat-depth", type=int, default=6, help="CatBoost depth")
    parser.add_argument(
        "--cat-l2-leaf-reg",
        type=float,
        default=3.0,
        help="CatBoost L2 leaf regularization",
    )
    parser.add_argument(
        "--cat-border-count", type=int, default=128, help="CatBoost border count"
    )
    parser.add_argument(
        "--cat-bagging-temperature",
        type=float,
        default=1.0,
        help="CatBoost bagging temperature",
    )

    return parser


class LGBParams:
    @staticmethod
    def get_params(args):
        return {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "verbosity": -1,
            "random_state": args.random_state,
            "lambda_l1": args.lgb_lambda_l1,
            "lambda_l2": args.lgb_lambda_l2,
            "max_depth": args.lgb_max_depth,
            "num_leaves": args.lgb_num_leaves,
            "feature_fraction": args.lgb_feature_fraction,
            "bagging_fraction": args.lgb_bagging_fraction,
            "bagging_freq": 1,
            "min_child_samples": args.lgb_min_child_samples,
            "seed": args.random_state,
        }


class XGBParams:
    @staticmethod
    def get_params(args):
        return {
            "tree_method": args.xgb_tree_method,
            "objective": "binary:logistic",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "verbosity": 1,
            "random_state": args.random_state,
            "max_depth": getattr(args, "xgb_max_depth", 6),
            "subsample": getattr(args, "xgb_subsample", 0.8),
            "colsample_bytree": getattr(args, "xgb_colsample_bytree", 0.8),
        }


class CatParams:
    @staticmethod
    def get_params(args):
        return {
            "objective": "CrossEntropy",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "verbose": 1,
            "random_state": args.random_state,
            "depth": getattr(args, "cat_depth", 6),
            "l2_leaf_reg": getattr(args, "cat_l2_leaf_reg", 3.0),
            "border_count": getattr(args, "cat_border_count", 128),
            "bagging_temperature": getattr(args, "cat_bagging_temperature", 1.0),
        }


# Environment detection
def get_config():
    """Automatically detect environment and return appropriate config"""
    if Path("/kaggle").exists():
        return KaggleConfig()
    else:
        return LocalConfig()


# Export the active configuration
Config = get_config()
