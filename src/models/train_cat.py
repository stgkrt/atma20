import warnings
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import BaseConfig, CatParams, KaggleConfig, get_cat_parser
from src.data_utils import (
    get_feature_columns,
    load_and_prepare_data,
    plot_predictions,
    save_results,
    setup_logging,
)
from src.metrics import competition_metrics, evaluate_and_report
from src.model_utils import (
    make_prediction,
    print_training_info,
)
from src.optuna_utils import get_optimizer
from src.split_fold import prepare_fold_data
from src.timer_utils import Timer

warnings.filterwarnings("ignore")


def fit_cat(df, config, params):
    """CatBoost を CrossValidation の枠組みで学習を行なう function"""
    import logging

    logger = logging.getLogger("cat_training")

    models = []
    evals_results_list = []
    feat_cols = get_feature_columns(df, config)
    print("feature columns = ", feat_cols)
    logger.info(f"Feature columns count: {len(feat_cols)}")

    # training data の target と同じだけのゼロ配列を用意
    oof_df = df[[config.target_col, "fold"]].copy()
    oof_df["oof_pred"] = -1
    target = []

    for fold in config.FOLDS:
        logger.info(f"Starting fold {fold} training...")
        # Prepare fold data using common function
        x_train, y_train, x_valid, y_valid, idx_valid = prepare_fold_data(
            df, fold, config, feat_cols
        )
        target.extend(y_valid)
        logger.info(f"Fold {fold} - Train: {len(x_train)}, Valid: {len(x_valid)}")

        with Timer(prefix="fit fold={} ".format(fold)):
            clf = CatBoostClassifier(**params)
            clf.fit(x_train, y_train, verbose=500)

        # cv 内で validation data とされた x_valid で予測をして oof_pred に保存していく
        # oof_pred は全部学習に使わなかったデータの予測結果になる → モデルの予測性能を見る指標として利用できる
        pred_i = clf.predict_proba(x_valid)[:, 1]
        oof_df["oof_pred"].iloc[idx_valid] = pred_i
        models.append(clf)

        score = competition_metrics(y_valid, pred_i)
        print(f" - fold{fold} - {score:.4f}")
        logger.info(f"Fold {fold} CV score: {score:.4f}")

    score = competition_metrics(oof_df[config.target_col], oof_df["oof_pred"].values)
    logger.info(f"Overall CV score: {score:.4f}")

    print("=" * 50)
    print(f"FINISH: Whole Score: {score:.4f}")
    return oof_df, target, models, evals_results_list, feat_cols


def plot_importance(models, train_df, feat_cols, config):
    """Plot feature importance for CatBoost models"""
    for idx, model in enumerate(models):
        importance = model.get_feature_importance(
            Pool(train_df[feat_cols].values, train_df[config.target_col].values),
            type="PredictionValuesChange",
        )
        importance_df = pd.DataFrame(
            {
                "features": train_df[feat_cols].columns,
                "importance": importance,
            }
        )

        print(f"Model {idx} - Top 10 Important Features:")
        print(importance_df.sort_values("importance", ascending=False).head(10))
        print()


def cat_main():
    """Main training function"""
    parser = get_cat_parser()
    args = parser.parse_args()

    # Create config from arguments
    if Path("/kaggle").exists():
        config = KaggleConfig.from_args(args)
    else:
        config = BaseConfig.from_args(args)

    # Setup logging
    logger = setup_logging(config, "cat")

    # Load and prepare data
    train_df, test_df = load_and_prepare_data(config)
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    # Check if using Optuna for hyperparameter optimization
    if config.use_optuna:
        print("Using Optuna for hyperparameter optimization...")
        logger.info("Starting Optuna hyperparameter optimization...")

        optimizer = get_optimizer(
            "cat", config, config.optuna_trials, config.random_state
        )
        best_params, best_score = optimizer.optimize(train_df, config.optuna_study_name)

        print(f"Best Optuna score: {best_score:.6f}")
        print(f"Best parameters: {best_params}")
        logger.info(f"Optuna optimization completed. Best score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")

        # Use optimized parameters
        params = best_params
    else:
        # Get default model parameters
        params = CatParams.get_params(args)

    # Print training info
    print_training_info(config, params, "CatBoost")
    logger.info(f"Training parameters: {params}")

    # Train model
    print("Training CatBoost model...")
    logger.info("Starting cross-validation training...")
    oof_df, target, models, evals_results, feat_cols = fit_cat(train_df, config, params)

    # Evaluate and report
    evaluate_and_report(oof_df, config, "CatBoost")

    # Plot feature importance
    print("Plotting feature importance...")
    plot_importance(models, train_df, feat_cols, config)

    # Make predictions on test set
    print("Making predictions...")
    logger.info("Making predictions on test set...")
    pred = make_prediction(models, test_df, feat_cols, model_type="sklearn")

    # Plot prediction distributions
    plot_predictions(pred, oof_df)

    # Save results and models
    logger.info("Saving results and models...")
    sub = save_results(pred, oof_df, config, "cat", models)

    logger.info("Training completed successfully!")

    return models, oof_df, pred, sub


if __name__ == "__main__":
    cat_main()
