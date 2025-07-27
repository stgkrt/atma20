import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import BaseConfig, KaggleConfig, XGBParams, get_xgb_parser
from src.data_utils import (
    get_feature_columns,
    load_and_prepare_data,
    plot_predictions,
    save_results,
)
from src.metrics import competition_metrics, evaluate_and_report
from src.model_utils import (
    make_prediction,
    print_training_info,
)
from src.split_fold import prepare_fold_data
from src.timer_utils import Timer

warnings.filterwarnings("ignore")


def fit_xgb(df, config, params):
    """XGBoost を CrossValidation の枠組みで学習を行なう function"""

    models = []
    evals_results_list = []
    feat_cols = get_feature_columns(df, config)
    print("feature columns = ", feat_cols)

    # training data の target と同じだけのゼロ配列を用意
    oof_df = df[[config.target_col, "fold"]].copy()
    oof_df["oof_pred"] = -1
    target = []

    for fold in config.FOLDS:
        # Prepare fold data using common function
        x_train, y_train, x_valid, y_valid, idx_valid = prepare_fold_data(
            df, fold, config, feat_cols
        )
        target.extend(y_valid)

        with Timer(prefix="fit fold={} ".format(fold)):
            clf = xgb.XGBClassifier(**params)
            clf.fit(x_train, y_train)

        # cv 内で validation data とされた x_valid で予測をして oof_pred に保存していく
        # oof_pred は全部学習に使わなかったデータの予測結果になる → モデルの予測性能を見る指標として利用できる
        pred_i = clf.predict_proba(x_valid)[:, 1]
        oof_df["oof_pred"].iloc[idx_valid] = pred_i
        models.append(clf)
        score = competition_metrics(y_valid, pred_i)
        print(f" - fold{fold} - {score:.4f}")

    score = competition_metrics(oof_df[config.target_col], oof_df["oof_pred"].values)

    print("=" * 50)
    print(f"FINISH: Whole Score: {score:.4f}")
    return oof_df, target, models, evals_results_list, feat_cols


def plot_importance(models):
    """Plot feature importance for XGBoost models"""
    for idx, model in enumerate(models):
        plt.figure(figsize=(8, 6))
        xgb.plot_importance(
            model,
            importance_type="gain",
            max_num_features=25,
            xlabel="Feature Importance",
            ylabel="Features",
        )
        plt.tight_layout()
        plt.show()


def xgb_main():
    """Main training function"""
    parser = get_xgb_parser()
    args = parser.parse_args()

    # Create config from arguments
    if Path("/kaggle").exists():
        config = KaggleConfig.from_args(args)
    else:
        config = BaseConfig.from_args(args)

    # Get model parameters
    params = XGBParams.get_params(args)

    # Print training info
    print_training_info(config, params, "XGBoost")

    # Load and prepare data
    train_df, test_df = load_and_prepare_data(config)

    # Train model
    print("Training XGBoost model...")
    oof_df, target, models, evals_results, feat_cols = fit_xgb(train_df, config, params)

    # Evaluate and report
    evaluate_and_report(oof_df, config, "XGBoost")

    # Plot feature importance
    print("Plotting feature importance...")
    plot_importance(models)

    # Make predictions on test set
    print("Making predictions...")
    pred = make_prediction(models, test_df, feat_cols, model_type="sklearn")

    # Plot prediction distributions
    plot_predictions(pred, oof_df)

    # Save results
    sub = save_results(pred, oof_df, config, "xgb")

    return models, oof_df, pred, sub
