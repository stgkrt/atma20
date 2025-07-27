#!/usr/bin/env python3
"""
特徴量ファイルを切り替え可能な実験スクリプト
既存のsrcモジュールを活用
"""

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Add the parent directory to sys.path so we can import src modules
sys.path.append(str(Path(__file__).parent))

from src.config import KaggleConfig
from src.timer_utils import Timer


def get_argument_parser():
    """実験用の引数パーサーを取得"""
    parser = argparse.ArgumentParser(description="ATMA20 Feature Experiment Script")

    # 実験設定
    parser.add_argument(
        "--exp-name", type=str, default="feature_experiment", help="実験名"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["original", "new"],
        default="new",
        help="特徴量タイプ (original: 既存, new: 新規)",
    )

    # データパス設定
    parser.add_argument(
        "--train-features",
        type=str,
        default="/kaggle/working/train_features.csv",
        help="学習用特徴量ファイルのパス",
    )
    parser.add_argument(
        "--test-features",
        type=str,
        default="/kaggle/working/test_features.csv",
        help="テスト用特徴量ファイルのパス",
    )

    # モデルパラメータ
    parser.add_argument("--n-folds", type=int, default=5, help="フォールド数")
    parser.add_argument("--random-state", type=int, default=42, help="ランダムシード")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="学習率")
    parser.add_argument("--num-leaves", type=int, default=31, help="葉の数")
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=100, help="早期停止ラウンド数"
    )
    parser.add_argument("--verbose-eval", type=int, default=100, help="評価表示間隔")

    return parser


def load_data_with_features(args):
    """特徴量タイプに応じてデータを読み込み"""
    print(f"Loading data with feature type: {args.feature_type}")

    if args.feature_type == "new":
        # 新しい特徴量ファイルを使用
        print(f"Loading new features from: {args.train_features}")
        train_df = pd.read_csv(args.train_features)
        test_df = pd.read_csv(args.test_features)

        # 特徴量カラムを取得（基本カラムを除く）
        feature_cols = [
            col
            for col in train_df.columns
            if col not in ["社員番号", "category", "target"]
        ]

    else:
        # 既存のデータ読み込み処理を使用
        from src.data_utils import load_and_prepare_data

        config = KaggleConfig()
        train_df, test_df = load_and_prepare_data(config)

        # 既存の特徴量カラムを取得
        feature_cols = [col for col in train_df.columns if col not in config.unused_col]

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target distribution: {train_df['target'].value_counts()}")

    return train_df, test_df, feature_cols


def prepare_features(train_df, test_df, feature_cols):
    """特徴量の前処理"""
    print("Preparing features...")

    # カテゴリエンコーディング（新特徴量の場合のみ）
    if "category_encoded" not in feature_cols:
        le_category = LabelEncoder()
        all_categories = pd.concat([train_df["category"], test_df["category"]]).unique()
        le_category.fit(all_categories)

        train_df["category_encoded"] = le_category.transform(train_df["category"])
        test_df["category_encoded"] = le_category.transform(test_df["category"])
        feature_cols.append("category_encoded")

    # 欠損値処理
    train_df[feature_cols] = train_df[feature_cols].fillna(-1)
    test_df[feature_cols] = test_df[feature_cols].fillna(-1)

    print(f"Final feature count: {len(feature_cols)}")

    return train_df, test_df, feature_cols


def create_employee_based_folds(train_df, n_splits=5, random_state=42):
    """社員ベースのフォールド作成"""
    print("Creating employee-based folds...")

    # 各社員の応募率を計算
    employee_stats = (
        train_df.groupby("社員番号")["target"].agg(["mean", "count"]).reset_index()
    )
    employee_stats["target_bin"] = pd.cut(employee_stats["mean"], bins=5, labels=False)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(employee_stats, employee_stats["target_bin"]))

    # フォールド情報を付与
    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(folds):
        val_employees = employee_stats.iloc[val_idx]["社員番号"].values
        train_df.loc[train_df["社員番号"].isin(val_employees), "fold"] = fold

    print("Fold distribution:")
    fold_counts = train_df["fold"].value_counts().sort_index()
    for fold, count in fold_counts.items():
        print(f"  Fold {fold}: {count} samples")

    return train_df


def train_lgb_with_cv(train_df, feature_cols, args):
    """LightGBMによるクロスバリデーション訓練"""
    print("Starting LightGBM cross-validation training...")

    # パラメータ設定
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": args.random_state,
        "n_jobs": -1,
    }

    models = []
    oof_predictions = np.zeros(len(train_df))
    feature_importance = pd.DataFrame()
    fold_scores = []

    for fold in range(args.n_folds):
        print(f"\nFold {fold + 1}/{args.n_folds}")
        print("-" * 30)

        # データ分割
        train_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_train = train_df.loc[train_idx, feature_cols]
        y_train = train_df.loc[train_idx, "target"]
        X_val = train_df.loc[val_idx, feature_cols]
        y_val = train_df.loc[val_idx, "target"]

        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

        # データセット作成
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        # モデル訓練
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["train", "val"],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping_rounds),
                lgb.log_evaluation(period=args.verbose_eval),
            ],
        )

        models.append(model)

        # OOF予測
        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred

        # Fold AUC
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

        # 特徴量重要度
        fold_importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importance(importance_type="gain"),
                "fold": fold,
            }
        )
        feature_importance = pd.concat(
            [feature_importance, fold_importance], ignore_index=True
        )

    # Overall結果
    overall_auc = roc_auc_score(train_df["target"], oof_predictions)
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)

    print("\nCross-Validation Results:")
    print(f"Overall AUC: {overall_auc:.5f}")
    print(f"Mean AUC: {mean_auc:.5f} ± {std_auc:.5f}")

    return models, oof_predictions, feature_importance, overall_auc


def analyze_feature_importance(feature_importance, top_n=20):
    """特徴量重要度分析"""
    print(f"\nTop {top_n} Important Features:")
    print("=" * 60)

    # 平均重要度を計算
    avg_importance = (
        feature_importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )

    for i, (feature, importance) in enumerate(avg_importance.head(top_n).items()):
        print(f"{i + 1:2d}. {feature:35s}: {importance:10.1f}")

    return avg_importance


def make_test_predictions(models, test_df, feature_cols):
    """テストデータの予測"""
    print("\nMaking predictions on test data...")

    predictions = np.zeros(len(test_df))

    for i, model in enumerate(models):
        pred = model.predict(test_df[feature_cols])
        predictions += pred / len(models)
        print(f"Model {i + 1} - mean: {pred.mean():.4f}, std: {pred.std():.4f}")

    print(
        f"Final predictions - mean: {predictions.mean():.4f}, std: {predictions.std():.4f}"
    )

    return predictions


def save_experiment_results(
    train_df,
    test_df,
    predictions,
    oof_predictions,
    feature_importance,
    overall_auc,
    args,
):
    """実験結果の保存"""
    output_dir = Path(f"/kaggle/working/{args.exp_name}")
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # 提出ファイル
    submission = test_df[["社員番号", "category"]].copy()
    submission["target"] = predictions
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved: {submission_path}")

    # OOF予測
    oof_df = pd.DataFrame(
        {
            "社員番号": train_df["社員番号"],
            "category": train_df["category"],
            "target": train_df["target"],
            "oof_pred": oof_predictions,
            "fold": train_df["fold"],
        }
    )
    oof_path = output_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved: {oof_path}")

    # 特徴量重要度
    importance_path = output_dir / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved: {importance_path}")

    # 実験メタデータ
    metadata = {
        "experiment_name": args.exp_name,
        "feature_type": args.feature_type,
        "cv_auc": overall_auc,
        "n_folds": args.n_folds,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "random_state": args.random_state,
    }

    metadata_path = output_dir / "experiment_metadata.txt"
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"Metadata saved: {metadata_path}")

    return submission_path, oof_path


def main():
    """メイン実行関数"""
    # 引数解析
    parser = get_argument_parser()
    args = parser.parse_args()

    print("ATMA20 Feature Experiment")
    print("=" * 50)
    print(f"Experiment: {args.exp_name}")
    print(f"Feature type: {args.feature_type}")
    print(f"Random seed: {args.random_state}")

    with Timer(prefix="Total execution time"):
        # データ読み込み
        with Timer(prefix="Data loading"):
            train_df, test_df, feature_cols = load_data_with_features(args)

        # 特徴量前処理
        with Timer(prefix="Feature preparation"):
            train_df, test_df, feature_cols = prepare_features(
                train_df, test_df, feature_cols
            )

        # フォールド作成
        with Timer(prefix="Fold creation"):
            train_df = create_employee_based_folds(
                train_df, args.n_folds, args.random_state
            )

        # モデル訓練
        with Timer(prefix="Model training"):
            models, oof_predictions, feature_importance, overall_auc = (
                train_lgb_with_cv(train_df, feature_cols, args)
            )

        # 特徴量重要度分析
        analyze_feature_importance(feature_importance)

        # テスト予測
        with Timer(prefix="Test prediction"):
            predictions = make_test_predictions(models, test_df, feature_cols)

        # 結果保存
        with Timer(prefix="Saving results"):
            submission_path, oof_path = save_experiment_results(
                train_df,
                test_df,
                predictions,
                oof_predictions,
                feature_importance,
                overall_auc,
                args,
            )

    print("\nExperiment completed successfully!")
    print(f"CV AUC: {overall_auc:.5f}")
    print(f"Results saved to: /kaggle/working/{args.exp_name}/")


if __name__ == "__main__":
    main()
