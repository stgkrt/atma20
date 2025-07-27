import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


def ensemble_submissions(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
    output_dir: str = "/kaggle/working",
    ensemble_name: Optional[str] = None,
    submission_filename: str = "submission.csv",
) -> str:
    """
    複数のモデルのsubmissionファイルを重み付きアンサンブルする

    Args:
        model_paths: モデル結果フォルダのパスのリスト
        weights: 各モデルの重み（指定しない場合は均等重み）
        output_dir: 出力先ディレクトリ
        ensemble_name: アンサンブル名（指定しない場合は自動生成）
        submission_filename: submissionファイル名

    Returns:
        作成されたアンサンブルフォルダのパス
    """

    # 重みの設定
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    else:
        # 重みを正規化
        weights = np.array(weights)
        weights = weights / weights.sum()

    if len(model_paths) != len(weights):
        raise ValueError("model_pathsとweightsの長さが一致しません")

    # アンサンブル名の生成
    if ensemble_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_names = [os.path.basename(path) for path in model_paths]
        ensemble_name = f"ensemble_{'_'.join(model_names)}_{timestamp}"

    # 出力フォルダの作成
    ensemble_dir = os.path.join(output_dir, ensemble_name)
    os.makedirs(ensemble_dir, exist_ok=True)

    print(f"アンサンブル実行: {ensemble_name}")
    print(f"モデル数: {len(model_paths)}")
    print(f"重み: {weights}")

    # submissionファイルの読み込みとアンサンブル
    submissions = []
    for i, (path, weight) in enumerate(zip(model_paths, weights)):
        submission_path = os.path.join(path, submission_filename)
        if not os.path.exists(submission_path):
            raise FileNotFoundError(
                f"submissionファイルが見つかりません: {submission_path}"
            )

        df = pd.read_csv(submission_path)
        print(
            f"モデル {i + 1} ({os.path.basename(path)}): {df.shape[0]} 行, 重み {weight:.4f}"
        )
        submissions.append(df)

    # アンサンブル実行（重み付き平均）
    ensemble_df = submissions[0].copy()
    target_col = ensemble_df.columns[0]  # 'target'カラム

    # 重み付き平均を計算
    ensemble_values = np.zeros(len(ensemble_df))
    for i, (df, weight) in enumerate(zip(submissions, weights)):
        ensemble_values += df[target_col].values * weight

    ensemble_df[target_col] = ensemble_values

    # アンサンブル結果の保存
    ensemble_submission_path = os.path.join(ensemble_dir, submission_filename)
    ensemble_df.to_csv(ensemble_submission_path, index=False)
    print(f"アンサンブル結果を保存: {ensemble_submission_path}")

    # アンサンブル情報をメタデータとして保存
    metadata = {
        "ensemble_name": ensemble_name,
        "model_paths": model_paths,
        "weights": weights.tolist() if hasattr(weights, "tolist") else weights,
        "timestamp": datetime.now().isoformat(),
        "submission_shape": ensemble_df.shape,
    }

    metadata_path = os.path.join(ensemble_dir, "ensemble_metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"メタデータを保存: {metadata_path}")
    print("アンサンブル完了!")

    return ensemble_dir


def ensemble_oof_predictions(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
    output_dir: str = "/kaggle/working",
    ensemble_name: Optional[str] = None,
    oof_filename: str = "oof_*.csv",
) -> str:
    """
    複数のモデルのOOF予測をアンサンブルする

    Args:
        model_paths: モデル結果フォルダのパスのリスト
        weights: 各モデルの重み（指定しない場合は均等重み）
        output_dir: 出力先ディレクトリ
        ensemble_name: アンサンブル名
        oof_filename: OOFファイル名のパターン

    Returns:
        作成されたアンサンブルフォルダのパス
    """
    import glob

    # 重みの設定
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()

    if ensemble_name is None:
        ensemble_name = f"ensemble_oof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ensemble_dir = os.path.join(output_dir, ensemble_name)
    os.makedirs(ensemble_dir, exist_ok=True)

    print(f"OOFアンサンブル実行: {ensemble_name}")

    # OOFファイルの読み込みとアンサンブル
    oof_predictions = []
    for i, (path, weight) in enumerate(zip(model_paths, weights)):
        oof_pattern = os.path.join(path, oof_filename)
        oof_files = glob.glob(oof_pattern)

        if not oof_files:
            print(f"警告: OOFファイルが見つかりません: {oof_pattern}")
            continue

        oof_path = oof_files[0]  # 最初にマッチしたファイルを使用
        df = pd.read_csv(oof_path)
        print(
            f"OOF {i + 1} ({os.path.basename(path)}): {df.shape[0]} 行, 重み {weight:.4f}"
        )
        oof_predictions.append(df)

    if not oof_predictions:
        raise FileNotFoundError("有効なOOFファイルが見つかりませんでした")

    # アンサンブル実行
    ensemble_oof = oof_predictions[0].copy()
    target_cols = [
        col
        for col in ensemble_oof.columns
        if "pred" in col.lower() or "target" in col.lower()
    ]

    for target_col in target_cols:
        ensemble_values = np.zeros(len(ensemble_oof))
        for df, weight in zip(oof_predictions, weights):
            if target_col in df.columns:
                ensemble_values += df[target_col].values * weight
        ensemble_oof[target_col] = ensemble_values

    # アンサンブルOOF結果の保存
    ensemble_oof_path = os.path.join(ensemble_dir, "oof_ensemble.csv")
    ensemble_oof.to_csv(ensemble_oof_path, index=False)
    print(f"アンサンブルOOF結果を保存: {ensemble_oof_path}")

    return ensemble_dir


def main():
    """
    使用例とテスト用のメイン関数
    """
    # 使用例1: 3つのモデルを均等重みでアンサンブル
    model_paths = [
        "/kaggle/working/experiment1_cat",
        "/kaggle/working/experiment1_lgb",
        "/kaggle/working/experiment1_xgb",
    ]

    print("=== 均等重みアンサンブル ===")
    ensemble_submissions(
        model_paths=model_paths, ensemble_name="ensemble_equal_weights"
    )

    # 使用例2: 重み付きアンサンブル（LightGBMを重視）
    weights = [0.2, 0.6, 0.2]  # cat: 0.2, lgb: 0.6, xgb: 0.2

    print("\n=== 重み付きアンサンブル ===")
    ensemble_submissions(
        model_paths=model_paths, weights=weights, ensemble_name="ensemble_lgb_focused"
    )

    # 使用例3: OOFアンサンブル
    print("\n=== OOFアンサンブル ===")
    ensemble_oof_predictions(
        model_paths=model_paths,
        weights=weights,
        ensemble_name="ensemble_oof_lgb_focused",
    )


if __name__ == "__main__":
    main()
