import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ensemble_submissions(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
    output_dir: str = "/kaggle/working",
    ensemble_name: Optional[str] = None,
    submission_filename: str = "submission.csv",
    include_oof_score: bool = True,
) -> str:
    """
    複数のモデルのsubmissionファイルを重み付きアンサンブルする

    Args:
        model_paths: モデル結果フォルダのパスのリスト
        weights: 各モデルの重み（指定しない場合は均等重み）
        output_dir: 出力先ディレクトリ
        ensemble_name: アンサンブル名（指定しない場合は自動生成）
        submission_filename: submissionファイル名
        include_oof_score: OOFスコアもアンサンブルして保存するかどうか

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
    target_col = ensemble_df.columns[-1]  # 最後の列が予測値（通常は'target'）

    # 重み付き平均を計算
    ensemble_values = np.zeros(len(ensemble_df))
    for i, (df, weight) in enumerate(zip(submissions, weights)):
        ensemble_values += df[target_col].values * weight

    ensemble_df[target_col] = ensemble_values

    # アンサンブル結果の保存
    ensemble_submission_path = os.path.join(ensemble_dir, submission_filename)
    ensemble_df.to_csv(ensemble_submission_path, index=False)
    print(f"アンサンブル結果を保存: {ensemble_submission_path}")

    # OOFスコアもアンサンブルする（オプション）
    oof_score = None
    oof_ensemble_path = None
    if include_oof_score:
        try:
            oof_score = ensemble_oof_predictions(
                model_paths=model_paths,
                weights=weights,
                output_dir=ensemble_dir,
                ensemble_name="oof_ensemble",
            )
            # アンサンブルされたOOFファイルからスコアを読み取り
            oof_file = os.path.join(ensemble_dir, "oof_ensemble", "oof_ensemble.csv")
            if os.path.exists(oof_file):
                oof_df = pd.read_csv(oof_file)
                if "target" in oof_df.columns and "oof_pred" in oof_df.columns:
                    oof_score = roc_auc_score(oof_df["target"], oof_df["oof_pred"])
                    print(f"アンサンブルOOFスコア: {oof_score:.6f}")
                oof_ensemble_path = oof_file
        except Exception as e:
            print(f"OOFアンサンブルをスキップ: {e}")

    # アンサンブル情報をメタデータとして保存
    metadata = {
        "ensemble_name": ensemble_name,
        "model_paths": model_paths,
        "weights": weights.tolist() if hasattr(weights, "tolist") else weights,
        "timestamp": datetime.now().isoformat(),
        "submission_shape": ensemble_df.shape,
        "oof_score": oof_score,
        "oof_ensemble_path": oof_ensemble_path,
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

    # OOFスコア計算（ターゲット列と予測列が存在する場合）
    oof_score = None
    if "target" in ensemble_oof.columns and "oof_pred" in ensemble_oof.columns:
        oof_score = roc_auc_score(ensemble_oof["target"], ensemble_oof["oof_pred"])
        print(f"アンサンブルOOFスコア: {oof_score:.6f}")

    # アンサンブルOOF結果の保存
    ensemble_oof_path = os.path.join(ensemble_dir, "oof_ensemble.csv")
    ensemble_oof.to_csv(ensemble_oof_path, index=False)
    print(f"アンサンブルOOF結果を保存: {ensemble_oof_path}")

    # メタデータにOOFスコアを含める
    metadata = {
        "ensemble_name": ensemble_name,
        "model_paths": model_paths,
        "weights": weights.tolist() if hasattr(weights, "tolist") else weights,
        "timestamp": datetime.now().isoformat(),
        "oof_score": oof_score,
        "oof_shape": ensemble_oof.shape,
    }

    metadata_path = os.path.join(ensemble_dir, "ensemble_metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"メタデータを保存: {metadata_path}")

    return ensemble_dir


def ensemble_with_complete_results(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
    output_dir: str = "/kaggle/working",
    ensemble_name: Optional[str] = None,
) -> str:
    """
    完全なアンサンブル結果（submission + OOF + スコア）を作成する

    Args:
        model_paths: モデル結果フォルダのパスのリスト
        weights: 各モデルの重み（指定しない場合は均等重み）
        output_dir: 出力先ディレクトリ
        ensemble_name: アンサンブル名

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_name = f"complete_ensemble_{timestamp}"

    ensemble_dir = os.path.join(output_dir, ensemble_name)
    os.makedirs(ensemble_dir, exist_ok=True)

    print(f"完全アンサンブル実行: {ensemble_name}")
    print(f"モデル数: {len(model_paths)}")
    print(f"重み: {weights}")

    # 1. Submissionアンサンブル
    submissions = []
    for i, (path, weight) in enumerate(zip(model_paths, weights)):
        submission_path = os.path.join(path, "submission.csv")
        if os.path.exists(submission_path):
            df = pd.read_csv(submission_path)
            submissions.append(df)
            print(f"Submission {i + 1} ({os.path.basename(path)}): {df.shape[0]} 行")

    if submissions:
        ensemble_submission = submissions[0].copy()
        target_col = ensemble_submission.columns[-1]  # 最後の列が予測値
        ensemble_values = np.zeros(len(ensemble_submission))

        for df, weight in zip(submissions, weights):
            ensemble_values += df[target_col].values * weight

        ensemble_submission[target_col] = ensemble_values
        ensemble_submission.to_csv(
            os.path.join(ensemble_dir, "submission.csv"), index=False
        )
        print("Submissionアンサンブル完了")

    # 2. OOFアンサンブル
    oof_predictions = []
    oof_scores = []

    for i, (path, weight) in enumerate(zip(model_paths, weights)):
        oof_pattern = os.path.join(path, "oof_*.csv")
        oof_files = glob.glob(oof_pattern)

        if oof_files:
            oof_path = oof_files[0]
            df = pd.read_csv(oof_path)
            oof_predictions.append(df)

            # 個別モデルのOOFスコア計算
            if "target" in df.columns and "oof_pred" in df.columns:
                score = roc_auc_score(df["target"], df["oof_pred"])
                oof_scores.append(score)
                print(f"OOF {i + 1} ({os.path.basename(path)}): スコア {score:.6f}")

    # OOFアンサンブル実行
    ensemble_oof_score = None
    if oof_predictions:
        ensemble_oof = oof_predictions[0].copy()

        if "oof_pred" in ensemble_oof.columns:
            ensemble_values = np.zeros(len(ensemble_oof))
            for df, weight in zip(oof_predictions, weights):
                if "oof_pred" in df.columns:
                    ensemble_values += df["oof_pred"].values * weight
            ensemble_oof["oof_pred"] = ensemble_values

            # アンサンブルOOFスコア計算
            if "target" in ensemble_oof.columns:
                ensemble_oof_score = roc_auc_score(
                    ensemble_oof["target"], ensemble_oof["oof_pred"]
                )
                print(f"アンサンブルOOFスコア: {ensemble_oof_score:.6f}")

        ensemble_oof.to_csv(
            os.path.join(ensemble_dir, "oof_predictions.csv"), index=False
        )
        print("OOFアンサンブル完了")

    # 3. 結果サマリー作成
    summary = {
        "ensemble_name": ensemble_name,
        "timestamp": datetime.now().isoformat(),
        "model_count": len(model_paths),
        "model_paths": model_paths,
        "weights": weights.tolist() if hasattr(weights, "tolist") else weights,
        "individual_oof_scores": oof_scores,
        "ensemble_oof_score": ensemble_oof_score,
        "score_improvement": ensemble_oof_score - np.mean(oof_scores)
        if ensemble_oof_score and oof_scores
        else None,
    }

    # サマリーファイル保存
    summary_path = os.path.join(ensemble_dir, "ensemble_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== アンサンブル結果サマリー ===\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        if oof_scores and ensemble_oof_score:
            f.write("\n=== スコア詳細 ===\n")
            for i, score in enumerate(oof_scores):
                f.write(f"モデル {i + 1}: {score:.6f}\n")
            f.write(f"アンサンブル: {ensemble_oof_score:.6f}\n")
            f.write(f"平均からの改善: {ensemble_oof_score - np.mean(oof_scores):.6f}\n")

    print(f"結果サマリーを保存: {summary_path}")
    print("完全アンサンブル完了!")

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

    print("=== 完全アンサンブル（推奨） ===")
    ensemble_with_complete_results(
        model_paths=model_paths, ensemble_name="complete_ensemble_example"
    )

    print("\n=== 均等重みアンサンブル ===")
    ensemble_submissions(
        model_paths=model_paths, ensemble_name="ensemble_equal_weights"
    )

    # 使用例2: 重み付きアンサンブル（LightGBMを重視）
    weights = [0.2, 0.6, 0.2]  # cat: 0.2, lgb: 0.6, xgb: 0.2

    print("\n=== 重み付きアンサンブル ===")
    ensemble_submissions(
        model_paths=model_paths, weights=weights, ensemble_name="ensemble_lgb_focused"
    )

    print("\n=== 重み付き完全アンサンブル ===")
    ensemble_with_complete_results(
        model_paths=model_paths,
        weights=weights,
        ensemble_name="complete_ensemble_lgb_focused",
    )

    # 使用例3: OOFアンサンブルのみ
    print("\n=== OOFアンサンブル ===")
    ensemble_oof_predictions(
        model_paths=model_paths,
        weights=weights,
        ensemble_name="ensemble_oof_lgb_focused",
    )


if __name__ == "__main__":
    main()
