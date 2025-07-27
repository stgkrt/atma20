#!/usr/bin/env python3
"""
コマンドライン用のアンサンブルスクリプト

使用例:
python run_ensemble.py --models experiment1_cat experiment1_lgb experiment1_xgb
python run_ensemble.py --models experiment1_cat experiment1_lgb experiment1_xgb --weights 0.2 0.6 0.2
python run_ensemble.py --models experiment1_cat experiment1_lgb --name "simple_ensemble"
"""

import argparse
import os
import sys

# srcディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble import ensemble_oof_predictions, ensemble_submissions


def main():
    parser = argparse.ArgumentParser(description="複数モデルのアンサンブル実行")

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="アンサンブルするモデルフォルダ名のリスト",
    )

    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="各モデルの重み（指定しない場合は均等重み）",
    )

    parser.add_argument(
        "--base-dir",
        default="/kaggle/working",
        help="モデルフォルダの基準ディレクトリ（デフォルト: /kaggle/working）",
    )

    parser.add_argument(
        "--output-dir",
        default="/kaggle/working",
        help="出力先ディレクトリ（デフォルト: /kaggle/working）",
    )

    parser.add_argument(
        "--name", default=None, help="アンサンブル名（指定しない場合は自動生成）"
    )

    parser.add_argument("--oof", action="store_true", help="OOFもアンサンブルする")

    args = parser.parse_args()

    # モデルパスの構築
    model_paths = [os.path.join(args.base_dir, model) for model in args.models]

    # 存在チェック
    for path in model_paths:
        if not os.path.exists(path):
            print(f"エラー: モデルフォルダが見つかりません: {path}")
            sys.exit(1)

    print("アンサンブル設定:")
    print(f"  モデル: {args.models}")
    print(f"  重み: {args.weights}")
    print(f"  出力先: {args.output_dir}")
    print(f"  アンサンブル名: {args.name}")
    print()

    # submissionのアンサンブル
    try:
        ensemble_dir = ensemble_submissions(
            model_paths=model_paths,
            weights=args.weights,
            output_dir=args.output_dir,
            ensemble_name=args.name,
        )
        print(f"\nsubmissionアンサンブル完了: {ensemble_dir}")

        # OOFのアンサンブル（オプション）
        if args.oof:
            print("\nOOFアンサンブルを実行中...")
            ensemble_oof_predictions(
                model_paths=model_paths,
                weights=args.weights,
                output_dir=args.output_dir,
                ensemble_name=args.name + "_oof" if args.name else None,
            )

    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
