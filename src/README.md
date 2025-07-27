# ATMA20 Training Scripts

このプロジェクトは、ノートブック形式のATMA20ベースラインコードを3つの独立したPythonスクリプト（LightGBM、XGBoost、CatBoost）に分割したものです。

## ファイル構成

```
src/
├── models/
│   ├── train_lgb.py   # LightGBM専用トレーニングスクリプト
│   ├── train_xgb.py   # XGBoost専用トレーニングスクリプト
│   └── train_cat.py   # CatBoost専用トレーニングスクリプト
├── utils.py           # 共通ユーティリティ関数
├── run_all.py         # 全モデルを統合実行するスクリプト
├── run_model.py       # 個別モデル実行用ラッパー
├── run_training.sh    # 実行用シェルスクリプト
├── config.py          # 設定ファイル（argparse対応）
└── README.md          # このファイル
```

## 使用方法

### 1. 個別モデルの実行

各モデルを個別に実行する場合（argparseでパラメータ調整可能）：

```bash
# LightGBMのみ実行（デフォルトパラメータ）
python src/models/train_lgb.py

# LightGBMのカスタムパラメータで実行
python src/models/train_lgb.py --learning-rate 0.05 --n-estimators 5000 --lgb-max-depth 8

# XGBoostのみ実行
python src/models/train_xgb.py

# XGBoostのカスタムパラメータで実行
python src/models/train_xgb.py --learning-rate 0.01 --xgb-tree-method hist

# CatBoostのみ実行
python src/models/train_cat.py

# CatBoostのカスタムパラメータで実行
python src/models/train_cat.py --learning-rate 0.02 --cat-depth 8

# ヘルプを表示（利用可能なパラメータ確認）
python src/models/train_lgb.py --help
python src/models/train_xgb.py --help
python src/models/train_cat.py --help
```

### 2. 統合スクリプトでの実行

`run_all.py`を使用して複数モデルを実行：

```bash
# 全モデル実行
python src/run_all.py

# 特定のモデルのみ実行
python src/run_all.py --models lgb xgb

# 全モデル実行後にアンサンブル作成
python src/run_all.py --ensemble
```

### 3. シェルスクリプトでの実行

便利なシェルスクリプトを使用：

```bash
# 全モデル実行
./src/run_training.sh

# LightGBMのみ実行
./src/run_training.sh -m lgb

# XGBoostのみ実行
./src/run_training.sh -m xgb

# CatBoostのみ実行
./src/run_training.sh -m cat

# 全モデル実行後にアンサンブル作成
./src/run_training.sh -e

# ヘルプ表示
./src/run_training.sh -h
```

## 利用可能なコマンドライン引数

### 共通パラメータ
- `--train-csv`: 訓練データCSVファイルのパス
- `--test-csv`: テストデータCSVファイルのパス
- `--sample-sub-csv`: サンプル提出ファイルのパス
- `--output-dir`: 結果出力ディレクトリ
- `--n-estimators`: 推定器の数 (デフォルト: 10000)
- `--learning-rate`: 学習率 (デフォルト: 0.1)
- `--random-state`: 乱数シード (デフォルト: 510)
- `--early-stopping-rounds`: 早期停止ラウンド数 (デフォルト: 50)
- `--verbose-eval`: 評価表示頻度 (デフォルト: 30)
- `--n-folds`: クロスバリデーションフォールド数 (デフォルト: 5)
- `--exp-name`: 実験名 (デフォルト: "experiment")
- `--exp-category`: 実験カテゴリ (デフォルト: "baseline")

### LightGBM専用パラメータ
- `--lgb-max-depth`: 最大深度 (デフォルト: 6)
- `--lgb-num-leaves`: 葉の数 (デフォルト: 128)
- `--lgb-feature-fraction`: 特徴量サンプリング率 (デフォルト: 0.8)
- `--lgb-bagging-fraction`: バギングサンプリング率 (デフォルト: 0.8)
- `--lgb-lambda-l1`: L1正則化 (デフォルト: 0.3)
- `--lgb-lambda-l2`: L2正則化 (デフォルト: 0.3)
- `--lgb-min-child-samples`: 最小子サンプル数 (デフォルト: 20)

### XGBoost専用パラメータ
- `--xgb-tree-method`: 木構築方法 (gpu_hist/hist/exact, デフォルト: gpu_hist)
- `--xgb-max-depth`: 最大深度 (デフォルト: 6)
- `--xgb-subsample`: サブサンプル率 (デフォルト: 0.8)
- `--xgb-colsample-bytree`: 列サンプリング率 (デフォルト: 0.8)

### CatBoost専用パラメータ
- `--cat-depth`: 深度 (デフォルト: 6)
- `--cat-l2-leaf-reg`: L2葉正則化 (デフォルト: 3.0)
- `--cat-border-count`: ボーダー数 (デフォルト: 128)
- `--cat-bagging-temperature`: バギング温度 (デフォルト: 1.0)

## 設定の変更

従来の`src/config.py`での固定設定に加えて、コマンドライン引数でパラメータを動的に変更できるようになりました：

### 設定の優先順位
1. コマンドライン引数（最優先）
2. config.pyのデフォルト値

### 環境別設定

## 出力ファイル

各実行後に以下のファイルが生成されます：

### 個別モデル実行時
- `ens_lgb.csv` / `ens_xgb.csv` / `ens_cat.csv`: 各モデルの予測結果
- `oof_lgb.csv` / `oof_xgb.csv` / `oof_cat.csv`: 各モデルのOOF予測
- `submission.csv`: 最後に実行したモデルの提出ファイル

### アンサンブル実行時
- `ensemble_submission.csv`: アンサンブル予測結果
- `ensemble_oof.csv`: アンサンブルOOF予測

## 環境要件

必要なPythonパッケージ：

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

## 特徴

1. **モジュール化**: 各モデルが独立したファイルで管理
2. **共通関数の分離**: `utils.py`で重複コードを排除
3. **設定の一元化**: `config.py`で設定を管理、argparseで動的変更可能
4. **環境自動検出**: Kaggle環境とローカル環境を自動判別
5. **アンサンブル機能**: 複数モデルの予測を組み合わせ
6. **実行時間計測**: 各フォールドの実行時間を表示
7. **特徴量重要度可視化**: 各モデルの重要な特徴量を表示
8. **コードの再利用性**: 共通処理を関数化して保守性向上

## utils.pyの共通関数

- `competition_metrics()`: 競技評価指標（ROC AUC）の計算
- `Timer`: 実行時間計測用コンテキストマネージャー
- `cut_fold()`: クロスバリデーション用フォールド作成
- `get_feature_columns()`: 特徴量カラムの取得
- `prepare_fold_data()`: フォールド別データ準備
- `make_prediction()`: モデル予測の実行と平均化
- `plot_predictions()`: 予測分布の可視化
- `save_results()`: 結果ファイルの保存
- `load_and_prepare_data()`: データの読み込みと前処理
- `print_training_info()`: トレーニング情報の表示
- `evaluate_and_report()`: 最終評価とレポート

## 元のノートブックからの変更点

1. **model_type変数の削除**: 各ファイルが特定のモデル専用
2. **設定の外部化**: ハードコードされた設定を`config.py`に移動
3. **関数の整理**: 共通処理を関数化
4. **エラーハンドリング**: より堅牢なエラー処理
5. **ログ出力の改善**: より詳細な進捗表示

## トラブルシューティング

### データファイルが見つからない場合
`src/config.py`でデータファイルのパスを確認・修正してください。

### GPU関連エラー
XGBoostのGPU使用でエラーが発生する場合は、`src/config.py`の`XGBParams`で`'tree_method': 'hist'`に変更してください。

### メモリ不足
大きなデータセットでメモリ不足が発生する場合は、各設定ファイルで`n_estimators`を小さくしてください。
