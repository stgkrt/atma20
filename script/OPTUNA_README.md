# Optuna ハイパーパラメーター最適化ガイド

このプロジェクトでは、LightGBM、XGBoost、CatBoostの各モデルでOptunaを使用したハイパーパラメーター最適化をサポートしています。

## 機能概要

### Optunaを使用する場合
- 自動的にハイパーパラメーターの最適化を実行
- TPESampler とMedianPrunerを使用した効率的な探索
- 各モデル用に最適化された探索空間
- Cross-Validationベースの評価

### 従来の方法（Optuna無し）
- 事前定義されたハイパーパラメーターを使用
- コマンドライン引数で調整可能
- 高速な実行

## 使用方法

### 1. Optunaを使用したハイパーパラメーター最適化

#### LightGBM
```bash
python src/models/train_lgb.py \
    --exp-name "experiment_lgb_optuna" \
    --use-optuna \
    --optuna-trials 100 \
    --optuna-study-name "lgb_optimization" \
    --n-folds 5
```

#### XGBoost
```bash
python src/models/train_xgb.py \
    --exp-name "experiment_xgb_optuna" \
    --use-optuna \
    --optuna-trials 100 \
    --optuna-study-name "xgb_optimization" \
    --n-folds 5
```

#### CatBoost
```bash
python src/models/train_cat.py \
    --exp-name "experiment_cat_optuna" \
    --use-optuna \
    --optuna-trials 100 \
    --optuna-study-name "cat_optimization" \
    --n-folds 5
```

### 2. 従来の方法（Optuna無し）

#### LightGBM
```bash
python src/models/train_lgb.py \
    --exp-name "experiment_lgb_traditional" \
    --learning-rate 0.1 \
    --lgb-num-leaves 128 \
    --lgb-max-depth 6 \
    --n-folds 5
```

#### XGBoost
```bash
python src/models/train_xgb.py \
    --exp-name "experiment_xgb_traditional" \
    --learning-rate 0.1 \
    --xgb-max-depth 6 \
    --xgb-subsample 0.8 \
    --n-folds 5
```

#### CatBoost
```bash
python src/models/train_cat.py \
    --exp-name "experiment_cat_traditional" \
    --learning-rate 0.1 \
    --cat-depth 6 \
    --cat-l2-leaf-reg 3.0 \
    --n-folds 5
```

## Optunaパラメーター

### 共通パラメーター
- `--use-optuna`: Optuna最適化を有効にする
- `--optuna-trials`: 最適化試行回数（デフォルト: 100）
- `--optuna-study-name`: 研究名（デフォルト: {exp_name}_optuna）

### 最適化対象パラメーター

#### LightGBM
- learning_rate: 0.01 ~ 0.3 (log scale)
- num_leaves: 10 ~ 300
- max_depth: 3 ~ 12
- min_child_samples: 5 ~ 100
- feature_fraction: 0.4 ~ 1.0
- bagging_fraction: 0.4 ~ 1.0
- bagging_freq: 1 ~ 7
- lambda_l1, lambda_l2: 1e-8 ~ 10.0 (log scale)

#### XGBoost
- learning_rate: 0.01 ~ 0.3 (log scale)
- max_depth: 3 ~ 12
- min_child_weight: 1 ~ 10
- subsample: 0.5 ~ 1.0
- colsample_bytree: 0.5 ~ 1.0
- reg_alpha, reg_lambda: 1e-8 ~ 10.0 (log scale)

#### CatBoost
- learning_rate: 0.01 ~ 0.3 (log scale)
- depth: 4 ~ 10
- l2_leaf_reg: 1 ~ 10
- border_count: 32 ~ 255
- bagging_temperature: 0.0 ~ 10.0
- random_strength: 1e-8 ~ 10.0 (log scale)

## 実行例

一括実行スクリプトも用意されています：

```bash
# 全モデルの最適化を実行
bash examples_optuna.sh
```

## 注意点

1. **実行時間**: Optunaを使用すると実行時間が大幅に増加します
2. **試行回数**: 最初は少ない試行回数（10-50）でテストすることを推奨
3. **メモリ使用量**: 大きなデータセットでは十分なメモリが必要
4. **結果の保存**: 最適化結果はログファイルに記録されます

## 出力

### Optuna使用時
- 最適化プロセスのログ
- 最良のスコアとパラメーター
- 最適化されたパラメーターでの最終モデル

### 従来の方法
- 事前定義パラメーターでの直接学習
- より高速な実行

## 推奨ワークフロー

1. 最初は従来の方法で動作確認
2. 小さな試行回数（10-20）でOptuna動作確認
3. 本格的な最適化で多くの試行回数を設定
4. 結果を比較して最適な手法を選択
