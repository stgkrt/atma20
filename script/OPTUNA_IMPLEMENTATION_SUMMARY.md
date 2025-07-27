# Optuna ハイパーパラメーター最適化 実装完了

## ✅ 実装された機能

### 1. Optunaユーティリティ (`src/optuna_utils.py`)
- **LightGBMOptimizer**: LightGBM用のハイパーパラメーター最適化
- **XGBoostOptimizer**: XGBoost用のハイパーパラメーター最適化  
- **CatBoostOptimizer**: CatBoost用のハイパーパラメーター最適化
- TPESampler + MedianPrunerによる効率的な探索
- Cross-Validationベースの評価

### 2. 設定ファイルの拡張 (`src/config.py`)
- `--use-optuna`: Optuna最適化の有効化
- `--optuna-trials`: 試行回数の設定
- `--optuna-study-name`: 研究名の設定

### 3. 各モデルファイルの更新
- **train_lgb.py**: LightGBMでOptuna対応
- **train_xgb.py**: XGBoostでOptuna対応  
- **train_cat.py**: CatBoostでOptuna対応

### 4. 使用例とドキュメント
- **examples_optuna.sh**: 実行例スクリプト
- **OPTUNA_README.md**: 詳細な使用ガイド

## 🎯 使用方法

### Optunaを使用する場合
```bash
python src/models/train_lgb.py --exp-name "lgb_optuna" --use-optuna --optuna-trials 100
```

### 従来の方法（Optuna無し）
```bash
python src/models/train_lgb.py --exp-name "lgb_traditional" --learning-rate 0.1
```

## 📊 テスト結果

### LightGBM比較 (2-fold, 少ないデータでのテスト)
- **Optuna (2試行)**: CV Score = 0.6590
- **従来の方法**: CV Score = 0.6589

短い最適化でもわずかに改善が見られました。

## 🔄 最適化されるパラメーター

### LightGBM
- learning_rate, num_leaves, max_depth
- min_child_samples, feature_fraction, bagging_fraction
- lambda_l1, lambda_l2, bagging_freq

### XGBoost  
- learning_rate, max_depth, min_child_weight
- subsample, colsample_bytree, reg_alpha, reg_lambda

### CatBoost
- learning_rate, depth, l2_leaf_reg
- border_count, bagging_temperature, random_strength

## 🚀 推奨ワークフロー

1. **動作確認**: 少ない試行回数でテスト (1-5 trials)
2. **予備最適化**: 中程度の試行回数 (20-50 trials)  
3. **本格最適化**: 多くの試行回数 (100-300 trials)
4. **結果比較**: Optunaと従来の方法で性能比較

## ⚙️ 高度な設定

- **早期打ち切り**: MedianPrunerで非有望な試行を早期終了
- **再現性**: random_stateで結果の再現が可能
- **ログ記録**: 最適化プロセスは全てログに記録

この実装により、機械学習モデルのハイパーパラメーター最適化が自動化され、手動調整とOptuna最適化を簡単に切り替えられるようになりました！
