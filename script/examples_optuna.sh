#!/bin/bash

# Optuna使用例

echo "=== Optuna Hyperparameter Optimization Examples ==="

# LightGBM with Optuna optimization (100 trials)
echo "Running LightGBM with Optuna optimization..."
python src/models/train_lgb.py \
    --exp-name "lgb_optuna_100trials" \
    --use-optuna \
    --optuna-trials 100 \
    --optuna-study-name "lgb_optimization" \
    --n-folds 5

# XGBoost with Optuna optimization (50 trials for faster testing)
echo "Running XGBoost with Optuna optimization..."
python src/models/train_xgb.py \
    --exp-name "xgb_optuna_50trials" \
    --use-optuna \
    --optuna-trials 50 \
    --optuna-study-name "xgb_optimization" \
    --n-folds 5

# CatBoost with Optuna optimization
echo "Running CatBoost with Optuna optimization..."
python src/models/train_cat.py \
    --exp-name "cat_optuna_50trials" \
    --use-optuna \
    --optuna-trials 50 \
    --optuna-study-name "cat_optimization" \
    --n-folds 5

# Traditional training without Optuna for comparison
echo "Running LightGBM without Optuna (traditional)..."
python src/models/train_lgb.py \
    --exp-name "lgb_traditional" \
    --n-folds 5

echo "=== All experiments completed! ==="
