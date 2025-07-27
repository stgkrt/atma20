#!/bin/bash

echo "=== Starting exp004 Optuna Hyperparameter Optimization Experiments ==="

# exp004_lgb: LightGBM with Optuna optimization
echo "Starting exp004_lgb: LightGBM with Optuna optimization..."
python src/models/train_lgb.py \
    --exp-name "exp004_lgb" \
    --use-optuna \
    --optuna-trials 100 \
    --optuna-study-name "exp004_lgb_optimization" \
    --n-folds 5 \
    --random-state 510

echo "exp004_lgb completed!"
echo "================================"

# exp004_xgb: XGBoost with Optuna optimization  
echo "Starting exp004_xgb: XGBoost with Optuna optimization..."
python src/models/train_xgb.py \
    --exp-name "exp004_xgb" \
    --use-optuna \
    --optuna-trials 100 \
    --optuna-study-name "exp004_xgb_optimization" \
    --n-folds 5 \
    --random-state 510

echo "exp004_xgb completed!"
echo "================================"

# exp004_cat: CatBoost with Optuna optimization
echo "Starting exp004_cat: CatBoost with Optuna optimization..."
python src/models/train_cat.py \
    --exp-name "exp004_cat" \
    --use-optuna \
    --optuna-trials 100 \
    --optuna-study-name "exp004_cat_optimization" \
    --n-folds 5 \
    --random-state 510

echo "exp004_cat completed!"
echo "================================"

echo "=== All exp004 experiments completed! ==="
echo "Results saved in:"
echo "- /kaggle/working/exp004_lgb/"
echo "- /kaggle/working/exp004_xgb/"  
echo "- /kaggle/working/exp004_cat/"
