#!/bin/bash

# Sample usage examples for ATMA20 training scripts

echo "========================================"
echo "ATMA20 Training - Usage Examples"
echo "========================================"

echo ""
echo "1. Individual model execution with default parameters:"
echo "   python src/run_model.py lgb"
echo "   python src/run_model.py xgb"
echo "   python src/run_model.py cat"

echo ""
echo "2. Individual model execution with custom parameters:"
echo "   python src/run_model.py lgb --learning-rate 0.05 --n-estimators 5000"
echo "   python src/run_model.py xgb --xgb-tree-method hist --learning-rate 0.01"
echo "   python src/run_model.py cat --cat-depth 8 --learning-rate 0.02"

echo ""
echo "3. Direct execution from models folder:"
echo "   python src/models/train_lgb.py --help"
echo "   python src/models/train_lgb.py --lgb-max-depth 8 --lgb-num-leaves 256"

echo ""
echo "4. Batch execution with run_all.py:"
echo "   python src/run_all.py"
echo "   python src/run_all.py --models lgb xgb"
echo "   python src/run_all.py --ensemble"

echo ""
echo "5. Using the shell script wrapper:"
echo "   ./src/run_training.sh"
echo "   ./src/run_training.sh -m lgb"
echo "   ./src/run_training.sh -e"

echo ""
echo "6. Custom data paths:"
echo "   python src/run_model.py lgb --train-csv /path/to/train.csv --test-csv /path/to/test.csv"

echo ""
echo "7. Experiment tracking:"
echo "   python src/run_model.py lgb --exp-name lgb_experiment_v1 --exp-category tuning"

echo ""
echo "========================================"
echo "Available parameters (use --help for full list):"
echo "========================================"
echo "Common: --learning-rate, --n-estimators, --random-state, --n-folds"
echo "LGB:    --lgb-max-depth, --lgb-num-leaves, --lgb-lambda-l1, --lgb-lambda-l2"
echo "XGB:    --xgb-tree-method, --xgb-max-depth, --xgb-subsample"
echo "CAT:    --cat-depth, --cat-l2-leaf-reg, --cat-border-count"
