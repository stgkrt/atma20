#!/bin/bash

echo "=== exp005: Sequential Optuna Experiments (Efficient) ==="
echo "Running models one by one with reduced trials for faster execution"
echo ""

# Configuration
TRIALS=30  # Reduced trials for faster execution
FOLDS=5
RANDOM_STATE=510

# exp005_lgb: LightGBM with Optuna optimization
echo "Starting exp005_lgb: LightGBM with Optuna optimization..."
echo "Configuration: $TRIALS trials, $FOLDS folds"
python src/models/train_lgb.py \
    --exp-name "exp005_lgb" \
    --use-optuna \
    --optuna-trials $TRIALS \
    --optuna-study-name "exp005_lgb_optimization" \
    --n-folds $FOLDS \
    --random-state $RANDOM_STATE

if [ $? -eq 0 ]; then
    echo "✓ exp005_lgb completed successfully!"
else
    echo "✗ exp005_lgb failed!"
fi
echo "================================"

# Wait a moment before next experiment
sleep 10

# exp005_xgb: XGBoost with Optuna optimization  
echo "Starting exp005_xgb: XGBoost with Optuna optimization..."
echo "Configuration: $TRIALS trials, $FOLDS folds"
python src/models/train_xgb.py \
    --exp-name "exp005_xgb" \
    --use-optuna \
    --optuna-trials $TRIALS \
    --optuna-study-name "exp005_xgb_optimization" \
    --n-folds $FOLDS \
    --random-state $RANDOM_STATE

if [ $? -eq 0 ]; then
    echo "✓ exp005_xgb completed successfully!"
else
    echo "✗ exp005_xgb failed!"
fi
echo "================================"

# Wait a moment before next experiment
sleep 10

# exp005_cat: CatBoost with Optuna optimization
echo "Starting exp005_cat: CatBoost with Optuna optimization..."
echo "Configuration: $TRIALS trials, $FOLDS folds"
python src/models/train_cat.py \
    --exp-name "exp005_cat" \
    --use-optuna \
    --optuna-trials $TRIALS \
    --optuna-study-name "exp005_cat_optimization" \
    --n-folds $FOLDS \
    --random-state $RANDOM_STATE

if [ $? -eq 0 ]; then
    echo "✓ exp005_cat completed successfully!"
else
    echo "✗ exp005_cat failed!"
fi
echo "================================"

echo ""
echo "=== exp005 Sequential Experiments Completed! ==="
echo "Results saved in:"
echo "- /kaggle/working/exp005_lgb/"
echo "- /kaggle/working/exp005_xgb/"  
echo "- /kaggle/working/exp005_cat/"

echo ""
echo "Checking results..."
for exp_dir in exp005_lgb exp005_xgb exp005_cat; do
    if [ -d "/kaggle/working/$exp_dir" ]; then
        echo "✓ $exp_dir: Directory exists"
        csv_files=$(ls "/kaggle/working/$exp_dir"/*.csv 2>/dev/null | wc -l)
        if [ "$csv_files" -gt 0 ]; then
            echo "  ✓ Output files: $csv_files CSV files generated"
        else
            echo "  ⚠ No CSV output files found"
        fi
    else
        echo "✗ $exp_dir: Directory not found"
    fi
done
