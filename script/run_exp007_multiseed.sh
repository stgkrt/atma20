#!/bin/bash

echo "=== exp007: Multi-Seed Optuna Experiments ==="
echo "Running each model with different random seeds for robustness analysis"
echo ""

# Configuration
TRIALS=30
FOLDS=5
SEEDS="42 510 111"  # Different seeds for robustness testing

# Function to run experiment for a specific model and seed
run_experiment() {
    local model=$1
    local seed=$2
    local exp_name="exp007_seed${seed}"
    local study_name="exp007_seed${seed}_optimization"

    model_upper=$(echo "$model" | tr '[:lower:]' '[:upper:]')
    echo "Starting ${exp_name}: ${model_upper} with seed ${seed}..."
    echo "Configuration: $TRIALS trials, $FOLDS folds, seed=$seed"
    
    python src/run_model.py $model \
        --exp-name "$exp_name" \
        --use-optuna \
        --optuna-trials $TRIALS \
        --optuna-study-name "$study_name" \
        --random-state $seed
    
    if [ $? -eq 0 ]; then
        echo "✓ ${exp_name} completed successfully!"
    else
        echo "✗ ${exp_name} failed!"
    fi
    echo "================================"
    
    # Short pause between experiments
    sleep 5
}

# Run experiments for each seed and model combination
for seed in $SEEDS; do
    echo ""
    echo "--- Starting experiments with seed=$seed ---"
    
    # LightGBM
    run_experiment "lgb" $seed
    
    # XGBoost
    run_experiment "xgb" $seed
    
    # CatBoost
    run_experiment "cat" $seed
    
    echo "--- Completed all models for seed=$seed ---"
    echo ""
done

echo ""
echo "=== exp006 Multi-Seed Experiments Completed! ==="
echo ""
echo "Results saved in directories:"
for seed in $SEEDS; do
    for model in lgb xgb cat; do
        exp_dir="exp006_seed${seed}_${model}"
        echo "- /kaggle/working/${exp_dir}/"
    done
done

