#!/bin/bash

echo "=== exp006: Multi-Seed Optuna Experiments ==="
echo "Running each model with different random seeds for robustness analysis"
echo ""

# Configuration
TRIALS=30  # Consistent with exp005
FOLDS=5
SEEDS="42 510 111"  # Different seeds for robustness testing

# Function to run experiment for a specific model and seed
run_experiment() {
    local model=$1
    local seed=$2
    local exp_name="exp006_seed${seed}_${model}"
    local study_name="exp006_seed${seed}_${model}_optimization"
    
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

echo ""
echo "=== Results Summary ==="
for seed in $SEEDS; do
    echo ""
    echo "Seed $seed results:"
    for model in lgb xgb cat; do
        exp_dir="exp006_seed${seed}_${model}"
        if [ -d "/kaggle/working/$exp_dir" ]; then
            echo "  ✓ $model: Directory exists"
            csv_files=$(ls "/kaggle/working/$exp_dir"/*.csv 2>/dev/null | wc -l)
            if [ "$csv_files" -gt 0 ]; then
                echo "    ✓ Output files: $csv_files CSV files"
                # Try to extract CV score from log if available
                log_file=$(find "/kaggle/working/$exp_dir" -name "*.log" 2>/dev/null | head -1)
                if [ -n "$log_file" ] && [ -f "$log_file" ]; then
                    cv_score=$(grep "Overall CV score" "$log_file" | tail -1 | grep -o "[0-9]\+\.[0-9]\+" | head -1)
                    if [ -n "$cv_score" ]; then
                        echo "    ✓ CV Score: $cv_score"
                    fi
                fi
            else
                echo "    ⚠ No CSV output files found"
            fi
        else
            echo "  ✗ $model: Directory not found"
        fi
    done
done
