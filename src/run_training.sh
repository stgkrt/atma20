#!/bin/bash

# ATMA20 Training Scripts Execution
# This script provides easy ways to run the training pipeline

set -e

# Change to the script directory
cd "$(dirname "$0")"

# Default values
MODEL="all"
ENSEMBLE=false

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL     Specify model to run: lgb, xgb, cat, or all (default: all)"
    echo "  -e, --ensemble        Create ensemble of all models"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all models"
    echo "  $0 -m lgb            # Run only LightGBM"
    echo "  $0 -m xgb            # Run only XGBoost"
    echo "  $0 -m cat            # Run only CatBoost"
    echo "  $0 -e                # Run all models and create ensemble"
    echo "  $0 -m \"lgb xgb\" -e   # Run LightGBM and XGBoost, then ensemble"
    echo ""
    echo "Individual model execution:"
    echo "  python models/train_lgb.py [args]  # Run LightGBM with custom args"
    echo "  python models/train_xgb.py [args]  # Run XGBoost with custom args"
    echo "  python models/train_cat.py [args]  # Run CatBoost with custom args"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--ensemble)
            ENSEMBLE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "========================================"
echo "ATMA20 Training Pipeline"
echo "========================================"
echo "Model(s): $MODEL"
echo "Ensemble: $ENSEMBLE"
echo "========================================"

# Prepare Python command
PYTHON_CMD="python run_all.py"

# Add model specification
if [ "$MODEL" != "all" ]; then
    PYTHON_CMD="$PYTHON_CMD --models $MODEL"
fi

# Add ensemble flag
if [ "$ENSEMBLE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --ensemble"
fi

echo "Executing: $PYTHON_CMD"
echo ""

# Execute the Python script
eval $PYTHON_CMD

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"

# List generated files
echo "Generated files:"
ls -la *.csv 2>/dev/null || echo "No CSV files found"
