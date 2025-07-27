from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .split_fold import cut_fold


def get_feature_columns(df, config):
    """Get feature columns excluding unused columns"""
    return [col for col in df.columns if col not in config.unused_col]


def plot_predictions(pred, oof_df):
    """Plot prediction distributions"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(pred, bins=100, density=True, alpha=0.5, label="Test")
    ax.hist(oof_df["oof_pred"], bins=100, density=True, alpha=0.5, label="OutOfFold")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()


def setup_logging(config, model_type):
    """Setup logging for training progress"""
    import logging
    from datetime import datetime

    # Create output directory if it doesn't exist
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Setup log file path - use experiment name if available
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = getattr(config, "EXP_NAME", "experiment")
    log_file = Path(config.OUTPUT_DIR) / f"{exp_name}_{model_type}_{timestamp}.log"

    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create logger
    logger = logging.getLogger(f"{model_type}_training")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers from this logger
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    logger.info(f"Starting {model_type} training...")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Log file: {log_file}")

    return logger


def save_models(models, config, model_type):
    """Save trained models to disk"""
    import pickle

    # Create models directory if it doesn't exist
    models_dir = Path(config.OUTPUT_DIR) / "models"
    models_dir.mkdir(exist_ok=True)

    # Save each fold model
    for fold, model in enumerate(models):
        model_path = models_dir / f"{model_type}_fold_{fold}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Save all models as a list
    all_models_path = models_dir / f"{model_type}_all_models.pkl"
    with open(all_models_path, "wb") as f:
        pickle.dump(models, f)

    print(f"Models saved to {models_dir}:")
    for fold in range(len(models)):
        print(f"  - {model_type}_fold_{fold}.pkl")
    print(f"  - {model_type}_all_models.pkl")

    return models_dir


def save_results(pred, oof_df, config, model_type, models=None):
    """Save prediction results and optionally models"""
    # Create output directory if it doesn't exist
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Save submission file
    sub = pd.DataFrame({"target": pred})
    sub_path = Path(config.OUTPUT_DIR) / f"ens_{model_type}.csv"
    sub.to_csv(sub_path, index=False)

    submission_path = Path(config.OUTPUT_DIR) / "submission.csv"
    sub.to_csv(submission_path, index=False)

    # Save OOF predictions
    oof_path = Path(config.OUTPUT_DIR) / f"oof_{model_type}.csv"
    oof_df.to_csv(oof_path, index=False)

    print(f"Results saved to {config.OUTPUT_DIR}:")
    print(f"  - ens_{model_type}.csv")
    print("  - submission.csv")
    print(f"  - oof_{model_type}.csv")

    # Save models if provided
    if models is not None:
        save_models(models, config, model_type)

    return sub


def load_and_prepare_data(config):
    """Load and prepare training and test data"""
    print("Loading data...")
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target distribution: {train_df['target'].value_counts()}")

    # Load feature data
    career = pd.read_csv(config.INPUT_DIR / "career.csv")
    dx = pd.read_csv(config.INPUT_DIR / "dx.csv")
    hr = pd.read_csv(config.INPUT_DIR / "hr.csv")
    overtime = pd.read_csv(config.INPUT_DIR / "overtime_work_by_month.csv")
    position = pd.read_csv(config.INPUT_DIR / "position_history.csv")

    # Create features
    features = create_features(train_df, test_df, career, dx, hr, overtime, position)

    # Merge features
    train_df = train_df.merge(features, on="社員番号", how="left")
    test_df = test_df.merge(features, on="社員番号", how="left")

    # Encode category
    le_category = LabelEncoder()
    all_categories = pd.concat([train_df["category"], test_df["category"]]).unique()
    le_category.fit(all_categories)

    train_df["category_encoded"] = le_category.transform(train_df["category"])
    test_df["category_encoded"] = le_category.transform(test_df["category"])

    # Create folds
    print("Creating folds...")
    train_df = cut_fold(train_df, config)

    # Fill missing values
    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)

    return train_df, test_df


def create_features(train, test, career, dx, hr, overtime, position):
    """Create features from auxiliary data"""
    print("Creating features...")

    # Get all unique employee IDs
    all_employees = pd.concat([train["社員番号"], test["社員番号"]]).unique()
    features = pd.DataFrame({"社員番号": all_employees})

    # 1. Career features
    if not career.empty:
        # Try to convert career responses to numeric
        career_processed = career.copy()

        # Get all columns except the first one (employee ID)
        career_cols = career.columns[1:]

        # Convert text responses to numeric values
        # Mapping typical Likert scale responses
        response_map = {
            "5 強くそう思う／とても当てはまる": 5,
            "4 そう思う／当てはまる": 4,
            "3 どちらとも言えない": 3,
            "2 あまりそう思わない／あまり当てはまらない": 2,
            "1 全くそう思わない／全く当てはまらない": 1,
        }

        # Apply mapping to all career columns
        for col in career_cols:
            career_processed[col] = career_processed[col].astype(str).map(response_map)
            # Fill any unmapped values with 3 (neutral)
            career_processed[col] = career_processed[col].fillna(3)

        # Calculate career metrics
        career_processed["career_sum"] = career_processed[career_cols].sum(axis=1)
        career_processed["career_mean"] = career_processed[career_cols].mean(axis=1)
        career_processed["career_max"] = career_processed[career_cols].max(axis=1)
        career_processed["career_min"] = career_processed[career_cols].min(axis=1)
        career_processed["career_std"] = career_processed[career_cols].std(axis=1)

        career_features = career_processed[
            [
                "社員番号",
                "career_sum",
                "career_mean",
                "career_max",
                "career_min",
                "career_std",
            ]
        ]
        features = features.merge(career_features, on="社員番号", how="left")

    # 2. DX training features
    if not dx.empty:
        dx_features = (
            dx.groupby("社員番号")
            .agg(
                {
                    "研修実施日": "count",
                    "研修カテゴリ": "nunique",
                }
            )
            .reset_index()
        )
        dx_features.columns = ["社員番号", "dx_count", "dx_category_count"]
        features = features.merge(dx_features, on="社員番号", how="left")

    # 3. HR training features
    if not hr.empty:
        hr_features = (
            hr.groupby("社員番号")
            .agg(
                {
                    "カテゴリ": ["count", "nunique"],
                }
            )
            .reset_index()
        )
        hr_features.columns = ["社員番号", "hr_count", "hr_category_count"]
        features = features.merge(hr_features, on="社員番号", how="left")

    # 4. Overtime features
    if not overtime.empty:
        overtime_features = (
            overtime.groupby("社員番号")["hours"]
            .agg(["sum", "mean", "max", "min", "std", "count"])
            .reset_index()
        )
        overtime_features.columns = [
            "社員番号",
            "overtime_sum",
            "overtime_mean",
            "overtime_max",
            "overtime_min",
            "overtime_std",
            "overtime_months",
        ]
        features = features.merge(overtime_features, on="社員番号", how="left")

    # 5. Position history features
    if not position.empty:
        # Latest position info
        latest_position = position.loc[position.groupby("社員番号")["year"].idxmax()]

        # Position changes count
        position_changes = position.groupby("社員番号").size().reset_index()
        position_changes.columns = ["社員番号", "position_changes"]

        # Encode work type and position
        le_work_type = LabelEncoder()
        le_position = LabelEncoder()

        latest_position["勤務区分_encoded"] = le_work_type.fit_transform(
            latest_position["勤務区分"]
        )
        latest_position["役職_encoded"] = le_position.fit_transform(
            latest_position["役職"]
        )

        position_features = latest_position[
            ["社員番号", "勤務区分_encoded", "役職_encoded"]
        ].merge(position_changes, on="社員番号", how="left")

        features = features.merge(position_features, on="社員番号", how="left")

    # Fill missing values
    features = features.fillna(0)

    print(f"Created features: {features.shape}")
    return features
