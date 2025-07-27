import numpy as np


def make_prediction(models, test_df, feat_cols, model_type="sklearn"):
    """Make predictions using trained models"""
    if model_type == "lgb":
        # LightGBM uses predict method directly
        pred = np.array([model.predict(test_df[feat_cols].values) 
                         for model in models])
    else:
        # XGBoost and CatBoost use predict_proba
        pred = np.array([model.predict_proba(test_df[feat_cols].values)[:, 1]
                         for model in models])
    
    # Average predictions across models
    pred = np.mean(pred, axis=0)
    return pred


def print_training_info(config, params, model_name):
    """Print training information"""
    print(f"Starting {model_name} training...")
    print(f"Config: {config.EXP_NAME}")
    print(f"Parameters: {params}")


