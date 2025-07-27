
from sklearn.model_selection import StratifiedKFold


def cut_fold(train_df_, config):
    """Create folds for cross validation"""
    fold = StratifiedKFold(n_splits=len(config.FOLDS))
    
    feat_cols = [col for col in train_df_.columns
                 if col not in config.unused_col]
    X = train_df_[feat_cols].values
    y = train_df_[config.target_col].values
    
    cv = fold.split(X, y)

    train_df_["fold"] = -1
    for fold_idx, (train_index, valid_index) in enumerate(cv):
        train_df_.loc[valid_index, "fold"] = fold_idx
    return train_df_

def prepare_fold_data(df, fold, config, feat_cols):
    """Prepare training and validation data for a specific fold"""
    train_df_ = df[df["fold"] != fold].reset_index(drop=True)
    valid_df_ = df[df["fold"] == fold].reset_index(drop=True)
    idx_valid = df[df["fold"] == fold].index.values
    
    x_train = train_df_[feat_cols]
    y_train = train_df_[config.target_col]
    x_valid = valid_df_[feat_cols]
    y_valid = valid_df_[config.target_col]
    
    return x_train, y_train, x_valid, y_valid, idx_valid


