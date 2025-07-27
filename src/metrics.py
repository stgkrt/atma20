from sklearn.metrics import roc_auc_score


def competition_metrics(y_true, pred):
    """Calculate competition metric (ROC AUC)"""
    return roc_auc_score(y_true, pred)

def evaluate_and_report(oof_df, config, model_name):
    """Evaluate final score and report results"""
    oof_score = competition_metrics(oof_df[config.target_col],
                                    oof_df["oof_pred"].values)
    print(f"Final OOF Score: {oof_score:.4f}")
    print(f"{model_name} training completed!")
    return oof_score
