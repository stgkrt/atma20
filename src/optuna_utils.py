"""
Optuna utilities for hyperparameter optimization
"""

import logging
import warnings

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")


class OptunaOptimizer:
    """Base class for Optuna optimization"""

    def __init__(self, config, n_trials=100, seed=42):
        self.config = config
        self.n_trials = n_trials
        self.seed = seed
        self.logger = logging.getLogger("optuna_optimizer")

    def create_study(self, study_name: str):
        """Create an Optuna study"""
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name
        )
        return study

    def optimize(self, df, study_name: str):
        """Run optimization"""
        study = self.create_study(study_name)

        def objective(trial):
            return self.objective(trial, df)

        self.logger.info(f"Starting optimization with {self.n_trials} trials...")
        study.optimize(objective, n_trials=self.n_trials)

        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best score: {study.best_value:.6f}")
        self.logger.info(f"Best params: {study.best_params}")

        return study.best_params, study.best_value

    def objective(self, trial, df):
        """Override this method in subclasses"""
        raise NotImplementedError


class LightGBMOptimizer(OptunaOptimizer):
    """LightGBM hyperparameter optimizer"""

    def suggest_params(self, trial):
        """Suggest hyperparameters for LightGBM"""
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": self.config.random_state
            if hasattr(self.config, "random_state")
            else self.seed,
            "seed": self.config.random_state
            if hasattr(self.config, "random_state")
            else self.seed,
            # Suggest hyperparameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        return params

    def objective(self, trial, df):
        """Objective function for LightGBM optimization"""
        import lightgbm as lgb

        from src.data_utils import get_feature_columns
        from src.metrics import competition_metrics
        from src.split_fold import prepare_fold_data

        params = self.suggest_params(trial)
        feat_cols = get_feature_columns(df, self.config)

        cv_scores = []

        for fold in self.config.FOLDS:
            x_train, y_train, x_valid, y_valid, _ = prepare_fold_data(
                df, fold, self.config, feat_cols
            )

            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

            # Use a smaller number of rounds for optimization
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_eval],
                num_boost_round=300,  # Further reduced for speed
                callbacks=[
                    lgb.early_stopping(20),  # Reduced early stopping
                    lgb.log_evaluation(0),  # Silent
                ],
            )

            pred = model.predict(x_valid)
            score = competition_metrics(y_valid, pred)
            cv_scores.append(score)

            # Prune unpromising trials
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(cv_scores) / len(cv_scores)


class XGBoostOptimizer(OptunaOptimizer):
    """XGBoost hyperparameter optimizer"""

    def suggest_params(self, trial):
        """Suggest hyperparameters for XGBoost"""
        params = {
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "random_state": self.config.random_state
            if hasattr(self.config, "random_state")
            else self.seed,
            "seed": self.config.random_state
            if hasattr(self.config, "random_state")
            else self.seed,
            # Suggest hyperparameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return params

    def objective(self, trial, df):
        """Objective function for XGBoost optimization"""
        import xgboost as xgb

        from src.data_utils import get_feature_columns
        from src.metrics import competition_metrics
        from src.split_fold import prepare_fold_data

        params = self.suggest_params(trial)
        feat_cols = get_feature_columns(df, self.config)

        cv_scores = []

        for fold in self.config.FOLDS:
            x_train, y_train, x_valid, y_valid, _ = prepare_fold_data(
                df, fold, self.config, feat_cols
            )

            dtrain = xgb.DMatrix(x_train, label=y_train)
            dvalid = xgb.DMatrix(x_valid, label=y_valid)

            # Use a smaller number of rounds for optimization
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=300,  # Further reduced for speed
                evals=[(dvalid, "valid")],
                early_stopping_rounds=20,  # Reduced early stopping
                verbose_eval=0,  # Silent
            )

            pred = model.predict(dvalid)
            score = competition_metrics(y_valid, pred)
            cv_scores.append(score)

            # Prune unpromising trials
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(cv_scores) / len(cv_scores)


class CatBoostOptimizer(OptunaOptimizer):
    """CatBoost hyperparameter optimizer"""

    def suggest_params(self, trial):
        """Suggest hyperparameters for CatBoost"""
        params = {
            "objective": "CrossEntropy",
            "eval_metric": "AUC",
            "verbose": 0,
            "random_state": self.config.random_state
            if hasattr(self.config, "random_state")
            else self.seed,
            # Suggest hyperparameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 10.0
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-8, 10.0, log=True
            ),
        }
        return params

    def objective(self, trial, df):
        """Objective function for CatBoost optimization"""
        from catboost import CatBoostClassifier, Pool

        from src.data_utils import get_feature_columns
        from src.metrics import competition_metrics
        from src.split_fold import prepare_fold_data

        params = self.suggest_params(trial)
        feat_cols = get_feature_columns(df, self.config)

        cv_scores = []

        for fold in self.config.FOLDS:
            x_train, y_train, x_valid, y_valid, _ = prepare_fold_data(
                df, fold, self.config, feat_cols
            )

            train_pool = Pool(x_train, y_train)
            valid_pool = Pool(x_valid, y_valid)

            # Use a smaller number of rounds for optimization
            model = CatBoostClassifier(
                iterations=300,  # Further reduced for speed
                early_stopping_rounds=20,  # Reduced early stopping
                **params,
            )

            model.fit(train_pool, eval_set=valid_pool, verbose=False)

            pred = model.predict_proba(x_valid)[:, 1]
            score = competition_metrics(y_valid, pred)
            cv_scores.append(score)

            # Prune unpromising trials
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(cv_scores) / len(cv_scores)


def get_optimizer(model_type: str, config, n_trials: int = 100, seed: int = 42):
    """Factory function to get the appropriate optimizer"""
    optimizers = {
        "lgb": LightGBMOptimizer,
        "xgb": XGBoostOptimizer,
        "cat": CatBoostOptimizer,
    }

    if model_type not in optimizers:
        raise ValueError(f"Unsupported model type: {model_type}")

    return optimizers[model_type](config, n_trials, seed)
