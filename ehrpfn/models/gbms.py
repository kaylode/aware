from typing import *
import json
import os
import os.path as osp
import optuna
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
import wandb
import xgboost as xgb
import catboost
import lightgbm
import numpy as np
from ehrpfn.utils.logger import LoggerObserver
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score
)
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.integration.wandb import WeightsAndBiasesCallback
LOGGER = LoggerObserver.getLogger("main")
from ehrpfn.utils.visualize import visualize_shap

# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"



def fit_xgboost(params, X_train, X_val, y_train, y_val, is_classification=True):
    if is_classification:
        num_classes = len(np.unique(y_train))
        objective_fn = "multi:softprob" if num_classes > 2 else "binary:logistic"
        params.update({
            'eval_metric': ['auc'],
            'objective': objective_fn,
        })
        model = xgb.XGBClassifier(**params)
    else:
        params.update({
            'eval_metric': ['rmse'],
            'objective': 'reg:squarederror',
        })
        model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, early_stopping_rounds=50)
    return model, params

def fit_catboost(params, X_train, X_val, y_train, y_val, is_classification:bool=True, cat_features=None):
    from catboost import CatBoostClassifier, CatBoostRegressor
    params.update({
        'verbose': 100,
    })
    if not is_classification:
        params.update({
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
        })
        clf = CatBoostRegressor(**params)
    else:
        clf = CatBoostClassifier(**params)  
    
    clf.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        cat_features=None, 
        early_stopping_rounds=50
    )
    return clf, params

def fit_lightgbm(params, X_train, X_val, y_train, y_val, is_classification:bool=True, cat_features=None):
    from lightgbm import LGBMClassifier, LGBMRegressor
    params.update({'verbose': 100})

    if not is_classification:
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
        })
        clf = LGBMRegressor(**params)
    else:
        clf = LGBMClassifier(**params)  

    clf.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        categorical_feature=cat_features,
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=50, verbose=False),
            lightgbm.log_evaluation(period=2000)
        ]
    )
    return clf, params


def fit_rf(params, X_train, X_val, y_train, y_val, is_classification:bool=True):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    params.update({'verbose': 1})
    if not is_classification:
        params.update({'n_jobs': -1, 'criterion': 'squared_error'})
        clf = RandomForestRegressor(**params)
    else:
        clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    return clf, params

def fit_svm(params, X_train, X_val, y_train, y_val):
    from sklearn.svm import LinearSVC, SVC
    params.update({'kernel': 'linear', 'probability': True, 'verbose': True})
    clf = SVC(**params)  
    clf.fit(X_train, y_train)
    return clf, params

def fit_logistics(params, X_train, X_val, y_train, y_val):
    from sklearn.linear_model import LogisticRegression
    params.update({'verbose': 1, 'penalty': 'l2',})
    clf = LogisticRegression(**params)  
    clf.fit(X_train, y_train)
    return clf, params

def fit_adaboost(params, X_train, X_val, y_train, y_val):
    from sklearn.ensemble import AdaBoostClassifier
    params.update({'random_state': 0})
    clf = AdaBoostClassifier(**params)
    clf.fit(X_train, y_train)
    return clf, params

def objective(trial, X_train, X_val, y_train, y_val, is_classification=True, method="xgboost", cat_features=None):

    wandb.run.config.update({
        'MODEL': {
            'MODEL_NAME': method,
        },
        'MODEL_NAME': method,
        'FOLD': os.environ.get("FOLD", None),
        'TASK_NAME': os.environ.get("TASK_NAME", None),
    })

    num_classes = len(np.unique(y_train))

    assert method in ["xgboost", "rf", "svm", "logistics", "adaboost", "catboost", "lightgbm"], f"{method} is not supported"

    if method == "xgboost":
        model, params = fit_xgboost({
            'n_estimators': trial.suggest_int('n_estimators', 100, 300), #The number of sequential trees to be modeled
            'max_depth': trial.suggest_int('max_depth', 1, 9), # The maximum depth of a tree.higher depth will allow model to learn relations very specific to a particular sample. Should be tuned
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0), # impact of each tree on the final outcome
            'gamma': trial.suggest_float('gamma', 0.001, 1.0), #This will anyways be tuned later.
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0), #This will anyways be tuned later.
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0), #This will anyways be tuned later.
        }, X_train, X_val, y_train, y_val, is_classification=is_classification)

    elif method == "catboost":
        model, params = fit_catboost({
            'iterations': trial.suggest_int('iterations', 100, 300),
            'depth': trial.suggest_int('depth', 1, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 1.0),
        }, X_train, X_val, y_train, y_val, is_classification=is_classification, cat_features=cat_features)

    elif method == "lightgbm":
        model, params = fit_lightgbm({
            'n_estimators': trial.suggest_int('n_estimators', 100, 300), #The number of sequential trees to be modeled
            'max_depth': trial.suggest_int('max_depth', 1, 9), # The maximum depth of a tree.higher depth will allow model to learn relations very specific to a particular sample. Should be tuned
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0), # impact of each tree on the final outcome
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0), #This will anyways be tuned later.
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0), #This will anyways be tuned later.
        }, X_train, X_val, y_train, y_val, is_classification=is_classification, cat_features=cat_features)

    elif method == "rf":
        model, params = fit_rf({
            'n_estimators': trial.suggest_int('n_estimators', 100, 300), #The number of sequential trees to be modeled
            'max_depth': trial.suggest_int('max_depth', 1, 9), # The maximum depth of a tree.higher depth will allow model to learn relations very specific to a particular sample. Should be tuned
            # 'criterion': trial.suggest_categorical("criterion", ["gini", "log_loss", "entropy"]),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 30), 
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 10),
        }, X_train, X_val, y_train, y_val, is_classification=is_classification)

    elif method == "svm":
        model, params = fit_svm({
            'gamma': trial.suggest_float('gamma', 0.001, 1.0),
            'C': trial.suggest_float('C', 0.001, 1.0),
        }, X_train, X_val, y_train, y_val, is_classification=is_classification)

    elif method == "logistics":
        model, params = fit_logistics({
            'C': trial.suggest_float('C', 0.001, 1.0),
        }, X_train, X_val, y_train, y_val, is_classification=is_classification)

    elif method == "adaboost":
        model, params = fit_adaboost({
            'n_estimators': trial.suggest_int('n_estimators', 100, 300), #The number of sequential trees to be modeled
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0), # impact of each tree on the final outcome
        }, X_train, X_val, y_train, y_val, is_classification=is_classification)

    else:
        raise NotImplementedError()
    
    # Validate the model
    if is_classification:
        preds = model.predict_proba(X_val)
        pred_labels = np.argmax(preds, axis=1)
    else:
        preds = model.predict(X_val)
        pred_labels = preds

    wandb.run.config.update(params)

    # Evaluate the model
    if not is_classification:
        rmse = np.sqrt(np.mean((y_val - pred_labels) ** 2))
        # Log the metrics to wandb
        wandb.run.log({
            "Validation/RMSE": rmse,
        })
        return float(rmse)
    else:
        f1score = f1_score(y_val, pred_labels, average='binary', pos_label=1)
        mcc_score = matthews_corrcoef(y_val, pred_labels)
        precision = precision_score(y_val, pred_labels, average='binary', pos_label=1)
        recall = recall_score(y_val, pred_labels, average='binary', pos_label=1)
        accuracy = accuracy_score(y_val, pred_labels)

        # Log the metrics to wandb
        wandb.run.log({
            "Validation/Accuracy": accuracy,
            "Validation/F1": f1score,
            "Validation/MCC": mcc_score,
            "Validation/Precision": precision,
            "Validation/Recall": recall,
        })
        return float(f1score)

class Tuner:
    def __init__(
        self,
        storage:str=None,
        study_name: str = None,
        n_trials: int = 100,
        direction: str = "maximize",
        pruner=None,
        sampler=None,
        save_dir: str = None,
        method: str = "xgboost",
        use_best_params: bool = False,
        wandb_kwargs: dict = None,
    ) -> None:
        
        if storage is not None:
            if storage.endswith(".log"):
                self.storage = JournalStorage(JournalFileStorage(storage))

        self.save_dir = save_dir
        self.study_name = study_name
        self.n_trials = n_trials
        self.direction = direction
        self.pruner = pruner
        self.sampler = sampler
        self.save_dir = save_dir
        self.method = method
        self.use_best_params = use_best_params
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=self.storage,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )

        # Initialise wandb callback
        self.wandb_kwargs = {
            "entity": wandb_kwargs.get("entity", "kaylode"),
            "project": wandb_kwargs.get("project", "tabpfn-ehr"),
            "reinit": True,
            "group": "tabpfn-ehr-optuna",
            "job_type": "optuna",
            "tags": [self.method, "optuna"] + wandb_kwargs.get("tags", []),
            "resume": "allow",
            "name": f"{self.method}",
            "dir": self.save_dir,
        }
        self.WANDB_CALLBACK = WeightsAndBiasesCallback(
            metric_name="Validation/F1", 
            wandb_kwargs=self.wandb_kwargs, 
            as_multirun=True
        )


    def tune(self, X_train, X_val, y_train, y_val, is_classification:bool=True, cat_features=None):
        wrapped_objective = lambda trial: objective(
            trial, X_train, X_val, y_train, y_val, 
            is_classification=is_classification,
            method=self.method, 
            cat_features=cat_features
        )

        decorator = self.WANDB_CALLBACK.track_in_wandb()
        wrapped_objective = decorator(wrapped_objective)

        if not self.use_best_params:
            try:
                self.study.optimize(wrapped_objective, n_trials=self.n_trials, callbacks=[self.WANDB_CALLBACK])
            except KeyboardInterrupt:
                LOGGER.text("KeyboardInterrupt", level=LoggerObserver.ERROR)
        best_trial = self.study.best_trial
        self.save_best_config(best_trial.params)

        wandb_kwargs = {
            "name": f"{self.method}",
            "entity": self.wandb_kwargs.get("entity", "kaylode"),
            "project": self.wandb_kwargs.get("project", "tabpfn-ehr"),
            "reinit": True,
            "group": "tabpfn-ehr-finetune",
            "job_type": "train",
            "tags": list(set([self.method, "optuna", "best"] + self.wandb_kwargs.get("tags", []))),
            "resume": "allow",
            "name": f"{self.method}",
            "dir": self.save_dir,
        }
        wandb.init(**wandb_kwargs)


        if self.save_dir is not None:
            leaderboard_df = self.leaderboard()
            leaderboard_df.to_csv(osp.join(self.save_dir, 'leaderboard.csv'))
            leaderboard_df.to_json(osp.join(self.save_dir, 'leaderboard.json'), orient='records')
            LOGGER.text(f"Leaderboard saved to {self.save_dir}/leaderboard.csv", level=LoggerObserver.INFO)
            figs = self.visualize("all")
            os.makedirs(osp.join(self.save_dir, 'figures'), exist_ok=True)
            for fig_name, fig in figs:
                fig.write_image(osp.join(self.save_dir, 'figures', f"{fig_name}.png"))
                LOGGER.text(f"{fig_name} plot saved to {self.save_dir}/{fig_name}.png", level=LoggerObserver.INFO)
        
        # Log the best trial to wandb
        wandb.run.config.update({
            'MODEL': {
                'MODEL_NAME': self.method,
            },
            'MODEL_NAME': self.method,
            'FOLD': os.environ.get("FOLD", None),
            'TASK_NAME': os.environ.get("TASK_NAME", None),
        })

        if self.method == "xgboost":
            best_model, params = fit_xgboost(best_trial.params, X_train, X_val, y_train, y_val, is_classification=is_classification)
        elif self.method == "catboost":
            best_model, params = fit_catboost(best_trial.params, X_train, X_val, y_train, y_val, is_classification=is_classification)
        elif self.method == "lightgbm":
            best_model, params = fit_lightgbm(best_trial.params, X_train, X_val, y_train, y_val, is_classification=is_classification)
        elif self.method == "rf":
            best_model, params = fit_rf(best_trial.params, X_train, X_val, y_train, y_val, is_classification=is_classification)
        elif self.method == "svm":
            best_model, params = fit_svm(best_trial.params, X_train, X_val, y_train, y_val, is_classification=is_classification)
        elif self.method == "logistics":
            best_model, params = fit_logistics(best_trial.params, X_train, X_val, y_train, y_val, is_classification=is_classification)
        elif self.method == 'adaboost':
            best_model, params = fit_adaboost(best_trial.params, X_train, X_val, y_train, y_val, is_classification=is_classification)
        else:
            raise NotImplementedError()
        
        wandb.run.config.update(params)
        
        # Evaluate the model
        if not is_classification:
            preds = best_model.predict(X_val)
            pred_labels = preds
        else:
            preds = best_model.predict_proba(X_val)
            pred_labels = np.argmax(preds, axis=1)

        if is_classification:
            f1score = f1_score(y_val, pred_labels, average='binary', pos_label=1)
            mcc_score = matthews_corrcoef(y_val, pred_labels)
            precision = precision_score(y_val, pred_labels, average='binary', pos_label=1)
            recall = recall_score(y_val, pred_labels, average='binary', pos_label=1)
            accuracy = accuracy_score(y_val, pred_labels)
            # Log the metrics to wandb
            wandb.run.log({
                "Validation/Accuracy": accuracy,
                "Validation/F1": f1score,
                "Validation/MCC": mcc_score,
                "Validation/Precision": precision,
                "Validation/Recall": recall,
            })
        else:
            rmse = np.sqrt(np.mean((y_val - pred_labels) ** 2))
            # Log the metrics to wandb
            wandb.run.log({
                "Validation/RMSE": rmse,
            })
        LOGGER.text(f"Best trial: {best_trial.number} with value: {best_trial.value}", level=LoggerObserver.INFO)

        if is_classification:
            fig = visualize_shap(
                best_model, 
                X_val, 
                feature_names=[],
                classnames=['0', '1'],
                plot_type="bar",
                plot_size=(12, 6),
                cross_validation=False
            )
            fig.savefig(osp.join(self.save_dir, 'shap_summary.png'), bbox_inches='tight')
            wandb.log({"Shap Summary": wandb.Image(fig)})
            LOGGER.text(f"Shap Summary plot saved to {self.save_dir}/shap_summary.png", level=LoggerObserver.INFO)
        else:
            LOGGER.text("Shap values are not supported for regression tasks.", level=LoggerObserver.WARNING)

        

        wandb.finish()

        return best_model
    
    def save_best_config(self, best_params: Dict):
        with open(os.path.join(self.save_dir, "best_config.json"), "w") as f:
            json.dump(best_params, f, indent=4)
        
        LOGGER.text(f"Best config saved to {self.save_dir}/best_config.json", level=LoggerObserver.INFO)

    def leaderboard(self):
        """Print leaderboard of all trials"""
        df = self.study.trials_dataframe()
        df.columns = [col.replace("user_attrs_", "") for col in df.columns]
        return df

    def visualize(self, plot: str, plot_params: dict = {}):
        """Visualize everything"""

        allow_plot_types = [
            "history",
            "contour",
            "edf",
            "intermediate_values",
            "parallel_coordinate",
            "param_importances",
            "slice",
        ]
        assert plot in ["all", *allow_plot_types], f"{plot} is not supported by Optuna"

        try:
            if plot == "history":
                fig = plot_optimization_history(self.study, **plot_params)
            elif plot == "contour":
                fig = plot_contour(self.study, **plot_params)
            elif plot == "edf":
                fig = plot_edf(self.study, **plot_params)
            elif plot == "intermediate_values":
                fig = plot_intermediate_values(self.study)
            elif plot == "parallel_coordinate":
                fig = plot_parallel_coordinate(self.study, **plot_params)
            elif plot == "param_importances":
                fig = plot_param_importances(self.study, **plot_params)
            elif plot == "slice":
                fig = plot_slice(self.study, **plot_params)
            elif plot == "all":
                fig = []
                for plot_type in allow_plot_types:
                    one_fig = self.visualize(plot_type, plot_params)
                    if one_fig is not None:
                        fig.append((plot_type, one_fig))
            else:
                LOGGER.text(
                    f"{plot} is not supported by Optuna", level=LoggerObserver.ERROR
                )
                raise ValueError()

            if plot != "all":
                wandb.log({f"Plot/{plot}": fig})
        
        except Exception as e:
            LOGGER.text(f"Plotting error: {e}", level=LoggerObserver.ERROR)
            return None

        return fig