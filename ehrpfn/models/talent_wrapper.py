from typing import *
from ehrpfn.models.TALENT.TALENT.model.utils import get_method

import json
import torch
import os.path as osp
from sklearn.base import BaseEstimator, ClassifierMixin
from argparse import Namespace
from TALENT.model.methods.base import check_softmax
from ehrpfn.utils.logger import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')
    

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'

def dict_to_namespace(d):
    """Recursively convert a dictionary to an argparse.Namespace."""
    if isinstance(d, dict):
        ns = Namespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    else:
        return d

class TALENTWrapper(BaseEstimator, ClassifierMixin):

    def __init__(
        self, model_name:str, 
        config_path:str=None, tune_config:str=None,
        ckpt_path:str=None, save_dir:str=None,
        categorical_feature_indices:List[int]=None, 
        valset=None, is_classification:bool=False, seed:int=None
    ):
        self.model_name = model_name
        self.categorical_feature_indices = categorical_feature_indices
        self.is_classification = is_classification
        self.valset = valset
        self.ckpt_path = ckpt_path

        if config_path is not None:
            args = json.load(open(config_path, 'r'))
            args['save_path'] = save_dir
            args['seed'] = seed
            args = dict_to_namespace(args)
            if tune_config is not None:
                tune_config = json.load(open(tune_config, 'r'))
                setattr(args, "config", tune_config)
        else:
            config_dir = "/home/mpham/workspace/source/ehrpfn/ehrpfn/models/TALENT/TALENT/configs"
            with open(f"{config_dir}/deep_configs.json", "r") as f:
                args = json.load(f)

            args['use_float'] = False
            args['save_path'] = save_dir

            with open(f"{config_dir}/default/" + model_name + '.json', "r") as f:
                default_para = json.load(f)

            args['seed'] = seed
            args['config'] = None

            if model_name in ["tabr", "modernNCA"]:
                args['cat_policy'] = "tabr_ohe"
                args['num_policy'] = "none"
                args['normalization'] = "none"
            elif model_name in ['tabicl']:
                args['cat_policy'] = "indices"
                args['num_policy'] = "none"
                args['normalization'] = "none"
                args['tune'] = False
            elif model_name in ['tabm','tabtransformer', 'trompt']:
                args['cat_policy'] = "indices"
            elif model_name in [ 'excelformer', 'tabnet', 'resnet']:
                args['cat_policy'] = "ordinal"
            elif model_name in ['beta']:
                args['cat_policy'] = "indices"
                args['num_policy'] = "none"
                args['normalization'] = "standard"
                args['num_nan_policy'] = 'mean'
                args['cat_nan_policy'] = 'new'

            
            args = dict_to_namespace(args)

            if tune_config is not None:
                tune_config = json.load(open(tune_config, 'r'))
                setattr(args, "config", tune_config)
            else:
                setattr(args, "config", default_para[model_name])

            # args.config = default_para[model_name]
            args.config['training']['n_bins'] = args.n_bins
            # Dump argparse_dict to a JSON file
            argparse_dict = vars(args)
            with open(osp.join(save_dir, 'args.json'), 'w') as f:
                json.dump(argparse_dict, f, indent=4)

        self.args = args
        self.method = get_method(model_name)(args, not is_classification)


    def fit(self, X, y, **kwargs):

        X_val, y_val = self.valset

        N_trainval, C_trainval = None, None

        if self.categorical_feature_indices is None or len(self.categorical_feature_indices) == 0:
            C_X_train = None
            C_X_val = None
            N_X_train = X.astype(float)
            N_X_val = X_val.astype(float)
            N_trainval = {
                'train': N_X_train,
                'val': N_X_val
            } 

        elif len(self.categorical_feature_indices) == X.shape[1]:
            C_X_train = X.astype(int)
            C_X_val = X_val.astype(int)
            N_X_train = None
            N_X_val = None
            C_trainval = {
                'train': C_X_train,
                'val': C_X_val
            }
        else:
            C_X_train = X[:, self.categorical_feature_indices].astype(int)
            N_X_train = X[:, [i for i in range(X.shape[1]) if i not in self.categorical_feature_indices]].astype(float)
            C_X_val = X_val[:, self.categorical_feature_indices].astype(int)
            N_X_val = X_val[:, [i for i in range(X_val.shape[1]) if i not in self.categorical_feature_indices]].astype(float)

            N_trainval = {
                'train': N_X_train,
                'val': N_X_val
            } 
            
            C_trainval = {
                'train': C_X_train,
                'val': C_X_val
            }

        y_trainval = {
            'train': y,
            'val': y_val
        }

        num_classes = len(set(y)) if self.is_classification else 1

        info = {
            "task_type": REGRESSION if not self.is_classification else BINCLASS if num_classes == 2 else MULTICLASS, 
            "n_num_features": N_X_train.shape[1] if N_X_train is not None else 0,
            "n_cat_features": C_X_train.shape[1] if C_X_train is not None else 0,
        }
        train_val_data = (N_trainval,C_trainval,y_trainval)

        if self.ckpt_path is not None and osp.isfile(self.ckpt_path):
            time_cost = self.method.fit(train_val_data, info, train = False) # Not training again, just for constructing model
        else:
            time_cost = self.method.fit(train_val_data, info)


    def predict(self, X):
        N_test, C_test = None, None

        if self.categorical_feature_indices is None or len(self.categorical_feature_indices) == 0:
            C_X_test = None
            N_X_test = X.astype(float)
            N_test = {'test': N_X_test}
        elif len(self.categorical_feature_indices) == X.shape[1]:
            C_X_test = X.astype(int)
            N_X_test = None
            C_test = {'test': C_X_test}
        else:
            C_X_test = X[:, self.categorical_feature_indices].astype(int)
            N_X_test = X[:, [i for i in range(X.shape[1]) if i not in self.categorical_feature_indices]].astype(float)
            N_test = {'test': N_X_test}
            C_test = {'test': C_X_test}
            
        test_data = (N_test,C_test)

        if self.ckpt_path is not None and osp.isfile(self.ckpt_path):
            try:
                self.method.model.load_state_dict(
                    torch.load(self.ckpt_path)['params']
                )
            except:
                try:
                    self.method.load_from(self.ckpt_path)
                except Exception as e:
                    LOGGER.text(f"Failed to load trained model. {e}", level=LoggerObserver.ERROR)

        predictions = self.method.predict_proba(test_data)

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if self.is_classification:
            predictions = check_softmax(predictions)

        return predictions

    # def get_params(self, deep=True):
    #     return {'param1': self.param1, 'param2': self.param2}

    # def set_params(self, **parameters):
    #     self.param1 = parameters.get('param1', self.param1)
    #     self.param2 = parameters.get('param2', self.param2)
    #     self.model = ModelClass(self.param1, self.param2)
    #     return self