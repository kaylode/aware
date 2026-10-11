from tqdm import tqdm
from TALENT.model.utils import get_deep_args,show_results,tune_hyper_parameters,get_method,set_seeds
import os
import os.path as osp
import json
from sklearn.model_selection import train_test_split
from TALENT.logger import LoggerObserver
from ehrmeds.static import (
    get_mimic_dataset, get_mimiciv_default_configs, get_mimiciv_hai_configs,
    get_eicu, get_eicu_default_configs, get_eicu2_default_configs,
    get_sjh_dataset, get_sjh_cpe_default_configs,
    get_openml_ehr, get_openml_ehr_default_configs,
    get_eicu_v2, get_mimiciv_v2,
    get_eicu_v2_default_configs, get_mimiciv_v2_default_configs,
)

LOGGER = LoggerObserver.getLogger('main')

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'


if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []
    args,default_para,opt_space = get_deep_args()

    dataset_name = args.data_name
    TASK = args.task
    SELECTION_METHOD = args.selection_method
    if SELECTION_METHOD == 'null':
        SELECTION_METHOD = None

    if dataset_name == "mimiciv":
        configs = get_mimiciv_default_configs(
            TASK=TASK,
            selection_method=SELECTION_METHOD,
        )
        dataset = get_mimic_dataset(**configs)
    elif dataset_name == "mimiciv_hai":
        configs = get_mimiciv_hai_configs(
            TASK=TASK,
            selection_method=SELECTION_METHOD,
        )
        dataset = get_mimic_dataset(**configs)
    elif dataset_name == "eicu":
        configs = get_eicu_default_configs(
            TASK=TASK, 
            selection_method=SELECTION_METHOD,
        )
        dataset = get_eicu(**configs)
    elif dataset_name == "eicu2":
        configs = get_eicu2_default_configs(
            TASK=TASK, 
            selection_method=SELECTION_METHOD,
        )
        dataset = get_eicu(**configs)
    elif dataset_name == "sjh_cpe":
        configs = get_sjh_cpe_default_configs(
            TASK=TASK,
            selection_method=SELECTION_METHOD,
        )
        dataset = get_sjh_dataset(**configs)
    elif dataset_name == "openml_ehr":
        if TASK is not None:
            force_ids = [int(TASK)]
        else:
            force_ids = None
        configs = get_openml_ehr_default_configs(
            force_ids=force_ids,
        )
        dataset = get_openml_ehr(**configs)[0]
        TASK = dataset['data_name']
    elif dataset_name == "eicu_hcai":
        configs = get_eicu_v2_default_configs(
            TASK=TASK,
            selection_method=SELECTION_METHOD,
        )
        dataset = get_eicu_v2(**configs)
    elif dataset_name == "mimiciv_hcai":
        configs = get_mimiciv_v2_default_configs(
            TASK=TASK,
            selection_method=SELECTION_METHOD,
        )
        dataset = get_mimiciv_v2(**configs)


    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    is_classification = True


    categorical_feature_indices = dataset['categorical_feature_indices']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=1702, 
        stratify=y_train if is_classification else None
    )

    N_trainval, C_trainval = None, None

    if categorical_feature_indices is None or len(categorical_feature_indices) == 0:
        C_X_train = None
        C_X_val = None
        N_X_train = X_train.astype(float)
        N_X_val = X_val.astype(float)
        N_trainval = {
            'train': N_X_train,
            'val': N_X_val
        } 

    elif len(categorical_feature_indices) == X_train.shape[1]:
        C_X_train = X_train.astype(int)
        C_X_val = X_val.astype(int)
        N_X_train = None
        N_X_val = None
        C_trainval = {
            'train': C_X_train,
            'val': C_X_val
        }
    else:
        C_X_train = X_train[:, categorical_feature_indices].astype(int)
        N_X_train = X_train[:, [i for i in range(X_train.shape[1]) if i not in categorical_feature_indices]].astype(float)
        C_X_val = X_val[:, categorical_feature_indices].astype(int)
        N_X_val = X_val[:, [i for i in range(X_val.shape[1]) if i not in categorical_feature_indices]].astype(float)

        N_trainval = {
            'train': N_X_train,
            'val': N_X_val
        } 

        C_trainval = {
            'train': C_X_train,
            'val': C_X_val
        }

    y_trainval = {
        'train': y_train,
        'val': y_val
    }

    y_test = {
        'test': y_test
    }

    N_test, C_test = None, None
    if categorical_feature_indices is None or len(categorical_feature_indices) == 0:
        C_X_test = None
        N_X_test = X_test.astype(float)
        N_test = {'test': N_X_test}
    elif len(categorical_feature_indices) == X_test.shape[1]:
        C_X_test = X_test.astype(int)
        N_X_test = None
        C_test = {'test': C_X_test}
    else:
        C_X_test = X_test[:, categorical_feature_indices].astype(int)
        N_X_test = X_test[:, [i for i in range(X_test.shape[1]) if i not in categorical_feature_indices]].astype(float)
        N_test = {'test': N_X_test}
        C_test = {'test': C_X_test}
        
    train_val_data = (N_trainval,C_trainval,y_trainval)
    test_data = (N_test,C_test, y_test)

    n_num_features = N_X_train.shape[1] if N_X_train is not None else 0
    n_cat_features = C_X_train.shape[1] if C_X_train is not None else 0

    task_type = REGRESSION if not is_classification else (BINCLASS if len(set(y_train)) == 2 else MULTICLASS)
    info = {
        "task_type": task_type, 
        "n_num_features": n_num_features,
        "n_cat_features": n_cat_features
    }
    LOGGER.text(f"Numeric features: {n_num_features}, Cat features: {n_cat_features}")

    if args.tune:
        args = tune_hyper_parameters(args,opt_space,train_val_data,info)
    else:
        best_value = 0
        best_metric = 'AUC'
        for seed in tqdm(range(args.seed_num)):
            args.seed = seed    # update seed  
            set_seeds(args.seed)
            method = get_method(args.model_type)(args, info['task_type'] == 'regression')
            time_cost = method.fit(train_val_data, info)    
            vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
            loss_list.append(vl)
            results_list.append(vres)
            time_list.append(time_cost)

            if best_metric in metric_name:
                metric_index = metric_name.index(best_metric)
                if vres[metric_index] > best_value:
                    best_value = vres[metric_index]
                    method.save(name='best')
                    # save config
                    with open(osp.join(args.save_path, f'args-best.json'), 'w') as f:
                        json.dump(args.config, f, sort_keys=True, indent=4)

    # show_results(args,info, metric_name,loss_list,results_list,time_list)