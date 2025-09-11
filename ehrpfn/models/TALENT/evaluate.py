import torch
import os.path as osp
from TALENT.model.utils import get_deep_args,show_results,tune_hyper_parameters,get_method,set_seeds

from ehrpfn.datasets import get_mimic_dataset
from sklearn.model_selection import train_test_split

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'


if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []
    args,default_para,opt_space = get_deep_args()


    ROOT="/home/mpham/workspace/source/ehrpfn"
    TASK="ICD SEPSIS"

    data_csv=f"{ROOT}/data/processed/mimiciv/{TASK}/all_data.csv"
    train_label_path=f"{ROOT}/data/processed/mimiciv/{TASK}/split/train_labels.csv"
    test_label_path=f"{ROOT}/data/processed/mimiciv/{TASK}/split/test_labels.csv"
    selection_method='shap_100'
    feature_selection_path=f"{ROOT}/outputs/tabpfn/selection/mimicpipe/{TASK}"
    process_file_path=None
    trajectory_file_path=None

    data = get_mimic_dataset(
        data_csv=data_csv,
        train_label_path=train_label_path,
        test_label_path=test_label_path,
        feature_selection_path=feature_selection_path,
        selection_method=selection_method,
    )

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    is_classification = True

    categorical_feature_indices = data['categorical_feature_indices']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=1702, 
        stratify=y_train if is_classification else None
    )

    C_X_train = X_train[:, categorical_feature_indices].astype(int)
    N_X_train = X_train[:, [i for i in range(X_train.shape[1]) if i not in categorical_feature_indices]].astype(float)
    C_X_val = X_val[:, categorical_feature_indices].astype(int)
    N_X_val = X_val[:, [i for i in range(X_val.shape[1]) if i not in categorical_feature_indices]].astype(float)
    C_X_test = X_test[:, categorical_feature_indices].astype(int)
    N_X_test = X_test[:, [i for i in range(X_test.shape[1]) if i not in categorical_feature_indices]].astype(float)

    N_trainval = {
        'train': N_X_train,
        'val': N_X_val
    } 
    N_test = {'test': N_X_test}

    C_trainval = {
        'train': C_X_train,
        'val': C_X_val
    }
    C_test = {'test': C_X_test}

    y_trainval = {
        'train': y_train,
        'val': y_val
    }
    y_test = {'test': y_test}
    info = {
        "task_type": REGRESSION if not is_classification else BINCLASS, 
        "n_num_features": N_X_train.shape[1],
        "n_cat_features": C_X_train.shape[1]
    }
    train_val_data = (N_trainval,C_trainval,y_trainval)
    test_data = (N_test,C_test,y_test)
    print(f"Num features: {N_X_train.shape[1]}, Cat features: {C_X_train.shape[1]}")

    set_seeds(args.seed)
    method = get_method(args.model_type)(args, info['task_type'] == 'regression')
    time_cost = method.fit(train_val_data, info, train = False)    
    model_name = args.evaluate_option

    breakpoint()
    method.model.load_state_dict(
        torch.load(
            osp.join(method.args.save_path, model_name + '-{}.pth'.format(str(method.args.seed)))
        )['params']
    )

    vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
    loss_list.append(vl)
    results_list.append(vres)
    time_list.append(0)

    show_results(args,info, metric_name,loss_list,results_list,time_list)