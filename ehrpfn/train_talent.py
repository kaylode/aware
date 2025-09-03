from tqdm import tqdm
from TALENT.model.utils import get_deep_args,show_results,tune_hyper_parameters,get_method,set_seeds

from ehrpfn.datasets import get_mimic_dataset, get_openml_ehr, get_eicu
from sklearn.model_selection import train_test_split
from ehrpfn.utils.logger import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'


if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []
    args,default_para,opt_space = get_deep_args()

    

    if args.data_name == 'mimicpipe' or args.data_name.startswith('mimiciv'):
        data = get_mimic_dataset(
            data_csv=args.data_csv,
            train_label_path=args.train_label_path,
            test_label_path=args.test_label_path,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
        )
    elif args.data_name == 'openml_ehr':
        data = get_openml_ehr(args.data_folder)
    elif args.data_name == 'eicu':
        data = get_eicu(
            task=args.task,
            csv_file=args.data_csv,
            label_file=args.label_path,
            train_admission_id_file=args.train_admission_id_file,
            val_admission_id_file=args.val_admission_id_file,
            test_admission_id_file=args.test_admission_id_file,
            selection_method=args.selection_method,
            feature_selection_path=args.feature_selection_path
        )
    else:
        raise ValueError(f"Data {args.data_name} is not supported. Please choose from ['mimiciv', 'sjh_cpe']")

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
    LOGGER.text(f"Num features: {N_X_train.shape[1]}, Cat features: {C_X_train.shape[1]}")



    if args.model_type in ["tabr", "modernNCA"]:
        args.cat_policy = "tabr_ohe"
        args.num_policy = "none"
        args.normalization = "none"
    elif args.model_type in ['tabicl']:
        args.cat_policy = "indices"
        args.num_policy = "none"
        args.normalization = "none"
        args.tune = False
    elif args.model_type in ['tabm','tabtransformer', 'trompt']:
        args.cat_policy = "indices"
    elif args.model_type in [ 'excelformer', 'tabnet', 'resnet']:
        args.cat_policy = "ordinal"

    if args.tune:
        args = tune_hyper_parameters(args,opt_space,train_val_data,info)
    else:
        for seed in tqdm(range(args.seed_num)):
            args.seed = seed    # update seed  
            set_seeds(args.seed)
            method = get_method(args.model_type)(args, info['task_type'] == 'regression')
            time_cost = method.fit(train_val_data, info)    
            vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
            loss_list.append(vl)
            results_list.append(vres)
            time_list.append(time_cost)

    # show_results(args,info, metric_name,loss_list,results_list,time_list)