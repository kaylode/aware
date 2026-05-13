
from .EHRPFN import EHRPFNClassifier, EHRPFNRegressor, EHRPFNModel
from .AWARE import (
    AWAREAdapter,
    AWAREEstimator,
    AWAREClassifier,
    AWARERegressor,
    RetrievalContextSampler,
    AWARECLSModel,
    TabPFNAWARECLSModel,
    TabDPTAWARECLSModel,
    BaseClsFinetune,
    TabPFNClsFinetune,
    TabDPTClsFinetune,
    KNNPFNClsFinetune,
    AWARECLSClassifier,
)
from .talent_wrapper import TALENTWrapper
from .knn import SimpleKNNClassifier, SimpleKNNRegressor
from .KNNPFNv1 import KNNPFN
from .tabpfn import TabPFNv1
from .tabicl import TabICL, TabICLClassifier
from tradml import TradMLTuner
from ehrret import get_retriever, get_ehr_retrieval_weight
from ehrret.encoding.coding.icd_tokenizer import ICDHierachicalTokenizer
from ehrpfn.utils import save_model
import os
import os.path as osp
import numpy as np
import time
import shutil
from ehrpfn.utils import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

def get_predictions(args, data, config_dict):
    """
    Given the args and data, initialize and return the appropriate model,
    """
    result_folder = config_dict['result_folder']
    dataset_name = data['data_name']
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    categorical_feature_indices = data['categorical_feature_indices']
    is_classification = data['is_classification']
    study_name = config_dict['study_name']
    num_classes = data['num_classes'] if is_classification else None
    X_train_original, X_test_original = data['X_train_original'], data['X_test_original']

    method_name = args.inference.model

    retriever_additional_fit = {
        'ori_feature': X_train_original.values.astype(np.float64),
    }
    retriever_additional_query = {
        'ori_feature': X_test_original.values.astype(np.float64),
    }

    if "raw_codes_train" in data and "raw_codes_test" in data:
        raw_codes_train = [" ".join(i) for i in data['raw_codes_train']]
        raw_codes_test = [" ".join(i) for i in data['raw_codes_test']]
        tokenizer = ICDHierachicalTokenizer()
        tokenized_train = tokenizer(raw_codes_train)
        tokenized_train = [" ".join([item for sublist in seq for item in sublist]) for seq in tokenized_train]
        tokenized_test = tokenizer(raw_codes_test)
        tokenized_test = [" ".join([item for sublist in seq for item in sublist]) for seq in tokenized_test]

        retriever_additional_fit['codes'] = raw_codes_train
        retriever_additional_fit['hierarchical_codes'] = tokenized_train
        retriever_additional_query['codes'] = raw_codes_test
        retriever_additional_query['hierarchical_codes'] = tokenized_test


    ## Init context retrievers
    retrieval_weights = None
    retrieval_method = args.retrieval.retrieval_method
    if args.retrieval.retrieval_method is not None:
        if args.retrieval.retrieval_method.endswith('cosine'):
            distance = 'cosine'
            args.retrieval.retrieval_method = args.retrieval.retrieval_method.replace('_cosine', '')
        elif args.retrieval.retrieval_method.lower().endswith('l2'):
            distance = 'L2'
            args.retrieval.retrieval_method = args.retrieval.retrieval_method.lower().replace('_l2', '')
        else:
            distance = args.retrieval.get('distance', 'L2')

        retrieval_method = args.retrieval.retrieval_method
        if args.retrieval.retrieval_method.startswith('weighted'):
            retrieval_method, weight_type = args.retrieval.retrieval_method.split('_')
            weight_json_path = args.retrieval.get('weight_json_path', None)
            if args.data.data_name == 'openml_ehr':
                if not osp.isfile(weight_json_path):
                    weight_json_path = weight_json_path.replace(str(data['did']), data['data_name'])
            retrieval_weights = get_ehr_retrieval_weight(
                weight_type, task=args.data.task, 
                feature_names=data.get('feature_names', None),
                json_weight_path=weight_json_path
            )
            
    field_to_use='raw_feature'
    if field_to_use == 'raw_feature':
        n_features = X_train.shape[1]
    else:
        n_features = X_train_original.shape[1]

    if args.data.num_features:
        n_features = min(n_features, args.data.num_features)

    if args.retrieval.retrieval_method is not None:

        if args.data.data_name == 'openml_ehr':
             pretrained_path = args.retrieval.pretrained_path.replace('openml_ehr', 'openml_ehr/'+dataset_name)
        else:
            pretrained_path = args.retrieval.pretrained_path

        retriever_dict = dict(
            k=args.retrieval.context_size,
            distance=distance,
            field_to_use=field_to_use,
            weights=retrieval_weights,
            n_features=n_features,
            task=args.data.task,
            embedding_dim=256,
		    dropout = 0.3,
            pretrained_path=pretrained_path,
            num_classes=num_classes
        )

        retriever = get_retriever(
            retrieval_method, 
            **retriever_dict
        )
    else:
        retriever = None

    training_time = None
    if method_name in ['xgboost', 'rf', 'catboost', 'lightgbm', 'knn', 'mlp', 'logistics']:
        
        from sklearn.model_selection import train_test_split

        if args.wandb.use_wandb:
            wandb_kwargs={
                "project": args.wandb.project,
                "entity": args.wandb.entity,
            }
        else:
            wandb_kwargs = None

        tune_config=getattr(args.inference, 'tune_config', None)
        if tune_config:
            use_best_params = True
            task_name = args.data.task or dataset_name
            study_name = f"{method_name}_{dataset_name}_{task_name}"
            _storage = osp.join(tune_config, f"{task_name}/{study_name}", 'optuna.log')
            # copy
            if osp.isfile(_storage):
                shutil.copy(_storage, result_folder)
                storage = _storage
            else:
                storage = osp.join(result_folder, "optuna.log")
                use_best_params = False
        else:
            storage = osp.join(result_folder, "optuna.log")
            use_best_params = False

        tuner = TradMLTuner(
            storage=storage,
            study_name=study_name,
            n_trials=5,
            direction="maximize" if is_classification else "minimize",
            save_dir=result_folder,
            method=args.inference.model,
            use_best_params=use_best_params,
            wandb_kwargs=wandb_kwargs,
            feature_names=data.get('feature_names', None),
            classnames=['0', '1'] if is_classification else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.1, 
            random_state=args.inference.seed, 
            stratify=y_train if is_classification else None
        )
        start_time = time.time()
        model = tuner.tune(
            X_train, X_val, y_train, y_val, 
            is_classification=is_classification, 
            cat_features=categorical_feature_indices,
            extra_params={'weight_fn': retrieval_weights}
        )
        training_time = time.time() - start_time

        save_model(model, osp.join(result_folder, 'model.pkl'))

        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)
        running_time = time.time() - start_time
    

    elif method_name == "knn-simple":
        from ehrpfn.models.knn import SimpleKNNClassifier, SimpleKNNRegressor

        if is_classification:
            model = SimpleKNNClassifier(retriever=retriever)
        else:
            model = SimpleKNNRegressor(retriever=retriever)
        start_time = time.time()
        model.fit(
            X_train, y_train,
            retriever_additional_fit = retriever_additional_fit,
            retriever_additional_query = retriever_additional_query,
        )
        training_time = time.time() - start_time
        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)
        running_time = time.time() - start_time

    elif method_name == "tabpfnv2":
        from ehrpfn.models.TabPFN import get_base_tabpfn

        model = get_base_tabpfn(
            categorical_feature_indices=categorical_feature_indices,
            is_classification=is_classification,
            device='cuda',
            seed=args.inference.seed
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)
        running_time = time.time() - start_time

    elif method_name == 'tabpfnv1':
        model = TabPFNv1(
            n_classes=num_classes, 
            context_size=args.retrieval.context_size, 
            device="cuda",
            retriever=retriever
        )
        start_time = time.time()
        model.fit(
            X_train, y_train,
            retriever_additional_fit = retriever_additional_fit,
            retriever_additional_query = retriever_additional_query,
        )
        training_time = time.time() - start_time
        
        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(
                X_test, 
                retriever_additional_query=retriever_additional_query
            )
        else:
            y_pred = model.predict(
                X_test,
                retriever_additional_query=retriever_additional_query
            )
        running_time = time.time() - start_time

    elif method_name == "knnpfnv1":
        model = TabPFNv1(
            n_classes=num_classes, 
            retriever=retriever,
            context_size=args.retrieval.context_size, 
            device="cuda"
        )
        start_time = time.time()
        model.fit(
            X_train, y_train,
            retriever_additional_fit = retriever_additional_fit,
            retriever_additional_query = retriever_additional_query,
        )
        training_time = time.time() - start_time
        
        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(
                X_test,
                retriever_additional_query=retriever_additional_query
            )
        else:
            y_pred = model.predict(
                X_test,
                retriever_additional_query=retriever_additional_query
            )
        running_time = time.time() - start_time



    elif method_name == "knnpfnv2":
        model = KNNPFN(
            retriever=retriever,
            is_classification=is_classification,
            categorical_feature_indices=categorical_feature_indices,
            seed=42,
            device='cuda',
            batch_size=args.retrieval.get('inf_batch_size', 512),
        )
        start_time = time.time()
        model.fit(
            X_train, y_train,
            retriever_additional_fit = retriever_additional_fit,
            retriever_additional_query = retriever_additional_query,
        )
        training_time = time.time() - start_time
        # save_fitted_tabpfn_model(model, osp.join(result_folder, 'model.tabpfn_fit'))
        
        start_time = time.time()
        if is_classification:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        running_time = time.time() - start_time

    elif method_name in ['tabdpt', 'ehrpfn'] or method_name.startswith('ehrpfn'):
        from ehrpfn.models import EHRPFNClassifier, EHRPFNRegressor

        if is_classification:
            model = EHRPFNClassifier(
                pretrained_path=args.inference.pretrained_path,
                n_ensembles=1, temperature=0.8, 
                inf_batch_size=args.retrieval.get('inf_batch_size', 512),
                compile = False,
                categorical_feature_indices=categorical_feature_indices,
                retriever=retriever,
                fit_retriever_before_fit=True
            )
        else:
            model = EHRPFNRegressor(
                pretrained_path=args.inference.pretrained_path,
                n_ensembles=1, temperature=0.8, 
                inf_batch_size=args.retrieval.get('inf_batch_size', 512),
                categorical_feature_indices=categorical_feature_indices,
                retriever=retriever,
                fit_retriever_before_fit=True
            )

        start_time = time.time()
        model.fit(
            X_train, y_train,
            retriever_additional_fit = retriever_additional_fit,
            retriever_additional_query = retriever_additional_query,
        )
        training_time = time.time() - start_time
        start_time = time.time()
        if is_classification:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        running_time = time.time() - start_time
    
    elif method_name in ['tabdpt-aware', 'knnpfn-aware']:
        if not is_classification:
            raise ValueError(f"{method_name} only supports classification tasks.")

        if method_name == 'tabdpt-aware':
            finetune = TabDPTClsFinetune(
                retriever=retriever,
                learning_rate=3e-4,
                epochs=10,
                batch_size=args.retrieval.get('inf_batch_size', 64),
                context_size=args.retrieval.context_size,
                patience=3,
                device=getattr(args.inference, 'device', 'cuda:0'),
                random_state=getattr(args.inference, 'seed', 42),
            )
        else:  # knnpfn_aware
            finetune = KNNPFNClsFinetune(
                retriever=retriever,
                learning_rate=3e-4,
                epochs=10,
                batch_size=args.retrieval.get('inf_batch_size', 64),
                context_size=args.retrieval.context_size,
                patience=3,
                device=getattr(args.inference, 'device', 'cuda:0'),
                random_state=getattr(args.inference, 'seed', 42),
            )

        model = AWARECLSClassifier(
            finetune=finetune,
            n_ensembles=5,
            temperature=getattr(args.inference, 'temperature', 1.0),
            infer_batch_size=args.retrieval.get('inf_batch_size', 256),
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred = model.predict_proba(X_test)
        running_time = time.time() - start_time

    elif method_name == 'tabicl':
        if is_classification:
            model = TabICLClassifier(
                n_estimators=1, 
                norm_methods='none',
                feat_shuffle_method='none',
                batch_size=args.retrieval.get('inf_batch_size', 512),                                           # process this many ensemble members together (reduce RAM usage)
                use_amp=True,
                allow_auto_download=True,                               # whether automatic download to the specified path is allowed
                checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                device='cuda',                                            # specify device for inference
                random_state=42,                                        # random seed for reproducibility
                n_jobs=None,                                            # number of threads to use for PyTorch
                verbose=True, 
            )
        else:
            raise ValueError("TabICL only supports classification tasks.")

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(X_test)
        else:
            raise ValueError("TabICL only supports classification tasks.")
        
        running_time = time.time() - start_time

    elif method_name in ['danets', 'beta', 'resnet', 'tabr', 'modernNCA', 'tabm', 'tabtransformer', 'tabnet', 'excelformer', 'trompt']:

        from ehrpfn.models.talent_wrapper import TALENTWrapper
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.1, 
            random_state=args.inference.seed, 
            stratify=y_train if is_classification else None
        )

        config_path = getattr(args.inference, 'config_path', None)
        talent_root_dir=getattr(args.inference, 'talent_root_dir', None)
        tune_config=getattr(args.inference, 'tune_config', None)

        if args.data.data_name == 'openml_ehr':
            if not osp.isfile(config_path):
                config_path = config_path.replace(str(data['did']), data['data_name'])
            if not osp.isfile(tune_config):
                tune_config = tune_config.replace(str(data['did']), data['data_name'])

        if osp.isfile(config_path):
            shutil.copy(config_path, osp.join(result_folder, 'args.json'))
        if osp.isfile(tune_config):
            shutil.copy(tune_config, osp.join(result_folder, f'{method_name}_tuned.json'))

        model = TALENTWrapper(
            model_name=method_name,
            config_path=config_path,
            talent_root_dir=talent_root_dir,
            tune_config=tune_config,
            context_size=args.retrieval.context_size, # for modernNCA and Tabr
            ckpt_path=args.inference.pretrained_path,
            save_dir=result_folder,
            categorical_feature_indices=categorical_feature_indices,
            valset=[X_val, y_val],
            is_classification=is_classification,
            seed=args.inference.seed,
            batch_size=args.retrieval.get('inf_batch_size', None),
        )
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred = model.predict(X_test)
        running_time = time.time() - start_time

    else:
        raise ValueError(f"Method {method_name} is not supported. Please choose from ['tabpfn', 'rftabpfn', 'xgboost', 'catboost', 'lightgbm']")
    
    # patient_col = 'subject_id'
    # admission_col = 'stay_id'
    # df_with_original_index = data['test_ids'].copy()
    # df_with_original_index['_original_index'] = df_with_original_index.index
    # df_sorted = df_with_original_index.sort_values([patient_col, admission_col]).reset_index(drop=True)
    # df_sorted['EPISODE_ID'] = df_sorted.groupby(patient_col).cumcount()
    # df_result = df_sorted.sort_values('_original_index').drop('_original_index', axis=1).reset_index(drop=True)
    # df_result['PREDICTION'] = y_pred.tolist()
    # df_result['TRUE_LABEL'] = data['y_test'].tolist()
    # output_episode_path = osp.join(result_folder, 'test_predictions_with_episode_ids.csv')
    # df_result.to_csv(output_episode_path, index=False)
    # LOGGER.text(f"Saved test predictions with EPISODE_ID to {output_episode_path}", level=LoggerObserver.INFO)
    
    return y_pred, training_time, running_time