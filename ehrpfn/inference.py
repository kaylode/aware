from typing import *
import os
import os.path as osp
import hydra
import time
import json
import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
import xgboost as xgb
from tabpfn import TabPFNClassifier, TabPFNRegressor

from ehrpfn.models.talent_wrapper import TALENTWrapper
from ehrpfn.models.gbms import Tuner
from ehrpfn.utils.sampling import synthetic_sampling
from ehrpfn.utils import load_model, save_model
from ehrpfn.utils.retrieval import get_retriever, get_retrieval_weight
from ehrpfn.models import get_tabpfn_knn_results
from ehrpfn.utils import FileLogger, LoggerObserver, seed_everything
from ehrpfn.utils.metrics import run_binary_classification_metrics, run_regression_metrics
from ehrpfn.datasets import (
    get_sjh_dataset, 
    get_mimic_dataset, 
    get_openml_ehr,
    get_eicu
)

LOGGER = LoggerObserver.getLogger("main")


def inference(args, data):

    (
        X_train, y_train, 
        X_test, y_test, 
        categorical_feature_indices
    ) = (
        data['X_train'], data['y_train'], 
        data['X_test'], data['y_test'], 
        data['categorical_feature_indices']
    )

    data_name = data.get('data_name', args.data_name)
    target_column = args.target_column if args.data_name != 'openml_ehr' else data_name

    result_folder = osp.join(args.output_folder, "results", args.data_name, target_column)
    os.makedirs(result_folder, exist_ok=True)

    X_embs_train = data.get('X_embs_train', None)
    X_embs_test = data.get('X_embs_test', None)

    if X_embs_train is not None:
        assert X_embs_train.shape[0] == X_train.shape[0], "X_embs_train must have the same number of samples as X_train"
        assert X_embs_test is not None, "X_embs_test must be provided if X_embs_train is provided"


    study_name = f"{args.method}_{data_name}_{target_column}"

    if args.selection_method is not None:
        study_name += f"_sel:{args.selection_method}"

    if (args.method.startswith('ehrpfn') or args.method.startswith('knn')) and args.retrieval_method is not None:
        study_name += f"_retrieval:{args.retrieval_method}"
    

    if args.sampling is not None:
        LOGGER.text(f"================= Performing SYNTHESIS SAMPLING =================", level=LoggerObserver.DEBUG)
        LOGGER.text(f"Before sampling: {X_train.shape}, {y_train.shape}", level=LoggerObserver.DEBUG)
        sampling_type, sampling_method = args.sampling.split('_')
        X_train, y_train = synthetic_sampling(
            X_train, y_train, 
            oversampling=sampling_method if sampling_type == "oversampling" else None,
            undersampling=sampling_method if sampling_type == "undersampling" else None,
            seed=args.seed,
            max_samples=10000 if args.method in ['tabpfn', 'rftabpfn'] else None,
            balance_negatives=args.balance_negatives, negative_ratio=args.negative_ratio
        )
        study_name += f"_sampl:{sampling_type}-{sampling_method}"
        LOGGER.text(f"After sampling: {X_train.shape}, {y_train.shape}", level=LoggerObserver.DEBUG)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    LOGGER.text(f"Data shape: {X_train.shape}, {X_test.shape}", level=LoggerObserver.DEBUG)


    ## Init context retrievers


    if args.retrieval_method is not None and args.retrieval_method.startswith('weighted'):
        retrieval_method, weight_type = args.retrieval_method.split('_')
        retrieval_weights = get_retrieval_weight(weight_type, data_name, target_column, feature_names=data.get('feature_names', None))
    else:
        retrieval_weights = None

    if args.retrieval_method is not None:
        if args.retrieval_method.startswith('weighted'):
            retrieval_method, weight_type = args.retrieval_method.split('_')
        else:
            retrieval_method = args.retrieval_method
        retriever = get_retriever(retrieval_method, k=args.context_size, distance='L2', weights=retrieval_weights, max_context_size=1024)
    else:
        retriever = None
    result_folder = osp.join(result_folder, study_name)
    os.makedirs(result_folder, exist_ok=True)

    # Save config
    with open(osp.join(result_folder, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=args, f=f, resolve=True)

    
    # Initialize the model

    if data.get('is_classification', None) is not None:
        is_classification = data['is_classification']
    else:
        if target_column in ['NEXT_ADMISSION_LOS', 'NEXT_LOS']:
            is_classification = False
        else:
            is_classification = True

    with open(osp.join(result_folder, 'metadata.json'), 'w') as f:
        json.dump({
            "categorical_feature_indices": categorical_feature_indices,
            "feature_names": data.get('feature_names', None),
            "num_classes": data.get('num_classes', None),
            'is_classification': is_classification,
            'data_name': data.get('data_name', None),
            'num_data_features': data.get('num_features', None),
            'class_distribution': data.get('class_distribution', None),
            'data_size': X_train.shape[0] + X_test.shape[0],
        }, f, indent=4, default=str)

    tags = [f"METHOD:{args['method']}", f"FEAT:{args.num_features}"]
    if args.data_name != 'openml_ehr':
        tags.append(f"TASK:{args['target_column']}")
    else:
        tags.append(f"TASK:{data_name}")
    if args.sampling is not None:
        tags.append(f"SAMPLING:{args['sampling']}")
    if args.selection_method is not None:
        tags.append(f"SELECTION:{args['selection_method']}")
    

    if args.method in ['xgboost', 'rf', 'catboost', 'lightgbm', 'knn', 'mlp', 'logistics']:
        
        from sklearn.model_selection import train_test_split

        if args.method == 'knn':
            if args.retrieval_method is not None:
                tags.append(f"RETRIEVAL:{args['retrieval_method']}")

        tuner = Tuner(
            storage=osp.join(result_folder, "optuna.log"),
            study_name=study_name,
            n_trials=args.n_trials,
            direction=args.direction,
            save_dir=result_folder,
            method=args.method,
            use_best_params=args.use_best_params,
            wandb_kwargs={
                "project": args.wandb.project,
                "entity": args.wandb.entity,
                "tags": tags,
            },
            feature_names=data.get('feature_names', None),
            classnames=['0', '1'] if is_classification else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=args.val_size, 
            random_state=args.seed, 
            stratify=y_train if is_classification else None
        )

        model = tuner.tune(
            X_train, X_val, y_train, y_val, 
            is_classification=is_classification, 
            cat_features=categorical_feature_indices,
            extra_params={'weight_fn': retrieval_weights}
        )
        save_model(model, osp.join(result_folder, 'model.pkl'))
        
        LOGGER.text(f"Model saved to {result_folder}/model.pkl", level=LoggerObserver.INFO)

        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(X_test)
            score = run_binary_classification_metrics(
                y_pred, 
                y_test,
            )
        else:
            y_pred = model.predict(X_test)
            score = run_regression_metrics(
                y_pred, 
                y_test,
            )
        running_time = time.time() - start_time
    

    elif args.method == "knn-simple":
        from ehrpfn.models.knn import SimpleKNNClassifier, SimpleKNNRegressor

        if args.retrieval_method is not None:
            tags.append(f"RETRIEVAL:{args['retrieval_method']}")

        if is_classification:
            model = SimpleKNNClassifier(retriever=retriever)
        else:
            model = SimpleKNNRegressor(retriever=retriever)
        model.fit(X_train, y_train)
        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(X_test)
            score = run_binary_classification_metrics(
                y_pred, 
                y_test,
            )
        else:
            y_pred = model.predict(X_test)
            score = run_regression_metrics(
                y_pred, 
                y_test,
            )
        running_time = time.time() - start_time

    elif args.method == "tabpfn":

        from ehrpfn.models.TabPFN.utils import (
            default_classifier_preprocessor_configs,
            default_regressor_preprocessor_configs,
        )
        
        if is_classification:
            if args.selection_method is not None:
                device = 'cuda'
            else:
                device = 'cpu'
            model = TabPFNClassifier(
                n_estimators=args.n_ensembles,
                ignore_pretraining_limits=True,
                categorical_features_indices=categorical_feature_indices,
                device=device,
                n_jobs=4,
                random_state=args.seed,
                memory_saving_mode=True,
                fit_mode = 'low_memory',
                inference_config={
                    "SUBSAMPLE_SAMPLES": args.context_size,
                    "FEATURE_SHIFT_METHOD": "shuffle",
                    'FINGERPRINT_FEATURE': False,
                    "CLASS_SHIFT_METHOD": "shuffle",
                    "PREPROCESS_TRANSFORMS": default_classifier_preprocessor_configs(),
                }  # Needs to be set low so that not OOM on fitting intermediate nodes
            )
        else:
            model = TabPFNRegressor(
                n_estimators=args.n_ensembles,
                ignore_pretraining_limits=True,
                categorical_features_indices=categorical_feature_indices,
                device="cpu",
                fit_mode = 'low_memory',
                random_state=args.seed,
                memory_saving_mode=True,
                n_jobs=4,
                inference_config={
                    "SUBSAMPLE_SAMPLES": args.context_size,
                    "FEATURE_SHIFT_METHOD": "shuffle",
                    'FINGERPRINT_FEATURE': False,
                    "CLASS_SHIFT_METHOD": "shuffle",
                    "PREPROCESS_TRANSFORMS": default_regressor_preprocessor_configs(),
                }  # Needs to be set low so that not OOM on fitting intermediate nodes
            )
        model.fit(X_train, y_train)
        # save_fitted_tabpfn_model(model, osp.join(result_folder, 'model.tabpfn_fit'))
        
        start_time = time.time()
        if is_classification:
            y_pred = model.predict_proba(X_test)
            score = run_binary_classification_metrics(
                y_pred, 
                y_test,
            )
        else:
            y_pred = model.predict(X_test)
            score = run_regression_metrics(
                y_pred, 
                y_test,
            )
        running_time = time.time() - start_time

    elif args.method == "tabpfn-retrieval":

        if args.retrieval_method is not None:
            tags.append(f"RETRIEVAL:{args['retrieval_method']}")

        start_time = time.time()
        all_predictions = get_tabpfn_knn_results(
            X_train, y_train, 
            X_test, 
            categorical_feature_indices, 
            is_classification, 
            retriever=retriever,
            n_ensembles=args.n_ensembles,
            seed=args.seed,
            X_embs_train=X_embs_train,
            X_embs_test=X_embs_test
        )

        if is_classification:
            score = run_binary_classification_metrics(
                all_predictions, 
                y_test,
            )
        else:
            score = run_regression_metrics(
                all_predictions, 
                y_test,
            )
        running_time = time.time() - start_time

    elif args.method in ['tabdpt', 'ehrpfn'] or args.method.startswith('ehrpfn'):
        from ehrpfn.models import EHRPFNClassifier, EHRPFNRegressor

        if args.pretrained_path is not None:
            _pretrained_name = osp.basename(osp.dirname(args.pretrained_path))
            for tok in _pretrained_name.split('-'):
                if tok in ['HEALTH', 'MIMICIV', 'SYNTHETIC']:
                    tags.append(f"PRETRAIN:{tok}")

        if args.retrieval_method is not None:
            tags.append(f"RETRIEVAL:{args['retrieval_method']}")

        if is_classification:
            model = EHRPFNClassifier(
                pretrained_path=args.pretrained_path,
                inf_batch_size=args.inf_batch_size,
                compile = False,
                categorical_feature_indices=categorical_feature_indices,
                retriever=retriever
            )
        else:
            model = EHRPFNRegressor(
                pretrained_path=args.pretrained_path,
                inf_batch_size=args.inf_batch_size,
                categorical_feature_indices=categorical_feature_indices,
                retriever=retriever
            )

        model.fit(X_train, y_train)
        start_time = time.time()
        if is_classification:
            y_pred = model.predict(
                X_test, 
                n_ensembles=args.n_ensembles,
                temperature=0.8, context_size=args.context_size, 
                X_embs_train=X_embs_train, 
                X_embs_test=X_embs_test
            )

            score = run_binary_classification_metrics(
                y_pred, 
                y_test,
            )
        else:
            y_pred = model.predict(
                X_test,
                context_size=args.context_size, 
                n_ensembles=args.n_ensembles,
                X_embs_train=X_embs_train, 
                X_embs_test=X_embs_test
            )
            score = run_regression_metrics(
                y_pred, 
                y_test,
            )
        running_time = time.time() - start_time


    elif args.method in ['resnet', 'tabr', 'modernNCA', 'tabm', 'tabicl', 'tabtransformer', 'tabnet', 'excelformer', 'trompt']:

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=args.val_size, 
            random_state=args.seed, 
            stratify=y_train if is_classification else None
        )

        model = TALENTWrapper(
            model_name=args.method,
            config_path=getattr(args, 'config_path', None),
            tune_config=getattr(args, 'tune_config', None),
            ckpt_path=args.pretrained_path,
            save_dir=result_folder,
            categorical_feature_indices=categorical_feature_indices,
            valset=[X_val, y_val],
            is_classification=is_classification,
            seed=args.seed
        )

        model.fit(X_train, y_train)

        start_time = time.time()
        y_pred = model.predict(X_test)
        if is_classification:
            score = run_binary_classification_metrics(
                y_pred, 
                y_test,
            )
        else:
            score = run_regression_metrics(
                y_pred, 
                y_test,
            )
        running_time = time.time() - start_time

    else:
        raise ValueError(f"Method {args.method} is not supported. Please choose from ['tabpfn', 'rftabpfn', 'xgboost', 'catboost', 'lightgbm']")


    LOGGER.text(f"Score: {score}", level=LoggerObserver.SUCCESS)


    
    wandb_kwargs = {
        "name": f"{args['method']}" if args.method not in ['tabdpt', 'tabpfn-retrieval', 'knn', 'knn-simple'] else f"{args.method}_{args.retrieval_method}",
        "entity": "kaylode",
        "project": "tabpfn-ehr",
        "reinit": True,
        "group": "ehrpfn-mimicpipe",
        "job_type": "test",
        "tags": tags,
        "resume": "allow",
        "dir": osp.join(result_folder),
    }
    wandb.init(**wandb_kwargs)

    saved_config = OmegaConf.to_container(args, resolve=True)


    if args.data_name == 'openml_ehr':
        saved_config['target_column'] = data.get('data_name', None)


    wandb.run.config.update(saved_config)
    wandb.run.config.update({
        'study_name': study_name,
        'task_name': target_column,
        'is_classification': is_classification,
        'num_classes': data.get('num_classes', None),
        'dataset_name': data.get('data_name', None),
        'num_data_features': data.get('num_features', None),
        'data_size': X_train.shape[0] + X_test.shape[0],
        'class_distribution': data.get('class_distribution', None)
    })
    # Log the metrics to wandb
    wandb.run.log(score)
    wandb.run.log({"inference_time": running_time})
    wandb.finish()

    # Dump score into JSON
    with open(osp.join(result_folder, 'score.json'), 'w') as f:
        json.dump(score, f, indent=4)

    LOGGER.text(f"Score: {score}", level=LoggerObserver.INFO)


@hydra.main(version_base=None, config_path="../configs", config_name="mimiciv")
def run(args: DictConfig):

    seed_everything(args.seed)

    LOGGER.subscribe(FileLogger(__name__, osp.join(args.output_folder, 'logs'), debug=True))

    if args.selection_method is not None:
        args.selection_method += f"_{args.num_features}"

    if args.data_name == 'mimicpipe' or args.data_name.startswith('mimiciv'):
        data = get_mimic_dataset(
            data_csv=args.data_csv,
            train_label_path=args.train_label_path,
            test_label_path=args.test_label_path,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            embedding_file_path=args.embedding_file_path,
        )
    elif args.data_name == 'openml_ehr':
        data = get_openml_ehr(args.data_folder)
    elif args.data_name == 'sjh_cpe':
        data_output_folder = osp.join(args.data_folder, args.data_name)
        os.makedirs(data_output_folder, exist_ok=True)
        data = get_sjh_dataset(
            input_folder=args.input_folder,
            output_folder=data_output_folder,
            target_col=args.target_column,
            max_length=args.max_length,
            num_chunks=args.num_chunks,
            patient_ids=None,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            embedding_file_path=args.embedding_file_path,
        )
    elif args.data_name == 'eicu':
        data = get_eicu(
            task=args.target_column,
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



    if isinstance(data, list):
        for d in data:
            # try:
            inference(args, d)
            # except Exception as e:
            #     LOGGER.text(f"Error occurred while processing {d['data_name']}: {e}", level=LoggerObserver.ERROR)
            #     breakpoint()
    else:
        inference(args, data)


if __name__ == "__main__":
    run()