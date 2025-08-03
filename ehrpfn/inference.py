import os
import os.path as osp
import hydra
import time
import json
from ehrpfn.models.gbms import Tuner
from ehrpfn.utils.validate import validate
from ehrpfn.utils.sampling import synthetic_sampling
from ehrpfn.datasets import get_mimic_dataset, get_sjh_dataset
from ehrpfn.utils import load_model, save_model
import xgboost as xgb

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.model.loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
)
from ehrpfn.utils.metrics import run_binary_classification_metrics, run_regression_metrics
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
from ehrpfn.utils import FileLogger, LoggerObserver, seed_everything
LOGGER = LoggerObserver.getLogger("main")



@hydra.main(version_base=None, config_path="../configs", config_name="mimiciv")
def run(args: DictConfig):

    seed_everything(args.seed)

    data_output_folder = osp.join(args.output_folder, "data", args.data_name)
    result_folder = osp.join(args.output_folder, "results", args.data_name, args.target_column)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(data_output_folder, exist_ok=True)


    LOGGER.subscribe(FileLogger(__name__, osp.join(args.output_folder, 'logs'), debug=True))

    if args.selection_method is not None:
        args.selection_method += f"_{args.num_features}"

    if args.data_name.startswith('mimiciv'):
        data = get_mimic_dataset(
            train_folder=args.train_folder,
            test_folder=args.test_folder,
            icu_ids=args.icu_ids,
            data_output_folder=data_output_folder,
            target_column=args.target_column,
            train_label_path=args.train_label_path,
            test_label_path=args.test_label_path,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            process_file_path=args.process_file_path,
            n_process_variants=args.n_process_variants,
            trajectory_file_path=args.trajectory_file_path,
            logger=LOGGER
        )
    elif args.data_name == 'sjh_cpe':
        data = get_sjh_dataset(
            input_folder=args.input_folder,
            output_folder=data_output_folder,
            target_col=args.target_column,
            max_length=args.max_length,
            num_chunks=args.num_chunks,
            patient_ids=None,

            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            process_file_path=args.process_file_path,
            n_process_variants=args.n_process_variants,
            trajectory_file_path=args.trajectory_file_path,

        )
    else:
        raise ValueError(f"Data {args.data_name} is not supported. Please choose from ['mimiciv', 'sjh_cpe']")


    (
        X_train, y_train, 
        X_test, y_test, 
        categorical_feature_indices
    ) = (
        data['X_train'], data['y_train'], 
        data['X_test'], data['y_test'], 
        data['categorical_feature_indices']
    )

    X_train_process = data.get('X_train_process', None)
    X_test_process = data.get('X_test_process', None)

    X_train_traj = data.get('X_train_traj', None)
    X_test_traj = data.get('X_test_traj', None)

    if X_train_process is not None:
        assert X_train_process.shape[0] == X_train.shape[0], "X_train_process must have the same number of samples as X_train"
        assert X_test_process.shape[0] == X_test.shape[0], "X_test_process must have the same number of samples as X_test"
    if X_train_traj is not None:
        assert X_train_traj.shape[0] == X_train.shape[0], "X_train_traj must have the same number of samples as X_train"
        assert X_test_traj is not None, "X_test_traj must be provided if X_train_traj is provided"

    study_name = f"{args.method}_{args.data_name}_{args.target_column}"

    if args.selection_method is not None:
        study_name += f"_sel:{args.selection_method}"

    if args.method == 'ehrpfn' and args.retrieval_method is not None:
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

    result_folder = osp.join(result_folder, study_name)
    os.makedirs(result_folder, exist_ok=True)

    # Save config
    with open(osp.join(result_folder, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=args, f=f, resolve=True)

    with open(osp.join(result_folder, 'metadata.json'), 'w') as f:
        json.dump({
            "categorical_feature_indices": categorical_feature_indices,
            "feature_names": data.get('feature_names', None)
        }, f, indent=4)
    # Initialize the model

    if args.target_column in ['NEXT_ADMISSION_LOS', 'NEXT_LOS']:
        is_classification = False
    else:
        is_classification = True

    tags = [f"METHOD:{args['method']}", f"TASK:{args['target_column']}", f"FEAT:{args.num_features}", f"PRETRAIN:{args.pretrain_mode}"]
    if args.sampling is not None:
        tags.append(f"SAMPLING:{args['sampling']}")
    if args.selection_method is not None:
        tags.append(f"SELECTION:{args['selection_method']}")
    

    if args.method in ['xgboost', 'rf', 'catboost', 'lightgbm']:
        
        if osp.exists(osp.join(result_folder, 'model.pkl')):
            LOGGER.text(f"Loading fitted model from {osp.join(result_folder, 'model.pkl')}", level=LoggerObserver.DEBUG)
            
            if args.method == 'xgboost':
                if is_classification:
                    model = xgb.XGBClassifier()
                else:
                    model = xgb.XGBRegressor()
                model = load_model(osp.join(result_folder, 'model.pkl'), model)
            else:
                model = load_model(osp.join(result_folder, 'model.pkl'))
        else:
            from sklearn.model_selection import train_test_split

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
                }
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=args.val_size, 
                random_state=args.seed, 
                stratify=y_train if is_classification else None
            )

            model = tuner.tune(X_train, X_val, y_train, y_val, is_classification=is_classification, cat_features=categorical_feature_indices)
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
    

    elif args.method == "tabpfn":
        
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
                inference_config={"SUBSAMPLE_SAMPLES": args.inf_batch_size}  # Needs to be set low so that not OOM on fitting intermediate nodes
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
                inference_config={"SUBSAMPLE_SAMPLES": args.inf_batch_size}  # Needs to be set low so that not OOM on fitting intermediate nodes
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

    elif args.method == "tabpfn-knn":

        from ehrpfn.utils.retrieval import FAISS
        from tqdm import tqdm
        retriever = FAISS(X_train)
        all_predictions = []
        

        start_time = time.time()
        for i in tqdm(range(X_test.shape[0])):
            item = X_test[i].reshape(1, -1)
            nn_indices = retriever.retrieve(item, k=1024)[0]
            X_context = X_train[nn_indices, :]
            y_context = y_train[nn_indices]
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
                    n_jobs=-1,
                    random_state=args.seed,
                    memory_saving_mode=True,
                )
            else:
                model = TabPFNRegressor(
                    n_estimators=args.n_ensembles,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=categorical_feature_indices,
                    device="cpu",
                    random_state=args.seed,
                    memory_saving_mode=True,
                    n_jobs=-1,
                )

            try:
                X_context = X_context + np.random.normal(0, 1e-6, X_context.shape)
                model.fit(X_context, y_context)
                if is_classification:
                    y_pred = model.predict_proba(item)
                    if y_pred.shape[1] != 2:
                        y_pred = np.zeros((1, 2))
                        max_freq_class = np.bincount(y_context).argmax()
                        y_pred[0, max_freq_class] = 1.0
                else:
                    y_pred = model.predict(item)
            except Exception as e:
                if is_classification:
                    y_pred = np.zeros((1, 2))
                    max_freq_class = np.bincount(y_context).argmax()
                    y_pred[0, max_freq_class] = 1.0
                else:
                    y_pred = np.zeros((1, model.n_classes_))
                    y_pred[0, -1] = -1
            
            all_predictions.append(y_pred)

        all_predictions = np.concatenate(all_predictions, axis=0)
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

    elif args.method == 'tabdpt':
        from ehrpfn.models import TabDPTClassifier, TabDPTRegressor

        if is_classification:
            model = TabDPTClassifier(
                model_name=f"{args.pretrain_mode}_{args.num_features}",
                inf_batch_size=args.inf_batch_size
            )
        else:
            model = TabDPTRegressor(
                model_name=f"{args.pretrain_mode}_{args.num_features}",
                inf_batch_size=args.inf_batch_size
            )

        model.fit(X_train, y_train)
        save_model(model, osp.join(result_folder, 'model.pkl'))
        start_time = time.time()
        if is_classification:
            y_pred = model.predict(
                X_test, 
                n_ensembles=args.n_ensembles,
                temperature=0.8, context_size=args.context_size
            )
            score = run_binary_classification_metrics(
                y_pred, 
                y_test,
            )
        else:
            y_pred = model.predict(
                X_test, 
                n_ensembles=args.n_ensembles,
                context_size=args.context_size
            )
            score = run_regression_metrics(
                y_pred, 
                y_test,
            )
        running_time = time.time() - start_time

    elif args.method == 'ehrpfn':
        from ehrpfn.models import EHRPFNClassifier, EHRPFNRegressor

        if args.retrieval_method is not None:
            tags.append(f"RETRIEVAL:{args['retrieval_method']}")

        if is_classification:
            model = EHRPFNClassifier(
                model_name=f"{args.pretrain_mode}_{args.num_features}",
                retrieval_method=args.retrieval_method, 
                inf_batch_size=args.inf_batch_size
            )
        else:
            model = EHRPFNRegressor(
                model_name=f"{args.pretrain_mode}_{args.num_features}",
                retrieval_method=args.retrieval_method,
                inf_batch_size=args.inf_batch_size
            )


        model.fit(X_train, y_train, X_processes=X_train_process, X_traj=X_train_traj)
        save_model(model, osp.join(result_folder, 'model.pkl'))
        start_time = time.time()
        if is_classification:
            y_pred = model.predict(
                X_test, 
                n_ensembles=args.n_ensembles,
                temperature=0.8, context_size=args.context_size, 
                X_processes=X_test_process, X_traj=X_test_traj
            )

            # predict(self, X: np.ndarray, n_ensembles: int = 1, context_size: int = 128, seed=None)
            score = run_binary_classification_metrics(
                y_pred, 
                y_test,
            )
        else:
            y_pred = model.predict(
                X_test,
                context_size=args.context_size, 
                n_ensembles=args.n_ensembles,
                X_processes=X_test_process, X_traj=X_test_traj
            )
            score = run_regression_metrics(
                y_pred, 
                y_test,
            )
        running_time = time.time() - start_time

    elif args.method == 'tabflex':
        from ehrpfn.models.ticl.ticl.prediction.tabflex import TabFlex

        # Initialize and train TabFlex model
        model = TabFlex()
        model.fit(X_train, y_train)

        start_time = time.time()
        if is_classification:
            y_pred = model.predict(X_test)
            score = run_binary_classification_metrics(y_pred, y_test)
        else:
            # Not supported for regression in TabFlex
            return
        running_time = time.time() - start_time
        
    elif args.method == 'tabicl':
        from tabicl import TabICLClassifier

        model = TabICLClassifier(
            n_estimators=args.n_ensembles,                                        # number of ensemble members
            norm_methods=["none", "power"],                         # normalization methods to try
            feat_shuffle_method="latin",                            # feature permutation strategy
            class_shift=True,                                       # whether to apply cyclic shifts to class labels
            outlier_threshold=4.0,                                  # z-score threshold for outlier detection and clipping
            softmax_temperature=0.9,                                # controls prediction confidence
            average_logits=True,                                    # whether ensemble averaging is done on logits or probabilities
            use_hierarchical=True,                                  # enable hierarchical classification for datasets with many classe
            batch_size=8,                                           # process this many ensemble members together (reduce RAM usage)
            use_amp=True,                                           # use automatic mixed precision for faster inference
            model_path=None,                                        # where the model checkpoint is stored
            allow_auto_download=True,                               # whether automatic download to the specified path is allowed
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",  # the version of pretrained checkpoint to use
            device='cuda',                                            # specify device for inference
            random_state=args.seed,                                        # random seed for reproducibility
            n_jobs=4,                                            # number of threads to use for PyTorch
            verbose=True,                                          # print detailed information during inference
        )

        model.fit(X_train, y_train)
        start_time = time.time()
        if is_classification:
            y_pred = model.predict(X_test)
            score = run_binary_classification_metrics(y_pred, y_test)
        else:
            # Not supported for regression in TabFlex
            return
        running_time = time.time() - start_time

    else:
        raise ValueError(f"Method {args.method} is not supported. Please choose from ['tabpfn', 'rftabpfn', 'xgboost', 'catboost', 'lightgbm']")


    LOGGER.text(f"Score: {score}", level=LoggerObserver.SUCCESS)


    
    wandb_kwargs = {
        "name": f"{args['method']}",
        "entity": "kaylode",
        "project": "tabpfn-ehr",
        "reinit": True,
        "group": "tabpfn-ehr-test",
        "job_type": "test",
        "tags": tags,
        "resume": "allow",
        "dir": osp.join(result_folder),
    }
    wandb.init(**wandb_kwargs)
    wandb.run.config.update(OmegaConf.to_container(args, resolve=True))
    wandb.run.config.update({
        'study_name': study_name,
    })
    # Log the metrics to wandb
    wandb.run.log(score)
    wandb.run.log({"inference_time": running_time})

    LOGGER.text(f"Score: {score}", level=LoggerObserver.INFO)


if __name__ == "__main__":
    run()