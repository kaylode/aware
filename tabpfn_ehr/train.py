import os
import os.path as osp
import hydra
import torch
import time
from tabpfn_ehr.gbms import Tuner
from tabpfn_ehr.validate import validate
from tabpfn_ehr.sampling import synthetic_sampling
from tabpfn_ehr.datasets import get_mimic_dataset, get_sjh_dataset

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.model.loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
)
from tabpfn_ehr.metrics import run_binary_classification_metrics, run_regression_metrics
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
from theseus import FileLogger, LoggerObserver, seed_everything
LOGGER = LoggerObserver.getLogger("main")



@hydra.main(version_base=None, config_path="../configs", config_name="mimiciv")
def run(args: DictConfig):

    seed_everything(args.seed)

    data_output_folder = osp.join(args.output_folder, "data", args.data_name)
    result_folder = osp.join(args.output_folder, "results", args.data_name, args.task_name)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(data_output_folder, exist_ok=True)


    LOGGER.subscribe(FileLogger(__name__, osp.join(args.output_folder, 'logs'), debug=True))

    if args.data_name == 'mimiciv':
        (
            X_train, y_train, 
            X_val, y_val, 
            X_test, y_test, 
            categorical_feature_indices
        ) = get_mimic_dataset(
            train_folder=args.train_folder,
            val_folder=args.val_folder,
            test_folder=args.test_folder,
            icu_ids=args.icu_ids,
            data_output_folder=data_output_folder,
            target_column=args.target_column,
            train_label_path=args.train_label_path,
            val_label_path=args.val_label_path,
            test_label_path=args.test_label_path,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            logger=LOGGER
        )
    elif args.data_name == 'sjh_cpe':
        (
            X_train, y_train, 
            X_val, y_val, 
            X_test, y_test, 
            categorical_feature_indices
        ) = get_sjh_dataset(
            input_folder=args.input_folder,
            output_folder=data_output_folder,
            target_col=args.target_column,
            max_length=args.max_length,
            num_chunks=args.num_chunks,
            use_data_sampler=False,
            use_excelformer=False,
            generalize_icd=True,
            use_contact_networks=True,
            fold=args.fold,
            patient_ids=None
        )
    else:
        raise ValueError(f"Data {args.data_name} is not supported. Please choose from ['mimiciv', 'sjh_cpe']")

    study_name = f"{args.method}_{args.data_name}_{args.task_name}"

    if args.selection_method is not None:
        study_name += f"_sel:{args.selection_method}"
    

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
        study_name += f"_{sampling_type}-{sampling_method}"
        LOGGER.text(f"After sampling: {X_train.shape}, {y_train.shape}", level=LoggerObserver.DEBUG)

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    LOGGER.text(f"Data shape: {X_train.shape}, {X_val.shape}, {X_test.shape}", level=LoggerObserver.DEBUG)

    result_folder = osp.join(result_folder, study_name, f"fold_{args.fold}")
    os.makedirs(result_folder, exist_ok=True)

    # Initialize the model

    if args.task_name in ['LOS']:
        is_classification = False
    else:
        is_classification = True

    if not osp.isfile(osp.join(result_folder, 'model.pkl')) and not osp.isfile(osp.join(result_folder, 'model.pt') and not osp.isfile(osp.join(result_folder, 'model.tabpfn_fit'))):

        if args.method in ['xgboost', 'catboost', 'lightgbm', 'rf']:
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
                    "tags": [f"task:{args.task_name}", 'tabpfn-ehr', 
                             f"fold:{args.fold}"],
                }
            )
            
            model = tuner.tune(X_train, X_val, y_train, y_val, is_classification=is_classification, cat_features=categorical_feature_indices)
        
        else:

            # Initialize the model based on method
            if args.method == "rftabpfn":
                clf_base = TabPFNClassifier(
                    ignore_pretraining_limits=True,
                    categorical_features_indices=categorical_feature_indices,
                    device=device,
                    n_jobs=4,
                    memory_saving_mode=True,
                    fit_mode = 'low_memory',
                )
                model = RandomForestTabPFNClassifier(
                    tabpfn=clf_base,
                    categorical_features=categorical_feature_indices.tolist(),
                    verbose=1,
                    show_progress=True,
                    max_predict_time=360,  # Will fit for one minute
                    fit_nodes=True,  # Whether or not to fit intermediate nodes
                    adaptive_tree=True,  # Whether or not to validate if adding a leaf helps or not
                )
                model.fit(X_train, y_train)
                
            elif args.method == "tabpfn":

                if is_classification:
                    if args.selection_method is not None:
                        device = 'cuda'
                    else:
                        device = 'cpu'
                    model = TabPFNClassifier(
                        ignore_pretraining_limits=True,
                        categorical_features_indices=categorical_feature_indices.tolist(),
                        device=device,
                        n_jobs=4,
                        memory_saving_mode=True,
                        fit_mode = 'low_memory',
                    )
                else:
                    model = TabPFNRegressor(
                        ignore_pretraining_limits=True,
                        categorical_features_indices=categorical_feature_indices.tolist(),
                        device="cpu",
                        fit_mode = 'low_memory',
                        memory_saving_mode=True,
                        n_jobs=4,
                        # inference_config={"SUBSAMPLE_SAMPLES": 10000}  # Needs to be set low so that not OOM on fitting intermediate nodes
                    )
                model.fit(X_train, y_train)

            elif args.method == 'tabicl':
                from tabicl import TabICLClassifier

                model = TabICLClassifier(
                    n_estimators=32,                                        # number of ensemble members
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

            elif args.method == 'beta':
                from tabpfn_ehr.models.BETA.beta.model.methods.Beta import BetaMethod
                config = {
                    "config": {
                        "model": {
                            "arch_type": "tabm-mini",
                            "k": 16,
                            "num_embeddings": {
                                "type": "PLREmbeddings",
                                "n_frequencies": 72,
                                "frequency_scale": 0.04,
                                "d_embedding": 32,
                                "lite": True
                            },
                            "backbone": {
                                "type": "MLP",
                                "n_blocks": 2,
                                "d_block":100,
                                "dropout": 0.15
                            }
                        },
                        "training": {
                            "lr": 0.003,
                            "weight_decay": 0.02
                        },
                        "general": {},
                    },
                    "dataset": "yeast",
                    "model_type": "Beta", 
                    "max_epoch": 200,
                    "batch_size": 256,
                    "normalization": "standard",
                    "num_nan_policy": "mean",
                    "cat_nan_policy": "new",
                    "cat_policy": "indices",
                    "num_policy": "none",
                    "n_bins": 2,
                    "cat_min_frequency": 0.0,
                    "n_trials": 100,
                    "seed": args.seed,
                    "workers": 0,
                    "gpu": "0",
                    "tune": False,
                    "retune": False,
                    "evaluate_option": "best-val",
                    "dataset_path": "data",
                    "model_path": "results_model" ,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }

                # Create info dictionary
                numerical_feature_indices = [i for i in range(X_train.shape[1]) if i not in categorical_feature_indices]
                info = {
                    'task_type': 'binclass' if (is_classification and len(np.unique(y_train)) == 2) else (
                        'multiclass' if is_classification else 'regression'
                    ),
                    'n_num_features': len(numerical_feature_indices),
                    'n_cat_features': len(categorical_feature_indices),
                    'n_classes': len(np.unique(y_train)) if is_classification else None,
                    'train_size': X_train.shape[0],
                    'val_size': X_val.shape[0], 
                    'test_size': X_test.shape[0],
                    'n_features': X_train.shape[1]
                }

                import argparse
                config = argparse.Namespace(**config)
                model = BetaMethod(config, is_regression=not is_classification)


                X_train_numerical = X_train[:, numerical_feature_indices]
                X_train_categorical = X_train[:, categorical_feature_indices]
                X_val_numerical = X_val[:, numerical_feature_indices]
                X_val_categorical = X_val[:, categorical_feature_indices]

                data = (
                    {'train': X_train_numerical, 'val': X_val_numerical},
                    {'train': X_train_categorical, 'val': X_val_categorical},
                    {'train': y_train, 'val': y_val}
                )

                time_cost = model.fit(data, info)
                predict_proba = lambda x: model.predict(x, info, model_name='best-val')[-1]
                setattr(model, 'predict_proba', predict_proba)
                # vl, vres, metric_name, predict_logits = model.predict(test_data, info, model_name='best-val')

            elif args.method == 'mothernet':
                from tabpfn_ehr.models.ticl.ticl.prediction import MotherNetClassifier, EnsembleMeta
                model = MotherNetClassifier(device='cuda', inference_device="cuda")
                model.fit(X_train, y_train)

            elif args.method == 'tabflex':
                from tabpfn_ehr.models.ticl.ticl.prediction.tabflex import TabFlex

                # Initialize and train TabFlex model
                model = TabFlex()
                model.fit(X_train, y_train)

            elif args.method == 'tabdpt':
                from tabpfn_ehr.models.TabDPT.src.tabdpt import TabDPTClassifier, TabDPTRegressor

                if is_classification:
                    model = TabDPTClassifier()
                else:
                    model = TabDPTRegressor()

                model.fit(X_train, y_train)
                # model.predict_proba = lambda X_test: model.predict_proba(X_test, temperature=0.8, context_size=1024)

            else:
                raise ValueError(f"Method {args.method} is not supported. Please choose from ['tabpfn', 'rftabpfn', 'xgboost', 'catboost', 'lightgbm']")

    
        if isinstance(model, torch.nn.Module):
            save_model(model, osp.join(result_folder, 'model.pt'))
        elif isinstance(model, TabPFNClassifier) or isinstance(model, TabPFNRegressor):
            save_fitted_tabpfn_model(model, osp.join(result_folder, 'model.tabpfn_fit'))
        else:
            save_model(model, osp.join(result_folder, 'model.pkl'))

        LOGGER.text(f"Model saved to {result_folder}/model.pkl", level=LoggerObserver.INFO)

    else:
        LOGGER.text(f"Model already exists at {result_folder}/model.pkl, skipping training.", level=LoggerObserver.INFO)

        if args.method == "tabpfn":
            # Later or on a CPU-only machine
            model = load_fitted_tabpfn_model("model.pkl", device="cpu")
        else:
            model = load_model(osp.join(result_folder, 'model.tabpfn_fit'))

        # if isinstance(model, TabPFNClassifier):
        #     model.device= "cpu"

    # Evaluate the model
    start_time = time.time()
    if is_classification:
        y_pred = model.predict_proba(X_test)
        # Evaluate the model
        score = run_binary_classification_metrics(
            y_pred, 
            y_test,
        )
    else:
        y_pred = model.predict(X_test)
        # Evaluate the model
        score = run_regression_metrics(
            y_pred, 
            y_test,
        )

    running_time = time.time() - start_time

    LOGGER.text(f"Score: {score}", level=LoggerObserver.SUCCESS)


    tags = [f"{args['method']}", f"task:{args['task_name']}", f"fold:{args['fold']}"]
    if args.sampling is not None:
        tags.append(f"sampling:{args['sampling']}")
    if args.selection_method is not None:
        tags.append(f"selection:{args['selection_method']}")

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


def save_model(model, output_path):
    # Save the model
    # import pickle
    import dill as pickle

    if hasattr(model, 'save_model'):
        # If the model has a save_model method, use it
        model.save_model(output_path)
    else:
        if output_path.endswith('.pt'):
            # If the output path is a .pt file, save the model using torch
            import torch
            torch.save(model.state_dict(), output_path)
        elif output_path.endswith('pkl'):
            with open(output_path, "wb") as f:
                pickle.dump(model, f)

    return


def load_model(input_path, model=None):
    # Load the model
    # import pickle
    import dill as pickle

    if model is not None and hasattr(model, 'load_model'):
        # If the model has a load_model method, use it
        model = model.load_model(input_path)
    else:
        if input_path.endswith('pt'):
            import torch
            state_dict = torch.load(input_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        elif input_path.endswith('pkl'):
            with open(input_path, "rb") as f:
                model = pickle.load(f)

    return model

if __name__ == "__main__":
    run()