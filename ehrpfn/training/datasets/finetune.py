import random
from collections import defaultdict

import numpy as np
import openml
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm
import os.path as osp
from tabulate import tabulate 
from ehrpfn.training.model import pad_x
from ehrpfn.training.utils import FAISS, DataPreprocessor

import sys
sys.path.append("..")  # Add parent directory to path for imports
from ehrpfn.datasets import get_mimic_dataset_v2

from ehrpfn.utils.logger import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')



class FullDataset(Dataset):
    """FullDataset class for loading and processing datasets.
    This class inherits from `torch.utils.data.Dataset` and is used to load datasets from OpenML,
    preprocess them, and provide samples for training a model.
    It supports both regression and classification tasks, with the ability to use k-NN for context sampling.
    """

    def __init__(self, device: str, config: DictConfig):
        """FullDataset initialization.

        Args:
            device (str): The device to use for tensor operations.
            config (DictConfig): The configuration for the dataset.
        """
        self.steps_per_epoch = (
            config.training.num_agg * config.training.num_model_updates * config.training.num_epochs
        )
        self.batch_size = config.training.batch_size
        self.device = device
        self.context_length = getattr(config.training, "seq_len", 1024)
        self.retrieval = getattr(config.data, "retrieval", True)
        self.max_feat = config.model.max_num_features
        self.y_reg_augment = config.data.y_reg_augment

        # Load dataset IDs from CSV file
        # TODO: make dataset_ids configurable
        self.dataset_ids = []
        self.open_ml_health_ids = []
        self.open_ml_general_ids = []
        if config.data.use_health_data:
            self.open_ml_health_ids = pd.read_csv(f"{config.data.openml.input_folder}/health_training_datasets.csv")["did"].values.ravel().tolist()
            self.dataset_ids.extend(self.open_ml_health_ids)

        if config.data.use_general_data:
            self.open_ml_general_ids = pd.read_csv(f"{config.data.openml.input_folder}/noleak_training_datasets.csv")["did"].values.ravel().tolist()
            self.dataset_ids.extend(self.open_ml_general_ids)

        if len(self.dataset_ids) > 0:
            self.dataset_ids = sorted(self.dataset_ids)
        self.datasets = defaultdict(dict)
        self.dataset_weights = []

        # Initialize dataset summary list
        dataset_summary = []

        # Load datasets and preprocess them
        for did in tqdm(self.dataset_ids):
            # download the dataset from OpenML
            dataset = openml.datasets.get_dataset(
                did,
                download_data=False,
                download_qualities=False,
                download_features_meta_data=False,
            )

            # Get the data as a pandas DataFrame
            X, y, _, _ = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )

            original_samples, original_features = X.shape

            # TODO: preprocessing is a bit different from the usual .fit_transform() since concatenation happens in the middle
            preprocessor = DataPreprocessor()
            # Preprocess the data
            X, y, cat_vals, _ = preprocessor.convert_cat2num(X, y)
            X = preprocessor.imputer.fit_transform(X)

            # Concatenate the target as the last column
            X = np.concatenate([X, y[:, None]], axis=1)
            X = preprocessor.scaler.fit_transform(X)

            if config.data.retrieval:
                faiss_knn = FAISS(X, metric="L2", use_hnsw=False)
            else:
                faiss_knn = None

            self.datasets[did] = {"X": X, "index": faiss_knn, "cat": cat_vals}

            # Add to summary
            dataset_summary.append([
                f"OpenML-{did}",
                dataset.name if hasattr(dataset, 'name') else f"Dataset-{did}",
                original_samples,
                original_features,
            ])

            if did in self.open_ml_health_ids:
                self.dataset_weights.append(5.0)
            else:
                self.dataset_weights.append(1.0)

        self.mimiciv_id = None # This did has not been used in the training set
        if config.data.use_mimiciv:
            #### Add MIMIC-IV
            mimic_dataset = get_mimic_dataset_v2(
                data_csv=config.data.mimiciv.data_csv,
                train_label_path=config.data.mimiciv.train_label_path,
                test_label_path=config.data.mimiciv.test_label_path,
                feature_selection_path=None,
                selection_method=None,
                process_file_path=None,
                trajectory_file_path=None,
            )

            self.mimiciv_id = 69 # This did has not been used in the training set
            self.dataset_ids.append(self.mimiciv_id)

            (
                X_train, categorical_feature_indices, feature_names
            ) = (
                mimic_dataset["X_train"], 
                mimic_dataset["categorical_feature_indices"], mimic_dataset["feature_names"]
            )

            cat_vals = categorical_feature_indices
            preprocessor = DataPreprocessor()
            X_train = preprocessor.imputer.fit_transform(X_train)
            X_train = preprocessor.scaler.fit_transform(X_train)

            self.datasets[self.mimiciv_id] = {"X": X_train, "index": None, "cat": cat_vals}

            # Add MIMIC-IV to summary
            dataset_summary.append([
                f"MIMIC-IV-PIPE-{self.mimiciv_id}",
                "MIMIC-IV-PIPE",
                X_train.shape[0],
                X_train.shape[1],
            ])
            self.dataset_weights.append(5.0)

        if config.data.use_synthetic:
            synthetic_ehrs = [
                pd.read_csv(f"{config.data.synthetic.input_folder}/synthea-pt30k-stroke-ml-table-sel-convert.csv"),
                pd.read_csv(f"{config.data.synthetic.input_folder}/synthea-pt30k4-lc-data-sel-convert.csv"),
            ]
            synthetic_names = ["Synthea-Stroke", "Synthea-LC"]
            for i, (ehr, name) in enumerate(zip(synthetic_ehrs, synthetic_names)):
                # Preprocess the synthetic EHRs
                preprocessor = DataPreprocessor()
                ehr = ehr.drop(columns=["ptnum"])
                X, y = ehr.iloc[:, 1:], ehr.iloc[:, 0]
                original_samples, original_features = X.shape
                X, y, cat_vals, _ = preprocessor.convert_cat2num(X, y)
                X = preprocessor.imputer.fit_transform(X)
                X = preprocessor.scaler.fit_transform(X)

                # Concatenate the target as the last column
                X = np.concatenate([X, y[:, None]], axis=1)

                if config.data.retrieval:
                    faiss_knn = FAISS(X, metric="L2", use_hnsw=False)
                else:
                    faiss_knn = None

                self.datasets[100 + i] = {"X": X, "index": faiss_knn, "cat": cat_vals}
                self.dataset_ids.append(100 + i)

                # Add to summary
                dataset_summary.append([
                    f"Synthetic-{100 + i}",
                    name,
                    original_samples - 1,  # Exclude ptnum column
                    original_features - 2,  # Exclude ptnum and target
                ])
                self.dataset_weights.append(5.0)

        self._print_dataset_summary(dataset_summary)

    def _print_dataset_summary(self, dataset_summary):
        """Print a summary table of all loaded datasets.
        Args:
            dataset_summary (list): List of dataset information.
        """
        headers = ["Dataset ID", "Dataset Name", "Samples", "Original Features"]
        
        LOGGER.text("\n" + "="*80)
        LOGGER.text("DATASET SUMMARY")
        LOGGER.text("="*80)
        LOGGER.text(tabulate(dataset_summary, headers=headers, tablefmt="grid"))
        LOGGER.text("="*80)
        
        # LOGGER.text total statistics
        total_datasets = len(dataset_summary)
        total_samples = sum(row[2] for row in dataset_summary)
        total_columns = sum(row[3] for row in dataset_summary)
        
        LOGGER.text(f"Total Datasets: {total_datasets}")
        LOGGER.text(f"Total Samples: {total_samples:,}")
        LOGGER.text(f"Total Features: {total_columns:,}")

        LOGGER.text("="*80 + "\n")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.steps_per_epoch * self.batch_size
