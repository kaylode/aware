from typing import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ehrpfn.datasets.tabdpt_datasets.openml import OpenMLDataset
from .utils import DataPreprocessor


class OpenMLEHRImbalance:
    def __init__(
        self,
        data_folder:str="data_splits",
        impute_method="mean",
        imbalance_ratio:float=1.0,
        seed:int=42
    ):
        """Initialize the FullEval class for evaluating tabular data.

        Args:
            device (str): The device to use for evaluation (e.g., "cuda:0" or "cpu").
            impute_method (str, optional): The imputation method to use for missing values. Defaults to "mean".
            use_retrieval (bool, optional): Whether to use retrieval-augmented generation. Defaults to False.
        """

        np.random.seed(seed)

        # loading classification dataset
        df_eval = pd.read_csv(f"{data_folder}/health_training_datasets_imbalance.csv")
        df_eval_cls = df_eval[df_eval["type"] == "cls"]
        df_eval_rgr = df_eval[df_eval["type"] == "rgr"]

        self.health_cls_ids = df_eval_cls[
            "did"
        ].values.tolist()
        self.health_cls_ids = sorted(self.health_cls_ids)

        self.health_rgr_ids = df_eval_rgr[
            "did"
        ].values.tolist()
        self.health_rgr_ids = sorted(self.health_rgr_ids)
        
        # get did (dataset ID) to tid (task ID) mapping
        did_tid_mapping = dict(
            zip(
                df_eval["did"],
                df_eval["tid"],
            )
        )

        self.datasets = {}
        for did in [*self.health_cls_ids, *self.health_rgr_ids]:
            if not did_tid_mapping[did] or np.isnan(did_tid_mapping[did]):
                dataset = OpenMLDataset("openml_dataset", openml_dataset_id=int(did))
                dataset.prepare_data("data", manual_target_column=df_eval[df_eval["did"] == did]["target_column"].values[0])
            else:
                dataset = OpenMLDataset("openml_dataset", openml_task_id=int(did_tid_mapping[did]))
                dataset.prepare_data("data")

            dataset_name = dataset.openml_dataset.name

          
            # categorical_feature_indices = [i for i, cat in enumerate(dataset.is_categorical) if cat]
            categorical_feature_indices = dataset.metadata["categorical_feature_inds"]

            X_train, y_train = dataset.train_instances()
            X_val, y_val = dataset.val_instances()

            X_train = np.concatenate([X_train, X_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)

            X_test, y_test = dataset.test_instances()

            # Create imbalanced training data
            X_train, y_train = self.generate_imbalanced_data(X_train, y_train, imbalance_ratio, seed)

            if dataset_name == 'COVID-19-Hospitals-Treatment-Plan':
                # This dataset has 11 classes, thus, convert label 10 to 9
                y_train[y_train == 10] = 9
                y_test[y_test == 10] = 9

            # TODO missing convert_cat2num step in full_dataset.py
            # Preprocess data
            preprocessor = DataPreprocessor(impute_method=impute_method)
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            is_classification = True if did in self.health_cls_ids else False

            feature_names = dataset.column_names

            # Map all classes to string first, then use LabelEncoder to convert back to integer, so as to reset the the values to correct range
            if is_classification:
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train.astype(str))
                y_test = label_encoder.transform(y_test.astype(str))
                num_classes = len(label_encoder.classes_)
                class_distribution = np.bincount(y_train)
            else:
                raise NotImplementedError("Data imbalance for regression is not implemented yet.")

            for i in categorical_feature_indices:
                if int(i) >= X_train.shape[1]:
                    raise ValueError(f"Categorical feature index {i} is out of bounds for input with {X_train.shape[1]} features.")

            self.datasets[did] = (dataset_name, X_train, X_test, y_train, y_test, categorical_feature_indices, is_classification, feature_names, num_classes, class_distribution)

    def generate_imbalanced_data(self, X, y, imbalance_ratio, seed):
        """
        Resample the dataset to create class imbalance. 
        The majority class will have the original number of samples, 
        while the minority class will have samples reduced to the specified imbalance ratio.
        """
        np.random.seed(seed)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_count = max(class_counts)
        min_count = min(class_counts)

        need_min_samples = int(max_count // imbalance_ratio)
        if need_min_samples > min_count:
            do_undersampling_max = True
        else:
            do_undersampling_max = False

        new_X, new_y = [], []
        for cls_ind, cls in enumerate(unique_classes):
            cls_indices = np.where(y == cls)[0]
            if class_counts[cls_ind] == max_count:
                # Majority class, keep all samples
                if do_undersampling_max:
                    n_samples = int(min_count * imbalance_ratio)
                    selected_indices = np.random.choice(cls_indices, n_samples, replace=False)
                else:
                    selected_indices = cls_indices
            else:
                if do_undersampling_max:
                    selected_indices = cls_indices
                else:
                    # Minority class, reduce samples
                    n_samples = int(max_count // imbalance_ratio)
                    selected_indices = np.random.choice(cls_indices, n_samples, replace=False)
            new_X.append(X[selected_indices])
            new_y.append(y[selected_indices])

            # Shuffle the new dataset
        combined = list(zip(np.vstack(new_X), np.hstack(new_y)))
        np.random.shuffle(combined)
        new_X[:], new_y[:] = zip(*combined)

        return np.array(new_X), np.array(new_y)

def get_openml_ehr_imbalance(data_folder:str, ratio:float=1.0, seed:int=42):

    data = OpenMLEHRImbalance(data_folder=data_folder, imbalance_ratio=ratio, seed=seed)

    return [{
        'did': did,
        'data_name': d[0],
        'X_train': d[1], 'y_train': d[3],
        'X_test': d[2], 'y_test': d[4],
        'categorical_feature_indices': d[5],
        'is_classification': d[6],
        'feature_names': d[7],
        'num_classes': d[8],
        'num_features': len(d[7]),
        'class_distribution': d[9]
    } for did, d in data.datasets.items()]

