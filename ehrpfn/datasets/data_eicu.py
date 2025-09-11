from typing import *
import numpy as np
import pandas as pd
import json
import os.path as osp
from sklearn.preprocessing import LabelEncoder
from ehrpfn.training.utils import DataPreprocessor

def select_features(x, y, selection_method:str, feature_selection_path:str=None, selected_indices=None):
    # Load top features
    if selection_method is not None:
        if selected_indices is None:
            if selection_method.startswith("random"):
                size = int(selection_method.split("_")[1]) if "_" in selection_method else 100
                # Random selection of features
                selected_indices = np.random.choice(x.shape[1], size=size, replace=False)
            else:
                assert feature_selection_path is not None, "feature_selection_path must be provided for feature selection"
                tmp = selection_method.split("_")
                if len(tmp) > 1:
                    selection_method, size = tmp[0], int(tmp[1])
                else:
                    size = 100
                feature_selection_path = osp.join(feature_selection_path, f"{selection_method}_feature_importance.json")
                with open(feature_selection_path, 'r') as f:
                    top_features = json.load(f)
                top_features = {k:v for k,v in sorted(top_features.items(), key=lambda item: item[1], reverse=False)}
                top_features = list(top_features.keys())[:size]
                selected_indices = [x.columns.tolist().index(col) for col in top_features if col in x.columns]
        x = x.iloc[:, selected_indices]

    feature_names = x.columns.tolist()
    return x, y, feature_names, selected_indices


def get_eicu(
        task:str,
        csv_file:str,
        label_file:str,
        train_admission_id_file:str,
        val_admission_id_file:str,
        test_admission_id_file:str,
        selection_method:str,
        feature_selection_path:str,
        embedding_file_path:str=None,
        impute_method="mean",
    ):
        """Initialize the FullEval class for evaluating tabular data.

        Args:
            device (str): The device to use for evaluation (e.g., "cuda:0" or "cpu").
            impute_method (str, optional): The imputation method to use for missing values. Defaults to "mean".
            use_retrieval (bool, optional): Whether to use retrieval-augmented generation. Defaults to False.
        """

        # loading classification dataset
        df = pd.read_csv(csv_file)
        label_df = pd.read_csv(label_file)

        train_ids = open(train_admission_id_file).read().splitlines()
        val_ids = open(val_admission_id_file).read().splitlines()
        train_ids = list(map(int, train_ids+val_ids))
        test_ids = open(test_admission_id_file).read().splitlines()
        test_ids = list(map(int, test_ids))

        X_train = df[df['ICU_ID'].isin(train_ids)].reset_index(drop=True)
        X_test = df[df['ICU_ID'].isin(test_ids)].reset_index(drop=True)
        y_train = label_df[label_df['ICU_ID'].isin(train_ids)][task].reset_index(drop=True)
        y_test = label_df[label_df['ICU_ID'].isin(test_ids)][task].reset_index(drop=True)
        train_ids_df = X_train[['PATIENT_ID', 'ADMISSION_ID', 'ICU_ID']].drop_duplicates().reset_index(drop=True)
        test_ids_df = X_test[['PATIENT_ID', 'ADMISSION_ID', 'ICU_ID']].drop_duplicates().reset_index(drop=True)
        X_train = X_train.drop(['ADMISSION_ID', 'ICU_ID', 'PATIENT_ID', 'HOSPITAL_ID'], axis=1)
        X_test = X_test.drop(['ADMISSION_ID', 'ICU_ID', 'PATIENT_ID', 'HOSPITAL_ID'], axis=1)
        
        X_train, y_train, feature_names, selected_indices = select_features(
            X_train, y_train, selection_method=selection_method, 
            feature_selection_path=feature_selection_path, 
            selected_indices=None
        )

        X_test, y_test, _, _ = select_features(
            X_test, y_test, selection_method=selection_method,
            feature_selection_path=feature_selection_path,
            selected_indices=selected_indices
        )
        
        categorical_feature_indices = [i for i, f in enumerate(feature_names) if df[f].dtype == 'object']
        class_distribution = y_train.value_counts(normalize=True).to_dict()

        y_train = y_train.values
        y_test = y_test.values

        # Encode categorical features
        label_encoders = {}
        for col in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
        
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        preprocessor = DataPreprocessor(impute_method=impute_method)
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        

        is_classification = True
        num_classes = 2

        if embedding_file_path is not None:
            X_train_traj = np.load(f"{embedding_file_path}/train_embeddings.npz")
            train_metadata_ids = X_train_traj['metadata'] # "subject_id", "hadm_id", "stay_id", "admission_id"
            train_embeddings = X_train_traj['embeddings']

            # Get embedding dimension
            embedding_dim = train_embeddings.shape[1]

            train_ids_df = train_ids_df.astype(
                {"PATIENT_ID": str, "ADMISSION_ID": str, "ICU_ID": str}
            )

            # Use left join to keep all patients from train_ids_df
            sub_train_df = pd.DataFrame({
                "PATIENT_ID": train_metadata_ids[:, 0],
                "ADMISSION_ID": train_metadata_ids[:, 2],
                "ICU_ID": train_metadata_ids[:, 1]
            }).drop_duplicates().reset_index()
            
            filtered_train_ids = pd.merge(
                train_ids_df,
                sub_train_df,
                on=["PATIENT_ID", "ADMISSION_ID", "ICU_ID"],
                how="left"  # Changed from "inner" to "left"
            )

            # Create embeddings array with zeros for missing patients
            train_embedding_matrix = np.zeros((len(filtered_train_ids), embedding_dim))
            
            # Fill in embeddings for patients that have them
            valid_indices = filtered_train_ids['index'].to_numpy()
            valid_mask = ~np.isnan(valid_indices.astype(float))  # Check for non-null indices
            
            if valid_mask.any():
                train_embedding_matrix[valid_mask] = train_embeddings[valid_indices[valid_mask].astype(int)]
            
            X_train_traj = train_embedding_matrix

            # Same logic for test data
            X_test_traj = np.load(f"{embedding_file_path}/test_embeddings.npz")
            test_metadata_ids = X_test_traj['metadata']
            test_embeddings = X_test_traj['embeddings']

            test_ids_df = test_ids_df.astype(
                {"PATIENT_ID": str, "ADMISSION_ID": str, "ICU_ID": str}
            )
            
            filtered_test_ids = pd.merge(
                test_ids_df,
                pd.DataFrame({
                    "PATIENT_ID": test_metadata_ids[:, 0],
                    "ADMISSION_ID": test_metadata_ids[:, 2],
                    "ICU_ID": test_metadata_ids[:, 1]
                }).drop_duplicates().reset_index(),
                on=["PATIENT_ID", "ADMISSION_ID", "ICU_ID"],
                how="left"  # Changed from "inner" to "left"
            )
            
            # Create embeddings array with zeros for missing patients
            test_embedding_matrix = np.zeros((len(filtered_test_ids), embedding_dim))
            
            # Fill in embeddings for patients that have them
            valid_indices = filtered_test_ids['index'].to_numpy()
            valid_mask = ~np.isnan(valid_indices.astype(float))  # Check for non-null indices
            
            if valid_mask.any():
                test_embedding_matrix[valid_mask] = test_embeddings[valid_indices[valid_mask].astype(int)]
            
            X_test_traj = test_embedding_matrix

        else:
            X_train_traj = None
            X_test_traj = None


        return {
            'data_name': 'eicu',
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_embs_train': X_train_traj,
            'X_embs_test': X_test_traj,
            'categorical_feature_indices': categorical_feature_indices,
            'is_classification': is_classification,
            'feature_names': feature_names,
            'num_classes': num_classes,
            'num_features': len(feature_names),
            'class_distribution': class_distribution
        }

