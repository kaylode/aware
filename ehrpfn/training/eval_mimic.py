from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import scipy
import torch
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ehrpfn.training.model import TabDPTModel, Task
from ehrpfn.training.tabdpt import TabDPTClassifier, TabDPTRegressor
from ehrpfn.training.utils import FAISS, DataPreprocessor

from ehrpfn.training.eval_full import FullEval
from ehrpfn.utils.logger import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

class MIMICIVOutcomesEval(FullEval):
    def __init__(
        self,
        device,
        max_feat,
        impute_method="mean",
        use_retrieval=False,
        data_path="",
        label_path=""
    ):
        """Initialize the FullMIMICIVEval class for evaluating MIMIC-IV tabular data.

        Args:
            device (str): The device to use for evaluation (e.g., "cuda:0" or "cpu").
            max_feat (int): The maximum number of features to use for evaluation.
            impute_method (str, optional): The imputation method to use for missing values. Defaults to "mean".
            use_retrieval (bool, optional): Whether to use retrieval-augmented generation. Defaults to False.
            data_path (str): Path to the MIMIC-IV data folder.
            labels_path (str): Path to the MIMIC-IV labels folder.
        """
        self.device = device
        self.max_feat = max_feat
        self.use_retrieval = use_retrieval
        
        # MIMIC-IV specific paths
        self.data_path = data_path
        self.label_path = label_path
        
        # Load MIMIC-IV datasets
        self.datasets = {}
        self.labels = ['WILL_READMIT_WITHIN_7_DAYS', 'NEXT_ADMISSION_LOS', 'DEATH']
                
        # Set dataset IDs for MIMIC-IV tasks
        self.cc18_dids = [0, 2]  
        self.ctr_dids = [1] 

        self._load_mimiciv_data(impute_method)


    def _load_mimiciv_data(self, impute_method="mean"):
        """Load MIMIC-IV data from parquet and CSV files."""
        LOGGER.text("Loading MIMIC-IV evaluation data...")

        for dataset_id, label in enumerate(self.labels):
            train_features_df = pl.read_parquet(f"{self.data_path}/train/data.parquet")
            train_labels_df = pl.read_csv(f"{self.label_path}/PATIENT_OUTCOME/{label}_train_labels.csv")
            test_features_df = pl.read_parquet(f"{self.data_path}/test/data.parquet")
            test_labels_df = pl.read_csv(f"{self.label_path}/PATIENT_OUTCOME/{label}_test_labels.csv")

            train_features_df = train_features_df.filter(pl.col("admission_id") > 0)
            train_features_df = train_features_df.join(
                train_labels_df,
                on=["subject_id", "admission_id"],
                how="inner"
            )
            train_features_df = train_features_df.filter((pl.col(label).is_not_null()))
            X_train, y_train = train_features_df.drop([label]), train_features_df[label]

            test_features_df = test_features_df.filter(pl.col("admission_id") > 0)
            test_features_df = test_features_df.join(
                test_labels_df,
                on=["subject_id", "admission_id"],
                how="inner"
            )
            test_features_df = test_features_df.filter((pl.col(label).is_not_null()))
            X_test, y_test = test_features_df.drop([label]), test_features_df[label]

            # Preprocess the data
            preprocessor = DataPreprocessor(impute_method=impute_method)

            X_train = X_train.to_pandas()
            y_train = y_train.to_pandas()
            X_test = X_test.to_pandas()
            y_test = y_test.to_pandas()

            X_train, y_train, cat_vals, les = preprocessor.convert_cat2num(X_train, y_train)
            X_test, y_test, _, _ = preprocessor.convert_cat2num(X_test, y_test, les=les)

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
           
            # Create FAISS index for retrieval
            faiss_knn = FAISS(X_train, use_hnsw=False, metric="L2")
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
  
            self.datasets[dataset_id] = (
                f"MIMIC-IV-{label}",
                faiss_knn,
                X_train_tensor,
                X_test_tensor,
                y_train_tensor,  
                y_test_tensor
            )
            

    @torch.no_grad()
    def eval(
        self,
        model: TabDPTModel,
        context_length: int = 1024,
        inf_batch_size=512,
        temperature=0.8,
        return_individual_perfs=False,
    ) -> tuple[dict, Optional[dict]]:
        """Evaluate the model on MIMIC-IV classification datasets.

        Args:
            model (TabDPTModel): The TabDPT model to evaluate.
            context_length (int, optional): Context length for evaluation. Defaults to 1024.
            inf_batch_size (int, optional): Inference batch size. Defaults to 512.
            temperature (float, optional): Temperature for prediction. Defaults to 0.8.
            return_individual_perfs (bool, optional): Whether to return individual performance metrics. Defaults to False.

        Returns:
            dict: Performance metrics for MIMIC-IV classification tasks.
            Optional[dict]: Individual performance metrics for each dataset if return_individual_perfs is True.
        """
        classifier = TabDPTClassifier(
            model=model,
            mode=Task.CLS,
            device=self.device,
            inf_batch_size=inf_batch_size,
            tensor_eval=True,
        )

        regressor = TabDPTRegressor(
            model=model,
            mode=Task.REG,
            device=self.device,
            inf_batch_size=inf_batch_size,
            tensor_eval=True,
        )

        cls_performance = defaultdict(lambda: defaultdict(list))
        reg_performance = defaultdict(lambda: defaultdict(list))

        final_perfs = {}
        individual_perfs = {}

        # Evaluation for MIMIC-IV classification datasets
        for did in tqdm(self.cc18_dids, desc="Evaluating MIMIC-IV datasets"):
            dataset_name, faiss_index, X_train, X_test, y_train, y_test = self.datasets[did]

            classifier.fit(X_train, y_train, faiss_index)
            pred_val = classifier.predict_proba(
                X_test,
                temperature=temperature,
                context_size=context_length,
                use_retrieval=self.use_retrieval,
            )

            # Calculate metrics
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, pred_val[:, 1])
            else:
                auc = roc_auc_score(y_test, pred_val, multi_class="ovo")

            f1 = f1_score(y_test, np.argmax(pred_val, axis=1), average="weighted")
            acc = accuracy_score(y_test, np.argmax(pred_val, axis=1))
            ce = torch.nn.functional.cross_entropy(
                torch.Tensor(pred_val).float(), torch.Tensor(y_test).long()
            )
            
            cls_performance["cls-mimiciv"][did] = [acc, f1, auc, ce.item()]
            individual_perfs[dataset_name] = [acc, f1, auc, ce.item()]
            
            LOGGER.text(f"\nMIMIC-IV [{dataset_name}] Performance:", level=LoggerObserver.SUCCESS)
            LOGGER.text({
                "ACC": acc,
                "F1": f1,
                "AUC": auc,
                "CE": ce.item()
            }, level=LoggerObserver.SUCCESS)

        # Calculate mean performance across all MIMIC-IV tasks
        cls_perfs = np.array(list(cls_performance["cls-mimiciv"].values()))
        cls_perfs_mean = cls_perfs.mean(0)

        final_perfs["cls-mimiciv-acc"] = cls_perfs_mean[0]
        final_perfs["cls-mimiciv-f1"] = cls_perfs_mean[1]
        final_perfs["cls-mimiciv-auc"] = cls_perfs_mean[2]
        final_perfs["cls-mimiciv-ce"] = cls_perfs_mean[3]
        
        # Evaluation for MIMIC-IV regression datasets
        for did in self.ctr_dids:
            # evaluation for regression datasets
            dataset_name, faiss_index, X_train, X_test, y_train, y_test = self.datasets[did]

            regressor.fit(X_train, y_train, faiss_index)
            pred_val = regressor.predict(
                X_test, context_size=context_length, use_retrieval=self.use_retrieval
            ).flatten()
            # scaler = StandardScaler()
            # y_test = scaler.fit_transform(y_test.reshape(-1, 1)).flatten()

            mse = np.mean((y_test.cpu().numpy() - pred_val) ** 2)
            correlation = scipy.stats.pearsonr(y_test.cpu().numpy(), pred_val.flatten())
            r2 = r2_score(y_test.cpu().numpy(), pred_val)

            reg_performance["reg-mimiciv"][did] = [mse, correlation[0], r2]
            individual_perfs[dataset_name] = [mse, correlation[0], r2]
            LOGGER.text(f"\nMIMIC-IV [{dataset_name}] Performance:", level=LoggerObserver.SUCCESS)
            LOGGER.text({
                "MSE": mse,
                "Correlation": correlation[0],
                "R2": r2
            }, level=LoggerObserver.SUCCESS)

        reg_perfs = np.array(list(reg_performance["reg-mimiciv"].values()))
        reg_perfs_mean = reg_perfs.mean(0)
        final_perfs["reg-mimiciv-mse"] = reg_perfs_mean[0]
        final_perfs["reg-mimiciv-cor"] = reg_perfs_mean[1]
        final_perfs["reg-mimiciv-r2"] = reg_perfs_mean[2]

        LOGGER.text(f"\nMIMIC-IV Average Performance:", level=LoggerObserver.SUCCESS)
        LOGGER.text(final_perfs, level=LoggerObserver.SUCCESS)

        if return_individual_perfs:
            return final_perfs, individual_perfs
        else:
            return final_perfs