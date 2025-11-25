from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler



class DataPreprocessor:
    def __init__(self, impute_method: str = "mean", encode_y: bool = False):
        """Initialize the DataPreprocessor.
        Parameters:
        - impute_method: str, method for imputing missing values (default is "mean").
        - encode_y: bool, whether to encode the target variable y (default is False).
        """
        self.impute_method = impute_method
        self.encode_y = encode_y
        # Initialize the imputer and scaler
        self.imputer = SimpleImputer(strategy=impute_method)
        self.scaler = StandardScaler()

    def convert_cat2num(
        self, X: pd.DataFrame, y: pd.Series | None = None, les: Optional[Dict[str, LabelEncoder]] = None
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        """Convert categorical columns to numerical values.

        Parameters:
        - X: DataFrame, input features.
        - y: Series or None, target variable.

        Returns:
        - X: numpy array, features with categorical columns converted to numerical.
        - y: numpy array, target variable with categorical values converted to numerical.
        - cat_vals: list of bool, indicating which columns were categorical.
        """
        cat_vals = []
        if les is None:
            les = {}
        # Convert categorical columns to numerical values using dataframe information
        for col in X.columns:
            if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col]):
                if col not in les:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    les[col] = le
                else:
                    le = les.get(col, None)
                    X[col] = le.transform(X[col])
    
                cat_vals.append(True)
            else:
                cat_vals.append(False)
        X = X.to_numpy().astype(np.float32)

        # Convert categorical target to numerical values
        if y is not None:
            if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
                if "target" not in les:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    les["target"] = le
                else:
                    le = les.get("target", None)
                    y = le.transform(y)

                cat_vals.append(True)
            else:
                y = y.to_numpy().astype(np.float32)
                cat_vals.append(False)
        else:
            y = X[:, -1]
            X = np.delete(X, -1, axis=1)
        return X, y, cat_vals, les

    def fit_transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Fit the imputer and scaler on the training data and transform it.

        Parameters:
            - X: numpy array, training features.

        Returns:
            - X: numpy array, transformed training features.
        """
        # Impute missing values
        X = self.imputer.fit_transform(X)

        # Scale the features
        X = self.scaler.fit_transform(X)

        return X

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Transform the test data using the fitted imputer and scaler.

        Parameters:
            - X: numpy array, testing features.

        Returns:
            - X: numpy array, transformed testing features.
        """
        # Impute missing values
        X = self.imputer.transform(X)

        # Scale the features
        X = self.scaler.transform(X)

        return X


def standardize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize the input tensor by subtracting the mean and dividing by the standard deviation.
    Args:
        tensor (torch.Tensor): Input tensor to standardize.
    Returns:
        torch.Tensor: Standardized tensor.
        torch.Tensor: Mean of the input tensor.
        torch.Tensor: Standard deviation of the input tensor.
    """
    y_means = tensor.mean(dim=0)
    y_stds = tensor.std(dim=0) + 1e-6
    return (tensor - y_means) / y_stds, y_means, y_stds

