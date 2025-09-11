import imblearn
import numpy as np


def synthetic_sampling(X, y, oversampling=None, undersampling=None, seed: int = 42, max_samples=None, 
                      balance_negatives=False, negative_ratio=1.0):
    """
    Function to perform synthetic sampling on the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        oversampling: Oversampling method
        undersampling: Undersampling method
        seed: Random seed
        max_samples: Maximum number of samples to keep after sampling
        balance_negatives: Whether to reduce negatives to balance with positives
        negative_ratio: Ratio of negatives to positives (1.0 = equal, 2.0 = 2x negatives)
    """

    # First, balance negatives if requested
    if balance_negatives:
        X, y = balance_negative_samples(X, y, negative_ratio=negative_ratio, seed=seed)

    sampler = None
    if oversampling is not None:
        if oversampling == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=seed)
        elif oversampling == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=seed)
        elif oversampling == "svmsmote":
            from imblearn.over_sampling import SVMSMOTE
            sampler = SVMSMOTE(random_state=seed)
        elif oversampling == "random":
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=seed)
        elif oversampling == "borderline":
            from imblearn.over_sampling import BorderlineSMOTE
            sampler = BorderlineSMOTE(random_state=seed)
        elif oversampling == "kmeans":
            from imblearn.over_sampling import KMeansSMOTE
            sampler = KMeansSMOTE(random_state=seed)

    elif undersampling is not None:
        if undersampling == "cnn":
            from imblearn.under_sampling import CondensedNearestNeighbour
            sampler = CondensedNearestNeighbour(random_state=seed)
        elif undersampling == "enn":
            from imblearn.under_sampling import EditedNearestNeighbours
            sampler = EditedNearestNeighbours()
        elif undersampling == "nearmiss":
            from imblearn.under_sampling import NearMiss
            sampler = NearMiss()
        elif undersampling == "onesided":
            from imblearn.under_sampling import OneSidedSelection
            sampler = OneSidedSelection(random_state=seed)
        elif undersampling == "cluster":
            from imblearn.under_sampling import ClusterCentroids
            sampler = ClusterCentroids(random_state=seed)
        elif undersampling == 'random':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=seed)
        else:
            raise ValueError("Invalid undersampling method")

    if sampler is not None:
        # Apply the sampling
        X, y = sampler.fit_resample(X, y)
    
    # Apply max_samples limit if specified
    if max_samples is not None and len(X) > max_samples:
        X, y = limit_samples(X, y, max_samples, seed=seed)
    
    return X, y


def balance_negative_samples(X, y, negative_ratio=1.0, seed=42):
    """
    Reduce the number of negative samples to balance with positive samples.
    
    Args:
        X: Feature matrix
        y: Target vector (0 = negative, 1 = positive)
        negative_ratio: Ratio of negatives to positives to maintain
        seed: Random seed
    
    Returns:
        X_balanced, y_balanced: Balanced feature matrix and target vector
    """
    # Get indices for positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    n_positives = len(pos_indices)
    n_negatives = len(neg_indices)
    
    print(f"Original distribution: {n_positives} positives, {n_negatives} negatives")
    
    # Calculate target number of negatives
    target_negatives = int(n_positives * negative_ratio)
    
    if target_negatives >= n_negatives:
        # No need to reduce negatives
        print(f"No reduction needed. Keeping all {n_negatives} negatives")
        return X, y
    
    # Randomly sample negatives
    np.random.seed(seed)
    selected_neg_indices = np.random.choice(neg_indices, target_negatives, replace=False)
    
    # Combine positive and selected negative indices
    balanced_indices = np.concatenate([pos_indices, selected_neg_indices])
    np.random.shuffle(balanced_indices)  # Shuffle to mix positives and negatives

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print(f"Balanced distribution: {n_positives} positives, {target_negatives} negatives")
    
    return X_balanced, y_balanced


def limit_samples(X, y, max_samples, seed=42, stratify=True):
    """
    Limit the number of samples to max_samples while maintaining class balance.
    
    Args:
        X: Feature matrix
        y: Target vector
        max_samples: Maximum number of samples to keep
        seed: Random seed
        stratify: Whether to maintain class balance
    """
    if len(X) <= max_samples:
        return X, y
    
    if stratify:
        from sklearn.model_selection import train_test_split
        X_limited, _, y_limited, _ = train_test_split(
            X, y, 
            train_size=max_samples,
            stratify=y,
            random_state=seed
        )
    else:
        # Random sampling without stratification
        np.random.seed(seed)
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_limited = X[indices]
        y_limited = y[indices]
    
    return X_limited, y_limited