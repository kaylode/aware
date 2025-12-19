from tabpfn.preprocessing import (
    PreprocessorConfig,
)

def default_classifier_preprocessor_configs(num_features: int=500) -> list[PreprocessorConfig]:
    """Default preprocessor configurations for classification."""
    return [
        PreprocessorConfig(
            "none",
            max_features_per_estimator=num_features
            # categorical_name="numeric",
        ),
    ]


def default_regressor_preprocessor_configs(num_features: int=500) -> list[PreprocessorConfig]:
    """Default preprocessor configurations for regression."""
    return [
        PreprocessorConfig("safepower", categorical_name="onehot", max_features_per_estimator = num_features),
    ]

def v2_5_default_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
    """Get the preprocessor configuration for classification in v2.5 of the model."""
    return [
        PreprocessorConfig(
            name="none",
            append_original=False,
            categorical_name="none",
            global_transformer_name=None,
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            name="none",
            categorical_name="numeric",
            max_features_per_estimator=500,
        ),
    ]


def v2_5_default_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
    """Get the preprocessor configuration for regression in v2.5 of the model."""
    return [
        PreprocessorConfig(
            name="quantile_uni_coarse",
            append_original="auto",
            categorical_name="numeric",
            global_transformer_name=None,
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            name="squashing_scaler_default",
            append_original=False,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            max_features_per_estimator=500,
        ),
    ]