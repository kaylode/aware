from tabpfn.preprocessing import (
    PreprocessorConfig,
)

def default_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
    """Default preprocessor configurations for classification."""
    return [
        PreprocessorConfig(
            "none",
            categorical_name="numeric",
            subsample_features=-1,
        ),
    ]


def default_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
    """Default preprocessor configurations for regression."""
    return [
        PreprocessorConfig("safepower", categorical_name="onehot"),
    ]
