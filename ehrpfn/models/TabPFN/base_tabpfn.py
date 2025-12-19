from .utils import (
	default_classifier_preprocessor_configs,
	default_regressor_preprocessor_configs
)

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import (
    ModelVersion,
)

def get_base_tabpfn(categorical_feature_indices, is_classification, device, seed=42):

	if is_classification:
		return TabPFNClassifier.create_default_for_version(
			version=ModelVersion.V2,
			n_estimators=1,
			ignore_pretraining_limits=True,
			categorical_features_indices=categorical_feature_indices,
			device=device,
			n_preprocessing_jobs=1,
			random_state=seed, 
			memory_saving_mode=True, 
			fit_mode='low_memory',
			inference_config={
				"FEATURE_SHIFT_METHOD": None,
				'FINGERPRINT_FEATURE': False,
				"CLASS_SHIFT_METHOD": None,
				"POLYNOMIAL_FEATURES": "no",
				"PREPROCESS_TRANSFORMS": default_classifier_preprocessor_configs(), #v2_5_default_classifier_preprocessor_configs(),
			}
		)
	else:
		return TabPFNRegressor.create_default_for_version(
			version=ModelVersion.V2,
			n_estimators=1,
			ignore_pretraining_limits=True,
			categorical_features_indices=categorical_feature_indices,
			device=device,
			random_state=seed,
			memory_saving_mode=True,
			n_preprocessing_jobs=1,
			fit_mode='low_memory',
			inference_config={
				"FEATURE_SHIFT_METHOD": None,
				'FINGERPRINT_FEATURE': False,
				"CLASS_SHIFT_METHOD": None,
				"POLYNOMIAL_FEATURES": "no",
				"PREPROCESS_TRANSFORMS": default_regressor_preprocessor_configs(),
			}
		)