from typing import *
from omegaconf import DictConfig
from ehrmeds import (
	get_sjh_dataset, 
	get_mimic_dataset, 
	get_openml_ehr,
	get_openml_ehr_imbalance,
	get_eicu,
	get_eicu_default_configs,
	get_mimiciv_default_configs,
	get_sjh_cpe_default_configs,
	get_openml_ehr_default_configs,
	get_openml_ehr_imbalance_default_configs
)
import numpy as np
import pandas as pd
from ehrpfn.utils import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")


def get_data(args: DictConfig):
	data_name = args.data.data_name
	if args.data.selection_method is not None:
		assert args.data.num_features is not None, "Please provide num_features when using feature selection."
		args.data.selection_method += f"_{args.data.num_features}"

	if data_name.startswith(('mimicpipe', 'mimiciv', 'mimic')):
		config_func = get_mimiciv_default_configs
		data_func = get_mimic_dataset
	elif data_name == 'openml_ehr':
		config_func = get_openml_ehr_default_configs
		data_func = get_openml_ehr
	elif data_name == 'openml_ehr_imbalance':
		config_func = get_openml_ehr_imbalance_default_configs
		data_func = get_openml_ehr_imbalance
	elif data_name == 'sjh_cpe':
		config_func = get_sjh_cpe_default_configs
		data_func = get_sjh_dataset
	elif data_name == 'eicu':
		config_func = get_eicu_default_configs
		data_func = get_eicu
	else:
		raise ValueError(f"Data {data_name} is not supported. Please choose from ['mimiciv', 'sjh_cpe']")

	configs = config_func(
		TASK=args.data.task,
		selection_method=args.data.selection_method,
		feature_selection_path=args.data.feature_selection_path,
	)
	data = data_func(**configs)
	return data


def prepare_data(data, args: DictConfig):
	"""
	Prepare data for inference based on the provided arguments.
	"""

	(
		X_train, y_train, 
		X_test, y_test, 
		categorical_feature_indices,
		feature_names 
	) = (
		data['X_train'], data['y_train'], 
		data['X_test'], data['y_test'], 
		data['categorical_feature_indices'],
		data.get('feature_names', None)
	)

	if args.data.get('remove_aggregated_time_series', False):
		time_series_cols = [i for i, col in enumerate(feature_names) if col.lower().startswith(('lab//', 'vital//', 'infusion//', 'chart//'))]
		LOGGER.text(f"Removing aggregated time series columns: {time_series_cols}", level=LoggerObserver.DEBUG)
		X_train = X_train[:, [i for i in range(X_train.shape[1]) if i not in time_series_cols]]
		X_test = X_test[:, [i for i in range(X_test.shape[1]) if i not in time_series_cols]]
		categorical_feature_indices = [i for i in categorical_feature_indices if i not in time_series_cols]
		feature_names = [feature_names[i] for i in range(len(feature_names)) if i not in time_series_cols]

	if args.data.sampling.do_sampling:
		from ehrmeds.utils.sampling import synthetic_sampling

		LOGGER.text(f"================= Performing SYNTHESIS SAMPLING =================", level=LoggerObserver.DEBUG)
		LOGGER.text(f"Before sampling: {X_train.shape}, {y_train.shape}", level=LoggerObserver.DEBUG)
		try:
			sampling_type, sampling_method = args.data.sampling.sampling_method.split('_')
			study_name += f"_sampl:{sampling_type}-{sampling_method}"
		except:
			sampling_type, sampling_method = None, None

		X_train, y_train = synthetic_sampling(
			X_train, y_train, 
			oversampling=sampling_method if sampling_type == "oversampling" else None,
			undersampling=sampling_method if sampling_type == "undersampling" else None,
			seed=args.inference.seed,
			balance_negatives=args.data.sampling.balance_negatives, 
			negative_ratio=args.data.sampling.negative_ratio
		)
		LOGGER.text(f"After sampling: {X_train.shape}, {y_train.shape}", level=LoggerObserver.DEBUG)
	
	if args.data.sampling.max_samples is not None:
		from ehrmeds.utils.sampling import limit_samples

		LOGGER.text(f"Limiting samples to {args.data.sampling.max_samples}", level=LoggerObserver.DEBUG)
		X_train, y_train, indices = limit_samples(
			X_train, y_train, 
			max_samples=args.data.sampling.max_samples,
			seed=args.inference.seed,
			stratify=True,
			return_indices=True
		)
		LOGGER.text(f"After limiting samples: {X_train.shape}, {y_train.shape}", level=LoggerObserver.DEBUG)
	

		for key in data:
			if key == 'X_train' or key == 'y_train':
				continue
			if isinstance(data[key], Iterable) and len(data[key]) == len(data['X_train']):
				if isinstance(data[key], np.ndarray):
					data[key] = data[key][indices]
				elif isinstance(data[key], list):
					data[key] = [data[key][i] for i in indices]
				elif isinstance(data[key], pd.DataFrame):
					data[key] = data[key].iloc[indices]
				else:
					raise ValueError(f"Unsupported data type for key {key} during sample limiting.")
	
	y_train = y_train.astype(int)
	y_test = y_test.astype(int)
	LOGGER.text(f"Data shape: {X_train.shape}, {X_test.shape}", level=LoggerObserver.DEBUG)

	data['X_train'] = X_train
	data['y_train'] = y_train
	data['X_test'] = X_test
	data['y_test'] = y_test
	data['categorical_feature_indices'] = categorical_feature_indices
	data['feature_names'] = feature_names
	data['data_size'] = X_train.shape[0] + X_test.shape[0]

	return data