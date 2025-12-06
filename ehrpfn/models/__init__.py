from .EHRPFN import EHRPFNClassifier, EHRPFNRegressor
from .talent_wrapper import TALENTWrapper
from .knn import SimpleKNNClassifier, SimpleKNNRegressor
from tradml import TradMLTuner
from ehrret import get_retriever, get_ehr_retrieval_weight
from ehrpfn.utils import save_model
import os
import os.path as osp
import time
from ehrpfn.utils import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

def get_predictions(args, data, config_dict):
	"""
	Given the args and data, initialize and return the appropriate model,
	"""
	result_folder = config_dict['result_folder']
	X_train = data['X_train']
	y_train = data['y_train']
	X_test = data['X_test']
	categorical_feature_indices = data['categorical_feature_indices']
	is_classification = data['is_classification']
	study_name = config_dict['study_name']
	method_name = args.inference.model

	## Init context retrievers
	if args.retrieval.retrieval_method is not None and args.retrieval.retrieval_method.startswith('weighted'):
		retrieval_method, weight_type = args.retrieval.retrieval_method.split('_')
		retrieval_weights = get_ehr_retrieval_weight(
			weight_type, task=args.data.task, 
			feature_names=data.get('feature_names', None),
			json_weight_path=args.retrieval.get('weight_json_path', None)
		)
	else:
		retrieval_weights = None
		retrieval_method = args.retrieval.retrieval_method

	if args.retrieval.retrieval_method is not None:
		retriever = get_retriever(
			retrieval_method, 
			k=args.retrieval.context_size,
			distance=args.retrieval.distance,
			weights=retrieval_weights,
			n_features=X_train.shape[1],
			pretrained_path=args.retrieval.pretrained_path,
		)
	else:
		retriever = None

	training_time = None
	if method_name in ['xgboost', 'rf', 'catboost', 'lightgbm', 'knn', 'mlp', 'logistics']:
		
		from sklearn.model_selection import train_test_split

		if args.wandb.use_wandb:
			wandb_kwargs={
				"project": args.wandb.project,
				"entity": args.wandb.entity,
			}
		else:
			wandb_kwargs = None

		tuner = TradMLTuner(
			storage=osp.join(result_folder, "optuna.log"),
			study_name=study_name,
			n_trials=5,
			direction="maximize" if is_classification else "minimize",
			save_dir=result_folder,
			method=args.inference.model,
			use_best_params=False,
			wandb_kwargs=wandb_kwargs,
			feature_names=data.get('feature_names', None),
			classnames=['0', '1'] if is_classification else None
		)
		X_train, X_val, y_train, y_val = train_test_split(
			X_train, y_train, 
			test_size=0.1, 
			random_state=args.inference.seed, 
			stratify=y_train if is_classification else None
		)
		start_time = time.time()
		model = tuner.tune(
			X_train, X_val, y_train, y_val, 
			is_classification=is_classification, 
			cat_features=categorical_feature_indices,
			extra_params={'weight_fn': retrieval_weights}
		)
		training_time = time.time() - start_time

		save_model(model, osp.join(result_folder, 'model.pkl'))

		start_time = time.time()
		if is_classification:
			y_pred = model.predict_proba(X_test)
		else:
			y_pred = model.predict(X_test)
		running_time = time.time() - start_time
	

	elif method_name == "knn-simple":
		from ehrpfn.models.knn import SimpleKNNClassifier, SimpleKNNRegressor

		if is_classification:
			model = SimpleKNNClassifier(retriever=retriever)
		else:
			model = SimpleKNNRegressor(retriever=retriever)
		start_time = time.time()
		model.fit(
			X_train, y_train,
		)
		training_time = time.time() - start_time
		start_time = time.time()
		if is_classification:
			y_pred = model.predict_proba(X_test)
		else:
			y_pred = model.predict(X_test)
		running_time = time.time() - start_time

	elif method_name == "tabpfn":
		from tabpfn import TabPFNClassifier, TabPFNRegressor

		from ehrpfn.models.TabPFN.utils import (
			default_classifier_preprocessor_configs,
			default_regressor_preprocessor_configs,
		)
		
		if is_classification:
			if args.data.selection_method is not None:
				device = 'cuda'
			else:
				device = 'cpu'
			model = TabPFNClassifier(
				n_estimators=8,
				ignore_pretraining_limits=True,
				categorical_features_indices=categorical_feature_indices,
				device=device,
				n_jobs=4,
				random_state=args.inference.seed,
				memory_saving_mode=True,
				fit_mode = 'low_memory',
				inference_config={
					"SUBSAMPLE_SAMPLES": 1024,
					"FEATURE_SHIFT_METHOD": "shuffle",
					'FINGERPRINT_FEATURE': False,
					"CLASS_SHIFT_METHOD": "shuffle",
					"PREPROCESS_TRANSFORMS": default_classifier_preprocessor_configs(),
				}  # Needs to be set low so that not OOM on fitting intermediate nodes
			)
		else:
			model = TabPFNRegressor(
				n_estimators=8,
				ignore_pretraining_limits=True,
				categorical_features_indices=categorical_feature_indices,
				device="cpu",
				fit_mode = 'low_memory',
				random_state=args.inference.seed,
				memory_saving_mode=True,
				n_jobs=4,
				inference_config={
					"SUBSAMPLE_SAMPLES": 1024,
					"FEATURE_SHIFT_METHOD": "shuffle",
					'FINGERPRINT_FEATURE': False,
					"CLASS_SHIFT_METHOD": "shuffle",
					"PREPROCESS_TRANSFORMS": default_regressor_preprocessor_configs(),
				}  # Needs to be set low so that not OOM on fitting intermediate nodes
			)

		start_time = time.time()
		model.fit(X_train, y_train)
		training_time = time.time() - start_time
		# save_fitted_tabpfn_model(model, osp.join(result_folder, 'model.tabpfn_fit'))
		
		start_time = time.time()
		if is_classification:
			y_pred = model.predict_proba(X_test)
		else:
			y_pred = model.predict(X_test)
		running_time = time.time() - start_time

	elif method_name == "tabpfn-retrieval":
		from ehrpfn.models import get_tabpfn_knn_results
		start_time = time.time()
		all_predictions = get_tabpfn_knn_results(
			X_train, y_train, 
			X_test, 
			categorical_feature_indices, 
			is_classification, 
			retriever=retriever,
			n_ensembles=8,
			seed=args.inference.seed,
		)
		running_time = time.time() - start_time

	elif method_name in ['tabdpt', 'ehrpfn'] or method_name.startswith('ehrpfn'):
		from ehrpfn.models import EHRPFNClassifier, EHRPFNRegressor

		if is_classification:
			model = EHRPFNClassifier(
				pretrained_path=args.inference.pretrained_path,
				inf_batch_size=512,
				compile = False,
				categorical_feature_indices=categorical_feature_indices,
				retriever=retriever
			)
		else:
			model = EHRPFNRegressor(
				pretrained_path=args.inference.pretrained_path,
				inf_batch_size=512,
				categorical_feature_indices=categorical_feature_indices,
				retriever=retriever
			)

		model.fit(X_train, y_train)
		start_time = time.time()
		if is_classification:
			y_pred = model.predict(
				X_test, 
				n_ensembles=8,
				temperature=0.8, context_size=1024, 
			)
		else:
			y_pred = model.predict(
				X_test,
				context_size=1024, 
				n_ensembles=8,
			)
		running_time = time.time() - start_time


	elif method_name in ['beta', 'resnet', 'tabr', 'modernNCA', 'tabm', 'tabicl', 'tabtransformer', 'tabnet', 'excelformer', 'trompt']:

		from ehrpfn.models.talent_wrapper import TALENTWrapper
		from sklearn.model_selection import train_test_split
		X_train, X_val, y_train, y_val = train_test_split(
			X_train, y_train, 
			test_size=0.1, 
			random_state=args.inference.seed, 
			stratify=y_train if is_classification else None
		)

		model = TALENTWrapper(
			model_name=method_name,
			config_path=getattr(args, 'config_path', None),
			tune_config=getattr(args, 'tune_config', None),
			ckpt_path=args.inference.pretrained_path,
			save_dir=result_folder,
			categorical_feature_indices=categorical_feature_indices,
			valset=[X_val, y_val],
			is_classification=is_classification,
			seed=args.inference.seed
		)
		start_time = time.time()
		model.fit(X_train, y_train)
		training_time = time.time() - start_time

		start_time = time.time()
		y_pred = model.predict(X_test)
		running_time = time.time() - start_time

	else:
		raise ValueError(f"Method {method_name} is not supported. Please choose from ['tabpfn', 'rftabpfn', 'xgboost', 'catboost', 'lightgbm']")
	return y_pred, training_time, running_time