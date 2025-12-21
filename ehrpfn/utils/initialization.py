import os
import os.path as osp
import wandb
import json
from omegaconf import OmegaConf
from typing import *
from ehrpfn.utils import FileLogger, LoggerObserver, seed_everything
LOGGER = LoggerObserver.getLogger("main")


def setup_inference(args, data):

	(
		data_size,
		categorical_feature_indices,
	) = (
		data['data_size'],
		data['categorical_feature_indices'],
	)

	data_name = args.data.data_name
	target_column = args.data.task if not args.data.data_name.startswith('openml_ehr') else data.get('data_name', None)

	result_folder = osp.join(args.inference.output_folder, args.data.data_name, target_column)
	os.makedirs(result_folder, exist_ok=True)

	model_name = args.inference.model
	study_name = f"{model_name}_{data_name}_{target_column}"
	tags = [f"METHOD:{model_name}", f"FEAT:{args.data.num_features}"]
	
	if not args.data.data_name.startswith('openml_ehr'):
		tags.append(f"TASK:{args.data.task}")
	else:
		tags.append(f"TASK:{target_column}")
	
	if args.data.selection_method is not None:
		tags.append(f"SELECTION:{args.data.selection_method}")
		study_name += f"_sel:{args.data.selection_method}"

	if args.data.get('remove_aggregated_time_series', False):
		tags.append(f"NOAGGTS")
		study_name += "_noaggts"

	if args.data.sampling.do_sampling:
		tags.append(f"SAMPLING:{args.data.sampling.sampling_method}")
		study_name += f"_sampl:{args.data.sampling.sampling_method}"


	if args.data.sampling.imbalance_ratio is not None:
		tags.append(f"IMBALANCE_RATIO:{args.data.sampling.imbalance_ratio}")
		study_name += f"_imbalance:{args.data.sampling.imbalance_ratio}"
	
	if model_name.startswith((('ehrpfn', 'knn', 'tabdpt'))) and args.retrieval.retrieval_method is not None:
		study_name += f"_retrieval:{args.retrieval.retrieval_method}"

		if args.retrieval.context_size is not None:
			study_name += f"_ctx:{args.retrieval.context_size}"
			tags.append(f"CTX:{args.retrieval.context_size}")

	result_folder = osp.join(result_folder, study_name)
	os.makedirs(result_folder, exist_ok=True)

	# Save config
	with open(osp.join(result_folder, 'config.yaml'), 'w') as f:
		OmegaConf.save(config=args, f=f, resolve=True)

	
	# Initialize the model
	if data.get('is_classification', None) is not None:
		is_classification = data['is_classification']
	else:
		if target_column in ['NEXT_ADMISSION_LOS', 'NEXT_LOS']:
			is_classification = False
		else:
			is_classification = True

	with open(osp.join(result_folder, 'metadata.json'), 'w') as f:
		json.dump({
			"categorical_feature_indices": categorical_feature_indices,
			"feature_names": data.get('feature_names', None),
			"num_classes": data.get('num_classes', None),
			'is_classification': is_classification,
			'data_name': data.get('data_name', None),
			'num_data_features': data.get('num_features', None),
			'class_distribution': data.get('class_distribution', None),
			'data_size': data_size,
		}, f, indent=4, default=str)

	
	if args.wandb.use_wandb:
		wandb_kwargs = {
			"name": f"{args.inference.model}" if args.inference.model not in ['tabdpt', 'tabpfn-retrieval', 'knn', 'knnpfn', 'knn-simple'] else f"{args.inference.model}_{args.retrieval.retrieval_method}",
			"entity": "kaylode",
			"project": "tabpfn-ehr",
			"reinit": True,
			"group": "inference",
			"job_type": "test",
			"tags": tags,
			"resume": "allow",
			"dir": osp.join(result_folder),
		}
		wandb.init(**wandb_kwargs)

	saved_config = OmegaConf.to_container(args, resolve=True)


	if args.data.data_name.startswith('openml_ehr'):
		saved_config['target_column'] = data.get('data_name', None)


	if args.wandb.use_wandb:
		wandb.run.config.update(saved_config)
		wandb.run.config.update({
			'study_name': study_name,
			'task_name': target_column,
			'is_classification': is_classification,
			'num_classes': data.get('num_classes', None),
			'dataset_name': data.get('data_name', None),
			'num_data_features': data.get('num_features', None),
			'data_size': data_size,
			'class_distribution': data.get('class_distribution', None)
		})
		
	return result_folder, study_name