import os
import os.path as osp
import json
import hydra
import torch as th
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import wandb
from source.models import MODEL_REGISTRY
from source.datasets import PatientOutcomeDataset

from theseus.base.utilities.getter import get_instance_recursively
from theseus.base.utilities.cuda import move_to
from theseus.base.datasets.dataloader import DataLoaderWithCollator
from theseus.base.utilities.loggers import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

import json
import os.path as osp
import numpy as np
from ehrpfn.utils.metrics import run_binary_classification_metrics


@hydra.main(version_base=None)
def main(config: DictConfig):
    os.makedirs(config['global'].save_dir, exist_ok=True)

    dataset = PatientOutcomeDataset(
        **config.data.dataset.test.args,
        split='test',
    )

    LOGGER.text(f"Dataset size: {len(dataset)}", level=LoggerObserver.INFO)
    LOGGER.text(f"Vocabulary size: {dataset.vocab_size}", level=LoggerObserver.INFO)
    LOGGER.text(f"Number of classes: {dataset.num_classes}", level=LoggerObserver.INFO)

    dataloader = DataLoaderWithCollator(
        dataset,
        collate_fn=dataset.collate_fn,
        **config.data.dataloader.val.args
    )

    vocab = dataset.vocab
    model = get_instance_recursively(
        config["model"],
        registry=MODEL_REGISTRY,
        vocab_size=len(vocab.stoi.keys()),
        num_classes=getattr(dataset, "num_classes", None),
        padding_idx = dataset.vocab.stoi['[PAD]'],
        cls_idx = dataset.vocab.stoi['[CLS]'],
        sep_idx = dataset.vocab.stoi['[SEP]'],
    )


    ckpt_dict = th.load(
        config['global'].resume,
        map_location="cpu", 
        mmap=True, 
        weights_only=False
    )

    class ModelWrapper(th.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward_batch(self, batch):
            # batch = move_to(batch, th.device('cuda'))
            # batch = {k: v.to(th.device('cuda')) for k, v in batch.items()}
            return self.model.forward_batch(batch)

        def forward(self, batch):
            return self.model(batch)

    model = ModelWrapper(model)
    model_state_dict = ckpt_dict['state_dict']
    model.load_state_dict(model_state_dict, strict=True)
    model.to('cuda')

    model.eval()
    LOGGER.text(f"Model loaded", level=LoggerObserver.INFO)

    wandb_kwargs = {
        "name": f"{config['global']['exp_name']}",
        "entity": "kaylode",
        "project": "nosos-meds",
        "reinit": True,
        "group": "mimiciv-hcai-test",
        "job_type": "test",
        "tags": [f"{config['global']['exp_name']}", 'icu', "mimiciv-hcai", f"task:{os.environ.get('TASK_NAME')}"],
        "resume": "allow",
        "dir": config['global'].save_dir,
    }
    wandb.init(**wandb_kwargs)
    wandb.run.config.update(OmegaConf.to_container(config, resolve=True))

    pbar = tqdm(total=len(dataset))
    probs = []
    targets = []
    patient_ids = []
    hadm_ids = []
    for i, batch in enumerate(dataloader):
        batch = move_to(batch, th.device('cuda'))
        target = batch['targets']

        with th.no_grad():
            outputs = model.forward_batch(batch)['outputs']
            outputs = outputs.softmax(dim=-1)

        probs.append(outputs.cpu().numpy())
        targets.append(target.cpu().numpy())
        patient_ids.extend(batch['inputs']['patient_ids'].cpu().numpy().tolist())
        hadm_ids.extend(batch['inputs']['hadm_ids'].cpu().numpy().tolist())
        
        pbar.update(len(target))
        pbar.set_description(f"Evaluating {i}/{len(dataloader)}")
    pbar.close()
    
    LOGGER.text(f"Start evaluating", level=LoggerObserver.INFO)
    probs = np.concatenate(probs, axis=0)
    targets = np.concatenate(targets, axis=0)

    import pickle
    with open(osp.join(config['global'].save_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump({
            'patient_ids': patient_ids,
            'hadm_ids': hadm_ids,
            'y_prob': probs,
            'y_test': targets
        }, f)

    results = run_binary_classification_metrics(probs, targets)
    wandb.run.log(results)

    LOGGER.text(f"Evaluation finished", level=LoggerObserver.INFO)
    LOGGER.text(json.dumps(results, indent=4), level=LoggerObserver.SUCCESS)

    # Save results
    results_path = osp.join(config['global'].save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    LOGGER.text(f"Results saved to {config['global'].save_dir}", level=LoggerObserver.INFO)
    wandb.finish()

if __name__ == "__main__":
    main()