import sys
from omegaconf import DictConfig
sys.path.append(".")
from source.datasets import DATASET_REGISTRY
from source.models import MODEL_REGISTRY
from source.criterion import LOSS_REGISTRY
from source.metrics import METRIC_REGISTRY
from theseus.base.pipeline import BasePipeline
from theseus.base.utilities.getter import get_instance_recursively
from theseus.base.utilities.loggers import LoggerObserver
from theseus.base.utilities.seed import seed_everything


class LightningPipeline(BasePipeline):
    """docstring for MLTrainer."""

    def __init__(self, opt: DictConfig):
        super(LightningPipeline, self).__init__(opt)
        self.opt = opt
        self.seed = self.opt["global"].get("seed", 1702)
        seed_everything(self.seed)

    def init_registry(self):
        super().init_registry()
        self.dataset_registry = DATASET_REGISTRY
        self.model_registry = MODEL_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.metric_registry = METRIC_REGISTRY
    
    def init_model(self):
        vocab = self.train_dataset.vocab
        model = get_instance_recursively(
            self.opt["model"],
            registry=self.model_registry,
            vocab_size=len(vocab.stoi.keys()),
            num_classes=getattr(self.train_dataset, "num_classes", None),
            padding_idx = self.train_dataset.vocab.stoi['[PAD]'],
            cls_idx = self.train_dataset.vocab.stoi['[CLS]'],
            sep_idx = self.train_dataset.vocab.stoi['[SEP]'],
        )

        num_params = model.num_parameters()
        log_dict = [
            {
                "tag": "model_num_params",
                "value": num_params,
                "type": LoggerObserver.SCALAR,
                "kwargs": {
                    'step': 0
                },
            }
        ]

        self.logger.log(log_dict)

        return model
    
    def init_metrics(self):
        self.metrics = get_instance_recursively(
            self.opt["metrics"],
            registry=self.metric_registry,
            tokens_of_interest=getattr(self.train_dataset, "tokens_of_interest", None)
        )