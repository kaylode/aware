from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY
from theseus.base.augmentations import TRANSFORM_REGISTRY


from .mlm import MaskedLanguageModelDataset

DATASET_REGISTRY.register(MaskedLanguageModelDataset)

from .outcome import PatientOutcomeDataset
DATASET_REGISTRY.register(PatientOutcomeDataset)