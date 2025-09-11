from theseus.base.models import MODEL_REGISTRY
from theseus.base.losses import LOSS_REGISTRY


# from .behrt import BertForMaskedLM
from .behrt import BertForMaskedLM, BEHRT
MODEL_REGISTRY.register(BertForMaskedLM)
MODEL_REGISTRY.register(BEHRT)

from .bigru import BiGRU
MODEL_REGISTRY.register(BiGRU)

from .ehrmamba import MambaPretrain, MambaFinetune
MODEL_REGISTRY.register(MambaPretrain)
MODEL_REGISTRY.register(MambaFinetune)

from .corebehrt import CorebehrtForPretraining, CorebehrtForFineTuning
MODEL_REGISTRY.register(CorebehrtForPretraining)
MODEL_REGISTRY.register(CorebehrtForFineTuning)