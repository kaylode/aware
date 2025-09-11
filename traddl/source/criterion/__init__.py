from theseus.base.losses import LOSS_REGISTRY

from .celoss import MLMCELoss
LOSS_REGISTRY.register(MLMCELoss)
