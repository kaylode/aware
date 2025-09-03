from .ntp import NextTokenPredictionMetric
from .mlm import MLMAccuracyMetric, AccuracyAtK
from theseus.base.metrics import METRIC_REGISTRY

METRIC_REGISTRY.register(NextTokenPredictionMetric)
METRIC_REGISTRY.register(MLMAccuracyMetric)
METRIC_REGISTRY.register(AccuracyAtK)
