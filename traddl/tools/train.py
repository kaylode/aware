import matplotlib as mpl
import torch._dynamo
torch._dynamo.config.suppress_errors = True
mpl.use("Agg")
from source.pipelines import LightningPipeline
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None)
def main(opts: DictConfig):
    pipeline = LightningPipeline(opts)
    pipeline.fit()

if __name__ == "__main__":
    main()