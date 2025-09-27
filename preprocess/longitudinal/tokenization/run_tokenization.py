import time
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from preprocess.longitudinal.tokenization.run_stage import run_stage
from preprocess.longitudinal.tokenization.utils import load_function

OmegaConf.register_new_resolver("default_if_null", lambda a, b: b if a is None else a)
OmegaConf.register_new_resolver("is_not_null", lambda v: v is not None)
OmegaConf.register_new_resolver("is_null", lambda v: v is None)


@hydra.main(version_base=None, config_path="./configs", config_name="mimiciv")
def main(cfg: DictConfig):
    if not (input_dir := Path(cfg.input_dir)).is_dir():
        raise FileNotFoundError(f"Data directory '{input_dir}' not found.")
    logger.info(cfg)
    worker = int(cfg.worker)
    logger.info(f"Running tokenization on '{worker}' workers...")
    
    dataset = cfg.dataset.name
    out_fn = f"{dataset}_{input_dir.name}" if cfg.out_fn is None else cfg.out_fn
    output_dir = Path(cfg.output_dir) / out_fn
    # Update the real output directory path
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tokenizing '{input_dir}' using the {dataset.upper()} preprocessing pipeline.")

    in_fps = list(input_dir.glob("*.parquet"))
    logger.info(f"Found {len(in_fps)} patient splits in '{input_dir}'.")
    assert in_fps, "No parquet files found!"

    i = 1
    for stage_cfg in cfg.dataset.stages:
        if stage_cfg.get("skip", False):
            logger.info(f"Skipping stage {stage_cfg.name}")
            continue

        out_dir = output_dir / f"{i:02}_{stage_cfg.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fps = [out_dir / fp.name for fp in in_fps]

        transform_fns = []
        if "transforms" in stage_cfg:
            for transform_name in stage_cfg.transforms:
                func = load_function(transform_name, f"preprocess.longitudinal.tokenization.{dataset}")
                transform_fns.append(func)
        else:
            try:
                func = load_function(stage_cfg.name, "preprocess.longitudinal.tokenization.common")
            except AttributeError:
                func = load_function(stage_cfg.name, f"preprocess.longitudinal.tokenization.{dataset}")
            transform_fns.append(func)

        logger.info(f"Running stage='{stage_cfg}'")
        run_stage(in_fps, out_fps, *transform_fns, worker=worker, **stage_cfg)

        # Make workers wait for lock file deletion to avoid moving on prematurely
        while any(out_fps[0].parent.glob("*cache/locks/*.json")):
            logger.info(f"Waiting...'")
            time.sleep(2)

        if "agg_to" not in stage_cfg:
            in_fps = out_fps

        i += 1

    

if __name__ == "__main__":
    main()
