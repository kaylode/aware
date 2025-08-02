import shutil
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import polars as pl
import os
import os.path as osp
from safetensors.torch import save_file
from nosos import TensorizedColumns
from nosos.utilities.vocabulary import Vocabulary


def tensorize(in_fp: str | Path | list, out_fp: str | Path, vocab: Vocabulary=None):
    selected_columns = ["subject_id", "times", "tokens"]

    df = pl.scan_parquet(in_fp)
    for column in list(set(["hadm_id", "icustay_id", "dicom_id", "admission_id"] + TensorizedColumns.list())):
        if column in df.columns:
            selected_columns.append(column)

    df = (
        df
        .with_columns(
            tokens=pl.col("code").replace_strict(vocab.stoi, return_dtype=pl.Int64, default=None),
            times=pl.col("time").cast(pl.Int64),
        )
        .select(*selected_columns)
        # Filter out negative times, dates before 1970 are most likely errors
        .filter(pl.col("times") >= 0).drop_nulls(["subject_id", "times", "tokens"])
        .collect()
    )
    

    patient_id_col = "subject_id"
    patient_df = (
        df.set_sorted(patient_id_col)
        .group_by(patient_id_col, maintain_order=True)
        .agg(len=pl.len())
        .select(patient_id_col, offsets=pl.col("len").cum_sum().shift(1, fill_value=0))
    )
    tensors = {
        "subject_ids": df["subject_id"].to_torch(),
        "tokens": df["tokens"].to_torch(),
        "times": df["times"].to_torch(),
        "patient_ids": patient_df[patient_id_col].to_torch(),
        "patient_offsets": patient_df["offsets"].to_torch(),
    }

    # TODO: This is extremely memory-inefficient.
    for mimic_col in list(set(["hadm_id", "icustay_id", "dicom_id", "admission_id"] + TensorizedColumns.list())):
        if mimic_col in df.columns:
            tensors[mimic_col] = df[mimic_col].to_torch()

    save_file(tensors, Path(out_fp).with_suffix(".safetensors"))

def tensorize_with_labels(in_fp: str | Path | list, out_fp: str | Path, vocab: Vocabulary, label_folder: str, split:str=None):
    selected_columns = ["subject_id", "times", "tokens"]

    df = pl.scan_parquet(in_fp)
    for column in list(set(["hadm_id", "icustay_id", "dicom_id", "admission_id"] + TensorizedColumns.list())):
        if column in df.columns:
            selected_columns.append(column)

    df = (
        df
        .with_columns(
            tokens=pl.col("code").replace_strict(vocab.stoi, return_dtype=pl.Int64, default=None),
            times=pl.col("time").cast(pl.Int64),
        )
        .select(*selected_columns)
        # Filter out negative times, dates before 1970 are most likely errors
        .filter(pl.col("times") >= 0).drop_nulls(["subject_id", "times", "tokens"])
        .collect()
    )

    # Load the labels from the specified folder
    task_names = os.listdir(label_folder)
    for task_name in task_names:
        if split == "train":
            for split_name in ["train", "val"]:
                task_label_csv = osp.join(label_folder, task_name, f"{split_name}_prediction_labels.csv")
                if not osp.isfile(task_label_csv):
                    continue

                task_label_df = pl.read_csv(task_label_csv)

                save_path = Path(label_folder) / task_name / split_name 
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = (save_path / in_fp.name).with_suffix(".safetensors")
                tensorize_split(
                    task_label_df,
                    df,
                    task_name,
                    save_path=save_path,
                )
        else:
            task_label_csv = osp.join(label_folder, task_name, "prediction_labels.csv")
            if not osp.isfile(task_label_csv):
                continue
            save_path = Path(label_folder) / task_name
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = (save_path / in_fp.name).with_suffix(".safetensors")
            task_label_df = pl.read_csv(task_label_csv)
            tensorize_split(
                task_label_df,
                df,
                task_name,
                save_path=save_path,
            )


def tensorize_split(task_label_df, df, task_name, save_path):
    task_label_df = task_label_df.with_columns(
        pl.col("prediction_label").cast(pl.Int32)
    )
    task_label_df = task_label_df.filter(pl.col("prediction_label").is_not_null())
    task_label_df = task_label_df.select(["subject_id", "admission_id", "prediction_label"])

    # Filter the main DataFrame based on the task labels
    new_df = df.join(
        task_label_df,
        left_on=["subject_id", "admission_id"],
        right_on=["subject_id", "admission_id"],
        how="inner",
    )

    patient_id_col = "subject_id"
    patient_df = (
        new_df.set_sorted(patient_id_col)
        .group_by(patient_id_col, maintain_order=True)
        .agg(len=pl.len())
        .select(patient_id_col, offsets=pl.col("len").cum_sum().shift(1, fill_value=0))
    )
    
    tensors = {
        "subject_ids": new_df["subject_id"].to_torch(),
        "tokens": new_df["tokens"].to_torch(),
        "times": new_df["times"].to_torch(),
        "labels": new_df["prediction_label"].to_torch(),
        "patient_ids": patient_df[patient_id_col].to_torch(),
        "patient_offsets": patient_df["offsets"].to_torch(),
    }

    # TODO: This is extremely memory-inefficient.
    for mimic_col in list(set(["hadm_id", "icustay_id", "dicom_id", "admission_id"] + TensorizedColumns.list())):
        if mimic_col in new_df.columns:
            tensors[mimic_col] = new_df[mimic_col].to_torch()

    
    save_file(tensors, save_path)
    logger.info(f"Saved tensorized data for task {task_name} to {save_path}")



@hydra.main(version_base=None, config_path="./configs", config_name="mimiciv")
def main(cfg: DictConfig):
    logger.info(cfg)
    worker = 1
    logger.info(f"Running tensorization on '{worker}' workers...")
    
    out_fn = cfg.out_fn
    output_dir = Path(cfg.output_dir) /out_fn
    # Update the real output directory path
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(cfg.input_dir)
    folder_names = os.listdir(input_dir)
    for folder_name in folder_names:
        if 'FinalizeData' in folder_name:
            input_dir = input_dir / folder_name
            logger.info(f"Using input directory: {input_dir}")
            break


    logger.info(f"Tensorizing '{input_dir}'")

    in_fps = list(input_dir.glob("*.parquet"))
    logger.info(f"Found {len(in_fps)} patient splits in '{input_dir}'.")
    assert in_fps, "No parquet files found!"

    vocab = None
    if cfg.use_vocab:
        vocab = Vocabulary.from_path(cfg.vocab)
        if (quantile_fp := Path(cfg.vocab) / cfg.quantiles_fn).exists():
            logger.info(f"Copying quantile breaks from {quantile_fp} to {output_dir}")
            try:
                shutil.copy(quantile_fp, output_dir / cfg.quantiles_fn)
            except shutil.SameFileError:
                logger.warning(f"Quantile breaks file already exists at {output_dir / cfg.quantiles_fn}")

        if (intervals_fp := Path(cfg.vocab) / cfg.intervals_fn).exists():
            logger.info(f"Copying intervals from {intervals_fp} to {output_dir}")
            try:
                shutil.copy(intervals_fp, output_dir / cfg.intervals_fn)
            except shutil.SameFileError:
                logger.warning(f"Quantile breaks file already exists at {output_dir / cfg.intervals_fn}")
        else:
            logger.warning(f"Intervals file not found at {intervals_fp}")

    if cfg.tensorize or cfg.tensorize_with_labels:
        for in_fp in in_fps:
            if cfg.tensorize_with_labels:
                tensorize_with_labels(
                    in_fp, output_dir, 
                    vocab, label_folder=cfg.label_folder, 
                    split=out_fn, 
                )
            else:
                tensorize(in_fp, output_dir /in_fp.name, vocab)

if __name__ == "__main__":
    main()
