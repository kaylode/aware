import shutil
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import polars as pl
import os
import os.path as osp
from safetensors.torch import save_file
from preprocess.longitudinal.tokenization.constants import TensorizedColumns
from ehrpfn.utils.vocabulary import Vocabulary
from sklearn.model_selection import train_test_split


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

def tensorize_with_labels(in_fp: str | Path | list, out_fp: str | Path, vocab: Vocabulary, label_df: pl.DataFrame, save_path:str=None):
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
    tensorize_split(
        label_df,
        df,
        save_path=save_path,
    )



def tensorize_split(task_label_df, df, save_path):
    task_label_df = task_label_df.with_columns(
        pl.col("label").cast(pl.Int32)
    )
    task_label_df = task_label_df.filter(pl.col("label").is_not_null())
    task_label_df = task_label_df.select(["subject_id", "hadm_id", "stay_id", "label"])

    # Filter the main DataFrame based on the task labels
    new_df = df.join(
        task_label_df,
        left_on=["subject_id", "icustay_id", 'hadm_id'],
        right_on=["subject_id", "stay_id", "hadm_id"],
        how="inner",
    )
    if new_df.is_empty():
        return

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
        "labels": new_df["label"].to_torch(),
        "patient_ids": patient_df[patient_id_col].to_torch(),
        "patient_offsets": patient_df["offsets"].to_torch(),
    }

    # TODO: This is extremely memory-inefficient.
    for mimic_col in list(set(["hadm_id", "icustay_id", "dicom_id", "admission_id"] + TensorizedColumns.list())):
        if mimic_col in new_df.columns:
            tensors[mimic_col] = new_df[mimic_col].to_torch()

    save_file(tensors, save_path)



def main_mimic():
    worker = 1
    logger.info(f"Running tensorization on '{worker}' workers...")
    
    # Update the real output directory path

    input_dirs = []
    for split in ['train', 'test']:
        input_dir = Path("/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv") / split
        folder_names = os.listdir(input_dir)
        for folder_name in folder_names:
            if 'FinalizeData' in folder_name:
                input_dir = input_dir / folder_name
                in_fps = list(input_dir.glob("*.parquet"))
                input_dirs.extend(in_fps)
                logger.info(f"Using input directory: {input_dir}")
                break

    in_fps = input_dirs
    logger.info(f"Tensorizing '{len(input_dirs)} files'")

    vocab = Vocabulary.from_path("/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv/train")
    all_dfs = []
    for in_fp in in_fps:
        selected_columns = ["subject_id", "times", "tokens"]
        df = pl.scan_parquet(in_fp)
        for column in list(set(["hadm_id", "icustay_id"] + TensorizedColumns.list())):
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
        all_dfs.append(df)

    df = pl.concat(all_dfs)
    df = df.sort(["subject_id", "admission_id"])
    df = df.with_columns(
        pl.col("icustay_id").forward_fill().backward_fill().over(["subject_id", "admission_id"]),
        pl.col("hadm_id").forward_fill().backward_fill().over(["subject_id", "admission_id"]),
    )

    # Sort

    # Split train val
    label_folder = "/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv"
    task_names = [i for i in os.listdir(label_folder) if '.' not in i]

    for task_name in task_names:
        train_label_csv = osp.join(label_folder, task_name, 'split', "train_labels.csv")
        labels = pl.read_csv(train_label_csv).unique()

        # Group by subject_id, aggregate by label with max, then train_test_split based on labels
        _labels = labels.group_by("subject_id").agg(pl.col("label").max().alias("label"))

        _train_labels, _val_labels = train_test_split(_labels, stratify=_labels["label"], test_size=0.2)
        train_labels = labels.join(_train_labels.select("subject_id"), on="subject_id", how="inner")
        val_labels = labels.join(_val_labels.select("subject_id"), on="subject_id", how="inner")

        test_label_csv = osp.join(label_folder, task_name, 'split', "test_labels.csv")
        test_labels = pl.read_csv(test_label_csv).unique()

        out_dir = "/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv/Tensors"
        save_path = Path(out_dir) / task_name
        save_path.mkdir(parents=True, exist_ok=True)
        for split, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:

            task_label_df = labels.filter(pl.col("label").is_not_null())
            task_label_df = task_label_df.select(["subject_id", "hadm_id", "stay_id", "label"])

            # Filter the main DataFrame based on the task labels
            new_df = df.join(
                task_label_df,
                left_on=["subject_id", "icustay_id", 'hadm_id'],
                right_on=["subject_id", "stay_id", "hadm_id"],
                how="inner",
            )
            if new_df.is_empty():
                return

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
                "labels": new_df["label"].to_torch(),
                "patient_ids": patient_df[patient_id_col].to_torch(),
                "patient_offsets": patient_df["offsets"].to_torch(),
            }

            # TODO: This is extremely memory-inefficient.
            for mimic_col in list(set(["hadm_id", "icustay_id", "dicom_id", "admission_id"] + TensorizedColumns.list())):
                if mimic_col in new_df.columns:
                    tensors[mimic_col] = new_df[mimic_col].to_torch()

            save_file(tensors, (Path(save_path)/split).with_suffix(".safetensors"))



def main_eicu():
    worker = 1
    logger.info(f"Running tensorization on '{worker}' workers...")
    
    # Update the real output directory path

    input_dirs = []
    for split in ['train', 'test']:
        input_dir = Path("/home/mpham/workspace/source/ehrpfn/data/processed_meds/eicu") / split
        folder_names = os.listdir(input_dir)
        for folder_name in folder_names:
            if 'FinalizeData' in folder_name:
                input_dir = input_dir / folder_name
                in_fps = list(input_dir.glob("*.parquet"))
                input_dirs.extend(in_fps)
                logger.info(f"Using input directory: {input_dir}")
                break

    in_fps = input_dirs
    logger.info(f"Tensorizing '{len(input_dirs)} files'")

    vocab = Vocabulary.from_path("/home/mpham/workspace/source/ehrpfn/data/processed_meds/eicu/train")
    all_dfs = []
    for in_fp in in_fps:
        selected_columns = ["subject_id", "times", "tokens"]
        df = pl.scan_parquet(in_fp)
        for column in list(set(["hadm_id", "icustay_id"] + TensorizedColumns.list())):
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
        all_dfs.append(df)

    df = pl.concat(all_dfs)
    df = df.sort(["subject_id", "admission_id"])
    df = df.with_columns(
        pl.col("hadm_id").forward_fill().backward_fill().over(["subject_id", "admission_id"]),
    )

    # Sort

    # Split train val
    label_folder = "/home/mpham/workspace/source/ehrpfn/data/processed/eicu"
    task_names = [i for i in os.listdir(label_folder) if '.' not in i]

    all_labels = pl.read_csv(osp.join(label_folder, "all_labels.csv"))

    for task_name in task_names:
        train_label_csv = osp.join(label_folder, task_name, "train_admission_ids.txt")
        train_labels = pl.read_csv(train_label_csv, has_header=False).unique()
        train_labels = all_labels.filter(pl.col("ICU_ID").is_in(train_labels))
        _task_name = task_name.split('task:')[-1]
        train_labels = train_labels.select([
            "ICU_ID", "ADMISSION_ID", "PATIENT_ID", "HOSPITAL_ID", f"{_task_name}"
        ])
        train_labels = train_labels.rename({f"{_task_name}": "label"})

        val_label_csv = osp.join(label_folder, task_name, "val_admission_ids.txt")
        val_labels = pl.read_csv(val_label_csv, has_header=False).unique()
        val_labels = all_labels.filter(pl.col("ICU_ID").is_in(val_labels))
        val_labels = val_labels.select([
            "ICU_ID", "ADMISSION_ID", "PATIENT_ID", "HOSPITAL_ID", f"{_task_name}"
        ])
        val_labels = val_labels.rename({f"{_task_name}": "label"})

        test_label_csv = osp.join(label_folder, task_name, "test_admission_ids.txt")
        test_labels = pl.read_csv(test_label_csv, has_header=False).unique()
        test_labels = all_labels.filter(pl.col("ICU_ID").is_in(test_labels))
        test_labels = test_labels.select([
            "ICU_ID", "ADMISSION_ID", "PATIENT_ID", "HOSPITAL_ID", f"{_task_name}"
        ])
        test_labels = test_labels.rename({f"{_task_name}": "label"})

        # Group by subject_id, aggregate by label with max, then train_test_split based on labels
        # _labels = labels.group_by("subject_id").agg(pl.col("label").max().alias("label"))
        out_dir = "/home/mpham/workspace/source/ehrpfn/data/processed_meds/eicu/Tensors"
        save_path = Path(out_dir) / task_name
        save_path.mkdir(parents=True, exist_ok=True)
        for split, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:

            # Filter the main DataFrame based on the task labels
            new_df = df.join(
                labels,
                left_on=["subject_id",  'hadm_id'],
                right_on=["ADMISSION_ID", "ICU_ID"],
                how="inner",
            )
            if new_df.is_empty():
                return

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
                "labels": new_df["label"].to_torch(),
                "patient_ids": patient_df[patient_id_col].to_torch(),
                "patient_offsets": patient_df["offsets"].to_torch(),
            }

            # TODO: This is extremely memory-inefficient.
            for mimic_col in list(set(["hadm_id",  "admission_id"] + TensorizedColumns.list())):
                if mimic_col in new_df.columns:
                    tensors[mimic_col] = new_df[mimic_col].to_torch()

            save_file(tensors, (Path(save_path)/split).with_suffix(".safetensors"))


if __name__ == "__main__":
    main_eicu()
    # main_mimic()
