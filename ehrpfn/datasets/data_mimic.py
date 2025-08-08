from typing import *
import json
import os
import os.path as osp
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from ehrpfn.utils.logger import LoggerObserver
from ehrpfn.datasets.label_encoder import LabelEncode
from pathlib import Path
from queue import Queue, Empty
from multiprocessing import Manager, Process, set_start_method, cpu_count
from ehrpfn.trajectories.cohort import DISEASE_GROUP_TO_ICDS

RARE_DISEASE_ICDS = [i for key, i in DISEASE_GROUP_TO_ICDS.items() if key in ['HCAI']]
RARE_DISEASE_ICDS = [item for sublist in RARE_DISEASE_ICDS for item in sublist]
RARE_DISEASE_ICDS.extend([i[:5] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:4] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:3] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS = list(set(RARE_DISEASE_ICDS))

LOGGER = LoggerObserver.getLogger("main")


CATEGORICAL_COLUMNS = [
    "ADMISSION_TYPE",
    "DISCHARGE_LOCATION",
    "ICU_ADMISSION",
    "ICU_DISCHARGE",
    "HOSPITAL_DISCHARGE",
    "MARITAL",
    "RACE",
    "INSURANCE",
    "GENDER",
    "NEXT_ADMISSION_WARD"
]
CODE_COLUMNS = [
    "ICD",
    "DRG",
    "ATC",
    "PROCEDURES",
    "TRANSFER"
]
NUMERICAL_COLUMNS = [
    "AGE",
    'LOS',
    "LAB",
    "VITAL",
    "INFUSION"
]

ALL = CATEGORICAL_COLUMNS + CODE_COLUMNS + NUMERICAL_COLUMNS

def convert_to_pivot_table(subject_df, columns:List[str], subject_id, frequency_code_set, numeric_code_set, text_code_set):
    """
    Convert to instances table
    """
    num_admissions = subject_df.select(pl.col("admission_id")).unique().to_numpy().flatten().tolist()
    final_results = {}
    
    for i in num_admissions:
        results = {col: None for col in columns}
        results['admission_id'] = i
        results['subject_id'] = subject_id
        admission_subject_df = subject_df.filter(pl.col("admission_id") == i)

        # drop row if code not in columns
        admission_subject_df = admission_subject_df.filter(
            pl.col("code").is_in(columns)
        )

        grouped = admission_subject_df.group_by("code").agg(
            pl.len().alias("count"),
            pl.first("numeric_value").alias("numeric_value"),
            pl.first("text_value").alias("text_value"),
        )

        grouped = grouped.with_columns(
            pl.col("count").fill_null(0).cast(pl.Int16),
            pl.col("numeric_value").fill_null(-100).cast(pl.Float32),
            pl.col("text_value") .fill_null("None").cast(pl.String)
        )

        for code, count, numeric_val, text_val in zip(
            grouped['code'], 
            grouped['count'],
            grouped['numeric_value'],
            grouped['text_value'],
        ):
            if code in frequency_code_set:
                results[code] = (str(count))
            elif code in numeric_code_set:
                results[code] = (str(numeric_val))
            elif code in text_code_set:
                results[code] = (str(text_val))
            else:
                results[code] = (str(count))

        for col in columns:
            if results[col] is None:
                if col in frequency_code_set:
                    results[col] = "0"
                elif col in numeric_code_set:
                    results[col] = "-100"
                else:
                    results[col] = "None"


        for key in results.keys():
            if key not in final_results:
                final_results[key] = []
            final_results[key].append(results[key])
    
    #check if length of each column is the same
    for key in final_results.keys():
        if len(final_results[key]) != len(final_results['subject_id']):
            LOGGER.text(f"Length of {key} is not the same as subject_id", level=LoggerObserver.ERROR)
            return None

    return final_results


def feat_engineering(df: pl.DataFrame) -> pl.DataFrame:
    """
    Feature engineering
    Cumsum specific columns based on subject_id and admission_id
    """
    LOGGER.text("Feature engineering...", level=LoggerObserver.INFO)

    # Sort the DataFrame by subject_id and admission_id
    df = df.sort(["subject_id", "admission_id"])
    # Shift codes back 1 admission. Bceause: we want to use previous codes to predict current admission diseases
    COUNT_BASED_COLUMNS = [
        col for i in CODE_COLUMNS for col in df.columns if col.startswith(i)
    ]
    code_df = df.select(COUNT_BASED_COLUMNS + ["subject_id", "admission_id"])
    for col in tqdm(COUNT_BASED_COLUMNS):
        code_df = code_df.with_columns(
            pl.col(col).shift(1).over(["subject_id"])
        )
        code_df = code_df.with_columns(
            pl.col(col).fill_null(0)
        )
        code_df = code_df.with_columns(
            pl.col(col).cum_sum().over(["subject_id"])
        )
    code_df = code_df.drop_nulls()

    df = df.select(
        [col for col in df.columns if col not in COUNT_BASED_COLUMNS]
    )

    df = df.join(
        code_df,
        on=["subject_id", "admission_id"],
        how="inner"
    )

    return df


def encoding_table(df: pl.DataFrame, save_encoder_mapping:str=None, load_encoder_mapping:str=None) -> pl.DataFrame:
    """
    Convert to instances table
    ICD codes, DRG codes, and MEDICATION codes to int64
    HOSPITAL_ADMISSION, HOSPITAL_DISCHARGE, and other codes to categorical int64

    """
    LOGGER.text("Encoding categorical columns", level=LoggerObserver.INFO)
    categorical_columns = [col for i in CATEGORICAL_COLUMNS for col in df.columns if col.startswith(i)]
    df[categorical_columns] = df[categorical_columns].cast(pl.String).fill_null("None")
    if save_encoder_mapping is not None:
        label_encoder = LabelEncode(
            column_names=categorical_columns,
            encoder_type = "le",
            engine="polars"
        )
        df = label_encoder.encode_corpus(df)
        label_encoder.save_pickle(save_encoder_mapping)
        label_encoder.save_json(save_encoder_mapping.replace(".pkl", ".json"))
    else:
        label_encoder = LabelEncode.from_pickle(
            load_encoder_mapping
        )
        df = label_encoder.encode_query(df)


    df[categorical_columns] = df[categorical_columns].cast(pl.Int8).fill_null(-100)

    LOGGER.text("Encoding numerical columns", level=LoggerObserver.INFO)
    numerical_columns = [col for i in NUMERICAL_COLUMNS for col in df.columns if col.startswith(i)]

    for col in numerical_columns:
        if col.startswith("AGE"):
            df = df.with_columns(
                pl.col(col).cast(pl.Float32).fill_null(strategy="forward").alias(col)
            )
    df[numerical_columns] = df[numerical_columns].cast(pl.Float32).fill_null(-100)

    LOGGER.text("Encoding code columns", level=LoggerObserver.INFO)
    code_columns = [col for i in CODE_COLUMNS for col in df.columns if col.startswith(i)]
    df[code_columns] = df[code_columns].cast(pl.Int16).fill_null(0)

    LOGGER.text("Casting id to integers", level=LoggerObserver.INFO)
    df = df.with_columns(
        pl.col("admission_id").cast(pl.Int32),
        pl.col("subject_id").cast(pl.Int32),
    )
    LOGGER.text("Finish encoding columns", level=LoggerObserver.INFO)

    return df

def get_unique_codes(df: pl.DataFrame, save_path:str) -> List[str]:
    unique_codes = df.select("code").unique().to_numpy().flatten()
    unique_codes = sorted(unique_codes)

    filter_unique_codes = []
    for code in unique_codes:
        prefix = code.split("//")[0]
        if prefix in ALL:
            filter_unique_codes.append(code)
    unique_codes = filter_unique_codes

    prefixes = set([code.split("//")[0] for code in unique_codes])

    for prefix in prefixes:
        if prefix not in ALL:
            LOGGER.text(f"Prefix {prefix} not in ALL", level=LoggerObserver.ERROR)
            raise ValueError(f"Prefix {prefix} not in ALL")

    save_path = osp.join(save_path, "unique_codes.txt")
    if not osp.isfile(save_path):
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            for code in unique_codes:
                f.write(f"{code.strip()}\n")
    else:
        with open(save_path, "r") as f:
            unique_codes = f.readlines()
            unique_codes = [code.strip() for code in unique_codes]

    return unique_codes

def numpify_dataframe(df: pl.DataFrame, columns:List[str], save_path:str, chunk_size:int, target_column:str=None) -> np.ndarray:
    subject_ids = sorted(df.select("subject_id").unique().to_numpy().flatten())
    results = []
    track_id = 0

    ## Check done 
    if len(os.listdir(save_path)) > 0:
        no_cache_runs = len(os.listdir(save_path))
        LOGGER.text(f"Loading data from {save_path}", level=LoggerObserver.INFO)
        cache_df = pl.scan_parquet(save_path).collect()
        cache_ids = cache_df.select("subject_id").unique().to_numpy().flatten()
        LOGGER.text(f"Continue from [{len(cache_ids)}/{len(subject_ids)}]", level=LoggerObserver.INFO)
        subject_ids = [subject_id for subject_id in subject_ids if subject_id not in cache_ids]
        track_id = no_cache_runs

        if len(subject_ids) == 0:
            LOGGER.text(f"All subjects have been processed", level=LoggerObserver.INFO)
            return cache_df
    LOGGER.text("Converting to parquet data...", level=LoggerObserver.INFO)

    FREQ_CODES_SET = set([c for c in columns if any(c.startswith(i) for i in CODE_COLUMNS)])

    # Remove HCAI codes from columns if target_column is one of the rare diseases
    if target_column in ['HCAI', 'SSI', 'SEPSIS', 'BSI', 'VAP', 'UTI']:
        columns = [col for col in FREQ_CODES_SET if col.split('//')[-1] not in RARE_DISEASE_ICDS]

    NUM_CODES_SET = set([c for c in columns if any(c.startswith(i) for i in NUMERICAL_COLUMNS)])
    TEXT_CODES_SET = set([c for c in columns if any(c.startswith(i) for i in CATEGORICAL_COLUMNS)])

    pbar = tqdm(total=len(subject_ids), position=0, desc="Processing subjects",  leave=True)
    for i, subject_id in enumerate(subject_ids):
        pbar.set_description(f"Processing chunk {track_id}-th...")
        subject_df = df.filter(pl.col("subject_id") == subject_id)
        x = convert_to_pivot_table(
            subject_df, 
            columns=columns, 
            subject_id=subject_id,
            frequency_code_set=FREQ_CODES_SET,
            numeric_code_set=NUM_CODES_SET,
            text_code_set=TEXT_CODES_SET
        )
        results.append(x)

        if len(results) >= chunk_size or i == len(subject_ids) - 1:
            pbar.set_description(f"Saving chunk {track_id}-th...")
            final_results = {}
            for key in results[0].keys():
                if key not in final_results.keys():
                    final_results[key] = []
            for result in results:
                for key in result.keys():
                    final_results[key].extend(result[key])
            write_data_to_parquet(
                save_path, final_results, 
                offset=track_id,
            )
            del results
            results = []
            track_id+=1

        pbar.update(1)
    pbar.close()

    if len(results) > 0:
        final_results = {}
        for key in results[0].keys():
            if key not in final_results.keys():
                final_results[key] = []
        for result in results:
            for key in result.keys():
                final_results[key].extend(result[key])
        write_data_to_parquet(
            save_path, final_results, 
            offset=track_id,
        )
    LOGGER.text(f"Save to {save_path}", level=LoggerObserver.SUCCESS)

    df = pl.scan_parquet(save_path).collect()
    return df

def _process_subject(progress_queue: Queue, wid, subject_ids: List[int], df: pl.DataFrame, columns: List[str]) -> Dict[str, List[float]]:

    FREQ_CODES_SET = set([c for c in columns if any(c.startswith(i) for i in CODE_COLUMNS)])
    NUM_CODES_SET = set([c for c in columns if any(c.startswith(i) for i in NUMERICAL_COLUMNS)])
    TEXT_CODES_SET = set([c for c in columns if any(c.startswith(i) for i in CATEGORICAL_COLUMNS)])

    for subject_id in subject_ids:
        subject_df = df.filter(pl.col("subject_id") == subject_id)
        result = convert_to_pivot_table(
            subject_df, columns=columns, subject_id=subject_id,
            frequency_code_set=FREQ_CODES_SET, 
            numeric_code_set=NUM_CODES_SET, 
            text_code_set=TEXT_CODES_SET
        )
        progress_queue.put(result)


def write_data_to_parquet(
    output_dir: str, results: list[dict], offset: int | None = None
):
    os.makedirs(output_dir, exist_ok=True)
    out_fp = Path(output_dir) / f"data_{offset}"

    pl.from_dict(results, strict=False).write_parquet(
        out_fp.with_suffix(".parquet"), use_pyarrow=True
    )


def numpify_dataframe_parallel(df: pl.DataFrame, columns: List[str], save_path: str, num_workers:int = 8, timeout:int=180, chunk_size:int=32000) -> np.ndarray:
    
    subject_ids = sorted(df.select("subject_id").unique().to_numpy().flatten())
    track_id = 0    

    ## Check done 
    if len(os.listdir(save_path)) > 0:
        no_cache_runs = len(os.listdir(save_path))
        LOGGER.text(f"Loading data from {save_path}", level=LoggerObserver.INFO)
        cache_df = pl.scan_parquet(save_path).collect()
        cache_ids = cache_df.select("subject_id").unique().to_numpy().flatten()
        LOGGER.text(f"Continue from [{len(cache_ids)}/{len(subject_ids)}]", level=LoggerObserver.INFO)
        subject_ids = [subject_id for subject_id in subject_ids if subject_id not in cache_ids]
        track_id = no_cache_runs

    num_workers = min(num_workers, cpu_count())
    temp =  np.array_split(np.array(subject_ids), num_workers)
    ind_chunks = [l.tolist() for l in temp]
    worker_ids = list(range(num_workers))
    df_chunks = [df.filter(pl.col("subject_id").is_in(subject_ids)) for subject_ids in ind_chunks]
    LOGGER.text(f"Parallelizing numpify_dataframe across {num_workers} CPUs...", level=LoggerObserver.INFO)
    results = []
    
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    with Manager() as mgr:
        progress_queue = mgr.Queue()
        processes = []
        for i in range(num_workers):
            proc = Process(
                target=_process_subject,
                args=(
                    progress_queue,
                    worker_ids[i],
                    ind_chunks[i],
                    df_chunks[i],
                    columns,
                ),
                name=f"Process_{i}",
            )
            processes.append(proc)

        for p in processes:
            p.start()
        try:
            progress_bar = tqdm(total=len(subject_ids), desc="Pivoting table...")
            for i in range(len(subject_ids)):
                progress_bar.set_description(f"Pivoting table {track_id}...")
                results.append(progress_queue.get(timeout=timeout))
                progress_bar.update(1)


                if len(results) >= chunk_size or i == len(subject_ids) - 1:
                    progress_bar.set_description(f"Saving chunk {track_id}-th...")
                    final_results = {}
                    for key in results[0].keys():
                        if key not in final_results.keys():
                            final_results[key] = []
                    for result in results:
                        for key in result.keys():
                            final_results[key].extend(result[key])
                    write_data_to_parquet(
                        save_path, final_results, 
                        offset=track_id,
                    )
                    del results
                    results = []
                    track_id+=1
            progress_bar.close()
        except Empty:
            LOGGER.text("Queue is empty, terminating processes...", LoggerObserver.WARN)
            for p in processes:
                if p.is_alive():
                    p.terminate()
        for p in processes:
            p.join()


    if len(results) > 0:
        final_results = {}
        for key in results[0].keys():
            if key not in final_results.keys():
                final_results[key] = []
        for result in results:
            for key in result.keys():
                final_results[key].extend(result[key])
        write_data_to_parquet(
            save_path, final_results, 
            offset=track_id,
        )

    LOGGER.text(f"Save to {save_path}", level=LoggerObserver.SUCCESS)
    df = pl.scan_parquet(save_path).collect()
    return df

def make_inputs_targets(df: pl.DataFrame, target_column:str, feature_selection_path:str=None, selection_method:str=None, selected_indices:List[int]=None) -> pl.DataFrame:
    """
    Given a Dataframe with schema like this
    ```
    subject_id | hadm_id | icustay_id | time | code | numeric_value | text_value
    ---------- | ------- | --------- | --------- | ---- | ----- | -------
    1234567890 | 1234567 | 1234567   | 2189-10-17 15:00:00 | HOSPITAL_DISCHARGE | null | "HOME"

    ```
    Preprocessing steps:
    1. Replace "HOSPITAL_ADMISSION" with
    """
    assert target_column in df.columns

    if target_column in ['NEXT_ADMISSION_LOS']:
        # Drop negative values
        df = df.filter(pl.col(target_column) >= 0)

    x, y = df.drop([target_column]), df[target_column]

    # Filter features
    try:
        ids_ = x[['subject_id', "admission_id"]].to_numpy()
    except:
        ids_ = None
    
    x = x.drop(["subject_id", "hadm_id", "icustay_id", "admission_id"], strict=False)

    if target_column in ['DEATH']:
        x = x.drop(['DISCHARGE_LOCATION'], strict=False)
    

    # Load top features
    if selection_method is not None:
        if selected_indices is None:
            if selection_method.startswith("random"):
                size = int(selection_method.split("_")[1]) if "_" in selection_method else 100
                # Random selection of features
                selected_indices = np.random.choice(x.shape[1], size=size, replace=False)
            else:
                assert feature_selection_path is not None, "feature_selection_path must be provided for feature selection"
                tmp = selection_method.split("_")
                if len(tmp) > 1:
                    selection_method, size = tmp[0], int(tmp[1])
                else:
                    size = 100
                feature_selection_path = osp.join(feature_selection_path, f"{selection_method}_feature_importance.json")
                with open(feature_selection_path, 'r') as f:
                    top_features = json.load(f)
                top_features = {k:v for k,v in sorted(top_features.items(), key=lambda item: item[1], reverse=False)}
                top_features = list(top_features.keys())[:size]
                selected_indices = [x.columns.index(col) for col in top_features if col in x.columns]
        x = x[:, selected_indices]

    feature_names = x.columns
    x = x.to_numpy()
    y = y.to_numpy()

    return (x, y), ids_, feature_names, selected_indices

def collect_parquet(
        input_path:str=None,
        output_path:str=None,
        split:str=None,
        patient_ids_filter=None,
        load_encoder_mapping:str=None,
        target_column:str=None
):
    os.makedirs(output_path, exist_ok=True)
    
    if input_path is not None:
        # Read in and process the DataFrame
        df = pl.scan_parquet(input_path).collect()
        if patient_ids_filter is not None:
            df = df.filter(
                pl.col("subject_id").is_in(patient_ids_filter)
            )

        df = df.select([
            "subject_id",
            "admission_id",
            "time",
            "code",
            "numeric_value",
            "text_value"
        ])


        # Get unique codes
        LOGGER.text("Getting unique codes...", level=LoggerObserver.INFO)
        save_path = osp.join(output_path, '1_unique_codes')
        unique_codes = get_unique_codes(df, save_path=save_path)

        # strip code column
        df = df.with_columns(
            pl.col("code").str.strip_chars()
        )


        # Make numpy data
        if split in ['train', 'val']:
            save_path = osp.join(output_path, '2_numpify', 'train')
        else:
            save_path = osp.join(output_path, '2_numpify', split)
        os.makedirs(save_path, exist_ok=True)
        # df = numpify_dataframe_parallel(
        #     df, unique_codes, 
        #     save_path=save_path,
        #     num_workers=8,
        #     timeout=300,
        #     chunk_size=3000
        # )
        df = numpify_dataframe(
            df, unique_codes, 
            save_path=save_path,
            chunk_size=3000,
            target_column=target_column
        )

    # Encoding table
    if split in ['train', 'val']:
        save_path = osp.join(output_path, '3_encoding', 'train')
    else:
        save_path = osp.join(output_path, '3_encoding', split)
    os.makedirs(save_path, exist_ok=True)
    _save_path = osp.join(save_path, "data.parquet")
    if osp.isfile(_save_path):
        LOGGER.text("Loading encoded data...", level=LoggerObserver.INFO)
        df = pl.read_parquet(_save_path, use_pyarrow=True)
    else:
        LOGGER.text("Encoding data...", level=LoggerObserver.INFO)
        if split == 'train':
            df = encoding_table(
                df,
                save_encoder_mapping=osp.join(save_path, "encoder_mapping.pkl"),
                load_encoder_mapping=None,
            )
        else:
            df = encoding_table(
                df,
                save_encoder_mapping=None,
                load_encoder_mapping=load_encoder_mapping,
            )

        # Split df and save to multiple

        df.write_parquet(_save_path, use_pyarrow=True)
    
    # Feature engineering
    # Make numpy data
    if split in ['train', 'val']:
        save_path = osp.join(output_path, '3.5_feat_engineering', 'train')
    else:
        save_path = osp.join(output_path, '3.5_feat_engineering', split)
    os.makedirs(save_path, exist_ok=True)
    save_path = osp.join(save_path, "data.parquet")

    if osp.isfile(save_path):
        LOGGER.text("Loading feature-engineered data...", level=LoggerObserver.INFO)
        df = pl.read_parquet(save_path, use_pyarrow=True)
    else:
        df = feat_engineering(
            df,
        )
        df.write_parquet(save_path, use_pyarrow=True)

    return df

def make_targets(
        df,
        target_column:str="WILL_READMIT_WITHIN_7_DAYS", 
        label_path:str=None, 
        feature_selection_path:str = None, 
        selection_method:str=None,
        selected_indices:List[int]=None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, List[str]]:

    # Make inputs and targets

    LOGGER.text(f"Original data shape {df.shape}", level=LoggerObserver.INFO)
    label_df = pl.read_csv(label_path, has_header=True)
    string_columns = [col for col in label_df.columns if label_df[col].dtype == pl.String]
    label_df = label_df.drop(string_columns)

    # Drop first admission of all patients
    df = df.filter(
        pl.col("admission_id") > 0
    )

    df = df.join(
        label_df,
        on=["subject_id", "admission_id"],
        how="inner"
    )

    # Filter None values in target column
    df = df.filter((
        pl.col(target_column).is_not_null()
    ))
    LOGGER.text(f"Filtered data shape {df.shape}", level=LoggerObserver.INFO)

    data_ids = df.select(["subject_id", "admission_id"]).to_numpy()

    (x, y), ids_, feature_names, selected_indices = make_inputs_targets(
        df,
        target_column=target_column,
        selection_method=selection_method,
        feature_selection_path=feature_selection_path,
        selected_indices=selected_indices
    )

    # np.save(save_npy_path, np.concatenate((x, y[:, np.newaxis]), axis=1))
    return (x, y), ids_, feature_names, selected_indices, data_ids


def get_categorical_feature_indices(feature_names: List[str]) -> List[int]:
    """
    Get the indices of categorical features in the feature_names list.
    """
    categorical_feature_indices = []
    for i, feature_name in enumerate(feature_names):
        if any(feature_name.startswith(prefix) for prefix in CATEGORICAL_COLUMNS):
            categorical_feature_indices.append(i)
    return categorical_feature_indices



def get_mimic_dataset(
        train_folder=None, test_folder=None, 
        train_label_path=None, test_label_path=None,
        icu_ids=None, 
        data_output_folder=None, 
        target_column=None,
        feature_selection_path=None, 
        selection_method=None,
        process_file_path:str=None,
        trajectory_file_path:str=None,
        n_process_variants:int=8,
        logger=None
    ):

    if icu_ids is not None:
        with open(icu_ids, 'r') as f:
            icu_patient_ids = f.readlines()
        icu_patient_ids = list(map(lambda x: int(x.strip()), icu_patient_ids))
    else:
        icu_patient_ids = None

    if train_folder is not None:
        folder_names = os.listdir(train_folder)
        for folder_name in folder_names:
            if 'FinalizeData' in folder_name:
                train_folder = osp.join(train_folder, folder_name)
                break
    if test_folder is not None:
        folder_names = os.listdir(test_folder)
        for folder_name in folder_names:
            if 'FinalizeData' in folder_name:
                test_folder = osp.join(test_folder, folder_name)
                break

    if logger:
        logger.text(f"================= Collecting TRAINING data =================", level=LoggerObserver.DEBUG)
    X_train = collect_parquet(
        input_path=train_folder,
        output_path=data_output_folder,
        split="train",
        patient_ids_filter=icu_patient_ids,
        target_column=target_column
    )

    if target_column is not None:
        train_data, _, feature_names, selected_indices, train_data_ids = make_targets(
            X_train,
            target_column  = target_column,
            label_path = train_label_path,
            feature_selection_path=feature_selection_path,
            selection_method=selection_method
        )
        X_train, y_train = train_data
    else:
        train_data_ids = X_train.select(["subject_id", "admission_id"]).to_numpy()
        X_train = X_train.drop(["subject_id", "hadm_id", "icustay_id", "admission_id"], strict=False)
        feature_names = X_train.columns
        X_train = X_train.to_numpy()
        y_train = None

    categorical_feature_indices = get_categorical_feature_indices(feature_names)

    if logger:
        logger.text(f"================= Collecting TESTING data =================", level=LoggerObserver.DEBUG)
    X_test = collect_parquet(
        input_path=test_folder,
        output_path=data_output_folder,
        split="test",
        patient_ids_filter=icu_patient_ids,
        load_encoder_mapping=osp.join(data_output_folder, "3_encoding/train/encoder_mapping.pkl")
    )

    if target_column is not None:
        test_data, _, _, _, test_data_ids = make_targets(
            X_test,
            target_column  = target_column,
            label_path = test_label_path,
            feature_selection_path=feature_selection_path,
            selection_method=selection_method,
            selected_indices=selected_indices
        )
        X_test, y_test = test_data
    else:
        test_data_ids = X_test.select(["subject_id", "admission_id"]).to_numpy()
        X_test = X_test.drop(["subject_id", "hadm_id", "icustay_id", "admission_id"], strict=False)
        X_test = X_test.to_numpy()
        y_test = None

    train_ids_df = pl.DataFrame({
        "subject_id": train_data_ids[:, 0],
        "admission_id": train_data_ids[:, 1]
    })

    test_ids_df = pl.DataFrame({
        "subject_id": test_data_ids[:, 0], 
        "admission_id": test_data_ids[:, 1]
    })

    if process_file_path is not None:
        X_train_process = pl.read_csv(f"{process_file_path}/process_pathways_encoded_train.csv")
        X_test_process = pl.read_csv(f"{process_file_path}/process_pathways_encoded_test.csv")

        # X_train_process = pl.read_csv(f"{process_file_path}/{n_process_variants}_clustered_cases_train.csv")
        # X_test_process = pl.read_csv(f"{process_file_path}/{n_process_variants}_clustered_cases_test.csv")

        # Filter using inner join
        X_train_process = train_ids_df.join(
            X_train_process,
            on=["subject_id", "admission_id"],
            how="left"
        )
        X_test_process = test_ids_df.join(
            X_test_process,
            on=["subject_id", "admission_id"],
            how="left"
        )


    else:
        X_train_process = None
        X_test_process = None

    if trajectory_file_path is not None:
        X_train_traj = np.load(f"{trajectory_file_path}/trajectories_train_embeddings.npz")
        train_metadata_ids = X_train_traj['metadata']
        train_embeddings = X_train_traj['embeddings']
        
        # Get embedding dimension
        embedding_dim = train_embeddings.shape[1]

        # Use left join to keep all patients from train_ids_df
        filtered_train_ids = train_ids_df.join(
            pl.DataFrame({
                "subject_id": train_metadata_ids[:, 0],
                "admission_id": train_metadata_ids[:, 1]
            }).with_row_index(),
            on=["subject_id", "admission_id"],
            how="left"  # Changed from "inner" to "left"
        )
        # Create embeddings array with zeros for missing patients
        train_embedding_matrix = np.zeros((len(filtered_train_ids), embedding_dim))
        
        # Fill in embeddings for patients that have them
        valid_indices = filtered_train_ids['index'].to_numpy()
        valid_mask = ~np.isnan(valid_indices.astype(float))  # Check for non-null indices
        
        if valid_mask.any():
            train_embedding_matrix[valid_mask] = train_embeddings[valid_indices[valid_mask].astype(int)]
        
        X_train_traj = train_embedding_matrix

        # Same logic for test data
        X_test_traj = np.load(f"{trajectory_file_path}/trajectories_test_embeddings.npz")
        test_metadata_ids = X_test_traj['metadata']
        test_embeddings = X_test_traj['embeddings']
        
        filtered_test_ids = test_ids_df.join(
            pl.DataFrame({
                "subject_id": test_metadata_ids[:, 0],
                "admission_id": test_metadata_ids[:, 1]
            }).with_row_index(),
            on=["subject_id", "admission_id"],
            how="left"  # Changed from "inner" to "left"
        )
        
        # Create embeddings array with zeros for missing patients
        test_embedding_matrix = np.zeros((len(filtered_test_ids), embedding_dim))
        
        # Fill in embeddings for patients that have them
        valid_indices = filtered_test_ids['index'].to_numpy()
        valid_mask = ~np.isnan(valid_indices.astype(float))  # Check for non-null indices
        
        if valid_mask.any():
            test_embedding_matrix[valid_mask] = test_embeddings[valid_indices[valid_mask].astype(int)]
        
        X_test_traj = test_embedding_matrix

    else:
        X_train_traj = None
        X_test_traj = None


    return {
        'X_train': X_train, 'y_train': y_train,  
        'X_test': X_test, 'y_test': y_test, 
        'X_train_process': X_train_process,
        'X_test_process': X_test_process,
        'X_train_traj': X_train_traj,
        'X_test_traj': X_test_traj,
        'categorical_feature_indices': categorical_feature_indices,
        'feature_names': feature_names,
    }
