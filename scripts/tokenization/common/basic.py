from typing import *
import pickle
from collections.abc import Sequence
from pathlib import Path
from cv2 import sort
import polars as pl

from nosos.utilities.constants import STATIC_DATA_FN
from nosos.utilities.constants import SpecialToken as ST
from nosos.utilities.vocabulary import Vocabulary
from ..patterns import MatchAndRevise, ScanAndAggregate
from ..utils import create_prefix_or_chain, static_class
from loguru import logger
from nosos.datasets.cohort.cohort import DISEASE_GROUP_TO_DETAILED_ICDS, REQUIRED_HAI_RELATED_MIMICIV_CODES

def filter_subject_ids(
    df: pl.DataFrame, *, subject_ids_path: str = None
):
    if subject_ids_path is None:
        return df
    
    if not Path(subject_ids_path).exists():
        raise ValueError(f"{subject_ids_path} does not exist")
    
    subject_ids = open(subject_ids_path, "r").readlines()
    subject_ids = list(map(lambda x: int(x.strip()), subject_ids))
    return df.filter(pl.col("subject_id").is_in(subject_ids))

def filter_codes(
    df: pl.DataFrame, *, codes_to_remove: Sequence[str], is_prefix: bool = False
) -> pl.DataFrame:
    expr = pl.col("code").cast(str).is_in(codes_to_remove)
    if is_prefix:
        expr = create_prefix_or_chain(codes_to_remove)
    return df.filter(~expr)

def filter_codes_retain(
    df: pl.DataFrame, *, codes_to_keep: Sequence[str], is_prefix: bool = False
) -> pl.DataFrame:
    expr = pl.col("code").cast(str).is_in(codes_to_keep)
    if is_prefix:
        expr = create_prefix_or_chain(codes_to_keep)
    return df.filter(expr)

def filter_short_los(
    df: pl.DataFrame, *, los_thresh_in_days: int = 2
) -> pl.DataFrame:
    """
    Filter out patients with length of stay less than los_thresh.
    """
    los_thresh_in_days = float(los_thresh_in_days)
    short_los_df = (
        df.filter(
            pl.col("code").str.starts_with('LOS')
        ).with_columns(
            is_short_los=pl.col("numeric_value").cast(pl.Float64) < los_thresh_in_days
        )
    )
    short_los_df = short_los_df.filter(
        pl.col("is_short_los") == True
    ).select(
        pl.col("subject_id"),
        pl.col("admission_id")
    ).unique()

    # Get the subject_ids and admission idof patients with short LOS
    short_los_subject_ids = short_los_df.select(
        pl.col("subject_id"),
        pl.col("admission_id")
    ).unique()
    
    # Filter out patients with short LOS
    df = df.join(
        short_los_subject_ids,
        on=["subject_id", "admission_id"],
        how="anti"
    )

    return df

def filter_by_age(
    df: pl.DataFrame, *, age_thresh_in_years: int = 18
):
    """
    Filter out patients with age less than age_thresh.
    """
    adult_df = (
        df.filter(
            pl.col("code").str.starts_with('AGE')
        ).with_columns(
            is_adult=pl.col("numeric_value").cast(pl.Int64()) >= age_thresh_in_years
        )
    )

    adult_df = adult_df.filter(
        pl.col("is_adult") == True
    ).select(
        pl.col("subject_id"),
        pl.col("admission_id")
    ).unique()

    # Get the subject_ids and admission idof patients with short LOS
    adult_subject_ids = adult_df.select(
        pl.col("subject_id"),
        pl.col("admission_id")
    ).unique()
    
    # Filter out patients with short LOS
    df = df.join(
        adult_subject_ids,
        on=["subject_id", "admission_id"],
        how="anti"
    )

    return df
    

def remove_consecutive_same_codes(
    df: pl.DataFrame,
    *,
    time_thresh: int = 5 * 60,  # 5 minutes in seconds
) -> pl.DataFrame:
    """
    Remove duplicate rows with the same code and time diff < 5m.
    """

    # Sort the DataFrame by subject_id, code, and time
    df = df.sort(by=["subject_id", "time", "code"])
    # Add time diff column
    df = df.with_columns(
        (pl.col("time").diff()).alias("time_diff")
    )

    condition = (
        (pl.col("subject_id") == pl.col("subject_id").shift(1)) &
        (pl.col("code") == pl.col("code").shift(1)) &
        (pl.col("time_diff") < time_thresh)  # 5 minutes in seconds
        # (pl.col("time") == pl.col("time").shift(1))
    )
    deduped_df = df.filter(~condition).drop("time_diff")
    return deduped_df

def write_required_codes(
    df: pl.DataFrame, *, output_dir: str, required_codes_fn: str
):
    """
    Write the required codes to a file.
    """

    required_codes = []
    for code in DISEASE_GROUP_TO_DETAILED_ICDS["HCAI"]:
        required_codes.append('ICD//CM//10//' + code)
    required_codes.extend(REQUIRED_HAI_RELATED_MIMICIV_CODES)
    
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / required_codes_fn, "w") as f:
        for code in required_codes:
            f.write(f"{code}\n")
    logger.info(f"Required codes written to {output_dir}/{required_codes_fn}")
    return df


def apply_vocab(df: pl.DataFrame, *, vocab: str | list[str] | None = None) -> pl.DataFrame:
    if vocab is None:
        return df
    elif isinstance(vocab, str):
        vocab = list(Vocabulary.from_path(vocab))
    return df.filter(pl.col("code").is_in(vocab))


@static_class
class CodeCounter(ScanAndAggregate):
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.col("code").value_counts()).unnest("code")

    def agg(self, in_fps: list, out_fp: str | Path) -> None:
        logger.info(f"Starting CodeCounter aggregation..")
        dfs = [pl.scan_parquet(fp) for fp in in_fps]
        df = dfs[0]
        for rdf in dfs[1:]:
            df = df.join(rdf, on="code", how="full", coalesce=True, join_nulls=True).select(
                "code", pl.sum_horizontal(pl.exclude("code"))
            )
        df.sort("count", descending=True).collect().write_csv(out_fp)
        logger.info(f"CodeCounter aggregation completed. Output saved to {out_fp}")

@static_class
class FinalizeData(ScanAndAggregate):
    sort_cols = MatchAndRevise.sort_cols
    def __call__(self, df: pl.DataFrame, columns:List[str]) -> pl.DataFrame:
        existing_columns = df.columns
        columns = [col for col in columns if col in existing_columns]
        return df.sort(by=self.sort_cols).select(columns)


@static_class
class StaticDataCollector(ScanAndAggregate):
    patient_id_col = MatchAndRevise.sort_cols[0]

    def __call__(self, df: pl.DataFrame, *, static_code_prefixes: list[str]) -> pl.DataFrame:
        self.static_code_prefixes = static_code_prefixes
        df = (
            df.select(self.patient_id_col, "code", pl.col("time").cast(pl.Int64))
            .filter(create_prefix_or_chain(static_code_prefixes))
            .group_by(
                self.patient_id_col,
                prefix=pl.col("code").str.split("//").list.get(0),
            )
            .agg("code", "time")
            .with_columns(pl.struct(code="code", time="time"))
            .pivot(index=self.patient_id_col, on="prefix", values="code")
            .with_columns(
                pl.when(pl.col(col_name).struct[0].is_null())
                .then(pl.struct(code=pl.lit([f"{col_name}//UNKNOWN"])))
                .otherwise(col_name)
                .alias(col_name)
                for col_name in static_code_prefixes
                if col_name != ST.DOB
            )
        )
        # maintain the order of columns, so that the output is deterministic
        return df.select(sorted(df.columns))

    def agg(self, in_fps: list, out_fp: str | Path, *, num_quantiles: int = 10) -> None:
        # TODO: Let's store it in parquet instead of pickle
        logger.info(f"Starting StaticDataCollector aggregation..")
        df = pl.read_parquet(in_fps)
        out_dict = df.rows_by_key(self.patient_id_col, named=True, unique=True)

        if getattr(self, "static_code_prefixes", None) is None:
            if Path(out_fp).with_name(STATIC_DATA_FN).exists(): return
            raise ValueError("static_code_prefixes must be set before calling agg")
        

        default_dict = {}
        for col_name in self.static_code_prefixes:
            if col_name != ST.DOB:
                default_dict[col_name] = {
                    'code': [f"{col_name}//UNKNOWN"],
                    'time': None
                }
            else:
                default_dict[ST.DOB] = {
                    'code': [ST.DOB],
                    'time': None
                }

        out_dict['default'] = default_dict

        with Path(out_fp).with_name(STATIC_DATA_FN).open("wb") as f:
            pickle.dump(out_dict, f)

        logger.info(f"StaticDataCollector aggregation completed. Output saved to {out_fp}")
