import polars as pl

from scripts.tokenization.constants import SpecialToken as ST
from ..patterns import MatchAndRevise
from .preprocessors import DemographicData


class GBMSDemographicData(DemographicData):
    @staticmethod
    @MatchAndRevise(apply_vocab=True, needs_vocab=True, prefix=["INSURANCE", "RACE", "MARITAL", "ADMISSION_TYPE", "DISCHARGE_LOCATION", "GENDER", "LOS", "AGE"])
    def assign_numeric_values(df: pl.DataFrame, *, vocab=None) -> pl.DataFrame:
        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("INSURANCE"))
                .then(pl.lit("INSURANCE"))
                .otherwise(pl.col("code"))
            ),
            text_value=(
                pl.when(pl.col("code").str.starts_with("INSURANCE"))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("text_value"))
            )
        )

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("GENDER"))
                .then(pl.lit("GENDER"))
                .otherwise(pl.col("code"))
            ),
            text_value=(
                pl.when(pl.col("code").str.starts_with("GENDER"))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("text_value"))
            )
        )

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("RACE"))
                .then(pl.lit("RACE"))
                .otherwise(pl.col("code"))
            ),
            text_value=(
                pl.when(pl.col("code").str.starts_with("RACE"))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("text_value"))
            )
        )
        
        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("MARITAL"))
                .then(pl.lit("MARITAL"))
                .otherwise(pl.col("code"))
            ),
            text_value=(
                pl.when(pl.col("code").str.starts_with("MARITAL"))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("text_value"))
            )
        )

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("ADMISSION_TYPE"))
                .then(pl.lit("ADMISSION_TYPE"))
                .otherwise(pl.col("code"))
            ),
            text_value=(
                pl.when(pl.col("code").str.starts_with("ADMISSION_TYPE"))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("text_value"))
            )
        )

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("DISCHARGE_LOCATION"))
                .then(pl.lit("DISCHARGE_LOCATION"))
                .otherwise(pl.col("code"))
            ),
            text_value=(
                pl.when(pl.col("code").str.starts_with("DISCHARGE_LOCATION"))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("text_value"))
            )
        )

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("LOS"))
                .then(pl.lit("LOS"))
                .otherwise(pl.col("code"))
            ),
            numeric_value=(
                pl.when(
                    pl.col("code").str.starts_with("LOS") & 
                    (pl.col("code").str.split('//').list.len() > 1)
                )
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("numeric_value"))
            )
        )

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("AGE"))
                .then(pl.lit("AGE"))
                .otherwise(pl.col("code"))
            ),
            numeric_value=(
                pl.when(
                    pl.col("code").str.starts_with("AGE") & 
                    (pl.col("code").str.split('//').list.len() > 1)
                )
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("numeric_value"))
            )
        )

        return df