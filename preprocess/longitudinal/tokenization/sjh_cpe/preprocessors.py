from datetime import time

import numpy as np
import polars as pl

from scripts.tokenization.constants import SpecialToken as ST, TOKEN_TYPES_IDS, TensorizedColumns
from ..patterns import MatchAndRevise
from ..utils import apply_vocab_to_multitoken_codes, unify_code_names


class DeathData:
    @staticmethod
    @MatchAndRevise(prefix=[ST.DEATH, ST.DISCHARGE], needs_resorting=True)
    def place_death_before_dc_if_same_time(df: pl.DataFrame) -> pl.DataFrame:
        gb_cols = MatchAndRevise.sort_cols
        idx_col = MatchAndRevise.index_col
        return (
            df.with_columns(
                # If time is 0 in datetime, it means there is no information about exact time;
                # then move the death to the end of the day (23:59:59)
                pl.when(pl.col("code") == str(ST.DEATH), pl.col("time").dt.time() == 0)
                .then(pl.col("time").dt.combine(time=time(23, 59, 59)))
                .otherwise("time")
            )
            .sort(pl.col("code").replace_strict(ST.DEATH, 0, default=1, return_dtype=pl.UInt8))
            .group_by(gb_cols, maintain_order=True)
            .agg(pl.col(idx_col).last(), pl.exclude(gb_cols, idx_col))
            .explode(pl.exclude(gb_cols, idx_col))
            .sort(by=idx_col)
            .select(df.columns)
        )


class DemographicData:
    @staticmethod
    @MatchAndRevise(prefix=ST.ADMISSION)
    def retrieve_demographics_from_hosp_adm(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list("code", pl.lit("MARITAL_STATUS"), pl.lit("RACE")),
            text_value=pl.concat_list("text_value", pl.col("marital_status"), pl.col("race")),
        ).explode("code", "text_value")

    @staticmethod
    @MatchAndRevise(prefix="RESIDENCY//")
    def process_residency(df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize residence data into five categories:
        - Irish: Irish residents outside Dublin
        - European: European residents excluding Ireland
        - American: North and South American residents
        - Dubliner: Dublin residents
        - Others: All other locations
        """
        # Define patterns for categorization
        dublin_pattern = r'DUBLIN \d+'
        irish_counties_pattern = r'CO '
        irish_cities = ['CORK', 'GALWAY', 'LIMERICK', 'WATERFORD']
        
        european_countries = [
            'ENGLAND', 'WALES', 'SCOTLAND', 'NORTH IRELAND',
            'FRANCE', 'GERMANY', 'SPAIN', 'ITALY', 'BELGIUM', 
            'HOLLAND', 'CZECH REPUBLIC', 'DENMARK', 'FINLAND',
            'NORWAY', 'SWEDEN', 'PORTUGAL', 'SLOVENIA', 
            'SWITZERLAND', 'LUXEMBOURG', 'LITHUANIA', 'MALTA',
            'AUSTRIA', 'ALL OTHER EUROPEAN'
        ]
        
        american_patterns = ['AMERICA NORTH', 'CANADA', 'AMERICA STH', 'CARRIBEAN']

        df = df.with_columns(
            text_value=pl.col("code").str.split("//").list[-1]
        )
        
        df = df.with_columns(
            code=pl.when(pl.col("text_value").str.contains(dublin_pattern) | (pl.col("text_value") == "DUBLIN"))
            .then(pl.lit("RESIDENCY//DUBLINER"))
            .when(pl.col("text_value").str.contains(irish_counties_pattern) | pl.col("text_value").str.contains("|".join(irish_cities)))
            .then(pl.lit("RESIDENCY//IRISH"))
            .when(pl.col("text_value").str.contains("|".join(european_countries)))
            .then(pl.lit("RESIDENCY//EUROPEAN"))
            .when(pl.col("text_value").str.contains("|".join(american_patterns)))
            .then(pl.lit("RESIDENCY//AMERICAN"))
            .otherwise(pl.lit("RESIDENCY//OTHERS"))
        )
        return df

    @staticmethod
    @MatchAndRevise(prefix="GENDER//", apply_vocab=True)
    def process_gender(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            code = pl.col("code").str.split("//")
        ).with_columns(
            code= pl.lit('GENDER//') + (
                pl.when(pl.col("code").list[1].str.starts_with("F"))
                .then(pl.lit("FEMALE"))
                .when(pl.col("code").list[1].str.starts_with("M"))
                .then(pl.lit("MALE"))
                .otherwise(pl.lit("OTHER"))
            )
        )
        return df

    @staticmethod
    @MatchAndRevise(prefix="RACE", apply_vocab=True)
    def process_race(df: pl.DataFrame) -> pl.DataFrame:
        race_unknown = ["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"]
        race_minor = [
            "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
            "AMERICAN INDIAN/ALASKA NATIVE",
            "MULTIPLE RACE/ETHNICITY",
        ]
        # every patient can have only one race assigned, so we can prioritize which one to keep
        race_priority_mapping = {"RACE//OTHER": 1, "RACE//UNKNOWN": 2}  # every other will get 0
        return (
            df.with_columns(
                code=pl.when(pl.col("text_value").str.to_uppercase().is_in(race_unknown))
                .then(pl.lit("UNKNOWN"))
                .when(pl.col("text_value").str.to_uppercase().is_in(race_minor))
                .then(pl.lit("OTHER"))
                .when(pl.col("text_value").str.to_uppercase() == "SOUTH AMERICAN")
                .then(pl.lit("HISPANIC"))
                .when(pl.col("text_value").str.to_uppercase() == "PORTUGUESE")
                .then(pl.lit("WHITE"))
                .when(pl.col("text_value").str.contains_any(["/", " "]))
                .then(pl.lit(None))
                .otherwise("text_value")
            )
            .with_columns(
                code=(
                    pl.lit("RACE//")
                    + pl.when(pl.col("code").is_null())
                    .then(pl.col("text_value").str.slice(0, pl.col("text_value").str.find("/| ")))
                    .otherwise("code")
                )
            )
            .group_by(MatchAndRevise.sort_cols[0], maintain_order=True)
            .agg(
                pl.col("code")
                .sort_by(
                    pl.col.code.replace_strict(
                        race_priority_mapping, default=0, return_dtype=pl.UInt8
                    )
                )
                .first(),
                pl.exclude("code").first(),
            )
            .select(df.columns)
        )

    @staticmethod
    @MatchAndRevise(prefix="MARITAL_STATUS", apply_vocab=True)
    def process_marital_status(df: pl.DataFrame) -> pl.DataFrame:
        return df.drop_nulls("text_value").with_columns(
            code=pl.lit("MARITAL//") + pl.col("text_value")
        )

    @staticmethod
    @MatchAndRevise(prefix="", needs_resorting=True)
    def add_age_offsets_as_column(df: pl.DataFrame) -> pl.DataFrame:
        """
        calculate age offsets based on the time between each activity with MEDS_BIRTH
        """
        df = df.sort(['subject_id', 'time']).with_columns(
            time_between_sessions = (
                pl.col("time")
                .diff().dt.total_days()
                .fill_null(0)
                .over("subject_id")
            )
        )

        df = df.with_columns(
            time_between_sessions = pl.col('time_between_sessions') // 365
        )

        # Cum sum the time_between_sessions
        df = df.with_columns(
            pl.col('time_between_sessions').cum_sum().over("subject_id").cast(pl.Int32).alias(TensorizedColumns.AGE_OFFSET)
        ).drop('time_between_sessions')

        return df

    @staticmethod
    @MatchAndRevise(prefix=[ST.DOB, ST.ADMISSION])
    def add_age_as_code(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate age based on the difference between the first and last admission dates.
        """
        prefix = "AGE//"

        # Pivot to get separate columns for START and END times
        df_pivot = df.pivot(
            values="time",
            index="subject_id",
            columns="code",
            aggregate_function="min"  # Use min in case there are multiple START/END
        )

        # Compute the time difference
        df_pivot = df_pivot.with_columns(
            (pl.col(ST.ADMISSION) - pl.col(ST.DOB)).alias("time_difference")
        )
        df_pivot = df_pivot.select(
            subject_id=pl.col("subject_id"),
            age=(pl.col("time_difference").dt.total_days() / 365).cast(pl.Int32),
        )

        # Join back with original df to get time_difference per subject_id
        df = df.join(df_pivot.select(["subject_id", "age"]), on="subject_id", how="left")
       

        df = df.with_columns(
            code=(
                pl.when(pl.col("code") == pl.lit(ST.DOB))
                .then(pl.concat_list([pl.lit(ST.DOB) , pl.lit(prefix) + pl.col("age").cast(pl.Utf8)]))
                .otherwise(pl.concat_list([pl.col("code"),]))
            )
        ).explode("code")

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with(prefix))
                .then(pl.lit("AGE"))
                .otherwise(pl.col("code"))
            ),
            numeric_value=(
                pl.when(pl.col("code").str.starts_with(prefix))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("numeric_value"))
            )
        ).drop('age')

        return df

    def add_los_as_code(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate length of stay (LOS) based on the difference between the first and last admission dates.
        """
        prefix = "LOS//"

        # Pivot to get separate columns for START and END times
        df_pivot = df.pivot(
            values="time",
            index=["subject_id", 'admission_id'],
            columns="code",
            aggregate_function="min"  # Use min in case there are multiple START/END
        )
        # Compute the time difference
        df_pivot = df_pivot.with_columns(
            (pl.col(ST.DISCHARGE) - pl.col(ST.ADMISSION)).alias("time_difference")
        )

        df_pivot = df_pivot.select(
            subject_id=pl.col("subject_id"),
            admission_id=pl.col("admission_id"),
            los=(pl.col("time_difference").dt.total_days()).cast(pl.Int32),
        )

        # Join back with original df to get time_difference per subject_id
        df = df.join(df_pivot.select(["subject_id", "admission_id", "los"]), on=["subject_id", "admission_id"], how="left")
        df = df.with_columns(
            code=(
                pl.when(
                    pl.col("code") == pl.lit(ST.ADMISSION)
                )
                .then(
                    pl.concat_list([
                        pl.lit(ST.ADMISSION), 
                        pl.lit(prefix) + pl.col("los").cast(pl.Utf8)])
                    )
                .otherwise(pl.concat_list([pl.col("code"),]))
            )
        ).explode("code")
        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with(prefix))
                .then(pl.lit("LOS"))
                .otherwise(pl.col("code"))
            ),
            numeric_value=(
                pl.when(pl.col("code").str.starts_with(prefix))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("numeric_value"))
            )
        ).drop('los')

        return df


class InpatientData:
    @staticmethod
    @MatchAndRevise(prefix="DRG//", apply_vocab=False)
    def process_drg_codes(df: pl.DataFrame) -> pl.DataFrame:
        # return df.filter(pl.col.code.str.starts_with("DRG//HCFA")).with_columns(
        #     code=pl.lit("DRG//") + pl.col.code.str.split("//").list[2].cast(int).cast(str)
        # )

        return df.filter(pl.col.code.str.starts_with("DRG//")).with_columns(
            code=pl.lit("DRG//") + pl.col.code.str.split("//").list[-1].cast(str)
        )

    @staticmethod
    @MatchAndRevise(prefix="DRG//", needs_counts=True, needs_vocab=True)
    def filter_drg_based_on_counts(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None,
        top_n: int = 200, keep_codes_file: str = None
    ) -> pl.DataFrame:
        
        if keep_codes_file is not None:
            codes = open(keep_codes_file, "r").readlines()
        else:
            codes = None

        known_icd_names = list(counts.keys())[:top_n] if vocab is None else vocab
        if codes is not None:
            known_icd_names += [code.strip() for code in codes]
        return df.filter(pl.col("code").is_in(known_icd_names))

    @staticmethod
    @MatchAndRevise(prefix=ST.ADMISSION)
    def process_hospital_admissions(df: pl.DataFrame) -> pl.DataFrame:
        scheduled_admissions = ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]
        return (
            df.with_columns(
                pl.col.code.str.split("//").list[0].alias("code"),
                pl.col.code.str.split("//").list[1].alias("text_value"),
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.lit("ADMISSION_TYPE//")
                    + pl.when(
                        pl.col("text_value").str.to_uppercase().str.ends_with("EMER.")
                        | (pl.col("text_value") == "URGENT")
                    )
                    .then(pl.lit("EMERGENCY"))
                    .when(pl.col("text_value").str.to_uppercase().is_in(scheduled_admissions))
                    .then(pl.lit("SCHEDULED"))
                    .otherwise(pl.lit("OBSERVATION")),
                    pl.lit("INSURANCE//") + pl.col("insurance"),
                ).alias("code")
            )
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix=[ST.DISCHARGE, "DIAGNOSIS//ICD//", "DRG//"])
    def process_hospital_discharges(df: pl.DataFrame) -> pl.DataFrame:
        """Currently must be run before processing diagnoses."""
        discharge_facilities = [
            "HEALTHCARE FACILITY",
            "SKILLED NURSING FACILITY",
            "REHAB",
            "CHRONIC/LONG TERM ACUTE CARE",
            "OTHER FACILITY",
        ]

        drg_following_diag = pl.col.code.str.starts_with(
            "DIAGNOSIS//ICD"
        ) & ~pl.col.code.str.starts_with("DRG//").shift(-1, fill_value=False)
        drg_following_disch = pl.col.code.str.starts_with(ST.DISCHARGE)

        if "edstay_id" in df.columns:
            # This means that it is MIMIC with ED extension, and diagnoses in addition come from
            # ED_OUT and in those situations DRG code should not be added
            drg_following_diag &= pl.col.edstay_id.is_null() & ~(
                pl.col.code.str.starts_with("DIAGNOSIS//ICD") & pl.col.edstay_id.is_null()
            ).shift(-1, fill_value=False)

            drg_following_disch &= pl.col.code.str.starts_with(ST.DISCHARGE).shift(
                -1, fill_value=True
            ) | pl.col.edstay_id.is_not_null().shift(-1, fill_value=True)
        else:
            drg_following_diag &= ~pl.col.code.str.starts_with("DIAGNOSIS//ICD").shift(
                -1, fill_value=False
            )
            drg_following_disch &= pl.col.code.str.starts_with(ST.DISCHARGE).shift(
                -1, fill_value=True
            )

        drg_missing_cond = drg_following_diag | drg_following_disch

        return (
            df.with_columns(
                text_value=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(pl.col.code.str.split("//").list[1])
                .otherwise("text_value")
            )
            .with_columns(
                code=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(
                    pl.concat_list(
                        pl.lit(ST.DISCHARGE),
                        (
                            pl.lit("DISCHARGE_LOCATION//")
                            + pl.when(pl.col("text_value").str.to_uppercase().is_in(discharge_facilities))
                            .then(pl.lit("HEALTHCARE_FACILITY"))
                            .when(pl.col("text_value").is_null())
                            .then(pl.lit("UNKNOWN"))
                            .when(pl.col("text_value").str.to_uppercase().str.contains("HOME"))
                            .then(pl.lit("HOME"))
                            .when(pl.col("text_value").str.to_uppercase().str.contains("DIED"))
                            .then(pl.lit("DIED"))
                            .otherwise(pl.lit("OTHER"))
                        ),
                    )
                )
                .otherwise(pl.concat_list("code")),
                drg_missing=drg_missing_cond,
            )
            .with_columns(
                code=pl.when("drg_missing")
                .then(pl.concat_list("code", pl.lit("DRG//UNKNOWN")))
                .otherwise("code")
            )
            .drop("drg_missing")
            .explode("code")
        ).filter(pl.col.code != "DISCHARGE_LOCATION//DIED")


    @staticmethod
    @MatchAndRevise(prefix="", needs_resorting=True)
    def reassign_admission_id(df: pl.DataFrame) -> pl.DataFrame:
        """
        Generalize the function to process the entire DataFrame with multiple subject_ids.
        Assign unique admission_ids for each subject_id based on the first occurrence of HOSPITAL_ADMISSION.

        Args:
            df (pl.DataFrame): The input DataFrame containing event logs.

        Returns:
            pl.DataFrame: A DataFrame with an additional column `admission_id` indicating the admission group.
        """
        # Assign unique admission_id for each subject_id based on the first occurrence of ST.ADMISSION

        df = df.sort(by=["subject_id", "time"], descending=[False, False])

        df = df.with_columns(
            (pl.when(pl.col("code").str.split("//").list[0] == ST.ADMISSION)
                .then(1)
                .otherwise(0))
            .alias("admission_flag")
        )

        # Compute cumulative sum of admission_flag over each subject_id to assign admission_id
        df = df.with_columns(
            pl.col("admission_flag").cum_sum().over("subject_id").alias(TensorizedColumns.ADMISSION_ID)
        )

        # Drop the intermediate admission_flag column
        df = df.drop("admission_flag")

        return df

    @staticmethod
    @MatchAndRevise(prefix=[ST.DOB, ST.ADMISSION])
    def calculate_age_column(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate age based on the difference between the first and last admission dates.
        """

        # Pivot to get separate columns for START and END times
        df_pivot = df.pivot(
            values="time",
            index="subject_id",
            columns="code",
            aggregate_function="min"  # Use min in case there are multiple START/END
        )

        # Compute the time difference
        df_pivot = df_pivot.with_columns(
            (pl.col(ST.ADMISSION) - pl.col(ST.DOB)).alias("time_difference")
        )
        df_pivot = df_pivot.select(
            subject_id=pl.col("subject_id"),
            age=(pl.col("time_difference").dt.total_days() / 365).cast(pl.Int32),
        )

        # Join back with original df to get time_difference per subject_id
        df = df.join(df_pivot.select(["subject_id", "age"]), on="subject_id", how="left")
       

        df = df.with_columns(
            code=(
                pl.when(pl.col("code") == pl.lit(ST.DOB))
                .then(pl.concat_list([pl.lit(ST.DOB) , pl.lit("AGE//") + pl.col("age").cast(pl.Utf8)]))
                .otherwise(pl.concat_list([pl.col("code"),]))
            )
        ).explode("code")

        df = df.with_columns(
            code=(
                pl.when(pl.col("code").str.starts_with("AGE//"))
                .then(pl.lit("AGE"))
                .otherwise(pl.col("code"))
            ),
            numeric_value=(
                pl.when(pl.col("code").str.starts_with("AGE//"))
                .then(pl.col("code").str.split('//').list[-1])
                .otherwise(pl.col("numeric_value"))
                .cast(pl.Int32)
            )
        )

        return df


class MeasurementData:
    @staticmethod
    @MatchAndRevise(prefix=["TEMPERATURE", "HEART_RATE", "RESPIRATORY_RATE", "O2_SATURATION"])
    def process_simple_measurements(df: pl.DataFrame) -> pl.DataFrame:
        # temperature can be missing, account for it
        return df.with_columns(
            code=pl.lit("VITAL//") + pl.col("code")
        )
    
    @staticmethod
    @MatchAndRevise(prefix="PAIN")
    def process_pain(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col.text_value.is_not_null())
            .with_columns(
                pl.col("text_value")
                .str.to_lowercase()
                .str.strip_chars(' +"')
                .str.strip_suffix("/10")
            )
            .with_columns(
                numeric_value=pl.when(pl.col.text_value.str.contains("-", literal=True))
                .then(
                    pl.col.text_value.str.split("-").list.first().str.strip_chars() + pl.lit(".5")
                )
                .when(pl.col.text_value.str.contains("crit|moaning"))
                .then(pl.lit("10"))
                .when(pl.col.text_value.str.contains("lot|all over|hurts|much"))
                .then(pl.lit("8"))
                .when(
                    pl.col.text_value.is_in(
                        ["yes", "mild", "moderate", "y", "pain", "uncomfortable"]
                    )
                )
                .then(pl.lit("5"))
                .when(pl.col.text_value.str.contains("little|not bad|some|discomfort"))
                .then(pl.lit("2"))
                .when(
                    pl.col.text_value.str.contains(r"s[lep]{3,4}|sedat|n\/a|resting")
                    | pl.col.text_value.is_in(["no", "no pain", "ok", "none", "comfortable"])
                )
                .then(pl.lit("0"))
                .otherwise(pl.col("text_value").str.replace_all(r"\D", ""))
                .cast(float, strict=False)
            )
            .filter(pl.col.numeric_value.is_between(0, 10))
            .with_columns(code=pl.lit("VITAL//PAIN"))
        )

    @staticmethod
    @MatchAndRevise(prefix="Blood Pressure")
    def process_blood_pressure(bp_df: pl.DataFrame) -> pl.DataFrame:
        return (
            bp_df.with_columns(
                code=pl.when(pl.col.numeric_value.is_null()).then(
                    pl.col("text_value").str.split_exact("/", 1)
                )
                # Hacky way to get the systolic and diastolic that come from ED extension
                .otherwise(
                    pl.struct(
                        field_0=pl.col.numeric_value.cast(int).cast(str),
                        field_1="text_value",
                    )
                )
            )
            .with_columns(
                code=pl.concat_list(
                    pl.lit("VITAL//BLOOD_PRESSURE_SBP"),
                    pl.lit("VITAL//BLOOD_PRESSURE_DBP"),
                ),
                numeric_value=pl.concat_list(
                    pl.col("code").struct[0].cast(float),
                    pl.col("code").struct[1].cast(float),
                ),
                text_value=pl.col.code.struct[0].cast(str)
                + pl.lit("/")
                + pl.col.code.struct[1].cast(str),
            )
            .explode("code", "numeric_value")
        )
    
    @staticmethod
    @MatchAndRevise(prefix=["VITAL//"])
    def make_quantiles(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list("code", pl.lit("VITAL//Q//") + pl.col("code").str.slice(len("VITAL//")))
        ).explode("code")


class DiagnosesData:
    @staticmethod
    @MatchAndRevise(prefix="DIAGNOSIS//ICD//")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col.code.str.split_exact("//", 3)).with_columns(
            code=pl.lit("ICD//CM//") + pl.col.code.struct[2] + pl.lit('//') + pl.col.code.struct[3],
            text_value =pl.col.code.struct[3], 
        )
    
    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//", needs_resorting=True)
    def append_special_tokens(df: pl.DataFrame) -> pl.DataFrame:
        ## The code is sorted in order of time. I want to filter out the separate sequence of ICD codes  
        #  and append [START_ICD] and [END_ICD] tokens to the beginning and end of the group
        # We would need to identify the segments of ICD codes in the data 
        # and then add the special tokens to the beginning and end of each segment

        # Sort the data by time, within each subject_id
        icd_codes = df.sort(
            by=["subject_id", "time"],
            descending=[False, False],
        )

        # Get the start and end of each segment
        # by getting the difference between the time of two consecutive rows,
        # also get the difference between the subject_id of two consecutive rows to identify the segments
        icd_codes = icd_codes.with_columns(
            time_diff=pl.col("time").diff(),
            subject_id_diff=pl.col("subject_id").diff(),
            reverse_time_diff=pl.col("time").shift(-1) - pl.col("time"),
            reverse_subject_id_diff=pl.col("subject_id").shift(-1, fill_value=1) - pl.col("subject_id"),
        ).with_columns(
            time_diff = pl.col('time_diff').dt.total_days().cast(pl.Int64).fill_null(1),
            subject_id_diff = pl.col('subject_id_diff').fill_null(1),
            reverse_time_diff = pl.col('reverse_time_diff').dt.total_days().cast(pl.Int64).fill_null(1),
            reverse_subject_id_diff = pl.col('reverse_subject_id_diff').fill_null(1),
        )

        # Get the start and end of each segment
        icd_codes = icd_codes.with_columns(
            is_start=pl.when((pl.col("time_diff") != 0) | (pl.col('subject_id_diff') != 0) ).then(1).otherwise(0),
            is_end=pl.when((pl.col("reverse_time_diff") != 0) | (pl.col('reverse_subject_id_diff') != 0) ).then(1).otherwise(0),
        )


        # append [START_ICD] and [END_ICD] tokens to the beginning and end of the group
        # by append [START_ICD] and [END_ICD] tokens to where is_start and is_end are 1
        # and then use explode to create a new row for each token
        # and then drop the is_start and is_end columns
        icd_codes = icd_codes.with_columns(
            code=pl.when(pl.col("is_start") == 1)
            .then(pl.lit(f"{ST.START_ICD}///") + pl.col("code"))
            .otherwise(pl.col("code"))
        ).with_columns(
            code=pl.when(pl.col("is_end") == 1)
            .then(pl.col("code") + pl.lit(f"///{ST.END_ICD}"))
            .otherwise(pl.col("code"))
        )

        icd_codes = icd_codes.with_columns(pl.col.code.str.split("///"))
        icd_codes = icd_codes.explode("code").drop("is_start", "is_end", "time_diff", "subject_id_diff", "reverse_time_diff", "reverse_subject_id_diff")
        # Drop the is_start and is_end columns  
        return icd_codes

        

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_cm_9_to_10_mapping

        icd_9_to_10 = get_icd_cm_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//CM//10").alias("code"),
                pl.col("text_value").replace_strict(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value").with_columns(
            code=pl.lit("ICD//CM//10") + '//' + pl.col("text_value"),
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        from ..mappings import get_icd_cm_code_to_name_mapping

        code_to_name = get_icd_cm_code_to_name_mapping()
        temp_cols = ["part1", "part2", "part3"]
        code_prefixes = ["", "3-6//", "SFX//"]
        code_slices = [(0, 3), (3, 3), (6,)]

        df = (
            icd10_df.with_columns(
                pl.lit("ICD//CM//10")
            ).with_columns(
                pl.col("text_value").str.slice(*code_slice).alias(col)
                for col, code_slice in zip(temp_cols, code_slices)
            )
            # .with_columns(pl.col(temp_cols[0]).replace_strict(code_to_name, default=None))
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(pl.lit(f"ICD//CM//{prefix}") + pl.col(col))
                .alias(col)
                for col, prefix in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )

        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list(temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//10", needs_counts=True, needs_vocab=True)
    def filter_based_on_counts(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None,
        top_n: int = 200, keep_codes_file: str = None
    ) -> pl.DataFrame:

        if keep_codes_file is not None:
            codes = open(keep_codes_file, "r").readlines()
        else:
            codes = None

        known_icd_names = list(counts.keys())[:top_n] if vocab is None else vocab
        if codes is not None:
            known_icd_names += [code.strip() for code in codes]
        return df.filter(pl.col("code").is_in(known_icd_names))


class ProcedureData:
    @staticmethod
    @MatchAndRevise(prefix="PROCEDURE//ICD//")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col.code.str.split("//"))
            .filter(pl.col.code.list[1] == "ICD")
            .with_columns(
                code=pl.lit("ICD//PCS//") + pl.col.code.list[2] + pl.lit('//') + pl.col.code.list[3],
                text_value=pl.col.code.list[3]
            )
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_pcs_9_to_10_mapping

        icd_9_to_10 = get_icd_pcs_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//PCS//10").alias("code"),
                pl.col("text_value").replace(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value").with_columns(
            code=pl.lit("ICD//PCS//10") + '//' + pl.col("text_value"),
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        df = icd10_df.with_columns(
             code=pl.lit("ICD//PCS//10")
        ).with_columns(
            pl.col("text_value").str.split_exact("", 6).alias("code")
        ).with_columns(
            code=pl.concat_list(
                pl.when(pl.col("code").struct[i] != "").then(
                    pl.lit("ICD//PCS//") + pl.col("code").struct[i]
                )
                for i in range(7)
            ).list.drop_nulls()
        )
        if vocab is not None:
            # all characters have to be in the vocab to keep the code
            df = df.filter(pl.col("code").list.eval(pl.element().is_in(vocab)).list.all())
        return df.explode("code").drop_nulls("code")
    
    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//10", needs_counts=True, needs_vocab=True)
    def filter_based_on_counts(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None,
        top_n: int = 200, keep_codes_file: str = None
    ) -> pl.DataFrame:

        if keep_codes_file is not None:
            codes = open(keep_codes_file, "r").readlines()
        else:
            codes = None

        known_icd_names = list(counts.keys())[:top_n] if vocab is None else vocab
        if codes is not None:
            known_icd_names += [code.strip() for code in codes]
        return df.filter(pl.col("code").is_in(known_icd_names))

    @staticmethod
    @MatchAndRevise(prefix=["PROCEDURES"], apply_vocab=True)
    def unify_procedures(
        df: pl.DataFrame
    ) -> pl.DataFrame:
        return df.drop_nulls("code").with_columns(unify_code_names(pl.col('code')))

    @staticmethod
    @MatchAndRevise(prefix=["PROCEDURES"], needs_counts=True, needs_vocab=True)
    def filter_procedures_based_on_counts(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None,
        top_n: int = 200, keep_codes_file: str = None
    ) -> pl.DataFrame:
        
        if keep_codes_file is not None:
            codes = open(keep_codes_file, "r").readlines()
        else:
            codes = None

        known_icd_names = list(counts.keys())[:top_n] if vocab is None else vocab
        if codes is not None:
            known_icd_names += [code.strip() for code in codes]
        return df.filter(pl.col("code").is_in(known_icd_names))


class MedicationData:

    @staticmethod
    @MatchAndRevise(prefix="MEDICATION//")
    def convert_to_atc(df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_mimic_drug_name_to_atc_mapping
        import Levenshtein

        drug_to_atc = get_mimic_drug_name_to_atc_mapping()


        def unify_code_names_str(code:str):
            return code.replace(" ", "_").replace(",", "").replace(".", "").strip('"').strip("\n").upper()

        def find_nearest_word(word, word_list, distance_fn):
            """
            Find the nearest word in the list to the given word using Levenshtein distance.
            """
            min_distance = float('inf')
            nearest_word = None
            for candidate in word_list:
                dist = distance_fn(word, candidate)
                if dist < min_distance:
                    min_distance = dist
                    nearest_word = candidate

            # If the distance is greater than a threshold, return None
            if min_distance > 10:
                return None
            return nearest_word

        def correct_typo_drug_name_based_on_distance(name:str, mapping:dict) -> str:
            """
            Replace the nearest ATC code in the name using the provided mapping.
            """

            if name is None: return None

            standardized_name = unify_code_names_str(name)
            standardized_mapping = {
                unify_code_names_str(k): v for k, v in mapping.items()
            }

            if standardized_name in standardized_mapping:
                return standardized_name
            else:
                # # Find the nearest word in the mapping keys
                nearest_word = find_nearest_word(standardized_name, list(standardized_mapping.keys()), Levenshtein.distance)
                if nearest_word:
                    return nearest_word
            return standardized_name

        all_medication_codes = df.select("code").unique().sort('code').to_numpy().flatten()
        all_medication_codes = [i.split("//")[1] for i in all_medication_codes]
        all_medication_codes = [unify_code_names_str(i) for i in all_medication_codes]
        new_drug_to_atc = {
            unify_code_names_str(k): v for k, v in drug_to_atc.items()
        }
        all_medication_codes = [i for i in all_medication_codes if i not in new_drug_to_atc.keys()]
        fixed_mapping = {
            i: correct_typo_drug_name_based_on_distance(i, new_drug_to_atc) for i in all_medication_codes
        }
        df = ((
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                pl.when(pl.col("code").list[2].str.to_titlecase() == "Administered")
                .then(None)
                .when(pl.col("code").list[1].str.to_uppercase() == "START")
                .then(pl.lit("MEDICATION_START"))
                .alias("code"),
                pl.when(pl.col("code").list[2].str.to_titlecase() == "Administered")
                .then(pl.col("code").list[1])
                .when(pl.col("code").list[1].str.to_uppercase() == "START")
                .then(pl.col("code").list[2])
                .alias("text_value"),
            )
        ).with_columns(
            text_value=(
                pl.when(pl.col("text_value").str.to_uppercase() == "UNK")
                .then(None)
                .otherwise(unify_code_names(pl.col("text_value")))
            )
        )
        .drop_nulls("text_value")
        .with_columns(
                text_value = pl.col("text_value").replace_strict(
                    fixed_mapping, 
                    default=pl.col("text_value"), 
                    return_dtype=pl.String
                )
            )
        .with_columns(
            text_value = pl.col("text_value").replace_strict(
                new_drug_to_atc, default=None, return_dtype=pl.List(pl.String)
            )
        )
        .with_columns(
            pl.concat_list(
                "code",
                pl.lit(None).cast(str).repeat_by(pl.col("text_value").list.len().cast(int) - 1),
            ).alias("code")
        )
        .drop_nulls("text_value")
        .explode("code", "text_value")
        ).with_columns(
            code=pl.lit("ATC//") + pl.col("text_value")
        )
        return df

    @staticmethod
    @MatchAndRevise(prefix="ATC", needs_vocab=True)
    def process_atc(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        from ..mappings import get_atc_code_to_desc
        code_to_desc = get_atc_code_to_desc()
        temp_cols = ["pfx", "4", "sfx"]
        code_prefixes = ["ATC//", "ATC//4//", "ATC//SFX//"]
        code_slices = [(0, 3), (3, 1), (4,)]
    
        df = (
            df.with_columns(
                pl.col("text_value").str.slice(*slice).alias(col)
                for col, slice in zip(temp_cols, code_slices)
            )
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(
                    pl.lit(pfx)
                    + pl.col(col)
                    + (
                        pl.lit("//") + pl.col(col).replace_strict(code_to_desc, default=None)
                        if pfx == code_prefixes[0]
                        else pl.lit("")
                    )
                )
                .alias(col)
                for col, pfx in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )
        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list("code", *temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )

    @staticmethod
    @MatchAndRevise(prefix="ATC//", needs_counts=True, needs_vocab=True)
    def filter_medication_based_on_counts(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None,
        top_n: int = 200, keep_codes_file: str = None
    ) -> pl.DataFrame:
        
        if keep_codes_file is not None:
            codes = open(keep_codes_file, "r").readlines()
        else:
            codes = None

        known_lab_names = list(counts.keys())[:top_n] if vocab is None else vocab
        if codes is not None:
            known_lab_names += [code.strip() for code in codes]
        return df.filter(pl.col("code").is_in(known_lab_names))

class ICUStayData:
    @staticmethod
    @MatchAndRevise(prefix="ICU_")
    def process(df: pl.DataFrame, *, num_quantiles: int = 10) -> pl.DataFrame:
        from ..mappings import get_stay_id_to_sofa_mapping

        stay_id_to_sofa = get_stay_id_to_sofa_mapping()
        min_value, max_value = min(stay_id_to_sofa.values()), max(stay_id_to_sofa.values())
        bins = np.linspace(min_value, max_value, num_quantiles + 1)
        values = [
            np.arange(np.ceil(left), np.floor(right) + 1)
            for left, right in zip(bins[:-1], bins[1:])
        ]

        # these are not real quantiles, the values are divided equidistantly
        value_to_quantile = {
            value: f"Q{i}" for i, values in enumerate(values, 1) for value in values
        }

        stay_id_to_sofa = {
            stay_id: value_to_quantile[sofa] for stay_id, sofa in stay_id_to_sofa.items()
        }
        return (
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                code=pl.col("code").list[0],
                text_value=pl.lit("ICU_TYPE//") + pl.col("code").list[1],
                sofa_quantiles=pl.col("icustay_id").replace_strict(stay_id_to_sofa, default=None),
            )
            .with_columns(
                pl.when(code=ST.ICU_ADMISSION)
                .then(
                    pl.concat_list(
                        "code",
                        "text_value",
                        pl.when(pl.col("sofa_quantiles").is_not_null())
                        .then(pl.concat_list(pl.lit(ST.SOFA), "sofa_quantiles"))
                        .otherwise([]),
                    ).alias("code")
                )
                .otherwise(pl.concat_list("code"))
            )
            .drop("sofa_quantiles")
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix="INFUSION//", apply_vocab=True)
    def retain_only_infusion_with_numeric_result(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(unify_code_names(pl.col('code')))
        )

    @staticmethod
    @MatchAndRevise(prefix="INFUSION//", needs_counts=True, needs_vocab=True)
    def filter_infusion_based_on_counts(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None,
        top_n: int = 40, keep_codes_file: str = None
    ) -> pl.DataFrame:
        
        if keep_codes_file is not None:
            codes = open(keep_codes_file, "r").readlines()
        else:
            codes = None

        known_infusion_names = list(counts.keys())[:top_n] if vocab is None else vocab
        if codes is not None:
            known_infusion_names += [code.strip() for code in codes]
        return df.filter(pl.col("code").is_in(known_infusion_names))

    @staticmethod
    @MatchAndRevise(prefix="INFUSION//")
    def make_infusion_quantiles(
        df: pl.DataFrame
    ) -> pl.DataFrame:
        prefix = "INFUSION//"
        return (
            df.with_columns(
                pl.concat_list(
                    "code", pl.lit(prefix + "Q//") + pl.col("code").str.slice(len(prefix))
                )
            ).explode("code")
        )


class TransferData:
    @staticmethod
    @MatchAndRevise(prefix="TRANSFER//", apply_vocab=True)
    def unify_transfer_names(df: pl.DataFrame) -> pl.DataFrame:
        ### Rename codes with TRANSFER//admit//DEP to TRANSFER//DEP
        ### Rename codes with TRANSFER//discharge//UNKNOWN to TRANSFER//UNKNOWN
        ### Rename codes with TRANSFER_TO//ED//Emergency Department to TRANSFER//Emergency Department
        df = (
            df.with_columns(unify_code_names(pl.col('code')))
            .with_columns(
                code=pl.col('code').str.split("//").list[0],
                text_value=pl.col('code').str.split("//").list[1],
            )
        )
        
        unique_transfer_codes = df.select("text_value").unique().to_numpy().flatten()
        RENAME_MAPPING = {
            "MED/SURG": "MEDICAL/SURGICAL",
            "NEURO": "NEUROLOGY",
            'SURGICAL_INTERMEDIATE': 'SURGERY',
            'THORACIC_SURGERY': 'SURGERY',
        }

        STRICT_MAPPING = {}
        for code in unique_transfer_codes:
            for key, value in RENAME_MAPPING.items():
                if code.startswith(key):
                    STRICT_MAPPING[code] = value
                    break
            

        # Check ICU in name
        WARD_NAMES = [
           "CARDIAC", "CARDIOLOGY", "CORONARY_CARE_UNIT",
            "DISCHARGE_LOUNGE", "EMERGENCY_DEPARTMENT", "HEMATOLOGY",
            "ICU", "LABOR_&_DELIVERY", "MEDICAL/SURGICAL",
            'MEDICINE', 'NEUROLOGY', 
            'NURSERY', 'OBSERVATION', 'OBSTETRICS', 'PACU', 'PSYCHIATRY',
            'ONCOLOGY', 'SPECIAL_CARE_NURSERY', 'SURGERY', 'TRANSPLANT',
            'UNKNOWN', 'VASCULAR'
        ]

        STRICT_MAPPING2 = {}
        for code in unique_transfer_codes:
            for ward in WARD_NAMES:
                if code.startswith(ward):
                    STRICT_MAPPING2[code] = ward
                    break
        
        df = (
            df
            .with_columns(
                text_value=pl.col('text_value').replace_strict(
                    STRICT_MAPPING, 
                    default=pl.col('text_value')
                )
            )
            .with_columns(
                text_value=pl.when(pl.col("text_value").str.contains("ICU"))
                .then(pl.lit("ICU"))
                .otherwise(pl.col("text_value"))
            )
            .with_columns(
                text_value=pl.col('text_value').replace_strict(
                    STRICT_MAPPING2, 
                    default=pl.col('text_value')
                )
            )
            .with_columns(
                code=pl.when(pl.col('text_value').str == "UNKNOWN") # Transfer to unknown equals to discharge, thus remove 
                .then(None)
                .otherwise(pl.lit("TRANSFER//") + pl.col('text_value'))
            )
        ).filter(pl.col("code").is_not_null())
        return df


class BMIData:
    @staticmethod
    @MatchAndRevise(prefix="BMI")
    def make_quantiles(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("text_value").cast(str).cast(float).alias("numeric_value"),
                pl.lit(None).alias("text_value"),
            )
            .filter(pl.col("numeric_value").is_between(10, 100))
            .with_columns(pl.concat_list(pl.lit("BMI"), pl.lit("BMI//Q")).alias("code"))
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix=["BMI", "Q"])
    def join_token_and_quantile(df: pl.DataFrame) -> pl.DataFrame:
        """TODO: This one should be done in transform_to_quantiles"""
        q_following_bmi_mask = (pl.col("code") == "BMI").shift(1)
        return df.with_columns(
            code=pl.when(q_following_bmi_mask)
            .then(pl.lit("BMI//") + pl.col("code"))
            .when(pl.col("code") == "BMI")
            .then(None)
            .otherwise("code")
        ).drop_nulls("code")


class LabData:
    @staticmethod
    @MatchAndRevise(prefix="LAB//", apply_vocab=True)
    def retain_only_test_with_numeric_result(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(unify_code_names(pl.col('code')))
        )
    
    @staticmethod
    @MatchAndRevise(prefix="LAB//", needs_counts=True, needs_vocab=True)
    def filter_based_on_counts(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None,
        top_n: int = 200, keep_codes_file:str = None
    ) -> pl.DataFrame:

        if keep_codes_file is not None:
            codes = open(keep_codes_file, "r").readlines()
        else:
            codes = None

        known_lab_names = list(counts.keys())[:top_n] if vocab is None else vocab
        if codes is not None:
            known_lab_names += [code.strip() for code in codes]
        return df.filter(pl.col("code").is_in(known_lab_names))

    @staticmethod
    @MatchAndRevise(prefix="LAB//")
    def make_quantiles(
        df: pl.DataFrame
    ) -> pl.DataFrame:
        # TODO: Mark outlier values, use for that some utilities from MEDS_transforms.
        # known_lab_names = list(counts.keys())[:200] if vocab is None else vocab
        return (
            df.with_columns(pl.concat_list("code", pl.lit("LAB//Q//") + pl.col("code").str.slice(5)))
            .explode("code")
        )


class HCPCSData:
    @staticmethod
    @MatchAndRevise(prefix="HCPCS//", apply_vocab=True)
    def unify_names(df: pl.DataFrame) -> pl.DataFrame:
        """This will just unify the code names."""
        return df


class PatientFluidOutputData:
    @staticmethod
    @MatchAndRevise(prefix="SUBJECT_FLUID_OUTPUT//", needs_vocab=True)
    def make_quantiles(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        if vocab is not None:
            df.filter(pl.col("code").is_in(vocab))

        prefix = "SUBJECT_FLUID_OUTPUT//"
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                pl.concat_list(
                    "code", pl.lit(prefix + "Q//") + pl.col("code").str.slice(len(prefix))
                )
            )
            .explode("code")
        )


class EdData:
    @staticmethod
    @MatchAndRevise(prefix="ED_REGISTRATION")
    def process_ed_registration(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list(
                "code",
                pl.lit("ED_TRANSPORT//")
                + pl.when(pl.col.text_value == "HELICOPTER")
                .then(pl.lit("OTHER"))
                .otherwise("text_value"),
            )
        ).explode("code")

    @staticmethod
    @MatchAndRevise(prefix="ACUITY")
    def process_ed_acuity(df: pl.DataFrame) -> pl.DataFrame:
        # TODO: This shouldn't be quantiles, but rather categories
        # TODO: it also can be missing
        return df.with_columns(
            code=pl.lit("VITAL//ED_ACUITY")
        )


class TokenTypeIDs:
    @staticmethod
    @MatchAndRevise(prefix="")
    def add_token_type_ids(df: pl.DataFrame) -> pl.DataFrame:
        """
        Add token type ids to the dataframe based on the mapping dictionary.
        """

        # Create a mapping dictionary for token type ids
        expr = "pl"
        for prefix, value in TOKEN_TYPES_IDS.items():
            expr = expr + f'.when(pl.col("code").str.starts_with("{prefix}")).then({value})'
        expr += ".otherwise(0)"

        # Create a new column for token type ids
        df = df.with_columns(
            eval(expr).alias(TensorizedColumns.TOKEN_TYPES)
        )

        return df

class AdmissionData:

    @staticmethod
    @MatchAndRevise(prefix="", needs_resorting=True) # Assuming resorting is needed after adding rows
    def add_missing_hospital_admission(df: pl.DataFrame) -> pl.DataFrame:
        """
        Add missing <HOSPITAL_ADMISSION> tokens to the dataframe over subject_id and admission_id.
        Look for each unique subject_id and admission_id and add the HOSPITAL_ADMISSION token
        to be beginning of the sequence if it is not already present, Assign the time
        as the 00:00:00 of the beginning date.
        """
        # df = df.with_row_index(MatchAndRevise.index_col)

        has_admission_df = (
            df
            .filter(pl.col("code") == ST.ADMISSION)
            .select([MatchAndRevise.index_col, "subject_id", "admission_id"])
        )
        # Get indexes of the admission that do not have the <HOSPITAL_ADMISSION> token
        # Which is in df but not in has_admission_df
        missing_admission_df = df.join(
            has_admission_df,
            on=["subject_id", "admission_id"],
            how="anti"
        )

        # Find first record of each missing admission to get timestamp
        missing_groups = (
            missing_admission_df.sort(["subject_id", "admission_id", "time"])
            .group_by(["subject_id", "admission_id"], maintain_order=True)
            .agg(
                pl.min("time").alias("min_time"),
                pl.first(MatchAndRevise.index_col).alias(MatchAndRevise.index_col)
            )
        )

        # Create new rows for the HOSPITAL_ADMISSION tokens
        admission_rows = missing_groups.with_columns(
            pl.col(MatchAndRevise.index_col).alias(MatchAndRevise.index_col),
            code=pl.lit(ST.ADMISSION),
            time=pl.col("min_time").dt.truncate("1d"),  # Set to 00:00:00 of the earliest date
            # Add other necessary columns with null values as needed
            **{
                col: pl.lit(None, dtype=dt) for col, dt in zip(df.columns, df.dtypes) if col not in [
                MatchAndRevise.index_col, "subject_id", "admission_id", "time", "code"
            ]}
        )

        # Select columns that match the original dataframe
        admission_rows = admission_rows.select([col for col in df.columns if col in admission_rows.columns])

        # Concatenate the new rows with the original dataframe and sort
        df = pl.concat([df, admission_rows], how="vertical").sort(
            [MatchAndRevise.index_col, "subject_id", "admission_id", "time"]
        )

        return df

    @staticmethod
    @MatchAndRevise(prefix="", needs_resorting=False)
    def add_admission_time_offsets_as_column(df: pl.DataFrame) -> pl.DataFrame:
        """
        calculate time offsets based on the time between each activity with nearest prior HOSPITAL_ADMISSION token
        """
        # First sort data by patient, admission_id, and time to ensure proper sequencing
        # Compute the time difference between consecutive events within the same admission
        df = (
            df.with_columns(
                is_admission_token = pl.when(pl.col("code") == ST.ADMISSION)
                                   .then(1)
                                   .otherwise(0)
            )
            .with_columns(
                time_diff = pl.col("time")
                         .diff()
                         .dt.total_days()
                         .fill_null(0)
                         .over(["subject_id", "admission_id"])
            )
            .with_columns(
                time_diff = pl.col('time_diff')
            )
            .with_columns(
                # Create a group identifier that changes after each admission token
                group_id = pl.col("is_admission_token").cum_sum().over(["subject_id", "admission_id"])
            )
            .with_columns(
            pl.when(pl.col("is_admission_token") == 1)
                .then(0)  # Reset to 0 at admission tokens
                .otherwise(pl.col("time_diff"))
                .cum_sum()
                .over(["subject_id", "admission_id", "group_id"])
                .cast(pl.Int32)
                .alias(TensorizedColumns.ADMISSION_TIME_OFFSET)
            )
        )
        
        # Drop the temporary columns
        df = df.drop('is_admission_token', 'time_diff', 'group_id')
        
        return df