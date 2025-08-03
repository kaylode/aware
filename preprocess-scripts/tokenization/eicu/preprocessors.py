import polars as pl
import json
from nosos.utilities.constants import SpecialToken as ST
from ..patterns import MatchAndRevise
from ..utils import  unify_code_names
import scripts.tokenization.mimic as mimiciv

class AdmissionData(mimiciv.AdmissionData):
    pass

class GBMSDemographicData(mimiciv.GBMSDemographicData):
    pass

class LabData(mimiciv.LabData):
    pass

class DiagnosesData(mimiciv.DiagnosesData):

    @staticmethod
    @MatchAndRevise(prefix="DIAGNOSIS//ICD//")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
                pl.col.code.str.split_exact("//", 3)
            ).with_columns(
                code=pl.lit("ICD//"),
                text_value = pl.col.code.struct[-1], 
        )
    
    @staticmethod
    @MatchAndRevise(prefix="ICD//")
    def convert_icd_9_to_10(df: pl.DataFrame) -> pl.DataFrame:
        from nosos.datasets.cohort.cohort import (
            JSON_ICD_CM_MAPPING, 
            JSON_ICD_PCS_MAPPING
        )
        from ..mappings import (
            get_icd_cm_9_to_10_mapping,
            get_icd_pcs_9_to_10_mapping
        )

        icd_cm_9_to_10 = get_icd_cm_9_to_10_mapping()
        icd_pcs_9_to_10 = get_icd_pcs_9_to_10_mapping()
        icd_cm_10s = json.load(open(JSON_ICD_CM_MAPPING, 'r')).keys()
        icd_pcs_10s = json.load(open(JSON_ICD_PCS_MAPPING, 'r')).keys()
        icd_cm_9s = icd_cm_9_to_10.keys()
        icd_pcs_9s = icd_pcs_9_to_10.keys()

        all_code_conversion = {
            **icd_cm_9_to_10,
            **icd_pcs_9_to_10,
        }

        df = df.with_columns(
            text_value=unify_code_names(pl.col("text_value"))
        )

        
        df = df.with_columns(
            code=pl.when(pl.col("text_value").is_in(icd_cm_9s))
            .then(pl.lit("ICD//CM"))
            .when(pl.col("text_value").is_in(icd_pcs_9s))
            .then(pl.lit("ICD//PCS"))
            .when(pl.col("text_value").is_in(icd_cm_10s))
            .then(pl.lit("ICD//CM"))
            .when(pl.col("text_value").is_in(icd_pcs_10s))
            .then(pl.lit("ICD//PCS"))
            .otherwise(pl.lit("ICD//CM")),
        ).with_columns(
            text_value = pl.col("text_value").replace_strict(
                all_code_conversion, 
                default=pl.col("text_value")
            ),
        ).with_columns(
            text_value = (
                pl.when(pl.col("text_value").str.len_chars() == 0)
                .then(None)
                .otherwise(pl.col("text_value"))
            )
        )
    
        df = df.drop_nulls("text_value").with_columns(
            code=pl.when(pl.col("code").str.contains("ICD//CM"))
            .then(pl.lit("ICD//CM//10") + pl.lit("//") + pl.col("text_value"))
            .when(pl.col("code").str.contains("ICD//PCS"))
            .then(pl.lit("ICD//PCS//10") + pl.lit("//") + pl.col("text_value"))
            .otherwise(pl.col("code"))
        )
        return df

class ProcedureData(mimiciv.ProcedureData):
    pass

class DeathData(mimiciv.DeathData):
    pass

class DemographicData(mimiciv.DemographicData):
    @staticmethod
    @MatchAndRevise(prefix="RACE", apply_vocab=True)
    def process_race(df: pl.DataFrame) -> pl.DataFrame:

        mapping = {
            'Native American': 'WHITE',
            'Hispanic': 'HISPANIC',
            'Caucasian': 'WHITE',
            'African American': 'OTHER',
            'Asian': 'OTHER',
            'Other/Unknown': 'UNKNOWN',
        }


        return (
            df.with_columns(
                code=pl.col("code").str.split('//')
            ).with_columns(
                code=pl.when(pl.col("code").list[1] == "")
                .then(pl.lit("UNKNOWN"))
                .otherwise(pl.col("code").list[1])
            )
        ).with_columns(
            code="RACE//"+pl.col("code").replace_strict(
                mapping, default=pl.lit("UNKNOWN"), 
                return_dtype=pl.String
            )
        )
   

class InpatientData(mimiciv.InpatientData):

    @staticmethod
    @MatchAndRevise(prefix=ST.ADMISSION)
    def process_hospital_admissions(df: pl.DataFrame) -> pl.DataFrame:

        return (
            df.with_columns(
                pl.col.code.str.split("//").list[0].alias("code"),
                pl.col.code.str.split("//").list[1].alias("text_value"),
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.when(pl.col("text_value").str.contains("ICU"))
                    .then(pl.lit("ICU_ADMISSION"))
                    .otherwise(
                        pl.lit("ADMISSION_TYPE//")
                        + pl.when(
                            pl.col("text_value").str.to_lowercase().str.starts_with("emergency")
                        )
                        .then(pl.lit("EMERGENCY"))
                        .otherwise(pl.lit("OBSERVATION"))
                    ),
                ).alias("code")
            )
            .explode("code")
        ).drop_nulls("code").with_columns(
            code=unify_code_names(pl.col("code")),
        )

    @staticmethod
    @MatchAndRevise(prefix=[ST.DISCHARGE])
    def process_hospital_discharges(df: pl.DataFrame) -> pl.DataFrame:
        """Currently must be run before processing diagnoses."""

        discharge_facilities = [
            "Skilled Nursing Facility",
            "Nursing Home",
            "Operating Room",
            "Rehabilitation",
        ]


        return (
            df.with_columns(
                pl.col.code.str.split("//").list[0].alias("code"),
                pl.col.code.str.split("//").list[1].alias("text_value"),
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.when(pl.col("text_value").str.contains("ICU"))
                    .then(pl.lit("ICU_DISCHARGE"))
                    .otherwise(
                        pl.lit("DISCHARGE_LOCATION//")
                            + pl.when(pl.col("text_value").is_in(discharge_facilities))
                            .then(pl.lit("HEALTHCARE_FACILITY"))
                            .when((pl.col("text_value").is_null()) | (pl.col("text_value") == ""))
                            .then(pl.lit("UNKNOWN"))
                            .when(pl.col("text_value").str.to_uppercase().str.contains("HOME"))
                            .then(pl.lit("HOME"))
                            .when(pl.col("text_value").str.to_uppercase().str.contains("DEATH"))
                            .then(pl.lit("DEATH"))
                            .otherwise(pl.lit("OTHER"))
                    )
                ).alias("code"))
            .explode("code")
        ).filter(pl.col.code != "DISCHARGE_LOCATION//DEATH")


class MeasurementData(mimiciv.MeasurementData):
    @staticmethod
    @MatchAndRevise(prefix=[
        "TEMPERATURE", "HEART_RATE", 
        "RESPIRATORY_RATE", "O2_SATURATION",
        "CVP", "ETCO2", "SYSTEMIC_SYSTOLIC", "SYSTEMIC_DIASTOLIC",
        "SYSTEMIC_MEAN", "PULM_ART_SYSTOLIC", "PULM_ART_DIASTOLIC", 
        "PULM_ART_MEAN", "ST1", "ST2", "ST3", "ICP",
        "CARDIAC_INPUT",
        "NONINVASIVE_SYSTOLIC",
        "SVRI",
        "CARDIAC_OUTPUT",
        "PVRI",
        "NONINVASIVE_MEAN",
        "PVR",
        "SVR",
        "NONINVASIVE_DIASTOLIC",
        "PAOP"
        ])
    def process_simple_measurements(df: pl.DataFrame) -> pl.DataFrame:
        # temperature can be missing, account for it

        mapping = {
            "NONINVASIVE_DIASTOLIC": "BLOOD_PRESSURE_DBP",
            "NONINVASIVE_SYSTOLIC": "BLOOD_PRESSURE_SBP",
        }

        return df.with_columns(
            code=pl.col("code").replace_strict(
                mapping, default=pl.col("code"), return_dtype=pl.String
            )
        ).with_columns(
            code=pl.lit("VITAL//") + pl.col("code")
        )


class MedicationData(mimiciv.MedicationData):
    pass

class ICUStayData(mimiciv.ICUStayData):
    pass