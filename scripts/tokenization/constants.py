from enum import StrEnum
from importlib.resources import files
from pathlib import Path

import os

class TensorizedColumns(StrEnum):
    """
    The columns that are tensorized and saved in the .safetensors files.
    """
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
    # SUBJECT_ID = "subject_id"
    # TIMES = "times"
    # TOKENS = "tokens"
    # PATIENT_IDS = "patient_ids"
    # PATIENT_OFFSETS = "patient_offsets"
    ADMISSION_ID = "admission_id"
    DICOM_ID = "dicom_id"
    HADM_ID = "hadm_id"
    ICU_STAY_ID = "icustay_id"
    LABELS = "labels"
    AGE_OFFSET = "age_offset"
    TOKEN_TYPES="token_type_ids"
    ADMISSION_TIME_OFFSET = "adm_time_offset"


TOKEN_TYPES_IDS = {
    "[CLS]": 0,
    "[SEP]": 0,
    "[MASK]": 0,
    "[PAD]": 0,
    "[UNK]": 0,
    "TIME//": 0,
    "LAB//": 1,
    "VITAL//": 2,
    "ED_ACUITY//": 2,
    "MEDICATION//": 3,
    "INFUSION//": 3,
    "ATC//": 3,
    "ICD//CM": 4,
    "DRG//": 4,
    "DIAGNOSIS//": 4,
    "ICD//PCS": 5,
    "PROCEDURE//": 5,
    "PROCEDURES//": 5,
    "ICU_ADMISSION//": 6,
    "ICU_DISCHARGE//": 6,
    "TRANSFER//": 6,
    "ADMISSION_TYPE//": 6,
    "DISCHARGE_LOCATION//": 6,
    "ED_REGISTRATION//": 6,
    "ED_OUT//": 6,
    "GENDER//": 7,
    "MARITAL//": 7,
    "INSURANCE//": 7,
    "RACE//": 7,
    "QUANTIZATION//": 8,
}

class ModelType(StrEnum):
    DECODER = "gpt2"
    ENC_DECODER = "baseline2"
    DEEPSEEKV2 = "deepseekv2"
    GPT2LocalAttention = "gpt2la"
    NOSOS = "nosos"

class SpecialToken(StrEnum):
    DOB = "MEDS_BIRTH"
    DEATH = "MEDS_DEATH"
    TIMELINE_END = "TIMELINE_END"

    # hospital adm and dc
    ADMISSION = "HOSPITAL_ADMISSION"
    DISCHARGE = "HOSPITAL_DISCHARGE"

    # only MIMIC
    ICU_ADMISSION = "ICU_ADMISSION"
    ICU_DISCHARGE = "ICU_DISCHARGE"
    ED_ADMISSION = "ED_REGISTRATION"
    ED_DISCHARGE = "ED_OUT"
    SOFA = "SOFA"

    ## HCAI
    START_ICD = "START_ICD"
    END_ICD = "END_ICD"

    ## SEPARATORS
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"
    PAD = "[PAD]"
    UNKNOWN = "[UNK]"


STATIC_DATA_FN = "static_data.pickle"


from enum import StrEnum


class Task(StrEnum):
    HOSPITAL_MORTALITY = "hospital_mortality"
    HOSPITAL_MORTALITY_SINGLE = "hospital_mortality_single"
    READMISSION = "readmission"

    DRG_PREDICTION = "drg"
    SOFA_PREDICTION = "sofa"
    ICU_MORTALITY = "icu_mortality"

    ICU_READMISSION = "icu_readmission"

    ICU_ADMISSION = "icu_admission"
    ICU_ADMISSION_SINGLE = "icu_admission_single"

    # from the ED benchmark paper
    ED_HOSPITALIZATION = "ed_hospitalization"
    ED_CRITICAL_OUTCOME = "ed_critical_outcome"
    # the one below is called "ED reattendance" in the ED-Benchmark paper
    ED_REPRESENTATION = "ed_representation"

    ICD_PREDICTION = "icd_prediction_this_admission"
    ICD_PREDICTION_NEXT_TW = "icd_prediction_next_time_window"


class Reason(StrEnum):
    GOT_TOKEN = "token_of_interest"
    KEY_ERROR = "key_error"
    TIME_LIMIT = "time_limit"
