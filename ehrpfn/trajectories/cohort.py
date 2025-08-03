from typing import *
import json
import os
import polars as pl
from tqdm import tqdm

from importlib.resources import files
from pathlib import Path
MAPPINGS_DIR: Path = files("preprocess-scripts.tokenization.maps")

_COMPLEMENTARY_CODE_TO_NAME = {
    "F32": "Major depressive disorder, single episode",
    "F32A": "Depression, unspecified",
    "G928": "Other toxic encephalopathy",
    "G929": "Unspecified toxic encephalopathy",
    "I5A": "Non-ischemic myocardial injury (non-traumatic)",
    "K31A": "Gastric intestinal metaplasia",
    "L24A9": "Irritant contact dermatitis due to friction or contact with body fluids",
    "L24B": "Irritant contact dermatitis related to stoma or fistula",
    "M350": "Sicca syndrome [Sjogren]",
    "P099": "Abnormal findings on neonatal screening, unspecified",
    "R051": "Acute cough",
    "R052": "Subacute cough",
    "R053": "Chronic cough",
    "R054": "Cough syncope",
    "R058": "Other specified cough",
    "R059": "Cough, unspecified",
    "U09": "Post COVID-19 condition",
    "U099": "Post COVID-19 condition, unspecified",
    "W458X": "Lid of can entering through skin",
    "Z09": "Encounter for follow-up examination after completed treatment",
    "Z2252": "Carrier of viral hepatitis",
}


def get_icd_cm_code_to_name_mapping():
    df = (
        pl.read_csv(MAPPINGS_DIR / "icd10cm-order-Jan-2021.csv.gz")
        .group_by("code")
        .agg(pl.col("long").first())
    )
    out_dict = dict(zip(df["code"], df["long"]))
    out_dict.update(_COMPLEMENTARY_CODE_TO_NAME)
    return out_dict


# Path to the JSON file containing ICD-10 codes and their corresponding disease names
JSON_ICD_MAPPING = MAPPINGS_DIR / 'icd10_mapping.json'
JSON_ICD_CM_MAPPING = MAPPINGS_DIR / 'icd10_cm_mapping.json'
JSON_ICD_PCS_MAPPING = MAPPINGS_DIR / 'icd10_pcs_mapping.json'
try:
    if not os.path.isfile(JSON_ICD_MAPPING):
        import pandas as pd
        df = pd.read_csv(MAPPINGS_DIR / "d_icd_procedures.csv.gz", compression='gzip')
        df2 = pd.read_csv(MAPPINGS_DIR / "d_icd_diagnoses.csv.gz", compression='gzip')

        df = df[df['icd_version'] == 10]
        df2 = df2[df2['icd_version'] == 10]

        df['icd_name'] = df.apply(lambda x: str(x['icd_code']), axis=1)
        df2['icd_name'] = df2.apply(lambda x: str(x['icd_code']), axis=1)
        jsons = df[['icd_name', 'long_title']].to_dict(orient='records')
        jsons2 = df2[['icd_name', 'long_title']].to_dict(orient='records')

        icd_psc_json_mapping = {
            i['icd_name']:i['long_title'] for i in jsons
        }
        icd_cm_json_mapping = {
            i['icd_name']:i['long_title'] for i in jsons2
        }
        icd_cm_json_mapping2 = get_icd_cm_code_to_name_mapping()
        icd_cm_json_mapping.update(icd_cm_json_mapping2)

        all_icd_json_mapping = {
            **icd_psc_json_mapping,
            **icd_cm_json_mapping
        }
        all_icd_json_mapping = {
            k:v for k,v in list(sorted(all_icd_json_mapping.items(), key=lambda item: item[0]))
        }

        json.dump(icd_cm_json_mapping, open(JSON_ICD_CM_MAPPING, 'w'), indent=4)
        json.dump(icd_psc_json_mapping, open(JSON_ICD_PCS_MAPPING, 'w'), indent=4)
        json.dump(all_icd_json_mapping, open(JSON_ICD_MAPPING, 'w'), indent=4)

except Exception as e:
    print(f"Error: {e}")
    print("Cannot find the JSON file. Please check the path.")
    pass

ICD_GENERAL_DISEASES_MAPPING = {
    "I50": 'HF',#"Heart failure",
    "N18": 'CKD',#"Chronic Kidney Disease",
    "J44": "COPD",
    "F32": 'DEPR',#"Major depressive disorder, single episode",
    "F23": 'DEPR',#"Psychotic disorder due to another medical condition",
    "F24": 'DEPR',#"Other specified acute and transient psychotic disorders",
    "C25": 'NEOPAN',#"Malignant neoplasm of pancreas, unspecified",
    "C22": 'NEOLIV',#"Malignant neoplasm of liver, unspecified",
    "C71": 'NEOBRAIN',#"Malignant neoplasm of brain, unspecified",
}

ICD_HCAI_MAPPING = {
    "T814": 'SSI', # "Infection following a procedure", 
    "T835": 'UTI', #"Infection and inflammatory reaction due to urinary catheter",
    "N390": 'UTI', #"Urinary tract infection, site not specified",
    "J95851": 'VAP', #''"Ventilator-associated pneumonia",
    "J13": 'VAP', #"Pneumonia due to Streptococcus pneumoniae",
    "J150": 'VAP', #"Pneumonia due to Klebsiella",
    "J15211": 'VAP', #"Pneumonia due to Methicillin-resistant Staphylococcus aureus (MRSA)",
    "T80211": 'CLABSI',#"Bloodstream infection due to a vascular catheter",
    "T80219": 'CLABSI', #"Other infection due to vascular catheter",
    "A419": 'SEPSIS'#"Sepsis, unspecified organism"
}

DISEASE_GROUP_TO_ICDS = {
    "HF": ["I50"],
    "CKD": ["N18"],
    "COPD": ["J44"],
    "DEPR": ["F32", "F23", "F24"],
    "CANCER_PAN": ["C25"],
    "CANCER_LIVER": ["C22"],
    "CANCER_BRAIN": ["C71"],
    "SEPSIS": ["A419", "A410", "A411", "A412", "A413", "A414", "A415", "A4151", "A4152", "A4153", "A4189", "A4181"],
    "SSI": ["T814", "T8189"],
    "UTI": ["T835", "N390", "N30", "N34", "N398", "N399", "Z87440", "O2340"],
    "VAP": ["J95851", "J13", "J150", "J15211"],
    "BSI": ["T80211", "T80219", "T80212", "T80218"],
}
DISEASE_GROUP_TO_ICDS["HCAI"] = (
    DISEASE_GROUP_TO_ICDS["SSI"] 
    + DISEASE_GROUP_TO_ICDS["SEPSIS"] 
    + DISEASE_GROUP_TO_ICDS["UTI"] 
    + DISEASE_GROUP_TO_ICDS["VAP"] 
    + DISEASE_GROUP_TO_ICDS["BSI"]
    + ['Y95', 'A047', "A4902", "B9562", "Z8614", "Z22322", "J15212", "A4102", "A0811"]
)

# Clostridioides difficile (C. diff) infections: C. diff is a bacterium that can cause severe diarrhea and colitis, particularly in patients who have been treated with antibiotics.
# Methicillin-resistant Staphylococcus aureus (MRSA): MRSA is a type of antibiotic-resistant bacteria that can cause serious infections, including bloodstream infections and pneumonia. 
# Gastrointestinal infections: Norovirus, a common cause of stomach flu, can spread easily in hospitals and other healthcare settings. 

GENERAL_DISEASES_ICD_MAPPING = {v: [] for k, v in ICD_GENERAL_DISEASES_MAPPING.items()}
for k, v in ICD_GENERAL_DISEASES_MAPPING.items():
    if v in GENERAL_DISEASES_ICD_MAPPING:
        GENERAL_DISEASES_ICD_MAPPING[v].append(k)
    else:
        GENERAL_DISEASES_ICD_MAPPING[v] = [k]


TIME_WINDOW_MAPPING = {
    "12 hours": "12h-18h",
    "24 hours": "1d-2d",
    "48 hours": "2d-4d",
    "7 days": "7d-12d",
    "1 month": "30d-2mt",
    "6 months": "=6mt",
}


## Functions to get cohort data from JSON dict that starts with codes in dict
def get_cohort_data_from_json(disease_code, icd_mapping_json:str=JSON_ICD_MAPPING):
    """
    Extracts cohort data from a JSON dictionary based on the provided disease code.
    
    Args:
        json_dict (dict): The JSON dictionary containing cohort data.
        disease_code (str): The ICD-10 code for the disease of interest.
        
    Returns:
        dict: A dictionary containing the cohort data for the specified disease code.
    """

    json_dict = json.load(
        open(icd_mapping_json, 'r')
    )
    
    cohort_data = {}
    for key, value in json_dict.items():
        if key.startswith(disease_code):
            cohort_data[key] = value
    return cohort_data


def translate_icdcm_tokens(icd_tokens:List):
    """
    Translates ICD tokens to their corresponding disease names.
    
    Args:
        icd_tokens (List): A list of ICD tokens. 
        For example, ["ICD//CM//I50", "ICD//CM//N18"] -> ["I50", "N18"]
        or, ["ICD//CM//I50", "ICD//CM//3-6//9", "ICD//CM//SFX//9"] -> ["I5099"]
        
    Returns:
        List: A list of translated disease names.
    """

    results = []
    current_code = ""
    for token in icd_tokens:
        if token.startswith("ICD//CM//"):
            tokens = token.split("//")
            if len(tokens) == 3:
                if current_code != "":
                    results.append(current_code)
                current_code = ""
                current_code += tokens[-1]
            elif len(tokens) > 3:
                current_code += tokens[-1]
        else:
            if current_code != "":
                results.append(current_code)
            
    return results

def translate_icdpcs_tokens(icd_tokens:List):
    """
    Translates ICD tokens to their corresponding disease names.
    For every 7 consecutive tokens start with "ICD//PCS//", concatenate the last token
    to the previous token.
    Args:
        icd_tokens (List): A list of ICD tokens. 
        For example, ["ICD//PCS//A", "ICD//PCS//1", "ICD//PCS//1", "ICD//PCS//1", "ICD//PCS//1", "ICD//PCS//1"] -> ["I50", "N18"]
        
    Returns:
        List: A list of translated disease names.
    """
    # Split list into chunks of 7
    chunk_size = 7
    chunks = [icd_tokens[i:i + chunk_size] for i in range(0, len(icd_tokens), chunk_size)]
    results = []
    for chunk in chunks:
        results.append(''.join([c[-1] for c in chunk if c.startswith("ICD//PCS//")]))
    return results
    


def tokenize_icd_cm_code(icd_code):
    code_prefixes = ["", "3-6//", "SFX//"]
    code_slices = [(0, 3), (3, 6), (6,7)]

    # Slice a string into parts
    code_parts = [icd_code[slice(*code_slice)] for code_slice in code_slices]
    # Add prefixes to the parts
    code_parts = [f"ICD//CM//{prefix}{part}" for part, prefix in zip(code_parts, code_prefixes) if part]
    # Combine the parts into a single string
    return code_parts



try:
    DISEASE_GROUP_TO_DETAILED_ICDS = {
        k: [x for i in v for x in get_cohort_data_from_json(i).keys()] 
        for k,v in DISEASE_GROUP_TO_ICDS.items()
    }
except:
    pass


REQUIRED_HAI_RELATED_MIMICIV_CODES = [
    'LAB//WHITE_BLOOD_CELLS//K/UL',
    'LAB//HEMOGLOBIN//G/DL',
    'LAB//PLATELET_COUNT//K/UL',
    'LAB//SODIUM_(SERUM)//MEQ/L',
    'LAB//POTASSIUM//MEQ/L',
    'LAB//CREATININE//MG/DL',
    'LAB//GLUCOSE_(SERUM)//MG/DL',
    'LAB//ALBUMIN//G/DL',
    'LAB//BILIRUBIN_TOTAL//MG/DL',
    'LAB//LACTATE//MMOL/L',
    'LAB//MAGNESIUM//MG/DL',
    'LAB//CALCIUM_TOTAL//MG/DL',
    'LAB//CHLORIDE_(SERUM)//MEQ/L',
    'LAB//BICARBONATE//MEQ/L',
    'LAB//PH//UNITS',
    'LAB//NEUTROPHILS//%',
    "LAB//BUN//MG/DL",
    "LAB//ARTERIAL_O2_PRESSURE//MMHG",
    'PROCEDURES//INVASIVE_VENTILATION',
    'PROCEDURES//FOLEY_CATHETER',
    'PROCEDURES//URINE_CULTURE',
    'PROCEDURES//BLOOD_CULTURED',
    'PROCEDURES//ARTERIAL_LINE',
    'PROCEDURES//HEMODIALYSIS',
    'PROCEDURES//DIALYSIS_CATHETER',
]

def get_hcai_dataset_indices(
    dataset,
):
    """
    Get data loaders for the HCAI dataset.
    
    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): The batch size for the data loader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): The number of workers for the data loader.
        pin_memory (bool): Whether to pin memory.
        drop_last (bool): Whether to drop the last batch if it's smaller than the batch size.
        
    Returns:
        DataLoader: The data loader for the HCAI dataset.
    """

    def create_loader(dataset, indices):
        yield from (dataset[i] for i in indices)

    dataloader = create_loader(dataset, range(len(dataset)))

    hcai_indexes = {
        key: [] for key in DISEASE_GROUP_TO_DETAILED_ICDS.keys()
    }
    pbar = tqdm(total=len(dataset), desc="Filtering HCAI samples from the dataset")
    for i, (_, ground_truth) in enumerate(dataloader):
        pbar.update(1)
        labels = ground_truth['expected']
        decoded_labels = translate_icdcm_tokens(labels)

        for disease_group, icds in DISEASE_GROUP_TO_DETAILED_ICDS.items():
            for label in decoded_labels:
                if label in icds:
                    hcai_indexes[disease_group].append(i)
                    break    
    pbar.close()
    return hcai_indexes