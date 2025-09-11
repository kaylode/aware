import os
import sys

src_path = os.path.abspath('../..')
print(src_path)
sys.path.append(src_path)
from src.utils import create_directory, dump_pickle, raw_data_path, processed_data_path
import pickle
import torch
import pandas as pd


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

import json

apacheapsvar = json.load(open("/home/mpham/workspace/source/ehrpfn/data/processed_muse/eicu/apacheapsvar.json"))
item_mapping = json.load(open("/home/mpham/workspace/source/ehrpfn/data/processed_muse/eicu/itemid_to_index.json"))
all_hosp_adm_dict = load_pickle("/home/mpham/workspace/source/ehrpfn/data/processed_muse/eicu/icu_stay_dict.pkl")

input_path = os.path.join(raw_data_path, "eicu-crd/2.0")
output_path = os.path.join(processed_data_path, "eicu")

from tqdm import tqdm
all_treatment = []

for icu_id in tqdm(all_hosp_adm_dict.keys()):
    row = []
    icu_stay = all_hosp_adm_dict[icu_id]
    types = icu_stay.trajectory[0]
    codes = icu_stay.trajectory[1]

    for code, type in zip(codes, types):
        if type == "treatment":
            all_treatment.extend(code.split('|'))

all_treatment = list(set(all_treatment))
print(len(all_treatment))

all_codes = []

for icu_id in all_hosp_adm_dict.keys():
    row = []
    icu_stay = all_hosp_adm_dict[icu_id]
    codes = icu_stay.diagnosis_codes

    for code in codes:
        all_codes.extend(code[-1])

all_codes = sorted(list(set(all_codes)))
print(len(all_codes))


APACHEVARS = [
    "intubated",
    "vent",
    "dialysis",
    "eyes_0",
    "eyes_1",
    "eyes_2",
    "eyes_3",
    "motor_0",
    "motor_1",
    "motor_2",
    "motor_3",
    "motor_4",
    "motor_5",
    "verbal_0",
    "verbal_1",
    "verbal_2",
    "verbal_3",
    "verbal_4",
    "meds",
    "urine",
    "wbc",
    "temperature",
    "respiratoryrate",
    "sodium",
    "heartrate",
    "meanbp",
    "ph",
    "hematocrit",
    "creatinine",
    "albumin",
    "pao2",
    "pco2",
    "bun",
    "glucose",
    "bilirubin",
    "fio2"
]

# Write to all_data.csv and label.csv
from tqdm import tqdm

# Convert to pandas DataFrame
demo_df = []
label_df = []

pbar = tqdm(total=len(all_hosp_adm_dict.keys()))

for idx, icu_id in enumerate(all_hosp_adm_dict.keys()):
    pbar.update(1)
    icu_stay = all_hosp_adm_dict[icu_id]
    age = str(icu_stay.age)
    gender = icu_stay.gender
    ethnicity = icu_stay.ethnicity
    diagnosis_codes = icu_stay.diagnosis_codes
    labvectors = icu_stay.labvectors
    # apacheapsvar = icu_stay.apacheapsvar
    # if apacheapsvar is None: continue
    admission_id = icu_stay.admission_id 
    patient_id = icu_stay.patient_id  # str
    hospital_id = icu_stay.hospital_id  # int
    lab_names = icu_stay.lab_names
    vital_names = icu_stay.vital_names
    vitalvectors = icu_stay.vitalvectors
    apacheapsvar_names = icu_stay.apacheapsvar_names
    mortality = float(icu_stay.mortality)
    readmission = float(icu_stay.readmission)


    types = icu_stay.trajectory[0]
    codes = icu_stay.trajectory[1]

    treatments = []
    for code, type in zip(codes, types):
        if type == "treatment":
            treatments.extend(code.split('|'))

    row = {
        'icu_id': icu_id,
        'admission_id': admission_id,
        'patient_id': patient_id,
        'hospital_id': hospital_id,
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
    }

    label = {
        'icu_id': icu_id,
        'admission_id': admission_id,
        'patient_id': patient_id,
        'hospital_id': hospital_id,
        'mortality': mortality,
        'readmission': readmission
    }

    for col_name, col_id in lab_names.items():
        for type in ['min', 'max', 'mean', 'std']:
            if labvectors is not None:
                if type == 'min':
                    row[f'LAB//{col_name}_{type}'] = labvectors[:, col_id].min()
                elif type == 'max':
                    row[f'LAB//{col_name}_{type}'] = labvectors[:, col_id].max()
                elif type == 'mean':
                    row[f'LAB//{col_name}_{type}'] = labvectors[:, col_id].mean()
                elif type == 'std':
                    row[f'LAB//{col_name}_{type}'] = labvectors[:, col_id].std()
            else:
                row[f'LAB//{col_name}_{type}'] = None

    for col_id, col_name in enumerate(vital_names):
        for type in ['min', 'max', 'mean', 'std']:
            if vitalvectors is not None:
                if type == 'min':
                    row[f'VITAL//{col_name}_{type}'] = vitalvectors[:, col_id].min()
                elif type == 'max':
                    row[f'VITAL//{col_name}_{type}'] = vitalvectors[:, col_id].max()
                elif type == 'mean':
                    row[f'VITAL//{col_name}_{type}'] = vitalvectors[:, col_id].mean()
                elif type == 'std':
                    row[f'VITAL//{col_name}_{type}'] = vitalvectors[:, col_id].std()
            else:
                row[f'VITAL//{col_name}_{type}'] = None



    for col_id, col_name in enumerate(all_treatment):
        if col_name in treatments:
            row[f'TREATMENT//{col_name}'] = 1
        else:
            row[f'TREATMENT//{col_name}'] = 0

    # for col_id, col_name in enumerate(APACHEVARS):
    #     if apacheapsvar is not None:
    #         row[f'APACHE//{col_name}'] = apacheapsvar[col_id]
    #     else:
    #         row[f'APACHE//{col_name}'] = None

    row.update({
        "ICD//" + k: 1 if k in diagnosis_codes else 0 for k in all_codes
    })

    row = {'_'.join(k.upper().replace(' - ', '_').split()): v for k, v in row.items()}
    label = {'_'.join(k.upper().replace(' - ', '_').split()): v for k, v in label.items()}

    demo_df.append(row)
    label_df.append(label)

pbar.close()


# After your for-loop (remove the break to process all ICU stays)
demo_df = pd.DataFrame(demo_df)
label_df = pd.DataFrame(label_df)

# Save to CSV
demo_df.to_csv("all_data.csv", index=False)
label_df.to_csv("all_labels.csv", index=False)