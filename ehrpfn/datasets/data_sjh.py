from typing import *
import json
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch
import numpy as np
from ehrpfn.utils import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')


ICD_COLUMNS = [
    "ICD10CMs_3",
    "ICD10PCSs_5",
    "ICD10CMs_3_SOFAR",
    "ICD10PCSs_5_SOFAR",
]

CATEGORICAL_COLUMNS = [
    "NEXT_ADMISSION_WARD",
    "ADMISSION WARD",
    "DISCHARGE WARD",           # Commented out - contains discharge information that may leak mortality/LOS
    "DISCHARGE SPECIALTY",      # Commented out - contains discharge information that may leak mortality
    "DISCHARGE DESCRIPTION",    # Commented out - explicitly describes discharge status (death/transfer)
    "DRG_CODE",
    "MDC_CODE",
    "EPISODE_CPE_SCREEN",      
    "EPISODE_ISOLATED",         # Commented out - isolation often results from infection status
    "GENDER",
    "AREA OF RESIDENCE",
    "INSIDE DUBLIN",
    "INSIDE IRELAND",
]


CONTACT_NETWORK_COLUMNS = [
    "WARD_DEGREE",
    "WARD_DEGREE_CENTRALITY",
    "WARD_INFECTED_DEGREE",
    "WARD_INFECTED_CONTACT_DAYS",
    "WARD_INFECTED_CONTACT_DAYS_CENTRALITY",
    "WARD_CLUSTERING_COEFFICIENT",
    "WARD_CLOSENESS_CENTRALITY",
    "WARD_INFECTED_CLOSENESS_CENTRALITY",
    "WARD_BETWEENNESS_CENTRALITY",
    "WARD_PAGERANK",
    "WARD_K_CORE_NUMBER"
]

NUMERICAL_COLUMNS = [
    "NUMBER_OF_CPE_SCREENS",    
    "NUM_DIAGNOSIS",
    "NUM_PROCEDURES",
    "NUM_WARDS",                
    "EPISODE_NUMBER_EMERGENCY_DEPARTMENT_VISITS",
    "EPISODE_NUMBER_ACUTE_DEPARTMENT_VISITS",
    # "EPISODE_MINIMUM_DAYS_SINCE/UNTIL_SCREEN",  # Commented out - related to CPE screening timing
    # "EPISODE_NUM_OVERLAPPING_SCREEN_DAYS",      # Commented out - related to CPE screening

    "LOS",                     
    "TIME_SINCE_LAST_ADMISSION",
    "READMISSION_SOFAR",
    "NUM_PROCEDURES_SOFAR",
    "NUM_DIAGNOSIS_SOFAR",
    "LOS_SOFAR",
    "NUM_WARDS_SOFAR",
    "EPISODE_NUMBER_ACUTE_DEPARTMENT_VISITS_SOFAR",
    "EPISODE_NUMBER_EMERGENCY_DEPARTMENT_VISITS_SOFAR",
    "NUMBER_OF_CPE_SCREENS_SOFAR", 
] + CONTACT_NETWORK_COLUMNS


DROP_COLUMNS = [

]


def select_features(x, y, selection_method:str, feature_selection_path:str=None, selected_indices=None):
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
                selected_indices = [x.columns.tolist().index(col) for col in top_features if col in x.columns]
        x = x.iloc[:, selected_indices]

    feature_names = x.columns.tolist()
    return x, y, feature_names, selected_indices

def get_sjh_dataset(
        input_folder: str, output_folder: str = None, target_col:str = None, 
        patient_ids: Optional[List[int]] = None,
        feature_selection_path: str = None,
        selection_method: str = None,
    ):

    use_columns = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + ['ICD10CMs_3', 'PIN_NO', 'HIPE_PATIENTID']
    if target_col is not None:
        use_columns.append(target_col)

    global DROP_COLUMNS
    CLASS_NAMES = None
    NUM_CLASSES = None
    match target_col:
        case None:
            pass
        case "WILL_READMIT_WITHIN_30_DAYS":
            target_col = "WILL_READMIT_WITHIN_30_DAYS"
            NUM_CLASSES = 2
            CLASS_NAMES = ["No", "Yes"]
            DROP_COLUMNS.extend([
                "NEXT_ADMISSION_WARD",
            ])
        case "WILL_READMIT_WITHIN_7_DAYS":
            target_col = "WILL_READMIT_WITHIN_7_DAYS"
            NUM_CLASSES = 2
            CLASS_NAMES = ["No", "Yes"]
            DROP_COLUMNS.extend([
                "NEXT_ADMISSION_WARD",
            ])
        case "NEXT_LOS":
            target_col = "NEXT_LOS"
            NUM_CLASSES = 1
            CLASS_NAMES = None
        case "DECEASED_NEXT_EPISODE":
            target_col = "DECEASED_NEXT_EPISODE"
            NUM_CLASSES = 2
            CLASS_NAMES = ["No", "Yes"]
            DROP_COLUMNS.extend([
                "NEXT_ADMISSION_WARD",
                "DISCHARGE DESCRIPTION",    # Commented out - explicitly describes discharge status (death/transfer)
            ])
        case 'EPISODE_CPE_INFECTION_FLAGGED':
            target_col = "EPISODE_CPE_INFECTION_FLAGGED"
            NUM_CLASSES = 2
            CLASS_NAMES = ["No", "Yes"]
            DROP_COLUMNS.extend([
                "NEXT_ADMISSION_WARD",
                "DISCHARGE WARD",           # Commented out - contains discharge information that may leak mortality/LOS
                "DISCHARGE SPECIALTY",      # Commented out - contains discharge information that may leak mortality
                "DISCHARGE DESCRIPTION",    # Commented out - explicitly describes discharge status (death/transfer)
                "EPISODE_CPE_SCREEN",
                "EPISODE_ISOLATED",         # Commented out - isolation often results from infection status
                "WARD_INFECTED_DEGREE",
                "WARD_INFECTED_CONTACT_DAYS",
                "WARD_INFECTED_CONTACT_DAYS_CENTRALITY",
                "WARD_INFECTED_CLOSENESS_CENTRALITY",
            ])
        case _:
            raise ValueError(f"Unknown target column: {target_col}")

    DROP_COLUMNS = list(set(DROP_COLUMNS))  # Remove duplicates

    for k in DROP_COLUMNS:
        if k in use_columns:
            use_columns.remove(k)

    patient_df = pd.read_csv(osp.join(input_folder, "clean/patients.csv"))
    episode_df = pd.read_csv(osp.join(input_folder, "clean/episodes.csv"))
    networks_df = pd.read_csv(osp.join(input_folder, "clean/networks.csv"))
    networks_df = networks_df.drop(columns=["BED_DAY_DATE_DATE"]).groupby(["PIN_NO"]).mean().reset_index()

    merged_df = pd.merge(
        episode_df,
        patient_df,
        how="left",
        on="PIN_NO",
    )
    merged_df = pd.merge(
        merged_df,
        networks_df,
        how="left",
        on=['PIN_NO'],
    )

    with open(osp.join(input_folder, f"splits/train_episodes_all.txt"), "r") as f:
        train_ids = f.read().splitlines()
        train_ids = list(map(float, train_ids))

    with open(osp.join(input_folder, f"splits/val_episodes_all.txt"), "r") as f:
        val_ids = f.read().splitlines()
        val_ids = list(map(float, val_ids))

    train_ids = list(set(train_ids + val_ids))

    with open(osp.join(input_folder, f"splits/test_episodes.txt"), "r") as f:
        test_ids = f.read().splitlines()
        test_ids = list(map(float, test_ids))

    # Remove id from train_ids if id is in test_ids
    train_ids = [i for i in train_ids if i not in test_ids]
    test_ids = [i for i in test_ids if i not in train_ids]

    # Fill Nans
    for col in merged_df.columns:
        if col in NUMERICAL_COLUMNS:
            merged_df[col] = merged_df[col].fillna(-100)
        elif col in CATEGORICAL_COLUMNS:
            merged_df[col] = merged_df[col].fillna("[UNK]")
        elif col in ICD_COLUMNS:
            merged_df[col] = merged_df[col].fillna("[UNK]")
    
    if target_col is not None:
        merged_df.dropna(subset=[target_col], inplace=True)

    # Only keep episodes where CPE screening was done
    if target_col == 'EPISODE_CPE_INFECTION_FLAGGED':
        merged_df = merged_df[merged_df['EPISODE_CPE_SCREEN'] == 1]

    train_df = merged_df[merged_df.HIPE_PATIENTID.isin(train_ids)]
    if patient_ids is None:
        test_df = merged_df[merged_df.HIPE_PATIENTID.isin(test_ids)]
    else:
        # If patient_ids is provided, use it to filter the data
        test_df = merged_df[merged_df.PIN_NO.isin(patient_ids)]

    train_df = train_df[use_columns]
    test_df = test_df[use_columns]

    # Remove unseen categorical values from train to val/test
    for col in CATEGORICAL_COLUMNS:
        if col not in use_columns: continue
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        unseen_test = set(test_df[col].unique()) - set(train_df[col].unique())

        if len(unseen_test) > 0:
            test_df.loc[test_df[col].isin(unseen_test), col] = "[UNK]"
        
    # Encode categorical columns
    from sklearn.preprocessing import OrdinalEncoder
    
    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in use_columns: continue
        le = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        train_df[col] = le.fit_transform(train_df[col].values.reshape(-1, 1)).flatten()
        label_encoders[col] = le
        test_df[col] = le.transform(test_df[col].values.reshape(-1, 1)).flatten()

    all_icdcm_codes_train = train_df['ICD10CMs_3'].values.astype(str).flatten().tolist()

    vocab = []
    for icdcm_str in all_icdcm_codes_train:
        icds = icdcm_str.split()
        for icd in icds:
            icd = icd.strip()
            vocab.append(icd)

    vocab = list(set(vocab))

    # Create vocabulary-based features
    def create_vocab_features(df, vocab, icd_column='ICD10CMs_3'):
        """Create features based on vocabulary frequency"""
        vocab_features = pd.DataFrame(0, index=df.index, columns=[f'ICD10//{code}' for code in vocab])
        
        for idx, icd_str in enumerate(df[icd_column].values.astype(str)):
            icds = icd_str.split()
            for icd in icds:
                icd = icd.strip()
                if icd in vocab:
                    vocab_features.iloc[idx][f'ICD10//{icd}'] += 1

        return vocab_features

    train_vocab_features = create_vocab_features(train_df, vocab)
    test_vocab_features = create_vocab_features(test_df, vocab)
    
    # Concatenate original dataframes with vocabulary features
    train_df = pd.concat([train_df, train_vocab_features], axis=1)
    test_df = pd.concat([test_df, test_vocab_features], axis=1)
    
    # Update use_columns to include vocab features
    vocab_columns = [f'ICD10//{code}' for code in vocab]
    use_columns.extend(vocab_columns)

    use_columns.remove('ICD10CMs_3')  # Remove the original ICD column if needed

    if target_col == 'EPISODE_CPE_INFECTION_FLAGGED':
        # Remove ICD columns related to CPE infection: ICD10//Z22
        use_columns = [col for col in use_columns if not col.startswith('ICD10//Z22')]
        

    train_df = train_df[use_columns]
    test_df = test_df[use_columns]

    if output_folder is None:
        pass
    else:
        if target_col is not None:
            output_folder = osp.join(output_folder, target_col)
        else:
            output_folder = osp.join(output_folder, "pretrain")
        os.makedirs(output_folder, exist_ok=True)

    LOGGER.text(f"Train dataset size: {len(train_df)}")
    LOGGER.text(f"Test dataset size: {len(test_df)}")

    if target_col is not None:
        LOGGER.text(f"Train dataset target distribution: {train_df[target_col].value_counts()}")
        LOGGER.text(f"Test dataset target distribution: {test_df[target_col].value_counts()}")
    
    train_ids_df = pl.DataFrame(train_df.copy()).select(["PIN_NO", "HIPE_PATIENTID"])
    test_ids_df = pl.DataFrame(test_df.copy()).select(["PIN_NO", "HIPE_PATIENTID"])


    feature_names = list(train_df.drop(columns=['PIN_NO', 'HIPE_PATIENTID', target_col]).columns)
    categorical_feature_indices = [
        i for i, col in enumerate(feature_names) if col in CATEGORICAL_COLUMNS
    ]


    y_train_df = train_df[target_col]
    y_test_df = test_df[target_col]
    X_train_df = train_df.drop(columns=['PIN_NO', 'HIPE_PATIENTID', target_col])
    X_test_df = test_df.drop(columns=['PIN_NO', 'HIPE_PATIENTID', target_col])

    X_train_df, y_train_df, feature_names, selected_indices = select_features(
        X_train_df, y_train_df, selection_method=selection_method, 
        feature_selection_path=feature_selection_path, 
        selected_indices=None
    )

    X_test_df, y_test_df, _, _ = select_features(
        X_test_df, y_test_df, selection_method=selection_method,
        feature_selection_path=feature_selection_path,
        selected_indices=selected_indices
    )

    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = y_train_df.values
    y_test = y_test_df.values
    label_dist = np.bincount(np.concatenate([y_train, y_test]))

    if output_folder is not None:
        with open(osp.join(output_folder, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=4)
    
    # Save label encoder mappings
    encoder_mappings = {}
    for col, encoder in label_encoders.items():
        encoder_mappings[col] = {
            'categories': [cat.tolist() for cat in encoder.categories_],
            'mapping': {str(cat): int(idx) for idx, cat in enumerate(encoder.categories_[0])}
        }
        
    if output_folder is not None:
        with open(osp.join(output_folder, "label_encoders.json"), "w") as f:
            json.dump(encoder_mappings, f, indent=4)
        
        # Save vocabulary
        with open(osp.join(output_folder, "vocab.json"), "w") as f:
            json.dump(vocab, f, indent=4)

    return {
        'X_train': X_train, 'y_train': y_train,  
        'X_test': X_test, 'y_test': y_test, 
        # 'X_embs_train': X_train_traj,
        # 'X_embs_test': X_test_traj,
        'categorical_feature_indices': categorical_feature_indices,
        'feature_names': feature_names,
        'train_ids': train_ids_df,
        'test_ids': test_ids_df,
        'class_distribution': label_dist,
        "is_classification": True,
        'num_classes': len(np.unique(y_train)),
        'num_features': len(feature_names),

    }


if __name__ == "__main__":
    dataset = get_sjh_dataset(
        input_folder="/home/mpham/workspace/source/ehrpfn/data/raw/hipe-processed-ultimate",
        output_folder="/home/mpham/workspace/source/ehrpfn/data/processed/sjh_cpe",
        target_col="EPISODE_CPE_INFECTION_FLAGGED",
        patient_ids=None,
        feature_selection_path=None,
        selection_method=None,
    )
