import json
import polars as pl
import os.path as osp
import os

# File paths
ICU_ITEMS_PATH = "/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/icu_items.json"
PROC_DATA_PATH_TEMPLATE = "/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/{TASK}/features/preproc_proc_icu.csv.gz"
TRANSFERS_PATH = "/home/mpham/spinning/data/mimic/mimiciv/3.1/hosp/transfers.csv.gz"

# Procedure codes to filter
PROCEDURE_CODES = [
    'GAUGE', 'CHEST_X-RAY', 'EKG', 'ARTERIAL_LINE', 'INVASIVE_VENTILATION',
    'BLOOD_CULTURED', 'MULTI_LUMEN', 'EXTUBATION', 'CT_SCAN', 'PICC_LINE',
    'ULTRASOUND', 'URINE_CULTURE', 'TRANSTHORACIC_ECHO', 'NASAL_SWAB',
    'INTUBATION', 'SPUTUM_CULTURE', 'FOLEY_CATHETER', 'X-RAY',
    'CORDIS/INTRODUCER', 'BRONCHOSCOPY', 'HEMODIALYSIS', 'DIALYSIS_CATHETER',
    'PA_CATHETER', 'NON-INVASIVE_VENTILATION', 'MAGNETIC_RESONANCE_IMAGING',
    'DIALYSIS_-_CRRT', 'STOOL_CULTURE', 'CHEST_TUBE_REMOVED'
]

# Ward mapping configurations
RENAME_MAPPING = {
    "MED/SURG": "MEDICAL/SURGICAL",
    "NEURO": "NEUROLOGY",
    'SURGICAL_INTERMEDIATE': 'SURGERY',
    'THORACIC_SURGERY': 'SURGERY',
}

WARD_NAMES = [
    "CARDIAC", "CARDIOLOGY", "CORONARY_CARE_UNIT", "DISCHARGE_LOUNGE",
    "EMERGENCY_DEPARTMENT", "HEMATOLOGY", "ICU", "LABOR_&_DELIVERY",
    "MEDICAL/SURGICAL", 'MEDICINE', 'NEUROLOGY', 'NURSERY', 'OBSERVATION',
    'OBSTETRICS', 'PACU', 'PSYCHIATRY', 'ONCOLOGY', 'SPECIAL_CARE_NURSERY',
    'SURGERY', 'TRANSPLANT', 'UNKNOWN', 'VASCULAR'
]


def load_icu_item_mapping():
    """Load and process ICU item mapping dictionary."""
    with open(ICU_ITEMS_PATH, 'r') as f:
        icu_items = json.load(f)
    return {k['itemid']: k['label'] for k in icu_items}


def process_activity_data(icu_item_mapping_dict):
    """Process procedure/activity data from ICU."""
    proc_df = pl.read_csv(PROC_DATA_PATH)
    
    activity_df = (
        proc_df
        .with_columns(
            pl.col("itemid").replace_strict(icu_item_mapping_dict).str.to_uppercase().str.replace_all(' ','_',literal=True).alias('item_name')
        )
        .with_columns(
            pl.col("item_name")
            .map_elements(
                lambda x: 'GAUGE' if 'GAUGE' in x else x,
                return_dtype=pl.String
            )
            .alias('item_name')
        )
        .filter(pl.col("item_name").is_in(PROCEDURE_CODES))
        .select(['subject_id', 'hadm_id', 'stay_id', 'starttime', 'item_name'])
        .rename({'item_name': 'activity'})
    )
    
    return activity_df


def create_mapping_dict(codes, mapping_rules):
    """Create a mapping dictionary based on prefix matching rules."""
    result_mapping = {}
    for code in codes:
        for key, value in mapping_rules.items():
            if code.startswith(key):
                result_mapping[code] = value
                break
    return result_mapping


def process_transfer_data():
    """Process transfer/ward data."""
    transfer_df = (
        pl.read_csv(TRANSFERS_PATH)
        .sort(['subject_id', 'hadm_id', 'intime'])
        .drop_nulls(subset=['hadm_id'])
        .select(['subject_id', 'hadm_id', 'intime', 'careunit'])
        .rename({
            'intime': 'starttime',
            'careunit': 'activity'
        })
        .with_columns(
            pl.col('activity')
            .map_elements(
                lambda x: x.upper().replace(' ', '_'),
                return_dtype=pl.String
            )
            .alias('activity')
        )
    )
    
    # Get unique transfer codes for mapping
    unique_transfer_codes = transfer_df.select("activity").unique().to_numpy().flatten()
    
    # Create mapping dictionaries
    strict_mapping_1 = create_mapping_dict(unique_transfer_codes, RENAME_MAPPING)
    strict_mapping_2 = create_mapping_dict(unique_transfer_codes, 
                                         {ward: ward if ward != 'UNKNOWN' else 'DISCHARGE' for ward in WARD_NAMES}
    )
    
    # Apply mappings
    transfer_df = (
        transfer_df
        .with_columns(
            activity=pl.col('activity').replace_strict(
                strict_mapping_1,
                default=pl.col('activity')
            )
        )
        .with_columns(
            activity=pl.when(pl.col("activity").str.contains("ICU"))
            .then(pl.lit("ICU"))
            .otherwise(pl.col("activity"))
        )
        .with_columns(
            activity=pl.col('activity').replace_strict(
                strict_mapping_2,
                default=pl.col('activity')
            )
        )
    )
    
    return transfer_df


def combine_and_aggregate_data(activity_df, transfer_df):
    """Combine activity and transfer data, then aggregate by patient admission."""
    process_df = (
        pl.concat([activity_df, transfer_df], how='diagonal')
        .sort(['subject_id', 'hadm_id', 'starttime'])
        .group_by(['subject_id', 'hadm_id'])
        .agg(
            num_activities=pl.len(),
            pathway=pl.col('activity').str.concat(delimiter=','),
            starttime=pl.col('starttime').min()
        )
    )

    process_df = process_df.sort(['subject_id', 'hadm_id', 'starttime'])

    # Add backward-looking pathway columns
    process_df = process_df.with_columns([
        # Get previous pathway (lag 1)
        pl.col("pathway").shift(1).over("subject_id").alias("prev_pathway_1"),
        # Get pathway from 2 steps back (lag 2)
        pl.col("pathway").shift(2).over("subject_id").alias("prev_pathway_2")
    ]).with_columns([
        # Concat current with 1 previous pathway
        pl.when(pl.col("prev_pathway_1").is_null())
        .then(pl.col("pathway"))
        .otherwise(pl.col("prev_pathway_1") + pl.lit(",") + pl.col("pathway"))
        .alias("pathway_with_prev_1"),
        
        # Concat current with 2 previous pathways
        pl.when(pl.col("prev_pathway_2").is_null() | pl.col("prev_pathway_1").is_null())
        .then(
            pl.when(pl.col("prev_pathway_1").is_null())
            .then(pl.col("pathway"))
            .otherwise(pl.col("prev_pathway_1") + pl.lit(";") + pl.col("pathway"))
        )
        .otherwise(
            pl.col("prev_pathway_2") + pl.lit(";") + 
            pl.col("prev_pathway_1") + pl.lit(";") + 
            pl.col("pathway")
        )
        .alias("pathway_with_prev_2")
    ])
    
    return process_df


def draw_process(process_df, top_n_variants=30, save_path=None):
    """
    Create and visualize a process mining graph from the aggregated process data.
    
    Args:
        process_df: DataFrame with columns ['subject_id', 'hadm_id', 'num_activities', 'pathway']
        top_n_variants: Number of top variants to visualize (default: 5)
    """
    import pm4py
    import pandas as pd
    
    # Convert polars to pandas for pm4py compatibility
    process_pd = process_df.to_pandas()
    
    # Create event log format from pathway data
    event_log_data = []
    
    for _, row in process_pd.iterrows():
        case_id = f"{row['subject_id']}_{row['hadm_id']}"
        activities = row['pathway'].split(',')  # Split the comma-separated string
        
        for i, activity in enumerate(activities):
            event_log_data.append({
                'case:concept:name': case_id,
                'concept:name': activity,
                'time:timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
                'org:resource': 'Hospital'
            })
    
    # Create DataFrame in pm4py event log format
    event_log_df = pd.DataFrame(event_log_data)
    
    # Convert to pm4py event log
    event_log = pm4py.convert_to_event_log(event_log_df)
    
    # Get variant statistics using the correct API
    variants_count = pm4py.get_variants(event_log)
    
    # Get top N most frequent variants
    top_variants = sorted(variants_count.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_variants]
    top_variant_strings = [v[0] for v in top_variants]
    
    print(f"Top {top_n_variants} most frequent process variants:")
    for i, (variant, traces) in enumerate(top_variants, 1):
        print(f"{i}. {variant} (Count: {len(traces)})")

    print(f"Coverage variants: {len(top_variants) * 100 / len(variants_count)} %")
    print(f"Coverage patients: {sum([len(i) for _,i in top_variants]) * 100 / sum([len(i) for _,i in variants_count.items()])} %")

    # Filter log to keep only top variants
    filtered_log = pm4py.filter_variants(event_log, top_variant_strings)
    
    # Discover and visualize the process
    dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_log)
    
    # Visualize the DFG
    # pm4py.view_dfg(dfg, start_activities, end_activities)

    VARIANTS_COVERAGE = len(top_variants) * 100 / len(variants_count)
    PATIENTS_COVERAGE = sum([len(i) for _,i in top_variants]) * 100 / sum([len(i) for _,i in variants_count.items()])

    # Optionally save the visualization
    pm4py.save_vis_dfg(dfg, start_activities, end_activities, save_path.format(VARIANTS_COVERAGE=VARIANTS_COVERAGE, PATIENTS_COVERAGE=PATIENTS_COVERAGE, N_VARIANTS=top_n_variants))

    return dfg, start_activities, end_activities, filtered_log


def main():
    global PROC_DATA_PATH
    global PROC_DATA_PATH_TEMPLATE
    global TASK
    # Constants
    N_VARIANTS = 50
# 'ICD SEPSIS', 'ICD BSI', 'ICD VAP', 'ICD UTI', 'ICD SSI',
    for TASK in ['Readmission', 'Mortality', 'Length of Stay']:
        PROC_DATA_PATH = PROC_DATA_PATH_TEMPLATE.format(TASK=TASK)
        SAVE_DIR = f"/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/{TASK}/process"
        os.makedirs(SAVE_DIR, exist_ok=True)
        SAVE_CSV = osp.join(SAVE_DIR, "process_data.csv")

        """Main processing pipeline."""
        # Load ICU item mapping
        icu_item_mapping_dict = load_icu_item_mapping()
        
        # Process activity data
        activity_df = process_activity_data(icu_item_mapping_dict)
        
        # Process transfer data
        transfer_df = process_transfer_data()

        # transfer_df2 = transfer_df.sort(['subject_id', 'hadm_id', 'starttime']).group_by(['subject_id', 'hadm_id']).agg(
        #     num_activities=pl.len(),
        #     pathway=pl.col('activity').str.concat(delimiter=',')
        # )
        # draw_process(transfer_df2, top_n_variants=30, save_path=osp.join(SAVE_DIR, "process_graph_nvar:{N_VARIANTS}_cvrgv:{VARIANTS_COVERAGE:.3f}_cvrgp:{PATIENTS_COVERAGE:.3f}.pdf"))
        # activity_df2 = activity_df.sort(['subject_id', 'hadm_id', 'starttime']).group_by(['subject_id', 'hadm_id']).agg(
        #     num_activities=pl.len(),
        #     pathway=pl.col('activity').str.concat(delimiter=',')
        # )
        # draw_process(activity_df2, top_n_variants=30, save_path=osp.join(SAVE_DIR, "activity_graph_nvar:{N_VARIANTS}_cvrgv:{VARIANTS_COVERAGE:.3f}_cvrgp:{PATIENTS_COVERAGE:.3f}.pdf"))
        # return 


        # Combine and aggregate
        process_df = combine_and_aggregate_data(activity_df, transfer_df)
        # Save process data (pathway is already a string)
        process_df.write_csv(SAVE_CSV)
        
        # Load train/test mappings
        train_id_df = pl.read_csv(f"/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/{TASK}/split/train_labels.csv")
        test_id_df = pl.read_csv(f"/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/{TASK}/split/test_labels.csv")

        train_process_df = train_id_df.join(process_df, on=['subject_id', 'hadm_id'], how='inner')
        train_process_df.write_csv(osp.join(SAVE_DIR, "process_pathways_encoded_train.csv"))
        test_process_df = test_id_df.join(process_df, on=['subject_id', 'hadm_id'], how='inner')
        test_process_df.write_csv(osp.join(SAVE_DIR, "process_pathways_encoded_test.csv"))

        # Draw process mining graph
        print("\nGenerating process mining visualization...")
        draw_process(process_df, top_n_variants=N_VARIANTS, save_path=osp.join(SAVE_DIR, "process_graph_nvar:{N_VARIANTS}_cvrgv:{VARIANTS_COVERAGE:.3f}_cvrgp:{PATIENTS_COVERAGE:.3f}.pdf"))
        draw_process(train_process_df, top_n_variants=N_VARIANTS, save_path=osp.join(SAVE_DIR, "train_process_graph_nvar:{N_VARIANTS}_cvrgv:{VARIANTS_COVERAGE:.3f}_cvrgp:{PATIENTS_COVERAGE:.3f}.pdf"))
        draw_process(test_process_df, top_n_variants=N_VARIANTS, save_path=osp.join(SAVE_DIR, "test_process_graph_nvar:{N_VARIANTS}_cvrgv:{VARIANTS_COVERAGE:.3f}_cvrgp:{PATIENTS_COVERAGE:.3f}.pdf"))


    return process_df


if __name__ == "__main__":
    main()

