from ehrpfn.datasets import (
    get_sjh_dataset, 
    get_mimic_dataset, 
    get_openml_ehr
)



def compute_imbalance_ratio(dataset):
    # Compute the class distribution
    class_counts = dataset['label'].value_counts()
    # Compute the imbalance ratio
    
    
    return imbalance_ratio

def run(args):
    if args.data_name == 'mimicpipe':
        data = get_mimic_dataset(
            data_csv=args.data_csv,
            train_label_path=args.train_label_path,
            test_label_path=args.test_label_path,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
            embedding_file_path=args.embedding_file_path,
        )
    elif args.data_name == 'sjh_cpe':
        data = get_sjh_dataset(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            target_col=args.target_column,
            patient_ids=None,
            feature_selection_path=args.feature_selection_path,
            selection_method=args.selection_method,
        )
    elif args.data_name == 'openml_ehr':
        data = get_openml_ehr(args.data_folder)
    else:
        raise ValueError(f"Data {args.data_name} is not supported. Please choose from ['mimiciv', 'sjh_cpe']")
    return data

class OpenMLEHRConfig:
    # Configuration class for OpenML EHR dataset.
    data_name: str = "openml_ehr"
    data_folder: str = "/home/mpham/workspace/source/ehrpfn/data/pretraining"


ROOT = "/home/mpham/workspace/source/ehrpfn"
TASK = 'WILL_READMIT_WITHIN_30_DAYS' #'EPISODE_CPE_INFECTION_FLAGGED' #
class MIMICIVConfig:
    # Main config variables
    data_name: str = "mimicpipe"
    data_csv: str = f"{ROOT}/data/processed/mimiciv/{TASK}/all_data.csv"
    train_label_path: str = f"{ROOT}/data/processed/mimiciv/{TASK}/split/train_labels.csv"
    test_label_path: str = f"{ROOT}/data/processed/mimiciv/{TASK}/split/test_labels.csv"
    embedding_file_path: str = None
    feature_selection_path: str = None
    selection_method: str = None
    target_column: str = f"{TASK}"


class SJHHIPEConfig:
    # Main config variables
    data_name: str = "sjh_cpe"
    input_folder: str = "/home/mpham/workspace/source/ehrpfn/data/raw/hipe-processed-ultimate"
    output_folder: str = "/home/mpham/workspace/source/ehrpfn/outputs/all"
    feature_selection_path: str = None
    selection_method: str = None
    target_column: str = f"{TASK}"

if __name__ == '__main__':
    # args = OpenMLEHRConfig()
    args = SJHHIPEConfig()
    # args = MIMICIVConfig()
    d = run(args)

    res = {}

    # for d in ds:
    class_distribution = d['class_distribution']
    is_classification = d['is_classification']
    if is_classification:
        imbalance_ratio = class_distribution.max() / class_distribution.min()
    print(imbalance_ratio)

    import json
    # json.dump(res, open('imbalance_ratio.json', 'w'), indent=4)