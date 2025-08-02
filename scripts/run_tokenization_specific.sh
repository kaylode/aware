#!/bin/bash -l

export N_WORKERS=1
export HYDRA_FULL_ERROR=1
clear

DATASET=$1

# ICU_IDS_PATH=null
ICU_IDS_PATH="$(pwd)/data/ehrfm/mimiciv-3.1/hcai_subject_ids.txt"

DATA_PATH="$(pwd)/data/ehrfm"
LOGGING_DIR="$(pwd)/outputs/nosos/logs"

case $DATASET in
mimiciv*)
    DATASET_VERSION="3.1"
    INPUT_DIR="${DATA_PATH}/mimiciv-${DATASET_VERSION}/tokenized_datasets/mimiciv/train/16_FinalizeData"
    TEST_INPUT_DIR="${DATA_PATH}/mimiciv-${DATASET_VERSION}/tokenized_datasets/mimiciv/test/15_FinalizeData"
    OUTPUT_DIR="${DATA_PATH}/mimiciv-${DATASET_VERSION}/tokenized_datasets/${DATASET}"

    ;;
eicu*)
    DATASET_VERSION="2.0"
    INPUT_DIR="${DATA_PATH}/eicu-${DATASET_VERSION}/tokenized_datasets/eicu/train/11_FinalizeData"
    TEST_INPUT_DIR="${DATA_PATH}/eicu-${DATASET_VERSION}/tokenized_datasets/eicu/test/10_FinalizeData"
    OUTPUT_DIR="${DATA_PATH}/eicu-${DATASET_VERSION}/tokenized_datasets/${DATASET}"

    ;;
*)
    echo "Wrong experiment: '$DATASET', available are: 'mimic', 'eicu', 'ehrshot'"
    exit 1
    ;;
esac


export LOGGING_DIR

script_body="
set -e

PYTHONPATH=. python scripts/tokenization/run_tokenization.py --config-name=$DATASET \
    -m input_dir=$INPUT_DIR \
    output_dir=$OUTPUT_DIR \
    out_fn=train \
    icu_ids_path=$ICU_IDS_PATH

PYTHONPATH=. python scripts/tokenization/build_vocab.py --config-name=$DATASET \
    -m vocab=$OUTPUT_DIR/train \
    output_dir=$OUTPUT_DIR \
    out_fn=train

PYTHONPATH=. python scripts/tokenization/run_tokenization.py --config-name=$DATASET \
    -m input_dir=$TEST_INPUT_DIR \
    vocab=$OUTPUT_DIR/train \
    output_dir=$OUTPUT_DIR \
    out_fn=test \
    icu_ids_path=$ICU_IDS_PATH

"



bash -c "${script_body}"
