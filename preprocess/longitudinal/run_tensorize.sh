#!/bin/bash -l

export N_WORKERS=1
export HYDRA_FULL_ERROR=1
clear

DATASET=$1
VOCAB_DATASET=$2
if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset> <vocab_dataset>"
    echo "Example: $0 mimiciv mimiciv"
    exit 1
fi

if [ -z "$VOCAB_DATASET" ]; then
    VOCAB_DATASET=$DATASET
fi


DATA_PATH="$(pwd)/data/ehrfm"
LOGGING_DIR="$(pwd)/outputs/nosos/logs"

case $DATASET in
mimiciv*)
    DATASET_VERSION="3.1"
    INPUT_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv"
    VOCAB_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv/train"

    ;;
eicu*)
    DATASET_VERSION="2.0"
    INPUT_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/eicu"
    VOCAB_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/eicu/train"

    ;;
*)
    echo "Wrong experiment: '$DATASET', available are: 'mimic', 'eicu', 'ehrshot'"
    exit 1
    ;;
esac


export LOGGING_DIR

script_body="
set -e

PYTHONPATH=. python preprocess/longitudinal/tokenization/run_tensorize.py --config-name=$DATASET \
    -m input_dir=$INPUT_DIR/ \
    vocab=$VOCAB_DIR \
    output_dir=$INPUT_DIR \
    out_fn=train
"



bash -c "${script_body}"
