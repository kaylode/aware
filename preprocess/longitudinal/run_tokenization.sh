#!/bin/bash -l

export N_WORKERS=1
export HYDRA_FULL_ERROR=1
clear

DATASET=$1

ICU_IDS_PATH=null

# DATA_PATH="$(pwd)/data"
DATA_PATH="/home/mpham/spinning/data/ehrfm"
LOGGING_DIR="$(pwd)/outputs/nosos/logs"

case $DATASET in
mimiciv*)
    DATASET_VERSION="3.1"
    INPUT_DIR="${DATA_PATH}/mimiciv-${DATASET_VERSION}/meds/data"
    OUTPUT_DIR="${DATA_PATH}/mimiciv-${DATASET_VERSION}/tokenized_datasets/${DATASET}"
    ;;
eicu*)
    DATASET_VERSION="2.0"
    INPUT_DIR="${DATA_PATH}/eicu-${DATASET_VERSION}/meds/data"
    OUTPUT_DIR="${DATA_PATH}/eicu-${DATASET_VERSION}/tokenized_datasets/${DATASET}"
    ;;
sjh-cpe*)
    DATASET_VERSION="1.0"
    # INPUT_DIR="${DATA_PATH}/sjh-cpe-${DATASET_VERSION}/meds/data"
    INPUT_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/sjh-cpe/sjh-cpe-1.0/meds/data"
    OUTPUT_DIR="${DATA_PATH}/sjh-cpe-${DATASET_VERSION}/tokenized_datasets/${DATASET}"

    ;;
*)
    echo "Wrong experiment: '$DATASET', available are: 'mimic', 'eicu', 'ehrshot'"
    exit 1
    ;;
esac

OUTPUT_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/${DATASET}"


export LOGGING_DIR

singularity_preamble="
export PATH=\$HOME/.local/bin:\$PATH

# Install nosos
cd /nosos
pip install \
    --no-deps \
    --no-index \
    --no-build-isolation \
    --user \
    -e  \
    . 1>/dev/null
"

script_body="
set -e

PYTHONPATH=. python preprocess/longitudinal/tokenization/run_tokenization.py --config-name=$DATASET \
    -m input_dir=$INPUT_DIR/train \
    output_dir=$OUTPUT_DIR \
    out_fn=train \
    icu_ids_path=$ICU_IDS_PATH

PYTHONPATH=. python preprocess/longitudinal/tokenization/build_vocab.py --config-name=$DATASET \
    -m vocab=$OUTPUT_DIR/train \
    output_dir=$OUTPUT_DIR \
    out_fn=train

PYTHONPATH=. python preprocess/longitudinal/tokenization/run_tokenization.py --config-name=$DATASET \
    -m input_dir=$INPUT_DIR/test \
    vocab=$OUTPUT_DIR/train \
    output_dir=$OUTPUT_DIR \
    out_fn=test \
    icu_ids_path=$ICU_IDS_PATH
"



module load singularity 2>/dev/null

if command -v singularity >/dev/null; then
    singularity exec \
        --contain \
        --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/nosos \
        --bind /mnt:/mnt \
        nosos.sif \
        bash -c "${singularity_preamble}${script_body}"
else
    echo Singularity not found, running using locally using bash.
    bash -c "${script_body}"
fi
