#!/bin/bash -l

export N_WORKERS=4
export HYDRA_FULL_ERROR=1

DATASET=$1

DATA_PATH="$(pwd)/data"
LOGGING_DIR="$(pwd)/outputs/nosos/logs"
TRAIN_RATIO=0.9
TEST_RATIO=0.1

clear
case $DATASET in
mimiciv*)
    DATASET_VERSION="3.1"
    N_SUBJECTS_PER_SHARD=20000
    input_dir="${DATA_PATH}/raw/mimic/mimiciv/${DATASET_VERSION}"
    suffix="ed"
    output_dir="${DATA_PATH}/raw/ehrfm/${DATASET}-${DATASET_VERSION}"
    ;;

eicu*)
    DATASET_VERSION="2.0"
    N_SUBJECTS_PER_SHARD=5000
    input_dir="${DATA_PATH}/raw/eicu-crd/${DATASET_VERSION}/"
    output_dir="${DATA_PATH}/ehrfm/${DATASET}-${DATASET_VERSION}"
    ;;
sjh-cpe*)
    DATASET_VERSION="1.0"
    N_SUBJECTS_PER_SHARD=5000
    input_dir="${DATA_PATH}/raw/sjh-cpe/${DATASET_VERSION}/"
    output_dir="${DATA_PATH}/processed_meds/sjh-cpe/${DATASET}-${DATASET_VERSION}"
    ;;
*)
    echo "### Wrong experiment: '$DATASET', available are: 'mimic', 'eicu', 'ehrshot' ###"
    exit 1
    ;;
esac

echo " ### Running pre-MEDS conversion on ${DATASET} dataset. ###"


# This makes the script fail if any internal script fails
set -e

echo "### Unsetting SLURM_CPU_BIND in case you're running this on a slurm interactive node with slurm parallelism ###"
unset SLURM_CPU_BIND


EVENT_CONVERSION_CONFIG_FP="$(pwd)/preprocess/longitudinal/meds/datasets/${DATASET}/event_conversion.yaml"
PIPELINE_CONFIG_FP="$(pwd)/preprocess/longitudinal/meds/configs/meds_transforms_pipeline.yaml"
PRE_MEDS_PY_FP="$(pwd)/preprocess/longitudinal/meds/datasets/${DATASET}/pre_MEDS.py"
MEDS_COHORT_DIR="${output_dir}/meds"
PRE_MEDS_DIR="${output_dir}/premeds"

# We export these variables separately from their assignment so that any errors during assignment are caught.
export EVENT_CONVERSION_CONFIG_FP
export PIPELINE_CONFIG_FP
export PRE_MEDS_PY_FP
export MEDS_COHORT_DIR
export PRE_MEDS_DIR
export LOGGING_DIR
export DATASET_VERSION
export DATASET_NAME=DATASET
export N_SUBJECTS_PER_SHARD
export TRAIN_RATIO
export TEST_RATIO

# echo "### Running pre-MEDS conversion. ###"
# python "$PRE_MEDS_PY_FP" \
#     input_dir="$input_dir" \
#     cohort_dir="$PRE_MEDS_DIR" \
#     do_overwrite=false

# if [ -z "$N_WORKERS" ]; then
#     echo "Setting N_WORKERS to 1 to avoid issues with the runners."
#     export N_WORKERS="1"
# fi

echo "### Running extraction pipeline. ###"
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" \
    stage_runner_fp=$(pwd)/preprocess/longitudinal/meds/configs/local_parallelism_runner.yaml
