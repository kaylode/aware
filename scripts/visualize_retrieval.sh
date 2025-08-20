export HYDRA_FULL_ERROR=1
ROOT=/home/mpham/workspace/source/ehrpfn
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

OUTPUT_DIR="${ROOT}/outputs/tabpfn/results"
TASK="ICD SEPSIS"

# Loop
PYTHONPATH=. python ehrpfn/xai/visualize_retrievalv2.py