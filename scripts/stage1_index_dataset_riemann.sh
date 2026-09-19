# Build KG-index and QA artifacts for one dataset on Linux.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
DATA_NAME="${1}"

cd "${PROJECT_ROOT}" || exit 1
source .venv/bin/activate

if [ -z "${DATA_NAME}" ]; then
    echo "Usage: bash scripts/stage1_index_dataset_riemann.sh <dataset_name>"
    exit 1
fi

python -m dgrag.workflow.stage1_index_dataset \
    dataset.root=./data \
    dataset.data_name="${DATA_NAME}"
