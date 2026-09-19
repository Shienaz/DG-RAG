#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-optimized}"
MODE="${2:-train}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-./data}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPU="${N_GPU:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
BATCH_PER_EPOCH="${BATCH_PER_EPOCH:-600}"
FAST_TEST="${FAST_TEST:-1}"
CHECKPOINT="${CHECKPOINT:-}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/benchmarks}"
PYTHON_BIN="${PYTHON_BIN:-}"
SOURCE_COMMIT="${SOURCE_COMMIT:-}"
EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD="${EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD:-}"

if [[ "${N_GPU}" -ne 2 ]]; then
    echo "This benchmark is normalized for exactly two GPUs; got N_GPU=${N_GPU}."
    exit 2
fi
if [[ "${MODE}" != "train" && "${MODE}" != "eval" ]]; then
    echo "MODE must be train or eval."
    exit 2
fi
if [[ "${MODE}" == "eval" && -z "${CHECKPOINT}" ]]; then
    echo "CHECKPOINT is required in eval mode."
    exit 2
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

if [[ -n "${PYTHON_BIN}" ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "PYTHON_BIN is not executable: ${PYTHON_BIN}"
        exit 2
    fi
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
    RUNNER=("${PYTHON_BIN}" -m torch.distributed.run)
elif command -v python >/dev/null 2>&1; then
    RUNNER=(python -m torch.distributed.run)
else
    echo "No Python interpreter on PATH. Set PYTHON_BIN to the python of your environment."
    exit 2
fi

if [[ "${ALLOW_BUSY_GPU}" != "1" ]] && \
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
        grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "A compute process is already using a GPU. Set ALLOW_BUSY_GPU=1 to override."
    exit 3
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT}/${LABEL}-${MODE}-${STAMP}"
mkdir -p "${RUN_ROOT}"

if [[ -n "${SOURCE_COMMIT}" ]]; then
    printf '%s\n' "${SOURCE_COMMIT}" > "${RUN_ROOT}/commit.txt"
elif git rev-parse HEAD > "${RUN_ROOT}/commit.txt" 2>/dev/null; then
    :
else
    echo "unknown" > "${RUN_ROOT}/commit.txt"
fi
if [[ -n "${PYTHON_BIN}" ]]; then
    "${PYTHON_BIN}" --version > "${RUN_ROOT}/python-version.txt"
else
    python --version > "${RUN_ROOT}/python-version.txt"
fi
"${RUNNER[@]}" --help >/dev/null
nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader > "${RUN_ROOT}/gpu-info.csv"

nvidia-smi \
    --query-gpu=timestamp,index,memory.used \
    --format=csv,noheader,nounits \
    -lms 500 > "${RUN_ROOT}/gpu-memory.csv" &
MONITOR_PID=$!
cleanup() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

COMMON_OVERRIDES=(
    "datasets.cfgs.root=${DATA_ROOT}"
    "datasets.cfgs.compute_structural_score=true"
    "datasets.cfgs.structural_score_alpha=0.2"
    "datasets.train_names=[hotpotqa_train_example]"
    "model.entity_model.branch_geometries=[lorentz,euclidean]"
    "model.entity_model.input_dim=64"
    "model.entity_model.hidden_dims=[64,64,64,64,64,64]"
    "model.entity_model.use_alignment_loss=false"
    "model.entity_model.use_decision_mutual_loss=true"
    "model.entity_model.decision_mutual_weight=0.01"
    "model.entity_model.decision_mutual_temperature=1.0"
    "model.entity_model.decision_mutual_divergence=js"
    "model.entity_model.use_topology_mutual_loss=true"
    "model.entity_model.topology_mutual_weight=0.005"
    "model.entity_model.topology_mutual_temperature=0.5"
    "model.entity_model.topology_mutual_divergence=js"
    "model.entity_model.topology_mutual_sample_size=64"
    "model.entity_model.topology_kernel_degree=2"
    "model.entity_model.topology_kernel_bias=0.0"
    "model.entity_model.topology_mask_self=true"
    "model.entity_model.use_edge_parallel=false"
    "model.entity_model.use_structure_gated_readout=false"
    "model.entity_model.specialization_weight=0.0"
    "train.distributed_batch_mode=ddp"
    "train.find_unused_parameters=false"
    "train.save_pretrained=false"
    "optimizer.lr=1e-4"
    "hydra.run.dir=${RUN_ROOT}/hydra"
)
if [[ -n "${EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD}" ]]; then
    COMMON_OVERRIDES+=(
        "model.entity_model.euclidean_edge_weight_requires_grad=${EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD}"
    )
fi

if [[ "${MODE}" == "train" ]]; then
    MODE_OVERRIDES=(
        "train.num_epoch=1"
        "train.batch_size=${TRAIN_BATCH_SIZE}"
        "train.batch_per_epoch=${BATCH_PER_EPOCH}"
        "train.fast_test=${FAST_TEST}"
        "train.save_best_only=true"
    )
else
    MODE_OVERRIDES=(
        "train.num_epoch=0"
        "train.batch_size=${TRAIN_BATCH_SIZE}"
        "train.fast_test=5000"
        "train.checkpoint=${CHECKPOINT}"
    )
fi

set +e
/usr/bin/time \
    -f 'elapsed_seconds=%e\nmax_rss_kb=%M' \
    -o "${RUN_ROOT}/time.txt" \
    "${RUNNER[@]}" \
        --nproc_per_node="${N_GPU}" \
        -m dgrag.workflow.stage2_kg_pretrain \
        --config-name stage2_kg_pretrain_riemann \
        "${COMMON_OVERRIDES[@]}" \
        "${MODE_OVERRIDES[@]}" \
        2>&1 | tee "${RUN_ROOT}/console.log"
STATUS=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT

awk -F',' '
{
    gpu=$2
    memory=$3
    gsub(/^[ \t]+|[ \t]+$/, "", gpu)
    gsub(/^[ \t]+|[ \t]+$/, "", memory)
    if (memory + 0 > peak[gpu]) {
        peak[gpu] = memory + 0
    }
}
END {
    for (gpu in peak) {
        printf "gpu_%s_peak_memory_mib=%d\n", gpu, peak[gpu]
    }
}' "${RUN_ROOT}/gpu-memory.csv" | sort > "${RUN_ROOT}/peak-memory.txt"

{
    echo "label=${LABEL}"
    echo "mode=${MODE}"
    echo "exit_status=${STATUS}"
    cat "${RUN_ROOT}/time.txt"
    cat "${RUN_ROOT}/peak-memory.txt"
    grep -E 'peak CUDA memory|mrr:|hits@[0-9]+:' \
        "${RUN_ROOT}/console.log" || true
} > "${RUN_ROOT}/summary.txt"

cat "${RUN_ROOT}/summary.txt"
exit "${STATUS}"
