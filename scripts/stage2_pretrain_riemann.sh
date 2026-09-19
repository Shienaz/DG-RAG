# Lorentz + Euclidean hybrid pretraining from scratch on Linux.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
N_GPU="${N_GPU:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
DISTRIBUTED_BATCH_MODE="${DISTRIBUTED_BATCH_MODE:-ddp}"
USE_EDGE_PARALLEL="${USE_EDGE_PARALLEL:-}"
EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD="${EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD:-true}"

if [ "${DISTRIBUTED_BATCH_MODE}" = "single_batch_edge_parallel" ] && [ -z "${USE_EDGE_PARALLEL}" ]; then
    USE_EDGE_PARALLEL=true
fi
USE_EDGE_PARALLEL="${USE_EDGE_PARALLEL:-false}"

cd "${PROJECT_ROOT}" || exit 1
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="${CUDA_HOME%:}"
if [ -n "${CUDA_HOME}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

torchrun --nproc-per-node=${N_GPU} -m dgrag.workflow.stage2_kg_pretrain \
    --config-name stage2_kg_pretrain_riemann \
    datasets.cfgs.root=./data \
    datasets.cfgs.compute_structural_score=true \
    datasets.cfgs.structural_score_alpha=0.2 \
    datasets.train_names=[musique_train0,hotpotqa_train_example,2wikimultihopqa_train0] \
    model.entity_model.branch_geometries=[lorentz,euclidean] \
    model.entity_model.use_alignment_loss=false \
    model.entity_model.alignment_weight=0.01 \
    model.entity_model.use_decision_mutual_loss=true \
    model.entity_model.decision_mutual_weight=0.01 \
    model.entity_model.decision_mutual_temperature=1.0 \
    model.entity_model.decision_mutual_divergence=js \
    model.entity_model.use_topology_mutual_loss=true \
    model.entity_model.topology_mutual_weight=0.005 \
    model.entity_model.topology_mutual_temperature=0.5 \
    model.entity_model.topology_mutual_divergence=js \
    model.entity_model.topology_mutual_sample_size=64 \
    model.entity_model.topology_kernel_degree=2 \
    model.entity_model.topology_kernel_bias=0.0 \
    model.entity_model.topology_mask_self=true \
    model.entity_model.use_edge_parallel="${USE_EDGE_PARALLEL}" \
    model.entity_model.edge_parallel_shard_strategy=contiguous \
    model.entity_model.euclidean_edge_weight_requires_grad="${EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD}" \
    model.entity_model.use_structure_gated_readout=false \
    model.entity_model.gate_tau_low=0.1 \
    model.entity_model.gate_tau_high=0.75 \
    model.entity_model.gate_beta=0.1 \
    model.entity_model.specialization_weight=0.0 \
    model.entity_model.specialization_margin=0.1 \
    train.num_epoch=10 \
    train.batch_size="${TRAIN_BATCH_SIZE}" \
    train.distributed_batch_mode="${DISTRIBUTED_BATCH_MODE}" \
    train.batch_per_epoch=30000 \
    train.fast_test=5000 \
    optimizer.lr=1e-4
