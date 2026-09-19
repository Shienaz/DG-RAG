#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the Lorentz + Euclidean hybrid retriever on QA data.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
N_GPU="${N_GPU:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
PRETRAINED_MODEL_PATH="${1:-}"
if [ "$#" -gt 0 ]; then
    shift
fi
EXTRA_ARGS=("$@")
DISTRIBUTED_BATCH_MODE="${DISTRIBUTED_BATCH_MODE:-single_batch_edge_parallel}"
USE_EDGE_PARALLEL="${USE_EDGE_PARALLEL:-}"
EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD="${EUCLIDEAN_EDGE_WEIGHT_REQUIRES_GRAD:-false}"
QEC_MODE="${QEC_MODE:-}"

if [ "${DISTRIBUTED_BATCH_MODE}" = "single_batch_edge_parallel" ] && [ -z "${USE_EDGE_PARALLEL}" ]; then
    USE_EDGE_PARALLEL=true
fi
if [ "${DISTRIBUTED_BATCH_MODE}" = "single_batch_edge_parallel" ] && [ -z "${QEC_MODE}" ]; then
    QEC_MODE=hard
fi
USE_EDGE_PARALLEL="${USE_EDGE_PARALLEL:-true}"
QEC_MODE="${QEC_MODE:-hard}"

cd "${PROJECT_ROOT}" || exit 1
source .venv/bin/activate

if [ -z "${PRETRAINED_MODEL_PATH}" ]; then
    echo "Usage: bash scripts/stage2_finetune_riemann.sh ${PROJECT_ROOT}/outputs/kg_pretrain_riemann/<DATE>/<TIME>/pretrained [hydra_overrides...]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT="${#VISIBLE_GPU_IDS[@]}"
if [ "${N_GPU}" -gt "${VISIBLE_GPU_COUNT}" ]; then
    echo "N_GPU=${N_GPU} but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes only ${VISIBLE_GPU_COUNT} GPU(s)."
    echo "Set CUDA_VISIBLE_DEVICES to ${N_GPU} devices, or reduce N_GPU."
    exit 1
fi
export CUDA_HOME="${CUDA_HOME%:}"
if [ -n "${CUDA_HOME}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

torchrun --nproc_per_node=${N_GPU} -m dgrag.workflow.stage2_qa_finetune \
    --config-name stage2_qa_finetune_riemann \
    datasets.cfgs.root=./data \
    datasets.cfgs.compute_structural_score=true \
    datasets.cfgs.structural_score_alpha=0.2 \
    'datasets.train_names=[hotpotqa_train_example,musique_train0,2wikimultihopqa_train0]' \
    'datasets.valid_names=[hotpotqa_test,musique_test,2wikimultihopqa_test]' \
    'model.entity_model.branch_geometries=[lorentz,euclidean]' \
    model.entity_model.input_dim=64 \
    'model.entity_model.hidden_dims=[64, 64, 64, 64, 64, 64]' \
    model.entity_model.use_alignment_loss=false \
    model.entity_model.alignment_weight=0.01 \
    model.entity_model.use_decision_mutual_loss=true \
    model.entity_model.decision_mutual_weight=0.005 \
    model.entity_model.decision_mutual_temperature=1.0 \
    model.entity_model.decision_mutual_divergence=js \
    model.entity_model.decision_mutual_teacher=fuse \
    model.entity_model.decision_mutual_branch_kl_weight=0.2 \
    model.entity_model.decision_mutual_candidate_mode=hard \
    model.entity_model.decision_mutual_topk=256 \
    model.entity_model.use_topology_mutual_loss=false \
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
    model.use_question_entity_contrastive=true \
    model.question_entity_contrastive_weight=0.1 \
    model.question_entity_contrastive_temperature=0.2 \
    model.question_entity_contrastive_mode="${QEC_MODE}" \
    model.question_entity_hard_negative_topk=256 \
    model.question_entity_hard_negative_exclude_question_entities=true \
    model.question_entity_hard_negative_random_fallback=true \
    model.branch_supervised_weight=0.03 \
    model.branch_supervised_bce_weight=0.3 \
    model.branch_supervised_listce_weight=0.7 \
    model.branch_supervised_adversarial_temperature=0.2 \
    train.checkpoint="${PRETRAINED_MODEL_PATH}" \
    train.num_epoch=20 \
    train.batch_size="${TRAIN_BATCH_SIZE}" \
    train.distributed_batch_mode="${DISTRIBUTED_BATCH_MODE}" \
    train.find_unused_parameters=false \
    optimizer.lr=5e-4 \
    "${EXTRA_ARGS[@]}"
