#!/usr/bin/env bash
set -euo pipefail

# Run stage3 QA inference for the fine-tuned Lorentz + Euclidean retriever.
# By default this runs HotpotQA, MuSiQue, and 2WikiMultihopQA sequentially.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
N_GPU="${N_GPU:-4}"
MODEL_PATH="${1:-${MODEL_PATH:-}}"
DATASETS_ARG="${2:-${DATASETS:-all}}"
LLM_NAME="${3:-${LLM_NAME:-deepseek-ai/DeepSeek-V3.2}}"
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "$#" -gt 0 ]; then
    shift
fi
EXTRA_ARGS=("$@")

DOC_TOP_K="${DOC_TOP_K:-5}"
DOC_RANKER="${DOC_RANKER:-idf_topk_ranker}"
DOC_RANKER_ENTITY_TOP_K="${DOC_RANKER_ENTITY_TOP_K:-}"
N_THREADS="${N_THREADS:-10}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-4}"
MUSIQUE_RETRIEVAL_BATCH_SIZE="${MUSIQUE_RETRIEVAL_BATCH_SIZE:-2}"
SAVE_RETRIEVAL="${SAVE_RETRIEVAL:-true}"
STRUCTURAL_SCORE_ALPHA="${STRUCTURAL_SCORE_ALPHA:-0.2}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.siliconflow.cn/v1}"
LLM_API_KEY_ENV="${LLM_API_KEY_ENV:-SILICONFLOW_API_KEY}"

cd "${PROJECT_ROOT}" || exit 1
source .venv/bin/activate

if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: bash scripts/stage3_qa_inference_riemann.sh <pretrained_model_path> [all|hotpotqa_test,musique_test,2wikimultihopqa_test] [llm_name] [hydra_overrides...]"
    echo "Example: DATASETS=all DOC_RANKER=idf_score_topk_ranker DOC_RANKER_ENTITY_TOP_K=20 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/stage3_qa_inference_riemann.sh ${PROJECT_ROOT}/outputs/qa_finetune_riemann/<DATE>/<TIME>/pretrained"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_HOME="${CUDA_HOME%:}"
elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

if [ "${DATASETS_ARG}" = "all" ]; then
    DATASET_LIST="hotpotqa_test musique_test 2wikimultihopqa_test"
else
    DATASET_LIST="${DATASETS_ARG//,/ }"
fi

DOC_RANKER_TOP_K_ARG=()
if [ -n "${DOC_RANKER_ENTITY_TOP_K}" ]; then
    DOC_RANKER_TOP_K_ARG=(doc_ranker.top_k="${DOC_RANKER_ENTITY_TOP_K}")
fi

resolve_dataset() {
    local raw_name="$1"
    case "${raw_name}" in
        hotpotqa|hotpotqa_test)
            DATA_NAME="hotpotqa_test"
            QA_PROMPT="hotpotqa"
            QA_EVALUATOR="hotpotqa"
            ;;
        musique|musique_test)
            DATA_NAME="musique_test"
            QA_PROMPT="musique"
            QA_EVALUATOR="musique"
            ;;
        2wiki|2wikimultihopqa|2wikimultihopqa_test)
            DATA_NAME="2wikimultihopqa_test"
            QA_PROMPT="2wikimultihopqa"
            QA_EVALUATOR="2wikimultihopqa"
            ;;
        *)
            echo "Unknown dataset '${raw_name}'. Use all, hotpotqa_test, musique_test, or 2wikimultihopqa_test."
            exit 1
            ;;
    esac
}

for DATASET in ${DATASET_LIST}; do
    resolve_dataset "${DATASET}"
    echo "===== Stage3 inference: ${DATA_NAME} ====="
    CURRENT_RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE}"
    if [ "${DATA_NAME}" = "musique_test" ]; then
        CURRENT_RETRIEVAL_BATCH_SIZE="${MUSIQUE_RETRIEVAL_BATCH_SIZE}"
    fi
    echo "Retrieval batch size: ${CURRENT_RETRIEVAL_BATCH_SIZE}"

    torchrun --nproc_per_node="${N_GPU}" -m dgrag.workflow.stage3_qa_inference \
        dataset.root=./data \
        dataset.data_name="${DATA_NAME}" \
        +dataset.compute_structural_score=true \
        +dataset.structural_score_alpha="${STRUCTURAL_SCORE_ALPHA}" \
        graph_retriever.model_path="${MODEL_PATH}" \
        doc_ranker="${DOC_RANKER}" \
        qa_prompt="${QA_PROMPT}" \
        qa_evaluator="${QA_EVALUATOR}" \
        llm.model_name_or_path="${LLM_NAME}" \
        llm.base_url="${LLM_BASE_URL}" \
        llm.api_key_env="${LLM_API_KEY_ENV}" \
        test.retrieval_batch_size="${CURRENT_RETRIEVAL_BATCH_SIZE}" \
        test.top_k="${DOC_TOP_K}" \
        test.n_threads="${N_THREADS}" \
        test.save_retrieval="${SAVE_RETRIEVAL}" \
        "${DOC_RANKER_TOP_K_ARG[@]}" \
        "${EXTRA_ARGS[@]}"
done
