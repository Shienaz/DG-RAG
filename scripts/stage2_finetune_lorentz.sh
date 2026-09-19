# Fine-tune the Lorentz single-branch GFM retriever on QA data.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
N_GPU=2
PRETRAINED_MODEL_PATH="${1}"

cd "${PROJECT_ROOT}" || exit 1
source .venv/bin/activate

if [ -z "${PRETRAINED_MODEL_PATH}" ]; then
    echo "Usage: bash scripts/stage2_finetune_lorentz.sh ${PROJECT_ROOT}/outputs/kg_pretrain_riemann/<DATE>/<TIME>/pretrained"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=${N_GPU} -m dgrag.workflow.stage2_qa_finetune \
    --config-name stage2_qa_finetune_lorentz \
    datasets.cfgs.root=./data \
    datasets.train_names=[hotpotqa_train_example] \
    datasets.valid_names=[hotpotqa_test,musique_test,2wikimultihopqa_test] \
    train.checkpoint="${PRETRAINED_MODEL_PATH}" \
    train.num_epoch=25 \
    train.batch_size=2 \
    optimizer.lr=5e-4 \
    model.use_question_relation_adapter=false \
    model.use_question_entity_contrastive=true \
    model.question_entity_contrastive_weight=0.1
