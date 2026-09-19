# IRCoT + GFM-RAG inference on QA tasks
N_GPU=1
DATA_ROOT="data"
DATA_NAME="hotpotqa" # hotpotqa musique 2wikimultihopqa
LLM="deepseek-ai/DeepSeek-V3.2"
MAX_STEPS=3
MAX_SAMPLE=-1
python -m dgrag.workflow.stage3_qa_ircot_inference \
    dataset.root=${DATA_ROOT} \
    llm.model_name_or_path=${LLM} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    agent_prompt=${DATA_NAME}_ircot \
    test.max_steps=${MAX_STEPS} \
    test.max_test_samples=${MAX_SAMPLE} \
    dataset.data_name=${DATA_NAME}_test
