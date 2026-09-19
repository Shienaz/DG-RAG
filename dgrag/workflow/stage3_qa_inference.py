import json
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from tqdm import tqdm

from dgrag import utils
from dgrag.datasets import QADataset
from dgrag.prompt_builder import QAPromptBuilder
from dgrag.ultra import query_utils
from dgrag.utils.distributed_batch import next_shared_batch

# A logger for this file
logger = logging.getLogger(__name__)


def _sample_id(value: int | torch.Tensor) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _uses_edge_parallel(model: nn.Module) -> bool:
    module = getattr(model, "module", model)
    entity_model = getattr(module, "entity_model", None)
    return bool(getattr(entity_model, "use_edge_parallel", False))


def _deduplicate_and_sort_predictions(predictions: list[dict]) -> list[dict]:
    predictions_by_id = {}
    for prediction in predictions:
        sample_id = _sample_id(prediction["id"])
        if sample_id not in predictions_by_id:
            prediction = dict(prediction)
            prediction["id"] = sample_id
            predictions_by_id[sample_id] = prediction
    return [predictions_by_id[i] for i in sorted(predictions_by_id)]


def _build_prediction_inputs(
    test_data: list[dict], retrieval_result: list[dict]
) -> list[tuple[dict, dict]]:
    retrieval_result = _deduplicate_and_sort_predictions(retrieval_result)
    return [
        (test_data[_sample_id(retrieval_doc["id"])], retrieval_doc)
        for retrieval_doc in retrieval_result
    ]


@torch.no_grad()
def doc_retrieval(
    cfg: DictConfig,
    model: nn.Module,
    qa_data: Dataset,
    device: torch.device,
) -> list[dict]:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    _, test_data = qa_data._data
    graph = qa_data.kg
    ent2docs = qa_data.ent2docs

    # Edge-parallel models shard graph edges across ranks and then all-reduce the
    # per-rank message passing result. All ranks must therefore see the same query
    # batch; sample-parallel DistributedSampler would mix different questions.
    shared_edge_parallel = world_size > 1 and _uses_edge_parallel(model)
    if shared_edge_parallel:
        if utils.is_main_process():
            logger.info(
                "Using shared-batch retrieval because the graph retriever enables "
                "edge-parallel message passing."
            )
        sampler = None
    else:
        sampler = torch_data.DistributedSampler(
            test_data, world_size, rank, shuffle=False
        )
    test_loader = torch_data.DataLoader(
        test_data, cfg.test.retrieval_batch_size, sampler=sampler
    )

    # Create doc retriever
    doc_ranker = instantiate(cfg.doc_ranker, ent2doc=ent2docs)

    if cfg.test.init_entities_weight:
        entities_weight = utils.get_entities_weight(ent2docs)
    else:
        entities_weight = None

    model.eval()
    all_predictions: list[dict] = []
    if shared_edge_parallel:
        batch_iterator = iter(test_loader) if utils.is_main_process() else None
        batch_iterable = tqdm(
            range(len(test_loader)), disable=not utils.is_main_process()
        )
    else:
        batch_iterator = None
        batch_iterable = tqdm(test_loader, disable=not utils.is_main_process())

    for batch_item in batch_iterable:
        batch = (
            next_shared_batch(batch_iterator)
            if shared_edge_parallel
            else batch_item
        )
        if batch is None:
            break
        batch = query_utils.cuda(batch, device=device)
        ent_pred = model(graph, batch, entities_weight=entities_weight)
        doc_pred = doc_ranker(ent_pred)  # Ent2docs mapping
        if not shared_edge_parallel or utils.is_main_process():
            idx = batch["sample_id"]
            all_predictions.extend(
                {"id": _sample_id(i), "ent_pred": e, "doc_pred": d}
                for i, e, d in zip(idx.cpu(), ent_pred.cpu(), doc_pred.cpu())
            )

    if shared_edge_parallel:
        sorted_predictions = _deduplicate_and_sort_predictions(all_predictions)
        utils.synchronize()
        return sorted_predictions

    # Gather the predictions across all processes
    if utils.get_world_size() > 1:
        gathered_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)
    else:
        gathered_predictions = [all_predictions]  # type: ignore

    sorted_predictions = _deduplicate_and_sort_predictions(
        [item for sublist in gathered_predictions for item in sublist]  # type: ignore
    )
    utils.synchronize()
    return sorted_predictions


def ans_prediction(
    cfg: DictConfig, output_dir: str, qa_data: Dataset, retrieval_result: list[dict]
) -> str:
    llm = instantiate(cfg.llm)
    doc_retriever = utils.DocumentRetriever(qa_data.doc, qa_data.id2doc)
    test_data = qa_data.raw_test_data
    id2ent = {v: k for k, v in qa_data.ent2id.items()}

    prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    prediction_inputs = _build_prediction_inputs(test_data, retrieval_result)

    def predict(qa_input: tuple[dict, dict]) -> dict | Exception:
        data, retrieval_doc = qa_input
        retrieved_ent_idx = torch.topk(
            retrieval_doc["ent_pred"], cfg.test.save_top_k_entity, dim=-1
        ).indices
        retrieved_ent = [id2ent[i.item()] for i in retrieved_ent_idx]
        retrieved_docs = doc_retriever(retrieval_doc["doc_pred"], top_k=cfg.test.top_k)

        message = prompt_builder.build_input_prompt(data["question"], retrieved_docs)

        response = llm.generate_sentence(message)
        if isinstance(response, Exception):
            return response
        else:
            return {
                "id": data["id"],
                "question": data["question"],
                "answer": data["answer"],
                "answer_aliases": data.get(
                    "answer_aliases", []
                ),  # Some datasets have answer aliases
                "response": response,
                "retrieved_ent": retrieved_ent,
                "retrieved_docs": retrieved_docs,
            }

    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        with ThreadPool(cfg.test.n_threads) as pool:
            for results in tqdm(
                pool.imap(predict, prediction_inputs),
                total=len(prediction_inputs),
            ):
                if isinstance(results, Exception):
                    logger.error(f"Error: {results}")
                    continue

                f.write(json.dumps(results) + "\n")
                f.flush()

    return os.path.join(output_dir, "prediction.jsonl")


@hydra.main(config_path="config", config_name="stage3_qa_inference", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode()
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    model, model_config = utils.load_model_from_pretrained(
        cfg.graph_retriever.model_path
    )
    qa_data = QADataset(
        **cfg.dataset,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
    )
    device = utils.get_device()
    model = model.to(device)

    qa_data.kg = qa_data.kg.to(device)
    qa_data.ent2docs = qa_data.ent2docs.to(device)

    if cfg.test.retrieved_result_path:
        retrieval_result = torch.load(cfg.test.retrieved_result_path, weights_only=True)
    else:
        if cfg.test.prediction_result_path:
            retrieval_result = None
        else:
            retrieval_result = doc_retrieval(cfg, model, qa_data, device=device)
    if utils.is_main_process():
        if cfg.test.save_retrieval and retrieval_result is not None:
            logger.info(
                f"Ranking saved to disk: {os.path.join(output_dir, 'retrieval_result.pt')}"
            )
            torch.save(
                retrieval_result, os.path.join(output_dir, "retrieval_result.pt")
            )
        if cfg.test.prediction_result_path:
            output_path = cfg.test.prediction_result_path
        else:
            output_path = ans_prediction(cfg, output_dir, qa_data, retrieval_result)

        # Evaluation
        evaluator = instantiate(cfg.qa_evaluator, prediction_file=output_path)
        metrics = evaluator.evaluate()
        query_utils.print_metrics(metrics, logger)
        return metrics


if __name__ == "__main__":
    main()
