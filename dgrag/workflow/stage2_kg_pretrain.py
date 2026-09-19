import logging
import math
import os
from itertools import islice

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.utils import data as torch_data
from torch_geometric.data import Data
from tqdm import tqdm

from dgrag import utils
from dgrag.datasets import KGDataset
from dgrag.ultra import tasks
from dgrag.utils import GraphDatasetLoader
from dgrag.utils.distributed_batch import (
    broadcast_object,
    expand_single_batch_for_bidirectional_negatives,
    next_shared_batch,
    require_distributed_single_batch_mode,
    use_single_batch_edge_parallel,
    validate_single_batch_edge_parallel_config,
)

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


def get_auxiliary_loss(model: nn.Module) -> torch.Tensor | None:
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "get_auxiliary_loss"):
        return raw_model.get_auxiliary_loss()
    return None


def get_auxiliary_stats(model: nn.Module) -> dict[str, float]:
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "get_auxiliary_stats_dict"):
        return raw_model.get_auxiliary_stats_dict()
    return {}


def find_first_non_finite_gradient(
    model: nn.Module,
) -> tuple[str, int] | None:
    raw_model = model.module if hasattr(model, "module") else model
    for name, parameter in raw_model.named_parameters():
        if parameter.grad is None:
            continue
        finite_mask = torch.isfinite(parameter.grad)
        if not finite_mask.all():
            return name, int((~finite_mask).sum().item())
    return None


def find_first_non_finite_parameter(
    model: nn.Module,
) -> tuple[str, int] | None:
    raw_model = model.module if hasattr(model, "module") else model
    for name, parameter in raw_model.named_parameters():
        finite_mask = torch.isfinite(parameter)
        if not finite_mask.all():
            return name, int((~finite_mask).sum().item())
    return None


def set_sampler_epoch_if_supported(sampler: object | None, epoch: int) -> None:
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def create_kgc_dataset(
    dataset: dict[str, KGDataset],
    batch_size: int,
    world_size: int,
    rank: int,
    is_train: bool = True,
    shuffle: bool = True,
    fast_test: None | int = None,
    shared_batch: bool = False,
) -> dict:
    data_name = dataset["data_name"]
    graph = dataset["data"][0]

    # The original triples is used for ranking evaluation
    val_filtered_data = Data(
        edge_index=graph.target_edge_index,
        edge_type=graph.target_edge_type,
        num_nodes=graph.num_nodes,
    )

    # Create a DataLoader for triples
    if not is_train and fast_test is not None:
        mask = torch.randperm(graph.target_edge_index.shape[1])[:fast_test]
        sampled_target_edge_index = graph.target_edge_index[:, mask]
        sampled_target_edge_type = graph.target_edge_type[mask]
        triples = torch.cat(
            [sampled_target_edge_index, sampled_target_edge_type.unsqueeze(0)]
        ).t()
    else:
        triples = torch.cat(
            [graph.target_edge_index, graph.target_edge_type.unsqueeze(0)]
        ).t()
    sampler = None
    if not shared_batch:
        sampler = torch_data.DistributedSampler(
            triples, world_size, rank, shuffle=shuffle
        )
    data_loader = torch_data.DataLoader(
        triples,
        batch_size,
        sampler=sampler,
        shuffle=shuffle if shared_batch else False,
    )
    val_filtered_data = Data(
        edge_index=graph.target_edge_index,
        edge_type=graph.target_edge_type,
        num_nodes=graph.num_nodes,
    )

    return {
        "data_name": data_name,
        "graph": graph,
        "val_filtered_data": val_filtered_data,
        "data_loader": data_loader,
    }


def train_and_validate(
    cfg: DictConfig,
    output_dir: str,
    model: nn.Module,
    dataset_loader: GraphDatasetLoader,
    device: torch.device,
    batch_per_epoch: int | None = None,
) -> None:
    if cfg.train.num_epoch == 0:
        return

    world_size = utils.get_world_size()
    rank = utils.get_rank()
    shared_batch = use_single_batch_edge_parallel(cfg)
    require_distributed_single_batch_mode(cfg)
    validate_single_batch_edge_parallel_config(cfg)

    optimizer = instantiate(cfg.optimizer, model.parameters())
    max_grad_norm = cfg.train.max_grad_norm if "max_grad_norm" in cfg.train else None
    start_epoch = 0
    # Load optimizer state and epoch if exists
    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(
                cfg.train.checkpoint, map_location="cpu", weights_only=True
            )
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            else:
                logger.warning(
                    f"Optimizer state not found in {cfg.train.checkpoint}, using default optimizer."
                )
            if "epoch" in state:
                start_epoch = state["epoch"]
                logger.warning(f"Resuming training from epoch {start_epoch}.")
        else:
            logger.warning(
                f"Checkpoint {cfg.train.checkpoint} does not exist, using default optimizer."
            )

    find_unused_parameters = (
        cfg.train.find_unused_parameters
        if "find_unused_parameters" in cfg.train
        else False
    )
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=find_unused_parameters,
        )
    else:
        parallel_model = model

    best_result = float("-inf")
    best_epoch = -1

    # How often to run the per-parameter non-finite guards. Kept out of the
    # shipped YAMLs so the composed configs stay byte-identical across ablations;
    # pass ++train.nonfinite_check_interval=N to thin them out.
    check_every = 1
    if "nonfinite_check_interval" in cfg.train:
        check_every = max(1, int(cfg.train.nonfinite_check_interval))

    batch_id = 0
    for i in range(start_epoch, cfg.train.num_epoch):
        epoch = i + 1
        parallel_model.train()

        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info(f"Epoch {epoch} begin")

        losses = []
        dataset_loader.set_epoch(
            epoch
        )  # Make sure the datasets order is the same across all processes
        for train_dataset in dataset_loader:
            train_dataset = create_kgc_dataset(
                train_dataset,
                cfg.train.batch_size,
                world_size,
                rank,
                is_train=True,
                shuffle=True,
                shared_batch=shared_batch,
            )
            data_name = train_dataset["data_name"]
            train_loader = train_dataset["data_loader"]
            set_sampler_epoch_if_supported(train_loader.sampler, epoch)
            train_graph = train_dataset["graph"].to(device)
            if shared_batch:
                total_batches = batch_per_epoch or len(train_loader)
                batch_iterator = iter(train_loader) if rank == 0 else None
                batch_iterable = tqdm(
                    range(total_batches),
                    desc=f"Training Batches: {data_name}: {epoch}",
                    total=total_batches,
                    disable=not utils.is_main_process(),
                )
            else:
                batch_iterator = None
                batch_iterable = tqdm(
                    islice(train_loader, batch_per_epoch),
                    desc=f"Training Batches: {data_name}: {epoch}",
                    total=batch_per_epoch,
                )
            for batch_item in batch_iterable:
                if shared_batch:
                    raw_batch = next_shared_batch(batch_iterator)
                    if raw_batch is None:
                        break
                    if rank == 0:
                        raw_batch = expand_single_batch_for_bidirectional_negatives(
                            raw_batch
                        )
                        sampled_batch = tasks.negative_sampling(
                            train_graph,
                            raw_batch.to(device),
                            cfg.task.num_negative,
                            strict=cfg.task.strict_negative,
                        ).cpu()
                    else:
                        sampled_batch = None
                    batch = broadcast_object(sampled_batch).to(device)
                else:
                    batch = batch_item.to(device)
                    batch = tasks.negative_sampling(
                        train_graph,
                        batch,
                        cfg.task.num_negative,
                        strict=cfg.task.strict_negative,
                    )
                pred = parallel_model(train_graph, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(
                            pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1
                        )
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                bce_loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                bce_loss = bce_loss.mean()
                auxiliary_loss = get_auxiliary_loss(parallel_model)
                auxiliary_stats = get_auxiliary_stats(parallel_model)
                loss = bce_loss if auxiliary_loss is None else bce_loss + auxiliary_loss

                if not torch.isfinite(loss).all():
                    logger.error(
                        "Non-finite loss detected on rank %s at batch %s in epoch %s",
                        rank,
                        batch_id,
                        epoch,
                    )
                    raise FloatingPointError("Non-finite loss encountered during KG pretraining")

                loss.backward()
                # Both guards walk every parameter and truth-test a CUDA tensor,
                # which forces one host synchronization per parameter (~2 x 140
                # for this model). That is pure overhead when the step itself is
                # sub-second. Default 1 preserves the historical every-step
                # behaviour; raise it via ++train.nonfinite_check_interval=N.
                if check_every == 1 or batch_id % check_every == 0:
                    bad_grad = find_first_non_finite_gradient(parallel_model)
                else:
                    bad_grad = None
                if bad_grad is not None:
                    bad_name, bad_count = bad_grad
                    logger.error(
                        "Non-finite gradient detected on rank %s at batch %s in epoch %s for parameter %s (%s bad values)",
                        rank,
                        batch_id,
                        epoch,
                        bad_name,
                        bad_count,
                    )
                    raise FloatingPointError(
                        f"Non-finite gradient encountered for parameter {bad_name}"
                    )

                if max_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(
                        parallel_model.parameters(), max_grad_norm
                    )
                    grad_norm = torch.as_tensor(grad_norm, device=device)
                    if not torch.isfinite(grad_norm).all():
                        logger.error(
                            "Non-finite gradient norm detected on rank %s at batch %s in epoch %s",
                            rank,
                            batch_id,
                            epoch,
                        )
                        raise FloatingPointError(
                            "Non-finite gradient norm encountered during KG pretraining"
                        )
                optimizer.step()
                if check_every == 1 or batch_id % check_every == 0:
                    bad_param = find_first_non_finite_parameter(parallel_model)
                else:
                    bad_param = None
                if bad_param is not None:
                    bad_name, bad_count = bad_param
                    logger.error(
                        "Non-finite parameter detected on rank %s at batch %s in epoch %s for parameter %s (%s bad values)",
                        rank,
                        batch_id,
                        epoch,
                        bad_name,
                        bad_count,
                    )
                    raise FloatingPointError(
                        f"Non-finite parameter encountered for parameter {bad_name}"
                    )
                optimizer.zero_grad()

                if utils.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.info(separator)
                    logger.info(f"binary cross entropy: {bce_loss:g}")
                    for key in [
                        "alignment_loss",
                        "decision_mutual_loss",
                        "topology_mutual_loss",
                        "specialization_loss",
                        "branch_score_gap",
                        "topology_sample_size",
                        "pos_score_gap",
                        "high_conf_lorentz_positive_rate",
                        "high_conf_euclidean_positive_rate",
                        "positive_residual_confidence",
                    ]:
                        if key in auxiliary_stats:
                            logger.info(f"{key.replace('_', ' ')}: {auxiliary_stats[key]:g}")
                losses.append(bce_loss.item())
                batch_id += 1

        if utils.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.info(separator)
            logger.info(f"Epoch {epoch} end")
            logger.info(line)
            logger.info(f"average binary cross entropy: {avg_loss:g}")

        utils.synchronize()
        if rank == 0:
            logger.info(separator)
            logger.info("Evaluate on valid")

        result = test(
            cfg,
            model,
            dataset_loader,
            device=device,
        )
        if rank == 0:
            if result > best_result:
                best_result = result
                best_epoch = epoch
                logger.info("Save checkpoint to model_best.pth")
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(output_dir, "model_best.pth"))
            if not cfg.train.save_best_only:
                logger.info(f"Save checkpoint to model_epoch_{epoch}.pth")
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(output_dir, f"model_epoch_{epoch}.pth"))
            logger.info(f"Best mrr: {best_result:g} at epoch {best_epoch}")
    utils.synchronize()
    if rank == 0:
        logger.info("Load checkpoint from model_best.pth")
    state = torch.load(
        os.path.join(output_dir, "model_best.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state["model"])
    utils.synchronize()


@torch.no_grad()
def test(
    cfg: DictConfig,
    model: nn.Module,
    test_dataset_loader: GraphDatasetLoader,
    device: torch.device,
    return_metrics: bool = False,
) -> float | dict:
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    shared_batch = use_single_batch_edge_parallel(cfg)

    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics = {}
    all_mrr = []
    test_dataset_loader.set_epoch(0)
    for test_dataset in test_dataset_loader:
        test_dataset = create_kgc_dataset(
            test_dataset,
            cfg.train.batch_size,
            world_size,
            rank,
            is_train=False,
            shuffle=False,
            fast_test=cfg.train.fast_test if "fast_test" in cfg.train else None,
            shared_batch=shared_batch,
        )
        test_data_name = test_dataset["data_name"]
        test_graph = test_dataset["graph"].to(device)
        test_loader = test_dataset["data_loader"]
        set_sampler_epoch_if_supported(test_loader.sampler, 0)
        filtered_data = test_dataset["val_filtered_data"].to(device)

        model.eval()
        rankings = []
        num_negatives = []
        tail_rankings, num_tail_negs = (
            [],
            [],
        )  # for explicit tail-only evaluation needed for 5 datasets
        if shared_batch:
            batch_iterator = iter(test_loader) if rank == 0 else None
            batch_iterable = tqdm(
                range(len(test_loader)),
                disable=not utils.is_main_process(),
            )
        else:
            batch_iterator = None
            batch_iterable = tqdm(test_loader)
        for batch_item in batch_iterable:
            batch = (
                next_shared_batch(batch_iterator)
                if shared_batch
                else batch_item
            )
            if batch is None:
                break
            batch = batch.to(device)
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred = model(test_graph, t_batch)
            h_pred = model(test_graph, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            if not shared_batch or rank == 0:
                rankings += [t_ranking, h_ranking]
                num_negatives += [num_t_negative, num_h_negative]

                tail_rankings += [t_ranking]
                num_tail_negs += [num_t_negative]

        if shared_batch:
            metrics = {}
            if rank == 0:
                ranking = torch.cat(rankings)
                num_negative = torch.cat(num_negatives)
                tail_ranking = torch.cat(tail_rankings)
                num_tail_neg = torch.cat(num_tail_negs)
                logger.info(f"{'-' * 15} Test on {test_data_name} {'-' * 15}")
                for metric in cfg.task.metric:
                    if "-tail" in metric:
                        _metric_name, direction = metric.split("-")
                        if direction != "tail":
                            raise ValueError("Only tail metric is supported in this mode")
                        _ranking = tail_ranking
                        _num_neg = num_tail_neg
                    else:
                        _ranking = ranking
                        _num_neg = num_negative
                        _metric_name = metric

                    if _metric_name == "mr":
                        score = _ranking.float().mean()
                    elif _metric_name == "mrr":
                        score = (1 / _ranking.float()).mean()
                    elif _metric_name.startswith("hits@"):
                        values = _metric_name[5:].split("_")
                        threshold = int(values[0])
                        if len(values) > 1:
                            num_sample = int(values[1])
                            fp_rate = (_ranking - 1).float() / _num_neg
                            score = 0
                            for i in range(threshold):
                                num_comb = (
                                    math.factorial(num_sample - 1)
                                    / math.factorial(i)
                                    / math.factorial(num_sample - i - 1)
                                )
                                score += (
                                    num_comb
                                    * (fp_rate**i)
                                    * ((1 - fp_rate) ** (num_sample - i - 1))
                                )
                            score = score.mean()
                        else:
                            score = (_ranking <= threshold).float().mean()
                    logger.info(f"{metric}: {score:g}")
                    metrics[metric] = score
                metrics["mrr"] = (1 / ranking.float()).mean()
                logger.info(separator)
            metrics_list = [metrics if rank == 0 else None]
            dist.broadcast_object_list(metrics_list, src=0)
            metrics = metrics_list[0]
            all_mrr.append(metrics["mrr"])
            all_metrics[test_data_name] = metrics
            continue

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)

        # ugly repetitive code for tail-only ranks processing
        tail_ranking = torch.cat(tail_rankings)
        num_tail_neg = torch.cat(num_tail_negs)
        all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size_t[rank] = len(tail_ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

        # obtaining all ranks
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank] : cum_size[rank]] = (
            num_negative
        )

        # the same for tails-only ranks
        cum_size_t = all_size_t.cumsum(0)
        all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
        all_ranking_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = (
            tail_ranking
        )
        all_num_negative_t = torch.zeros(
            all_size_t.sum(), dtype=torch.long, device=device
        )
        all_num_negative_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = (
            num_tail_neg
        )
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

        metrics = {}
        if rank == 0:
            logger.info(f"{'-' * 15} Test on {test_data_name} {'-' * 15}")
            for metric in cfg.task.metric:
                if "-tail" in metric:
                    _metric_name, direction = metric.split("-")
                    if direction != "tail":
                        raise ValueError("Only tail metric is supported in this mode")
                    _ranking = all_ranking_t
                    _num_neg = all_num_negative_t
                else:
                    _ranking = all_ranking
                    _num_neg = all_num_negative
                    _metric_name = metric

                if _metric_name == "mr":
                    score = _ranking.float().mean()
                elif _metric_name == "mrr":
                    score = (1 / _ranking.float()).mean()
                elif _metric_name.startswith("hits@"):
                    values = _metric_name[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (_ranking - 1).float() / _num_neg
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = (
                                math.factorial(num_sample - 1)
                                / math.factorial(i)
                                / math.factorial(num_sample - i - 1)
                            )
                            score += (
                                num_comb
                                * (fp_rate**i)
                                * ((1 - fp_rate) ** (num_sample - i - 1))
                            )
                        score = score.mean()
                    else:
                        score = (_ranking <= threshold).float().mean()
                logger.info(f"{metric}: {score:g}")
                metrics[metric] = score
        mrr = (1 / all_ranking.float()).mean()
        all_mrr.append(mrr)
        all_metrics[test_data_name] = metrics
        if rank == 0:
            logger.info(separator)
    avg_mrr = sum(all_mrr) / len(all_mrr)
    return avg_mrr if not return_metrics else all_metrics


@hydra.main(config_path="config", config_name="stage2_kg_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.train.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        output_dir = HydraConfig.get().runtime.output_dir
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        output_dir_list = [output_dir]
    else:
        output_dir_list = [None]
    if utils.get_world_size() > 1:
        dist.broadcast_object_list(
            output_dir_list, src=0
        )  # Use the output dir from rank 0
    output_dir = output_dir_list[0]

    shared_batch = use_single_batch_edge_parallel(cfg)
    # Initialize the datasets in the each process, make sure they are processed
    if cfg.datasets.init_datasets:
        if shared_batch:
            rel_emb_dim_list = (
                utils.init_multi_dataset(cfg, 1, 0) if utils.get_rank() == 0 else []
            )
            if utils.get_world_size() > 1:
                rel_emb_dim_objects = [rel_emb_dim_list]
                dist.broadcast_object_list(rel_emb_dim_objects, src=0)
                rel_emb_dim_list = rel_emb_dim_objects[0]
        else:
            rel_emb_dim_list = utils.init_multi_dataset(
                cfg, utils.get_world_size(), utils.get_rank()
            )
        rel_emb_dim = set(rel_emb_dim_list)
        assert len(rel_emb_dim) == 1, (
            "All datasets should have the same relation embedding dimension"
        )
    else:
        assert cfg.datasets.feat_dim is not None, (
            "If datasets.init_datasets is False, cfg.datasets.feat_dim must be set"
        )
        rel_emb_dim = {cfg.datasets.feat_dim}

    device = utils.get_device()
    model = instantiate(cfg.model, rel_emb_dim=rel_emb_dim.pop())

    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(
                cfg.train.checkpoint, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state["model"])
        # Try to load the model from the remote dictionary
        else:
            model, _ = utils.load_model_from_pretrained(cfg.train.checkpoint)

    model = model.to(device)

    if utils.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(line)
        logger.info(f"Number of parameters: {num_params}")

    train_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.train_names,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )

    train_and_validate(
        cfg,
        output_dir,
        model,
        dataset_loader=train_dataset_loader,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )

    if utils.get_rank() == 0:
        logger.info(separator)
        logger.info("Evaluate on valid")
    test(cfg, model, train_dataset_loader, device=device)

    # Save the model into the format for QA inference
    if utils.is_main_process() and cfg.train.save_pretrained:
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)

    # Shutdown the dataset loaders
    train_dataset_loader.shutdown()

    utils.synchronize()
    utils.cleanup()


if __name__ == "__main__":
    main()
