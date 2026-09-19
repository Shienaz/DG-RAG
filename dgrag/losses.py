from abc import ABC, abstractmethod
from typing import Any
import math

import torch
from torch import distributed as dist
from torch.nn import functional as F  # noqa:N812


class BaseLoss(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        pass


class BCELoss(BaseLoss):
    def __init__(
        self, adversarial_temperature: float = 0, *args: Any, **kwargs: Any
    ) -> None:
        self.adversarial_temperature = adversarial_temperature

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        is_positive = target > 0.5
        is_negative = target <= 0.5
        num_positive = is_positive.sum(dim=-1)
        num_negative = is_negative.sum(dim=-1)

        neg_weight = torch.zeros_like(pred)
        neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(
            num_positive
        )

        if self.adversarial_temperature > 0:
            from dgrag.ultra.variadic import variadic_softmax

            with torch.no_grad():
                logit = pred[is_negative] / self.adversarial_temperature
                neg_weight[is_negative] = variadic_softmax(logit, num_negative)
        else:
            neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(
                num_negative
            )
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()
        return loss


class ListCELoss(BaseLoss):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        target_sum = target.sum(dim=-1)
        non_zero_target_mask = target_sum != 0
        target_sum = target_sum[non_zero_target_mask]
        pred = pred[non_zero_target_mask]
        target = target[non_zero_target_mask]
        pred_prob = torch.sigmoid(pred)
        pred_prob_sum = pred_prob.sum(dim=-1, keepdim=True)
        loss = -torch.log((pred_prob / (pred_prob_sum + 1e-5)) + 1e-5) * target
        loss = loss.sum(dim=-1) / target_sum
        loss = loss.mean()
        return loss


class GeometryAlignmentLoss(BaseLoss):
    """Symmetric InfoNCE-style alignment loss between two embedding views."""

    def __init__(
        self, temperature: float = 0.2, *args: Any, **kwargs: Any
    ) -> None:
        self.temperature = temperature

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        if pred.size(0) <= 1 or target.size(0) <= 1:
            return pred.new_zeros(())

        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = pred @ target.transpose(0, 1)
        logits = logits / self.temperature

        labels = torch.arange(pred.size(0), device=pred.device)
        loss_left = F.cross_entropy(logits, labels)
        loss_right = F.cross_entropy(logits.transpose(0, 1), labels)
        return (loss_left + loss_right) / 2


class BranchDecisionMutualLoss(BaseLoss):
    """Mutual loss between Lorentz and Euclidean branch logits."""

    def __init__(
        self,
        temperature: float = 1.0,
        detach_target: bool = True,
        divergence: str = "kl",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if divergence not in {"kl", "js"}:
            raise ValueError("divergence must be 'kl' or 'js'")
        self.temperature = temperature
        self.detach_target = detach_target
        self.divergence = divergence

    @staticmethod
    def _js_from_log_probs(
        log_prob_left: torch.Tensor, log_prob_right: torch.Tensor
    ) -> torch.Tensor:
        log_mean = torch.logaddexp(log_prob_left, log_prob_right) - math.log(2.0)
        loss_left = F.kl_div(
            log_mean,
            log_prob_left,
            reduction="batchmean",
            log_target=True,
        )
        loss_right = F.kl_div(
            log_mean,
            log_prob_right,
            reduction="batchmean",
            log_target=True,
        )
        return 0.5 * (loss_left + loss_right)

    def _peer_divergence(
        self, log_prob_h: torch.Tensor, log_prob_e: torch.Tensor
    ) -> torch.Tensor:
        if self.divergence == "kl":
            target_h = log_prob_h.detach() if self.detach_target else log_prob_h
            target_e = log_prob_e.detach() if self.detach_target else log_prob_e
            loss_h = F.kl_div(
                log_prob_h, target_e, reduction="batchmean", log_target=True
            )
            loss_e = F.kl_div(
                log_prob_e, target_h, reduction="batchmean", log_target=True
            )
            return 0.5 * (loss_h + loss_e)

        if not self.detach_target:
            return self._js_from_log_probs(log_prob_h, log_prob_e)
        loss_h = self._js_from_log_probs(log_prob_h, log_prob_e.detach())
        loss_e = self._js_from_log_probs(log_prob_e, log_prob_h.detach())
        return 0.5 * (loss_h + loss_e)

    def _teacher_divergence(
        self,
        log_prob_h: torch.Tensor,
        log_prob_e: torch.Tensor,
        log_prob_teacher: torch.Tensor,
    ) -> torch.Tensor:
        target_teacher = (
            log_prob_teacher.detach() if self.detach_target else log_prob_teacher
        )
        if self.divergence == "kl":
            teacher_loss_h = F.kl_div(
                log_prob_h,
                target_teacher,
                reduction="batchmean",
                log_target=True,
            )
            teacher_loss_e = F.kl_div(
                log_prob_e,
                target_teacher,
                reduction="batchmean",
                log_target=True,
            )
            return 0.5 * (teacher_loss_h + teacher_loss_e)

        teacher_loss_h = self._js_from_log_probs(target_teacher, log_prob_h)
        teacher_loss_e = self._js_from_log_probs(target_teacher, log_prob_e)
        return 0.5 * (teacher_loss_h + teacher_loss_e)

    def __call__(
        self,
        score_h: torch.Tensor,
        score_e: torch.Tensor,
        teacher_score: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        branch_kl_weight: float = 0.2,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        zero = (score_h.sum() + score_e.sum()) * 0
        if teacher_score is not None:
            zero = zero + teacher_score.sum() * 0

        if score_h.size(-1) <= 1 or score_e.size(-1) <= 1:
            return zero
        if branch_kl_weight < 0:
            raise ValueError("branch_kl_weight must be non-negative")

        if candidate_mask is not None:
            if candidate_mask.shape != score_h.shape:
                raise ValueError("candidate_mask must have the same shape as scores")
            candidate_mask = candidate_mask.to(device=score_h.device, dtype=torch.bool)
            valid_row = candidate_mask.sum(dim=-1) >= 2
            if not bool(valid_row.any()):
                return zero
            score_h = score_h[valid_row].masked_fill(
                ~candidate_mask[valid_row], torch.finfo(score_h.dtype).min
            )
            score_e = score_e[valid_row].masked_fill(
                ~candidate_mask[valid_row], torch.finfo(score_e.dtype).min
            )
            if teacher_score is not None:
                teacher_score = teacher_score[valid_row].masked_fill(
                    ~candidate_mask[valid_row], torch.finfo(teacher_score.dtype).min
                )

        log_prob_h = F.log_softmax(score_h / self.temperature, dim=-1)
        log_prob_e = F.log_softmax(score_e / self.temperature, dim=-1)
        peer_loss = self._peer_divergence(log_prob_h, log_prob_e)

        if teacher_score is None:
            return peer_loss * (self.temperature**2)

        log_prob_teacher = F.log_softmax(teacher_score / self.temperature, dim=-1)
        teacher_loss = self._teacher_divergence(
            log_prob_h,
            log_prob_e,
            log_prob_teacher,
        )
        return (teacher_loss + branch_kl_weight * peer_loss) * (self.temperature**2)


class TopologyMutualLearningLoss(BaseLoss):
    """Mutual loss between sampled topology distributions from two branches."""

    def __init__(
        self,
        temperature: float = 0.5,
        sample_size: int = 64,
        kernel_degree: int = 2,
        kernel_bias: float = 0.0,
        mask_self: bool = True,
        detach_target: bool = True,
        divergence: str = "kl",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if kernel_degree < 1:
            raise ValueError("kernel_degree must be positive")
        if divergence not in {"kl", "js"}:
            raise ValueError("divergence must be 'kl' or 'js'")
        self.temperature = temperature
        self.sample_size = sample_size
        self.kernel_degree = kernel_degree
        self.kernel_bias = kernel_bias
        self.mask_self = mask_self
        self.detach_target = detach_target
        self.divergence = divergence
        self.last_sample_size = 0

    def _sample(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = pred.reshape(-1, pred.size(-1))
        target = target.reshape(-1, target.size(-1))
        if pred.size(0) <= self.sample_size:
            self.last_sample_size = pred.size(0)
            return pred, target

        index = torch.randperm(pred.size(0), device=pred.device)[: self.sample_size]
        self.last_sample_size = index.numel()
        return pred[index], target[index]

    def _log_topology_distribution(self, embedding: torch.Tensor) -> torch.Tensor:
        kernel = embedding @ embedding.transpose(0, 1)
        kernel = (kernel + self.kernel_bias).pow(self.kernel_degree)
        logits = kernel / self.temperature
        if self.mask_self and logits.size(0) > 1:
            eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
        return F.log_softmax(logits, dim=-1)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pred, target = self._sample(pred, target)
        if pred.size(0) <= 1 or target.size(0) <= 1:
            return (pred.sum() + target.sum()) * 0

        pred_log_dist = self._log_topology_distribution(pred)
        target_log_dist = self._log_topology_distribution(target)
        pred_target = pred_log_dist.detach() if self.detach_target else pred_log_dist
        target_target = (
            target_log_dist.detach() if self.detach_target else target_log_dist
        )

        if self.divergence == "js":
            if not self.detach_target:
                return BranchDecisionMutualLoss._js_from_log_probs(
                    pred_log_dist, target_log_dist
                )
            loss_pred = BranchDecisionMutualLoss._js_from_log_probs(
                pred_log_dist, target_target
            )
            loss_target = BranchDecisionMutualLoss._js_from_log_probs(
                target_log_dist, pred_target
            )
            return 0.5 * (loss_pred + loss_target)

        loss_pred = F.kl_div(
            pred_log_dist, target_target, reduction="batchmean", log_target=True
        )
        loss_target = F.kl_div(
            target_log_dist, pred_target, reduction="batchmean", log_target=True
        )
        return 0.5 * (loss_pred + loss_target)


class TopologySpecializationLoss(BaseLoss):
    """Positive-only specialization loss weighted by high-confidence branch gates."""

    def __init__(self, margin: float = 0.1, *args: Any, **kwargs: Any) -> None:
        self.margin = margin

    def __call__(
        self,
        score_h_pos: torch.Tensor,
        score_e_pos: torch.Tensor,
        weight_h_pos: torch.Tensor,
        weight_e_pos: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if score_h_pos.numel() == 0:
            return score_h_pos.new_zeros(())

        confidence = weight_h_pos + weight_e_pos
        loss = weight_h_pos * F.softplus(self.margin - (score_h_pos - score_e_pos))
        loss = loss + weight_e_pos * F.softplus(
            self.margin - (score_e_pos - score_h_pos)
        )
        return loss.sum() / confidence.sum().clamp_min(1.0)


class QuestionEntityContrastiveLoss(BaseLoss):
    """InfoNCE between question embeddings and entity prototypes."""

    def __init__(
        self, temperature: float = 0.2, *args: Any, **kwargs: Any
    ) -> None:
        self.temperature = temperature
        self.last_global_batch_size = 0
        self.last_local_valid_size = 0
        self.last_hard_negative_count = 0.0

    def _distributed_enabled(self) -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _gather_valid_with_local_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        local_count = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
        count_list = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(count_list, local_count)
        counts = [int(count.item()) for count in count_list]
        max_count = max(counts)
        if max_count == 0:
            return tensor.new_zeros((0, tensor.size(-1)))

        padded = tensor.new_zeros((max_count, tensor.size(-1)))
        if tensor.size(0) > 0:
            padded[: tensor.size(0)] = tensor

        gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, padded.detach())
        gathered[dist.get_rank()] = padded
        return torch.cat(
            [rank_tensor[:count] for rank_tensor, count in zip(gathered, counts)],
            dim=0,
        )

    def __call__(
        self,
        question_repr: torch.Tensor,
        node_tangent: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if question_repr.size(0) == 0:
            return (question_repr.sum() + node_tangent.sum()) * 0

        if negative_mask is not None:
            return self._hard_negative_loss(
                question_repr,
                node_tangent,
                positive_mask,
                negative_mask,
            )

        positive_mask = positive_mask.to(node_tangent.dtype)
        valid_mask = positive_mask.sum(dim=-1) > 0
        self.last_local_valid_size = int(valid_mask.sum().detach().item())
        if valid_mask.sum() == 0 and not self._distributed_enabled():
            self.last_global_batch_size = 0
            return (question_repr.sum() + node_tangent.sum()) * 0

        question_repr = question_repr[valid_mask]
        node_tangent = node_tangent[valid_mask]
        positive_mask = positive_mask[valid_mask]

        positive_proto = (node_tangent * positive_mask.unsqueeze(-1)).sum(dim=1)
        positive_proto = positive_proto / positive_mask.sum(dim=-1, keepdim=True).clamp_min(
            1.0
        )

        if self._distributed_enabled():
            question_repr = self._gather_valid_with_local_grad(question_repr)
            positive_proto = self._gather_valid_with_local_grad(positive_proto)

        self.last_global_batch_size = int(question_repr.size(0))
        if question_repr.size(0) <= 1:
            return (question_repr.sum() + positive_proto.sum()) * 0

        question_repr = F.normalize(question_repr, dim=-1)
        positive_proto = F.normalize(positive_proto, dim=-1)

        logits = question_repr @ positive_proto.transpose(0, 1)
        logits = logits / self.temperature
        labels = torch.arange(question_repr.size(0), device=question_repr.device)
        return F.cross_entropy(logits, labels)

    def _hard_negative_loss(
        self,
        question_repr: torch.Tensor,
        node_tangent: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        positive_mask = positive_mask.to(node_tangent.dtype)
        negative_mask = negative_mask.to(device=node_tangent.device, dtype=torch.bool)
        valid_mask = (positive_mask.sum(dim=-1) > 0) & negative_mask.any(dim=-1)
        self.last_local_valid_size = int(valid_mask.sum().detach().item())
        self.last_global_batch_size = self.last_local_valid_size
        if not bool(valid_mask.any()):
            self.last_hard_negative_count = 0.0
            return (question_repr.sum() + node_tangent.sum()) * 0

        question_repr = F.normalize(question_repr[valid_mask], dim=-1)
        node_tangent = F.normalize(node_tangent[valid_mask], dim=-1)
        positive_mask = positive_mask[valid_mask]
        negative_mask = negative_mask[valid_mask]

        positive_proto = (node_tangent * positive_mask.unsqueeze(-1)).sum(dim=1)
        positive_proto = positive_proto / positive_mask.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        positive_proto = F.normalize(positive_proto, dim=-1)
        positive_logit = (question_repr * positive_proto).sum(dim=-1, keepdim=True)

        negative_logits = torch.einsum("bd,bnd->bn", question_repr, node_tangent)
        negative_logits = negative_logits.masked_fill(
            ~negative_mask, torch.finfo(negative_logits.dtype).min
        )
        self.last_hard_negative_count = float(
            negative_mask.sum(dim=-1).float().mean().detach().item()
        )

        logits = torch.cat([positive_logit, negative_logits], dim=-1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
