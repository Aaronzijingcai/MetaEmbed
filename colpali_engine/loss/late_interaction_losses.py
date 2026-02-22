# instead of Max-Sum, we implement another late interaction method named as multi-token
# solution, where we use Sum-Sum so each of the token can contribute to the final score

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from colpali_engine.utils.dist_utils import rank0_print
from torch.nn import CrossEntropyLoss


class MultiTokenModule(torch.nn.Module):
    """
    Multi-token loss with late interactions.

    Args:
        max_batch_size (int): Maximum batch size for pre-allocating index buffer.
        tau (float): Temperature for smooth-max approximation.
        norm_tol (float): Tolerance for score normalization bounds.
        filter_threshold (float): Ratio threshold for pos-aware negative filtering.
        filter_factor (float): Multiplicative factor to down-weight high negatives.
    """

    def __init__(
        self,
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        self.register_buffer(
            "idx_buffer", torch.arange(max_batch_size), persistent=False
        )
        self.tau = tau
        self.norm_tol = norm_tol
        self.filter_threshold = filter_threshold
        self.filter_factor = filter_factor

    def _get_idx(self, batch_size: int, offset: int, device: torch.device):
        """
        Retrieve index and positive index tensors for in-batch losses.
        """
        idx = self.idx_buffer[:batch_size].to(device)
        return idx, idx + offset

    def _smooth_max(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute smooth max via log-sum-exp along a given dimension.
        """
        return self.tau * torch.logsumexp(scores / self.tau, dim=dim)

    # remove normalization -- our query is with a fixed length

    def _aggregate(
        self,
        scores_raw: torch.Tensor,
        use_smooth_max: bool,
        dim_max: int,
        dim_sum: int,
    ) -> torch.Tensor:
        """
        Aggregate token-level scores into document-level.

        Args:
            scores_raw (Tensor): Raw scores tensor.
            use_smooth_max (bool): Use smooth-max if True.
            dim_max (int): Dimension to perform max/logsumexp.
            dim_sum (int): Dimension to sum over after max.
        """
        if use_smooth_max:
            return self._smooth_max(scores_raw, dim=dim_max).sum(dim=dim_sum)
        return scores_raw.amax(dim=dim_max).sum(dim=dim_sum)

    def _filter_high_negatives(
        self, scores: torch.Tensor, pos_idx: torch.Tensor
    ) -> None:
        """
        Down-weight negatives whose score exceeds a fraction of the positive score.

        Args:
            scores (Tensor): In-batch score matrix [B, B].
            pos_idx (Tensor): Positive indices for each query in batch.
        """
        batch_size = scores.size(0)
        idx = self.idx_buffer[:batch_size].to(scores.device)
        pos_scores = scores[idx, pos_idx]
        thresh = self.filter_threshold * pos_scores.unsqueeze(1)
        mask = scores > thresh
        mask[idx, pos_idx] = False
        scores[mask] *= self.filter_factor


class NewColbertLoss(ColbertModule):
    """
    InfoNCE loss for late interaction (ColBERT) without explicit negatives.

    Args:
        temperature (float): Scaling factor for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        mrl_groups: List[
            Tuple[int, int, float]
        ] = None,  # (num_query_tokens, num_doc_tokens, weight)
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering

        self.mrl_groups = mrl_groups
        if self.mrl_groups:
            rank0_print(f"Using mrl_groups = {self.mrl_groups}")
        self.ce_loss = CrossEntropyLoss()

    def get_raw_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ):
        return torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)

    def get_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        offset: int = 0,
    ):
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = self.get_raw_scores(query_embeddings, doc_embeddings)
        return self.get_sub_loss_by_raw(raw, lengths, offset)

    def get_sub_loss_by_raw(self, raw, lengths, offset=0):
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        return self.ce_loss(scores / self.temperature, pos_idx)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        offset: int = 0,
        temperature: Optional[torch.Tensor] = None,
        # if given, learnable temperature is maintained
    ):
        """
        Compute ColBERT InfoNCE loss over a batch of queries and documents
        Optionally, MRL on tokens could be used.
        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B * num_gpus, Nd, D]
            offset (int): Offset for positive doc indices (multi-GPU).

        Returns:
            Tensor: Scalar loss value.
        """
        # if mrl_groups are provided, compute sub-losses and return a dict of losses.
        if self.mrl_groups is not None:
            loss = 0.0
            loss_stats = {}
            raw = self.get_raw_scores(query_embeddings, doc_embeddings)
            for num_query_tokens, num_doc_tokens, weight in self.mrl_groups:
                # reuse raw to avoid duplicated matrix multiplication
                sub_raw = raw[:, :, :num_query_tokens, :num_doc_tokens]
                lengths = (query_embeddings[:, :num_query_tokens, 0] != 0).sum(dim=1)
                subloss = self.get_sub_loss_by_raw(sub_raw, lengths, offset)
                loss += subloss * weight
                loss_stats[f"mrl_{num_query_tokens}_{num_doc_tokens}_{weight}"] = (
                    subloss
                )

            # normalize loss by number of groups
            loss = loss / len(self.mrl_groups)

            return loss, loss_stats

        # otherwise, compute the full loss.
        return self.get_loss(query_embeddings, doc_embeddings, offset)


# In-batch version of ColBERT CE loss with explicit negatives appended in the matrix.
# should perfectly aligned with MoCa multiple negatives design
class ColbertInBatchNegativeLoss(ColbertModule):
    """
    InfoNCE loss with explicit negative documents -- aligned with MoCa setting

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        mrl_groups: List[Tuple[int, int, float]] = None,
    ):

        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()
        self.mrl_groups = mrl_groups
        if self.mrl_groups:
            rank0_print(f"Using mrl_groups = {self.mrl_groups}")

    # NOTE: doc_embeddings is gathered, and neg_doc_embeddings can have
    # multiple negatives per query. Assuming them with the same B is not correct.
    def get_raw_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
    ):
        # return torch.einsum(
        #     "bnd,bsd->bns", query_embeddings, doc_embeddings
        # ), torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings)
        return torch.einsum(
            "bnd,csd->bcns", query_embeddings, doc_embeddings
        ), torch.einsum("bnd,gsd->bgns", query_embeddings, neg_doc_embeddings)

    def get_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ):
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw, neg_raw = self.get_raw_scores(
            query_embeddings, doc_embeddings, neg_doc_embeddings
        )
        return self.get_sub_loss_by_raw(raw, neg_raw, lengths, offset)

    def get_sub_loss_by_raw(
        self,
        raw,
        neg_raw,
        lengths,
        offset=0,
    ):
        pos_scores = self._aggregate(
            raw, self.use_smooth_max, dim_max=3, dim_sum=2
        )  # [B, B * num_gpus]
        neg_scores = self._aggregate(
            neg_raw, self.use_smooth_max, dim_max=3, dim_sum=2
        )  # [B, B * neg_ratio]

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        batch_size = pos_scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, pos_scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(pos_scores, pos_idx)

        batch_size = pos_scores.size(0)
        # assert batch_size == neg_scores.size(0), f"{batch_size} != {neg_scores.size(0)}"

        neg_ratio = neg_scores.size(1) // neg_scores.size(0)
        # print("neg_ratio: ", neg_ratio)

        # select negative scores to [B, neg_ratio] and concat them with positive scores
        neg_selected = neg_scores.view(  # [B, B, neg_ratio]
            batch_size, batch_size, neg_ratio
        )[
            torch.arange(batch_size, device=neg_scores.device),  # pick own row
            torch.arange(batch_size, device=neg_scores.device),
        ]  # along the second dim  -->  [B, neg_ratio]

        scores = torch.cat(
            [pos_scores, neg_selected], dim=1
        )  # [B, B * num_gpus + neg_ratio]

        # DEBUG: dump to check the everything
        # torch.save(
        #     {
        #         "scores": scores,
        #         "pos_scores": pos_scores,
        #         "neg_scores": neg_scores,
        #         "neg_selected": neg_selected,
        #     },
        #     f"debug_tensors/loss_part_{dist.get_rank()}.pt",
        # )

        return self.ce_loss(scores / self.temperature, pos_idx)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ):
        """
        Compute InfoNCE loss with explicit negatives and optional in-batch term.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B * num_gpus, Nd, D] positive docs
            neg_doc_embeddings (Tensor): [B * neg_ratio, Nneg, D] negative docs
            offset (int): Positional offset for in-batch CE.

        Returns:
            Tensor: Scalar loss.
        """
        loss_stats = {}
        if self.mrl_groups is not None:
            loss = 0.0
            raw, neg_raw = self.get_raw_scores(
                query_embeddings, doc_embeddings, neg_doc_embeddings
            )
            for num_query_tokens, num_doc_tokens, weight in self.mrl_groups:
                # reuse raw to avoid duplicated matrix multiplication
                sub_raw = raw[:, :, :num_query_tokens, :num_doc_tokens]
                neg_sub_raw = neg_raw[:, :, :num_query_tokens, :num_doc_tokens]
                lengths = (query_embeddings[:, :num_query_tokens, 0] != 0).sum(dim=1)
                subloss = self.get_sub_loss_by_raw(
                    sub_raw, neg_sub_raw, lengths, offset
                )
                loss += subloss * weight
                loss_stats[
                    f"mrl_negce_{num_query_tokens}_{num_doc_tokens}_{weight}"
                ] = subloss

            # normalize loss by number of groups
            loss = loss / len(self.mrl_groups)

        else:
            loss = self.get_loss(
                query_embeddings, doc_embeddings, neg_doc_embeddings, offset
            )

        return loss if not loss_stats else (loss, loss_stats)


# THIS is not aligned with MoCa -- as it additionally encourage the positive one more time
# still is should worth a try
class NewColbertNegativeCELoss(ColbertModule):
    """
    InfoNCE loss with explicit negative documents

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
        in_batch_term (bool): Add in-batch CE term.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        # in_batch_term: bool = False,
        in_batch_term_weight: Optional[float] = None,  # enable this in MoCa training
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        mrl_groups: List[Tuple[int, int, float]] = None,
    ):

        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        # self.in_batch_term = in_batch_term
        self.in_batch_term_weight = in_batch_term_weight
        if self.in_batch_term_weight is not None:
            assert (
                0.0 < self.in_batch_term_weight <= 1.0
            ), f"in_batch_term_weight must be between 0.0 and 1.0, got {self.in_batch_term_weight}"

        self.mrl_groups = mrl_groups
        if self.mrl_groups:
            rank0_print(f"Using mrl_groups = {self.mrl_groups}")
        # inner_loss is a in_batch loss together with the positive & negative loss
        self.inner_loss = NewColbertLoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
            mrl_groups=mrl_groups,
        )

    # NOTE: doc_embeddings is gathered, and neg_doc_embeddings can have
    # multiple negatives per query. Assuming them with the same B is not correct.
    def get_raw_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
    ):
        """
        Keep 4-dim in raw scores for MaxSim operations
        """
        # return torch.einsum(
        #     "bnd,bsd->bns", query_embeddings, doc_embeddings
        # ), torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings)
        return torch.einsum(
            "bnd,csd->bcns", query_embeddings, doc_embeddings
        ), torch.einsum("bnd,gsd->bgns", query_embeddings, neg_doc_embeddings)

    def get_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ):
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw, neg_raw = self.get_raw_scores(
            query_embeddings, doc_embeddings, neg_doc_embeddings
        )
        return self.get_sub_loss_by_raw(raw, neg_raw, lengths, offset)

    def get_sub_loss_by_raw(
        self,
        raw,
        neg_raw,
        lengths,
        offset=0,
    ):
        pos_scores = self._aggregate(
            raw, self.use_smooth_max, dim_max=3, dim_sum=2
        )  # [B, B * num_gpus]  -> use offset & diag to get the effective pos_scores of [B]
        neg_scores = self._aggregate(
            neg_raw, self.use_smooth_max, dim_max=3, dim_sum=2
        )  # [B, B * neg_ratio] -> use something to get effective neg_scores of [B]

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        batch_size = pos_scores.size(0)
        assert batch_size == neg_scores.size(0), f"{batch_size} != {neg_scores.size(0)}"
        # idx, pos_idx = self._get_idx(batch_size, offset, pos_scores.device)

        scores = pos_scores.diagonal(offset=offset).unsqueeze(1)  # [B, 1]
        neg_ratio = neg_scores.size(1) // neg_scores.size(0)
        # print("neg_ratio: ", neg_ratio)
        neg_selected = neg_scores.view(  # [B, B, neg_ratio]
            batch_size, batch_size, neg_ratio
        )[
            torch.arange(batch_size, device=neg_scores.device),  # pick own row
            torch.arange(batch_size, device=neg_scores.device),
        ]  # along the second dim  # [B, neg_ratio]

        # broadcast scores to [B, neg_ratio]
        sub_loss = F.softplus((scores - neg_selected) / self.temperature).mean()

        return sub_loss

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ):
        """
        Compute InfoNCE loss with explicit negatives and optional in-batch term.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B * num_gpus, Nd, D] positive docs
            neg_doc_embeddings (Tensor): [B * neg_ratio, Nneg, D] negative docs
            offset (int): Positional offset for in-batch CE.

        Returns:
            Tensor: Scalar loss.
        """
        loss_stats = {}
        if self.mrl_groups is not None:
            loss = 0.0
            raw, neg_raw = self.get_raw_scores(
                query_embeddings, doc_embeddings, neg_doc_embeddings
            )
            for num_query_tokens, num_doc_tokens, weight in self.mrl_groups:
                # reuse raw to avoid duplicated matrix multiplication
                sub_raw = raw[:, :, :num_query_tokens, :num_doc_tokens]
                neg_sub_raw = neg_raw[:, :, :num_query_tokens, :num_doc_tokens]
                lengths = (query_embeddings[:, :num_query_tokens, 0] != 0).sum(dim=1)
                subloss = self.get_sub_loss_by_raw(
                    sub_raw, neg_sub_raw, lengths, offset
                )
                loss += subloss * weight
                loss_stats[
                    f"mrl_negce_{num_query_tokens}_{num_doc_tokens}_{weight}"
                ] = subloss
            # normalize loss by number of groups
            loss = loss / len(self.mrl_groups)

        else:
            loss = self.get_loss(
                query_embeddings, doc_embeddings, neg_doc_embeddings, offset
            )

        if self.in_batch_term_weight is not None:
            loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)

            if isinstance(loss_ib, tuple):
                loss_ib, loss_ib_stats = loss_ib
                loss_stats.update(loss_ib_stats)

            # loss = (loss + loss_ib) / 2
            loss = (
                loss * (1 - self.in_batch_term_weight)
                + loss_ib * self.in_batch_term_weight
            )

        return loss if not loss_stats else (loss, loss_stats)


class NewColbertPairwiseCELoss(ColbertModule):
    """
    Pairwise loss for ColBERT (no explicit negatives).

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        in_batch_term_weight: Optional[float] = None,  # added myself
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        mrl_groups: List[Tuple[int, int, float]] = None,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.in_batch_term_weight = in_batch_term_weight
        self.mrl_groups = mrl_groups
        if self.mrl_groups:
            rank0_print(f"Using mrl_groups = {self.mrl_groups}")
        if self.in_batch_term_weight is not None:
            assert (
                0.0 < self.in_batch_term_weight <= 1.0
            ), f"in_batch_term_weight must be between 0.0 and 1.0, got {self.in_batch_term_weight}"
            self.inner_loss = NewColbertLoss(
                temperature=temperature,
                normalize_scores=normalize_scores,
                use_smooth_max=use_smooth_max,
                pos_aware_negative_filtering=pos_aware_negative_filtering,
                max_batch_size=max_batch_size,
                tau=tau,
                norm_tol=norm_tol,
                filter_threshold=filter_threshold,
                filter_factor=filter_factor,
                mrl_groups=mrl_groups,
            )

    def get_raw_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ):
        return torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)

    def get_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        offset: int = 0,
    ):
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = self.get_raw_scores(query_embeddings, doc_embeddings)
        return self.get_sub_loss_by_raw(raw, lengths, offset)

    def get_sub_loss_by_raw(self, raw, lengths, offset=0):
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)
        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        pos_scores = scores.diagonal(offset=offset)
        top2 = scores.topk(2, dim=1).values  # (bs, 2) -- dim-1 has 1 pos and 1 hard neg
        neg_scores = torch.where(top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0])
        return F.softplus((neg_scores - pos_scores) / self.temperature).mean()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        offset: int = 0,
    ):
        """
        Compute pairwise softplus loss over in-batch document pairs.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B, Nd, D]
            offset (int): Positional offset for positives.

        Returns:
            Tensor: Scalar loss value.
        """
        loss_stats = {}
        if self.mrl_groups is not None:
            loss = 0.0
            raw = self.get_raw_scores(query_embeddings, doc_embeddings)

            for num_query_tokens, num_doc_tokens, weight in self.mrl_groups:
                # reuse raw to avoid duplicated matrix multiplication
                sub_raw = raw[:, :, :num_query_tokens, :num_doc_tokens]
                lengths = (query_embeddings[:, :num_query_tokens, 0] != 0).sum(dim=1)
                subloss = self.get_sub_loss_by_raw(sub_raw, lengths, offset)
                loss += subloss * weight
                loss_stats[
                    f"mrl_pairwisece_{num_query_tokens}_{num_doc_tokens}_{weight}"
                ] = subloss

            # normalize loss by number of groups
            loss = loss / len(self.mrl_groups)
        else:
            loss = self.get_loss(query_embeddings, doc_embeddings, offset)

        if self.in_batch_term_weight is not None:
            loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)
            if isinstance(loss_ib, tuple):
                loss_ib, loss_ib_stats = loss_ib
                loss_stats.update(loss_ib_stats)

            loss = (
                loss * (1 - self.in_batch_term_weight)
                + loss_ib * self.in_batch_term_weight
            )

        return loss if not loss_stats else (loss, loss_stats)


class NewColbertPairwiseNegativeCELoss(ColbertModule):
    """
    Pairwise loss with explicit negatives and optional in-batch term.
    Possiblya a good alternative to use as MoCa did.
    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
        in_batch_term (bool): Add pairwise in-batch loss.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        # in_batch_term: bool = False,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        mrl_groups: List[Tuple[int, int, float]] = None,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.in_batch_term_weight = in_batch_term_weight
        assert (
            self.in_batch_term_weight >= 0 and self.in_batch_term_weight <= 1
        ), f"Invalid in_batch_term_weight: {self.in_batch_term_weight}"

        self.mrl_groups = mrl_groups

        self.inner_pairwise = NewColbertPairwiseCELoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
            mrl_groups=mrl_groups,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute pairwise softplus loss with explicit negatives and optional in-batch term.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B * num_gpus, Nd, D] positive docs
            neg_doc_embeddings (Tensor): [B, Nneg, D] negative docs
            offset (int): Positional offset for positives.

        Returns:
            Tensor: Scalar loss value.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        # this is problematic as it assumes len(query) == len(doc) and do not consider gathered doc
        # pos_raw = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings)
        # pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)
        # Use local (non-gathered) doc_embeddings for positive scores
        local_doc_embeddings = doc_embeddings[
            offset : offset + query_embeddings.size(0), ...
        ]
        pos_raw = torch.einsum("bnd,bsd->bns", query_embeddings, local_doc_embeddings)
        pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)

        # negative has no problem, as one sample's negative does not affect other samples
        neg_raw = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings)
        neg_scores = self._aggregate(neg_raw, self.use_smooth_max, dim_max=2, dim_sum=1)

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        loss = F.softplus((neg_scores - pos_scores) / self.temperature).mean()

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_pairwise(query_embeddings, doc_embeddings, offset)
            loss = (
                loss * (1 - self.in_batch_term_weight)
                + loss_ib * self.in_batch_term_weight
            )

        return loss
