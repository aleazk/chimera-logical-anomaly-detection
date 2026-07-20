"""Reusable operand-level Chimera construction utilities."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .semantics import apply_binary_operator


@dataclass(frozen=True)
class ChimeraBatch:
    """Mixed operand features and exact Boolean targets."""

    features: torch.Tensor
    targets: torch.Tensor
    permutation: torch.Tensor


def cyclic_derangement(batch_size: int, *, device=None, generator=None) -> torch.Tensor:
    """Return a random cyclic permutation with no fixed points when ``batch_size > 1``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    base = torch.arange(batch_size, device=device)
    if batch_size == 1:
        return base
    shift = int(torch.randint(1, batch_size, (1,), device=device, generator=generator).item())
    return (base + shift) % batch_size


def build_chimera_batch(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    left_truth: torch.Tensor,
    right_truth: torch.Tensor,
    op_code: int,
    *,
    left_negated: bool = False,
    right_negated: bool = False,
    permutation: torch.Tensor | None = None,
) -> ChimeraBatch:
    """Mix right operands across samples and compute exact logical targets.

    Features are concatenated as ``[left, left_neg_bit, right, right_neg_bit]``.
    Negation bits use the same convention as inference: ``1`` means negated and
    ``0`` means non-negated.
    """
    if left_features.ndim != 2 or right_features.ndim != 2:
        raise ValueError("features must have shape (B, F)")
    if left_features.shape != right_features.shape:
        raise ValueError("left and right feature tensors must have equal shape")
    batch_size = left_features.shape[0]
    if left_truth.shape[0] != batch_size or right_truth.shape[0] != batch_size:
        raise ValueError("truth tensors must share the feature batch dimension")

    if permutation is None:
        permutation = cyclic_derangement(batch_size, device=left_features.device)
    if permutation.shape != (batch_size,):
        raise ValueError("permutation must have shape (B,)")

    left_t = left_truth.bool()
    right_t = right_truth.bool()[permutation]
    if left_negated:
        left_t = ~left_t
    if right_negated:
        right_t = ~right_t
    targets = apply_binary_operator(op_code, left_t, right_t).float()

    left_flag = torch.full(
        (batch_size, 1), float(left_negated), device=left_features.device, dtype=left_features.dtype
    )
    right_flag = torch.full(
        (batch_size, 1), float(right_negated), device=left_features.device, dtype=left_features.dtype
    )
    features = torch.cat(
        [left_features, left_flag, right_features[permutation], right_flag], dim=1
    )
    return ChimeraBatch(features=features, targets=targets, permutation=permutation)
