"""Chimera training and neural evaluation of logical rule graphs."""

from .semantics import (
    OP_AND,
    OP_IFF,
    OP_IMPLIES,
    OP_OR,
    apply_binary_operator,
    compute_truth_value_hard,
)
from .chimera import ChimeraBatch, build_chimera_batch, cyclic_derangement

__all__ = [
    "OP_AND",
    "OP_IFF",
    "OP_IMPLIES",
    "OP_OR",
    "apply_binary_operator",
    "compute_truth_value_hard",
    "ChimeraBatch",
    "build_chimera_batch",
    "cyclic_derangement",
]

__version__ = "0.1.0"
