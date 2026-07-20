"""Exact Boolean semantics used to supervise neural rule gates."""
from __future__ import annotations

from typing import Iterable

OP_IFF = 1
OP_IMPLIES = 2
OP_AND = 3
OP_OR = 4


def compute_truth_value_hard(op_code: int, child_vals: Iterable[int | bool]) -> int:
    """Evaluate one logical operator on hard child truth values.

    Operator codes follow the rule-graph convention used throughout the project:
    ``1=IFF``, ``2=IMPLIES``, ``3=AND``, ``4=OR``.
    """
    vals = [bool(v) for v in child_vals]
    if op_code == OP_IFF:
        if not vals:
            return 1
        return int(all(v == vals[0] for v in vals))
    if op_code == OP_IMPLIES:
        if len(vals) != 2:
            raise ValueError("IMPLIES requires exactly two operands")
        left, right = vals
        return int((not left) or right)
    if op_code == OP_AND:
        return int(all(vals))
    if op_code == OP_OR:
        return int(any(vals))
    raise ValueError(f"Unknown logical operator code: {op_code}")


def apply_binary_operator(op_code: int, left, right):
    """Vector-friendly Boolean operator for Python bools or PyTorch bool tensors."""
    if op_code == OP_IFF:
        return left == right
    if op_code == OP_IMPLIES:
        return (~left) | right
    if op_code == OP_AND:
        return left & right
    if op_code == OP_OR:
        return left | right
    raise ValueError(f"Unknown logical operator code: {op_code}")
