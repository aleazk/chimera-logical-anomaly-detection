import pytest

from chimera_logic.semantics import (
    OP_AND,
    OP_IFF,
    OP_IMPLIES,
    OP_OR,
    compute_truth_value_hard,
)


@pytest.mark.parametrize(
    "left,right,expected",
    [(0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1)],
)
def test_implication_truth_table(left, right, expected):
    assert compute_truth_value_hard(OP_IMPLIES, [left, right]) == expected


@pytest.mark.parametrize(
    "op,vals,expected",
    [
        (OP_AND, [1, 1], 1),
        (OP_AND, [1, 0], 0),
        (OP_OR, [0, 0], 0),
        (OP_OR, [0, 1], 1),
        (OP_IFF, [1, 1], 1),
        (OP_IFF, [1, 0], 0),
    ],
)
def test_other_truth_tables(op, vals, expected):
    assert compute_truth_value_hard(op, vals) == expected


def test_implication_rejects_wrong_arity():
    with pytest.raises(ValueError):
        compute_truth_value_hard(OP_IMPLIES, [1])
