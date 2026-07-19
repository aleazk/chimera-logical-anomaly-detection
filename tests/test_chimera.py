import torch

from chimera_logic.chimera import build_chimera_batch, cyclic_derangement
from chimera_logic.semantics import OP_AND, OP_IMPLIES


def test_derangement_has_no_fixed_points():
    perm = cyclic_derangement(8)
    assert torch.all(perm != torch.arange(8))


def test_chimera_can_create_missing_implication_violation():
    left_features = torch.tensor([[1.0], [0.0]])
    right_features = torch.tensor([[1.0], [0.0]])
    left_truth = torch.tensor([1, 0])
    right_truth = torch.tensor([1, 0])
    perm = torch.tensor([1, 0])

    batch = build_chimera_batch(
        left_features,
        right_features,
        left_truth,
        right_truth,
        OP_IMPLIES,
        permutation=perm,
    )
    assert batch.targets.tolist() == [0.0, 1.0]


def test_chimera_and_targets_are_exact():
    feat = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    truth = torch.tensor([0, 1, 0, 1])
    perm = torch.tensor([1, 2, 3, 0])
    batch = build_chimera_batch(feat, feat, truth, truth, OP_AND, permutation=perm)
    expected = (truth.bool() & truth.bool()[perm]).float()
    assert torch.equal(batch.targets, expected)
    assert batch.features.shape == (4, 6)


def test_negation_flags_are_zero_one():
    feat = torch.zeros(2, 3)
    truth = torch.tensor([0, 1])
    batch = build_chimera_batch(
        feat,
        feat,
        truth,
        truth,
        OP_AND,
        left_negated=True,
        right_negated=False,
        permutation=torch.tensor([1, 0]),
    )
    assert torch.equal(batch.features[:, 3], torch.ones(2))
    assert torch.equal(batch.features[:, -1], torch.zeros(2))
