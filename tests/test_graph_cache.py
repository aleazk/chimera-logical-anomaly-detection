import pytest

dgl = pytest.importorskip("dgl")
torch = pytest.importorskip("torch")

from chimera_logic.trainer import CacheConfig, LevelwiseTrainer, SubtreeCache


def make_graph(op=3, swapped=False):
    src = torch.tensor([1, 0] if swapped else [0, 1])
    dst = torch.tensor([2, 2])
    g = dgl.graph((src, dst), num_nodes=3)
    g.ndata["mask"] = torch.tensor([1, 1, 0])
    g.ndata["x"] = torch.tensor([1, 2, 0])
    g.ndata["y"] = torch.tensor([0, 0, op])
    g.edata["neg"] = torch.tensor([1, 1])
    g.edata["pos"] = torch.tensor([0, 1])
    return g


def test_commutative_key_is_permutation_invariant(tmp_path):
    cache = SubtreeCache(CacheConfig(str(tmp_path)))
    t1 = LevelwiseTrainer(make_graph(3, False), 2, cache, feat_dim=4)
    t2 = LevelwiseTrainer(make_graph(3, True), 2, cache, feat_dim=4)
    t1._enc_fprint = t2._enc_fprint = "same"
    assert t1.cache_key(2) == t2.cache_key(2)
