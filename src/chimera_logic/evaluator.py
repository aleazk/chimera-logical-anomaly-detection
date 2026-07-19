# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:44:16 2025

@author: asc007
"""

# evaluator_dgl.py
# Evaluator for fixed logic trees represented as DGL graphs
# Now supports: (a) computing HARD ground-truth labels from per-concept leaf labels,
#               (b) computing SOFT predictions from per-concept classifiers,
#               (c) end-to-end API to return both truth and prediction tensors.
# Graph conventions:
#   - ndata['mask'] in {0,1}: 1=leaf (atomic proposition), 0=internal op node
#   - ndata['y'] op codes: 1=IFF, 2=IMPLIES (left->right), 3=AND, 4=OR (read at internal nodes)
#   - ndata['x'] concept ids for leaves in {1..N}, and 0 for non-leaves
#   - edata['neg'] in {-1,+1}: edge-level negation of child value

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import dgl

from .semantics import compute_truth_value_hard

# ---------------------------
# Hard logic (label creation)
# ---------------------------

def topo_nodes(g: dgl.DGLGraph) -> List[int]:
    order: List[int] = []
    for batch in dgl.topological_nodes_generator(g):
        if isinstance(batch, torch.Tensor):
            order.extend(batch.tolist())
        elif isinstance(batch, (tuple, list)):
            for b in batch:
                order.extend(b.tolist())
        else:
            raise TypeError("Unknown batch type from topological_nodes_generator")
    return order


def fill_node_vector_from_concepts(g: dgl.DGLGraph, concept_values: torch.Tensor) -> torch.Tensor:
    """Map per-concept values to a per-node vector using ndata['x'] concept ids.
    concept_values: shape (N,) or (B,N) with values in {0,1} (hard) or [0,1] (soft)
    Returns: node_vec of shape (num_nodes,) or (B,num_nodes)
    """
    cid = g.ndata['x'].long()  # 0 for non-leaves; 1..N for leaves
    num_nodes = g.num_nodes()
    if concept_values.dim() == 1:
        N = concept_values.shape[0]
        node_vec = torch.zeros(num_nodes, dtype=concept_values.dtype, device=concept_values.device)
        leaf_mask = cid > 0
        if leaf_mask.any():
            idx = cid[leaf_mask] - 1  # 0-based
            node_vec[leaf_mask] = concept_values[idx]
        return node_vec
    elif concept_values.dim() == 2:
        B, N = concept_values.shape
        node_vec = torch.zeros(B, num_nodes, dtype=concept_values.dtype, device=concept_values.device)
        leaf_mask = cid > 0
        if leaf_mask.any():
            # Expand indexes per batch
            idx = (cid[leaf_mask] - 1).unsqueeze(0).expand(B, -1)  # (B, L)
            node_vec[:, leaf_mask] = torch.gather(concept_values, 1, idx)
        return node_vec
    else:
        raise ValueError("concept_values must have shape (N,) or (B,N)")


def propagate_truth_values_hard_from_leaf_labels(g: dgl.DGLGraph, leaf_labels_by_concept: torch.Tensor) -> torch.Tensor:
    """Compute hard boolean truth labels bottom-up given per-concept leaf labels.
    leaf_labels_by_concept: shape (N,) with {0,1} for each concept id 1..N.
    Returns: truth_value per node (num_nodes,) in {0,1}, also stored in g.ndata['truth_value'].
    """
    mask = g.ndata['mask']
    y = g.ndata['y']
    # Initialize node truth with leaves filled from concept labels
    truth_value = fill_node_vector_from_concepts(g, leaf_labels_by_concept).long().clone()

    order = topo_nodes(g)
    for node in order:
        if mask[node].item() == 1:
            continue
        src, dst, eid = g.in_edges(node, form='all')
        child_vals: List[int] = []
        for s, e in zip(src.tolist(), eid.tolist()):
            val = int(truth_value[s].item())
            if 'neg' in g.edata and int(g.edata['neg'][e].item()) == -1:
                val = int(not val)
            child_vals.append(val)
        truth_value[node] = compute_truth_value_hard(int(y[node].item()), child_vals)

    g.ndata['truth_value'] = truth_value
    return truth_value


# ----------------------------------------
# Soft evaluator (neural, uncertainty-aware)
# ----------------------------------------
class BinaryGate(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.stack([a, b], dim=-1)
        return self.net(x).squeeze(-1)


class VariadicReducer(nn.Module):
    def __init__(self, gate: BinaryGate):
        super().__init__()
        self.gate = gate
    def forward(self, vals: List[torch.Tensor]) -> torch.Tensor:
        out = vals[0]
        for v in vals[1:]:
            out = self.gate(out, v)
        return out


class EvaluatorDGL(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_and = BinaryGate()
        self.gate_or = BinaryGate()
        self.gate_implies = BinaryGate()
        self.gate_iff = BinaryGate()
        self.reduce_and = VariadicReducer(self.gate_and)
        self.reduce_or = VariadicReducer(self.gate_or)
        self.reduce_iff = VariadicReducer(self.gate_iff)

    def _root_index(self, g: dgl.DGLGraph) -> int:
        out_deg = g.out_degrees()
        roots = torch.nonzero(out_deg == 0, as_tuple=False).flatten()
        assert len(roots) == 1, "Graph should have exactly one root"
        return int(roots[0].item())

    def forward(self, g: dgl.DGLGraph, leaf_probs_nodevec: torch.Tensor) -> torch.Tensor:
        """leaf_probs_nodevec: (num_nodes,) with [0,1] at leaves; ignored at internals."""
        device = next(self.parameters()).device
        num_nodes = g.num_nodes()
        mask = g.ndata['mask'].to(device)
        y = g.ndata['y'].to(device)
        neg = g.edata['neg'].to(device) if 'neg' in g.edata else torch.ones(g.num_edges(), dtype=torch.long, device=device)

        node_probs = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        # initialize leaves
        node_probs[mask == 1] = leaf_probs_nodevec[mask == 1].to(device)

        order = topo_nodes(g)
        for node in order:
            if mask[node].item() == 1:
                continue
            src, dst, eid = g.in_edges(node, form='all')
            child_vals: List[torch.Tensor] = []
            for s, e in zip(src.tolist(), eid.tolist()):
                v = node_probs[s]
                if int(neg[e].item()) == -1:
                    v = 1.0 - v
                child_vals.append(v)
            op = int(y[node].item())
            if op == 3:      # AND
                out = self.reduce_and(child_vals) if len(child_vals) > 1 else child_vals[0]
            elif op == 4:    # OR
                out = self.reduce_or(child_vals) if len(child_vals) > 1 else child_vals[0]
            elif op == 2:    # IMPLIES
                out = self.gate_implies(child_vals[0], child_vals[1]) if len(child_vals) == 2 else torch.ones((), device=device)
            elif op == 1:    # IFF
                out = self.reduce_iff(child_vals) if len(child_vals) > 1 else child_vals[0]
            else:
                raise ValueError(f"Unknown operator code: {op}")
            node_probs[node] = out
        return node_probs, self._root_index(g)


# ----------------------------------------
# Leaf classifier bank and end-to-end API
# ----------------------------------------
class LeafClassifierBank(nn.Module):
    """Holds one classifier per concept id in {1..N}. Each classifier maps raw input → prob.
    You can swap these with real perception models.
    """
    def __init__(self, concept_classifiers: Dict[int, nn.Module]):
        super().__init__()
        # normalize keys to str for ModuleDict, keep int map for convenience
        self._keys = {int(k): str(int(k)) for k in concept_classifiers.keys()}
        self.models = nn.ModuleDict({self._keys[i]: m for i, m in concept_classifiers.items()})

    def forward_per_concept(self, raw_input: torch.Tensor) -> Dict[int, torch.Tensor]:
        return {i: self.models[self._keys[i]](raw_input) for i in self._keys}

    def vectorize(self, raw_input: torch.Tensor, N: int) -> torch.Tensor:
        """Return a length-N vector of probabilities (for a single sample)."""
        out = torch.zeros(N, dtype=torch.float32, device=raw_input.device)
        per = self.forward_per_concept(raw_input)
        for i, p in per.items():
            if 1 <= i <= N:
                out[i-1] = p.squeeze()  # assume scalar prob
        return out


def compute_truth_and_predictions(
    g: dgl.DGLGraph,
    leaf_labels_by_concept: torch.Tensor,  # shape (N,), hard 0/1
    raw_input: torch.Tensor,               # shape compatible with leaf classifiers
    leaf_bank: LeafClassifierBank,
    evaluator: EvaluatorDGL,
):
    """Given a DGL logic graph and (i) hard labels per concept, (ii) raw input for prediction,
    return both: hard ground-truth per node and soft predicted probs per node.
    """
    device = next(evaluator.parameters()).device
    g = g.to(device)
    N = int(leaf_labels_by_concept.shape[0])

    # HARD: propagate ground truth from leaf concept labels
    truth = propagate_truth_values_hard_from_leaf_labels(g, leaf_labels_by_concept.to(device))  # (num_nodes,)

    # SOFT: get leaf predictions from the bank and propagate through evaluator
    leaf_probs_concept = leaf_bank.vectorize(raw_input.to(device), N)  # (N,)
    leaf_probs_nodevec = fill_node_vector_from_concepts(g, leaf_probs_concept)  # (num_nodes,)
    node_probs, root = evaluator(g, leaf_probs_nodevec)

    return truth, node_probs, root


# ---------------------------
# Minimal dummy leaf classifier for demo
# ---------------------------
class DummyLeaf(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze(-1)


# ---------------------------
# Example wiring (replace with your data pipeline)
# ---------------------------
if __name__ == "__main__":
    # Assume graph `g` already has ndata['mask'], ndata['y'], ndata['x'], edata['neg'].
    # You provide: (1) leaf concept labels (N,), (2) a raw input tensor for prediction.
    graphs: List[dgl.DGLGraph] = []  # TODO: populate with your tiny_sst graphs

    # Build a small leaf bank for N concepts
    N = 5
    bank = LeafClassifierBank({i: DummyLeaf(8) for i in range(1, N+1)})
    evaluator = EvaluatorDGL()

    if graphs:
        g = graphs[0]
        leaf_labels = torch.randint(0, 2, (N,))         # hard labels for concepts
        raw_x = torch.randn(1, 8)                        # raw input (batch=1)
        truth, probs, root = compute_truth_and_predictions(g, leaf_labels, raw_x, bank, evaluator)
        print("Root idx:", root)
        print("Truth per node:", truth)
        print("Preds per node:", probs)
    else:
        print("Provide your DGL graphs and plug real leaf classifiers + concept labels.")
