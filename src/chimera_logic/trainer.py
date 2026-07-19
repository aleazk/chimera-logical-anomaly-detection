# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:56:14 2025

@author: asc007
"""

# -*- coding: utf-8 -*-
"""
trainer.py — Feature-aware, level-wise trainer with lineage-aware caching and tqdm.

Key features:
- Trains bottom-up (leaves → root), with gates that map child features (+neg flags) → parent feature → parent prob.
- Cache keys are **lineage-aware**:
    * Leaves:  LEAF(concept_id | enc=<encoder_fingerprint>)
    * Internals: OPk(child_keys with order/neg as per edges) | arch=SubtreeGateRep_v2 | F=<feat_dim>
  so a cached gate is only reused when the **entire feature lineage** matches (including the leaf-bank encoder).
- Optionally restrict caching to **leaf-root nodes** (children are leaves) for maximal safety.
- Keeps a runtime registry of trained gates for this trainer instance (across epochs/levels).

Dependencies:
- evaluator.py: propagate_truth_values_hard_from_leaf_labels, fill_node_vector_from_concepts, topo_nodes
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import os
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from tqdm import tqdm, trange

from .evaluator import (
    propagate_truth_values_hard_from_leaf_labels,
    fill_node_vector_from_concepts,
    topo_nodes,
)

# ------------------------------
# Operator properties / helpers
# ------------------------------
COMMUTATIVE_OPS = {1, 3, 4}  # IFF, AND, OR are commutative; IMPLIES (2) is ordered

def _edge_order_for_implies(g: dgl.DGLGraph, node: int, src: torch.Tensor, eid: torch.Tensor) -> List[int]:
    """Ordered child indices for IMPLIES. If edata['pos'] exists, use it; else sort by src id."""
    if 'pos' in g.edata:
        return sorted(range(len(src)), key=lambda i: int(g.edata['pos'][eid[i]].item()))
    return sorted(range(len(src)), key=lambda i: int(src[i].item()))

# ------------------------------
# (Topology-only) signature helper (unchanged)
#   Keep this around for logging/visualization if other scripts import it.
#   Caching uses lineage-aware keys defined inside LevelwiseTrainer.cache_key(...)
# ------------------------------
def subtree_signature(g: dgl.DGLGraph, node: int) -> str:
    mask = g.ndata['mask']; x = g.ndata['x']; y = g.ndata['y']
    if int(mask[node].item()) == 1:
        return f"LEAF({int(x[node].item())})"
    op = int(y[node].item())
    src, _, eid = g.in_edges(node, form='all')
    parts: List[str] = []
    if op == 2:
        for i in _edge_order_for_implies(g, node, src, eid):
            s = int(src[i].item()); e = int(eid[i].item())
            cs = subtree_signature(g, s)
            if 'neg' in g.edata and int(g.edata['neg'][e].item()) == -1:
                cs = '!' + cs
            parts.append(cs)
    else:
        for s, e in zip(src.tolist(), eid.tolist()):
            cs = subtree_signature(g, s)
            if 'neg' in g.edata and int(g.edata['neg'][e].item()) == -1:
                cs = '!' + cs
            parts.append(cs)
        parts.sort()
    return f"OP{op}({','.join(parts)})"

# ------------------------------
# Depth/levels
# ------------------------------
def compute_depths(g: dgl.DGLGraph) -> Tuple[List[List[int]], List[int]]:
    order = topo_nodes(g)
    depth = [0] * g.num_nodes()
    for node in order:
        if int(g.ndata['mask'][node].item()) == 1:
            depth[node] = 0
        else:
            src, _, _ = g.in_edges(node, form='all')
            depth[node] = 1 + max(depth[int(s.item())] for s in src)
    L = max(depth) if depth else 0
    levels: List[List[int]] = [[] for _ in range(L + 1)]
    for n, d in enumerate(depth):
        levels[d].append(n)
    return levels, depth

# ------------------------------
# Gate
# ------------------------------
class SubtreeGateRep(nn.Module):
    """child features (+neg flag) -> parent feature -> parent prob"""
    def __init__(self, arity: int, feat_dim: int, hidden: int = 512, edge_feat_dim: int = 1):
        super().__init__()
        self.arity = arity
        self.feat_dim = feat_dim
        self.edge_feat_dim = edge_feat_dim
        in_dim = arity * (feat_dim + edge_feat_dim)
        hidden = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

    def forward(self, child_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(child_flat)                 # (B, F)
        p = self.head(h).squeeze(-1)             # (B,)
        return h, p

# ------------------------------
# Cache
# ------------------------------
@dataclass
class CacheConfig:
    root_dir: str = "subtree_cache"

class SubtreeCache:
    def __init__(self, cfg: CacheConfig):
        self.dir = cfg.root_dir
        os.makedirs(self.dir, exist_ok=True)

    def path(self, signature: str) -> str:
        sig_hash = hashlib.sha256(signature.encode()).hexdigest()[:32]
        return os.path.join(self.dir, f"{sig_hash}.pt")

    def has(self, signature: str) -> bool:
        return os.path.exists(self.path(signature))

    def load(self, signature: str, arity: int, feat_dim: int) -> SubtreeGateRep:
        pkg = torch.load(self.path(signature), map_location="cpu")
        gate = SubtreeGateRep(arity=arity, feat_dim=feat_dim)
        gate.load_state_dict(pkg['state_dict'])
        # Shape sanity
        assert pkg.get('arity', arity) == arity and pkg.get('feat_dim', feat_dim) == feat_dim, "Cache shape mismatch"
        return gate

    def save(self, signature: str, gate: SubtreeGateRep) -> None:
        pkg = {'state_dict': gate.state_dict(), 'arity': gate.arity, 'feat_dim': gate.feat_dim}
        torch.save(pkg, self.path(signature))

# ------------------------------
# Encoder fingerprint
# ------------------------------
def _hash_state_dict(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:24]

def _hash_modules(mods) -> str:
    h = hashlib.sha256()
    for m in mods:
        h.update(_hash_state_dict(m.state_dict()).encode())
    return h.hexdigest()[:24]

def encoder_fingerprint(leaf_bank, manual_tag: str | None = None) -> str:
    """
    Returns a short fingerprint string that changes whenever the **encoder’s weights**
    (or any obvious feature-producing submodule) change.

    Falls back to hashing the entire leaf_bank state_dict if we can't find named submodules.
    You can also force a tag via `manual_tag` to tie multiple runs explicitly.
    """
    if manual_tag:
        return f"manual_{manual_tag}"

    # Collect common feature-producing submodules if present.
    candidates = []
    for attr in ["encoder", "backbone", "proj", "cnn", "trunk", "feature_extractor", "body"]:
        m = getattr(leaf_bank, attr, None)
        if isinstance(m, nn.Module):
            candidates.append(m)

    if candidates:
        return _hash_modules(candidates)

    # Last resort: hash the whole model (safe, but invalidates cache whenever *any* head changes)
    try:
        return _hash_state_dict(leaf_bank.state_dict())
    except Exception:
        # Truly last resort: hash the repr (still stable across runs of same code)
        return hashlib.sha256(repr(leaf_bank).encode()).hexdigest()[:24]

# ------------------------------
# Trainer (lineage-aware caching)
# ------------------------------
class LevelwiseTrainer:
    def __init__(self, g: dgl.DGLGraph, N_concepts: int, cache: SubtreeCache,
                 feat_dim: Optional[int] = None, device: str = "cpu",
                 cache_leafroot_only: bool = False,
                 lineage_aware: bool = True,
                 arch_tag: str = "SubtreeGateRep_v2"):
        self.g = g
        self.N = N_concepts
        self.cache = cache
        self.device = torch.device(device)
        self.levels, self.depth = compute_depths(g)
        self.feat_dim = feat_dim
        self.cache_leafroot_only = cache_leafroot_only
        self.lineage_aware = lineage_aware
        self.arch_tag = arch_tag
        # computed at train-time from leaf_bank
        self._enc_fprint: Optional[str] = None
        # gates trained in this run
        self.runtime_gates: Dict[int, SubtreeGateRep] = {}

    # ---- util: children are leaves? ----
    def _children_are_leaves(self, node: int) -> bool:
        src, _, _ = self.g.in_edges(node, form='all')
        m = self.g.ndata['mask']
        return all(int(m[int(s.item())].item()) == 1 for s in src)

    # ---- lineage-aware cache key ----
    def cache_key(self, node: int, memo: Optional[Dict[int, str]] = None) -> str:
        """
        Build a lineage-aware key for subtree rooted at `node`.
        Leaves include leaf concept id and encoder fingerprint.
        Internals include op code, ordered/negated child keys, arch tag, and feature dim.
        """
        g = self.g
        if memo is None:
            memo = {}

        if node in memo:
            return memo[node]

        mask = g.ndata['mask']; x = g.ndata['x']; y = g.ndata['y']

        if int(mask[node].item()) == 1:
            cid = int(x[node].item())
            if self.lineage_aware:
                if self._enc_fprint is None:
                    raise RuntimeError("encoder fingerprint not initialized; call train() first or set manually.")
                key = f"LEAF(c{cid}|enc={self._enc_fprint})"
            else:
                key = f"LEAF(c{cid})"
            memo[node] = key
            return key

        op = int(y[node].item())
        src, _, eid = g.in_edges(node, form='all')

        child_keys: List[str] = []
        if op == 2:  # IMPLIES: ordered
            order = _edge_order_for_implies(g, node, src, eid)
            for i in order:
                s = int(src[i].item()); e = int(eid[i].item())
                ck = self.cache_key(s, memo)
                if 'neg' in g.edata and int(g.edata['neg'][e].item()) == -1:
                    ck = '!' + ck
                child_keys.append(ck)
        else:
            for s, e in zip(src.tolist(), eid.tolist()):
                ck = self.cache_key(s, memo)
                if 'neg' in g.edata and int(g.edata['neg'][e].item()) == -1:
                    ck = '!' + ck
                child_keys.append(ck)
            child_keys.sort()

        arch = f"|arch={self.arch_tag}"
        Ftag = f"|F={self.feat_dim}" if self.feat_dim is not None else ""
        key = f"OP{op}({','.join(child_keys)}){arch}{Ftag}"
        memo[node] = key
        return key

    # ---- leaf features & probs ----
    def _leaf_feats_probs(self, images: torch.Tensor, leaf_bank) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(leaf_bank, 'forward_probs'):
            raise AttributeError("leaf_bank must define forward_probs(images)->(B,N)")
        if hasattr(leaf_bank, 'encoder'):
            z = leaf_bank.encoder(images)
            if isinstance(z, torch.Tensor) and z.dim() == 4:
                z = torch.flatten(z, 1)
            elif z.dim() != 2:
                raise ValueError("encoder must return (B,F) or (B,C,1,1)")
            leaf_feats = z.unsqueeze(1).expand(-1, self.N, -1)
        elif hasattr(leaf_bank, '_encode'):
            z = leaf_bank._encode(images)
            leaf_feats = z.unsqueeze(1).expand(-1, self.N, -1)
        else:
            raise AttributeError("leaf_bank must expose an encoder/_encode for features")
        leaf_probs = leaf_bank.forward_probs(images)
        if self.feat_dim is None:
            self.feat_dim = leaf_feats.shape[-1]
        return leaf_feats, leaf_probs

    # ---- concat child feats + neg bits ----
    def _gather_child_flat(self, node: int, node_feats: torch.Tensor) -> torch.Tensor:
        g = self.g
        src, _, eid = g.in_edges(node, form='all')
        op = int(g.ndata['y'][node].item()) if int(g.ndata['mask'][node].item()) == 0 else None
        if op == 2:
            order = _edge_order_for_implies(g, node, src, eid)
            idx_list = [(int(src[i].item()), int(eid[i].item())) for i in order]
        else:
            idx_list = list(zip(src.tolist(), eid.tolist()))
        pieces: List[torch.Tensor] = []
        for s, e in idx_list:
            neg_flag = 1.0 if ('neg' in g.edata and int(g.edata['neg'][e].item()) == -1) else 0.0
            neg_col = torch.full((node_feats.shape[0], 1), neg_flag, device=node_feats.device)
            pieces.append(torch.cat([node_feats[:, s, :], neg_col], dim=1))
        return torch.cat(pieces, dim=1)

    # ---- train ----
    def train(self,
              dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
              leaf_bank,
              epochs_per_level: int = 3,
              lr: float = 1e-3,
              use_soft_leaves: bool = True,
              verbose: bool = True,
              use_tqdm: bool = True,
              negatives: str = "true_only",
              train_missing_only: bool = False) -> None:
        g = self.g.to(self.device)
        bce = torch.nn.BCELoss()

        # root node index (used to filter true normals)

        out_deg = g.out_degrees()
        root = int(torch.nonzero(out_deg == 0, as_tuple=False).flatten()[0].item())

        def _order_children_for_node(node, src, eids):
            # prefer edata['pos'] if present (0=left, 1=right), else keep order
            if "pos" in g.edata:
                pos = g.edata["pos"][eids].tolist()
                return [i for _, i in sorted(zip(pos, range(len(eids))))]
            return list(range(len(eids)))


        # init encoder fingerprint for lineage keys
        if self.lineage_aware and (self._enc_fprint is None):
            self._enc_fprint = encoder_fingerprint(leaf_bank)

        # iterate depths 1..L
        level_iter = range(1, len(self.levels))
        if use_tqdm:
            level_iter = trange(1, len(self.levels), desc="[Trainer] levels", leave=True)

        for depth_idx in level_iter:
            nodes_at_level = self.levels[depth_idx]
            if verbose and not use_tqdm:
                print(f"=== Training level {depth_idx} with {len(nodes_at_level)} node(s) ===")

            node_gates: Dict[int, SubtreeGateRep] = {}
            node_opts: Dict[int, optim.Optimizer] = {}

            frozen_nodes = set()  # nodes with existing cached/runtime gates
            train_nodes = set()   # nodes to optimize this run

            skip_level = False  # missing-only: if no trainable nodes, skip dataloader/epochs

            epoch_iter = range(1, epochs_per_level + 1)
            if use_tqdm:
                epoch_iter = trange(1, epochs_per_level + 1, desc=f"  [L{depth_idx}] epochs", leave=False)

            for ep in epoch_iter:
                total_loss = 0.0
                n_batches = 0

                batch_iter = dataset
                if use_tqdm:
                    batch_iter = tqdm(dataset, desc=f"    [L{depth_idx} E{ep}] batches", leave=False)

                for j, (images, leaf_labels) in enumerate(batch_iter):
                    #if j > 2:
                        #break                    #-------------------------------DEBUG-------------------------
                    if images.dim() == 3:
                        images = images.unsqueeze(0)
                    images = images.to(self.device)

                    if leaf_labels.dim() == 1:
                        leaf_labels = leaf_labels.unsqueeze(0)
                    leaf_labels = leaf_labels.to(self.device)
                    B = images.shape[0]

                    # hard truths
                    truths = []
                    for b in range(B):
                        tv = propagate_truth_values_hard_from_leaf_labels(g, leaf_labels[b])
                        truths.append(tv.unsqueeze(0))
                    truths = torch.cat(truths, dim=0).float().to(self.device)

                    # leaf feats & probs
                    leaf_bank.eval()
                    with torch.no_grad():
                        leaf_feats, leaf_probs = self._leaf_feats_probs(images, leaf_bank)

                    # init per-node tensors
                    num_nodes = g.num_nodes()
                    F = self.feat_dim or leaf_feats.shape[-1]
                    node_feats = torch.zeros(B, num_nodes, F, device=self.device)
                    node_probs = torch.zeros(B, num_nodes, device=self.device)

                    # place leaves
                    leaf_prob_nodes = fill_node_vector_from_concepts(g, leaf_probs)
                    node_probs[:, g.ndata['mask'].bool()] = leaf_prob_nodes[:, g.ndata['mask'].bool()]
                    cid = g.ndata['x'].long()
                    leaf_idx = torch.nonzero(cid > 0, as_tuple=False).flatten()
                    if leaf_idx.numel() > 0:
                        cids0 = (cid[leaf_idx] - 1).tolist()
                        node_feats[:, leaf_idx, :] = leaf_feats[:, cids0, :]

                    # prepare/load gates
                    if not node_gates:
                        for node in nodes_at_level:
                            arity = len(g.in_edges(node)[0])
                            key = self.cache_key(node)
                            can_cache_here = (not self.cache_leafroot_only) or self._children_are_leaves(node)

                            # Prefer runtime gate; else load from cache; else create new.
                            gate = self.runtime_gates.get(node, None)
                            from_runtime = gate is not None
                            from_cache = False

                            if gate is None and can_cache_here and self.cache.has(key):
                                gate = self.cache.load(key, arity=arity, feat_dim=F)
                                from_cache = True

                                if verbose and not use_tqdm:
                                    print(f"[cache] loaded gate for node {node} | key={key}")
                            elif gate is None:
                                gate = SubtreeGateRep(arity=arity, feat_dim=F)
                                if verbose and not use_tqdm:
                                    print(f"[init ] new gate for node {node} | arity={arity} F={F} | key={key}")

                            gate.to(self.device)
                            node_gates[node] = gate

                            # Missing-only mode: don't optimize gates that already exist.
                            if train_missing_only and (from_runtime or from_cache):
                                frozen_nodes.add(node)
                            else:
                                train_nodes.add(node)
                                node_opts[node] = optim.Adam(gate.parameters(), lr=lr)

                        if verbose:
                            print(f"[Trainer] L{depth_idx}: train={len(train_nodes)} frozen={len(frozen_nodes)} (missing-only={train_missing_only})")

                        # Missing-only fast path: if everything at this level is cached, don't iterate the dataset.
                        if train_missing_only and (len(train_nodes) == 0):
                            for n in nodes_at_level:
                                # register cached/runtime gates for later levels; keep on CPU like the normal save path
                                self.runtime_gates[n] = node_gates[n].cpu()
                            if verbose:
                                print(f"[Trainer] L{depth_idx}: all cached -> skipping dataloader for this level.")
                            skip_level = True
                            try:
                                batch_iter.close()  # close tqdm if present
                            except Exception:
                                pass
                            break

                    # propagate already-trained lower levels
                    for d in range(1, depth_idx):
                        for n in self.levels[d]:
                            gate_n = self.runtime_gates.get(n, None)
                            if gate_n is None:
                                can_cache_here = (not self.cache_leafroot_only) or self._children_are_leaves(n)
                                if can_cache_here:
                                    key_n = self.cache_key(n)
                                    if self.cache.has(key_n):
                                        gate_n = self.cache.load(key_n, arity=len(g.in_edges(n)[0]), feat_dim=F)
                                        self.runtime_gates[n] = gate_n
                            if gate_n is None:
                                continue
                            gate_n = gate_n.to(self.device).eval()
                            with torch.no_grad():
                                child_flat = self._gather_child_flat(n, node_feats)
                                par_feat, par_prob = gate_n(child_flat)
                                node_feats[:, n, :] = par_feat
                                node_probs[:, n] = par_prob

                    # train current level
                    batch_loss = 0.0
                    for n in nodes_at_level:
                        # If requested, only train gates that are missing from runtime/cache.
                        # For already-cached gates we still need to run a forward pass to feed higher nodes.
                        if train_missing_only and (n in frozen_nodes):
                            gate = node_gates[n].to(self.device).eval()
                            with torch.no_grad():
                                if negatives != "true_only":
                                    src, _, eids = g.in_edges(n, form="all")
                                    assert len(src) == 2, "binary node expected"
                                    if "pos" in g.edata:
                                        pos = g.edata["pos"][eids].tolist()  # 0=left,1=right
                                        order = [i for _, i in sorted(zip(pos, range(2)))]
                                    else:
                                        order = [0, 1]
                                    left_id  = int(src[order[0]].item())
                                    right_id = int(src[order[1]].item())

                                    neg_left  = (g.edata["neg"][eids[order[0]]].item() == -1) if "neg" in g.edata else False
                                    neg_right = (g.edata["neg"][eids[order[1]]].item() == -1) if "neg" in g.edata else False

                                    feat_left  = node_feats[:, left_id, :]
                                    feat_right = node_feats[:, right_id, :]
                                    Fdim = node_feats.size(-1)

                                    flag_vec = torch.tensor(
                                        [1.0 if neg_left else 0.0, 1.0 if neg_right else 0.0],
                                        device=self.device, dtype=torch.float32
                                    ).view(1, 2)
                                    flags_all = flag_vec.expand(B, -1)
                                    x_same_all = torch.cat([feat_left, feat_right, flags_all], dim=1)  # (B, 2F+2)
                                    par_feat_all, pred_all = gate(x_same_all)
                                    node_feats[:, n, :] = par_feat_all
                                    node_probs[:, n] = pred_all
                                else:
                                    child_flat = self._gather_child_flat(n, node_feats)
                                    par_feat, pred = gate(child_flat)
                                    node_feats[:, n, :] = par_feat
                                    node_probs[:, n] = pred
                            continue

                        if negatives != "true_only":

                            # ---- identify children & edge attributes ----
                            src, dst, eids = g.in_edges(n, form="all")
                            assert len(src) == 2, "binary node expected"
                            if "pos" in g.edata:
                                pos = g.edata["pos"][eids].tolist()  # 0=left,1=right
                                order = [i for _, i in sorted(zip(pos, range(2)))]
                            else:
                                order = [0, 1]
                            left_id  = src[order[0]].item()
                            right_id = src[order[1]].item()

                            neg_left  = (g.edata["neg"][eids[order[0]]].item() == -1) if "neg" in g.edata else False
                            neg_right = (g.edata["neg"][eids[order[1]]].item() == -1) if "neg" in g.edata else False
                            op = int(g.ndata["y"][n].item())  # 1=IFF, 2=IMPLIES, 3=AND, 4=OR

                            # ---- truths (same-image) with edge negations applied to children ----
                            t_node  = truths[:, n].bool().to(self.device)               # (B,)
                            t_left  = truths[:, left_id].bool().to(self.device)
                            t_right = truths[:, right_id].bool().to(self.device)
                            if neg_left:  t_left  = ~t_left
                            if neg_right: t_right = ~t_right
                            is_normal = truths[:, root].bool().to(self.device)          # rule holds for the image

                            # ---- features per child (already computed upstream into node_feats) ----
                            feat_left  = node_feats[:, left_id, :]   # (B,F)
                            feat_right = node_feats[:, right_id, :]  # (B,F)
                            F = node_feats.size(-1)

                            # two scalar edge-negation bits appended to every pair (0/1)
                            flag_vec = torch.tensor(
                                [1.0 if neg_left else 0.0, 1.0 if neg_right else 0.0],
                                device=self.device, dtype=torch.float32
                            ).view(1, 2)

                            def _pack_pair(feat_l, feat_r, mask):
                                """Return cat([feat_l[mask], feat_r[mask], flags]) with shape (?, 2F+2)."""
                                if mask.dtype == torch.bool:
                                    Bm = int(mask.sum().item())
                                    if Bm == 0:
                                        return torch.empty(0, 2*F + 2, device=self.device)
                                    flags = flag_vec.expand(Bm, -1)
                                    return torch.cat([feat_l[mask], feat_r[mask], flags], dim=1)
                                else:
                                    idx = mask
                                    Bm = idx.numel()
                                    if Bm == 0:
                                        return torch.empty(0, 2*F + 2, device=self.device)
                                    flags = flag_vec.expand(Bm, -1)
                                    return torch.cat([feat_l[idx], feat_r[idx], flags], dim=1)

                            # =========================
                            # SAME-IMAGE components
                            # =========================
                            same_pos_mask = is_normal & t_node      # node true & whole image normal
                            same_neg_mask = ~t_node                 # node false

                            x_same_pos = _pack_pair(feat_left, feat_right, same_pos_mask)
                            y_same_pos = torch.ones(x_same_pos.size(0), device=self.device)

                            x_same_neg = _pack_pair(feat_left, feat_right, same_neg_mask)
                            y_same_neg = torch.zeros(x_same_neg.size(0), device=self.device)

                            # =========================
                            # CHIMERA components
                            # =========================
                            Bsz = truths.size(0)
                            if Bsz > 1:
                                shift = torch.randint(1, Bsz, (1,), device=self.device).item()  # ensure j != i
                                perm = (torch.arange(Bsz, device=self.device) + shift) % Bsz
                            else:
                                perm = torch.arange(Bsz, device=self.device)

                            tl_chi  = t_left
                            tr_chi  = t_right[perm]

                            if op == 1:   op_chi = (tl_chi == tr_chi)        # IFF
                            elif op == 2: op_chi = (~tl_chi) | tr_chi        # IMPLIES
                            elif op == 3: op_chi = tl_chi & tr_chi           # AND
                            elif op == 4: op_chi = tl_chi | tr_chi           # OR
                            else:         raise ValueError(f"unknown op code {op}")

                            x_chi_pos = _pack_pair(feat_left, feat_right[perm],  op_chi)
                            y_chi_pos = torch.ones(x_chi_pos.size(0), device=self.device)

                            x_chi_neg = _pack_pair(feat_left, feat_right[perm], ~op_chi)
                            y_chi_neg = torch.zeros(x_chi_neg.size(0), device=self.device)

                            # =========================
                            # Assemble according to mode
                            # =========================
                            xs, ys = [], []
                            if negatives == "ad_strict":
                                # true positives (same-image normals) + chimera negatives ONLY
                                if x_same_pos.numel() > 0: xs.append(x_same_pos); ys.append(y_same_pos)
                                if x_chi_neg.numel()  > 0: xs.append(x_chi_neg);  ys.append(y_chi_neg)

                            elif negatives == "ad_chimera_pos":
                                # true positives (same-image normals) + chimera (both positives and negatives)
                                if x_same_pos.numel() > 0: xs.append(x_same_pos); ys.append(y_same_pos)
                                if x_chi_neg.numel()  > 0: xs.append(x_chi_neg);  ys.append(y_chi_neg)
                                if x_chi_pos.numel() > 0: xs.append(x_chi_pos); ys.append(y_chi_pos)

                            elif negatives == "chimera_plus_true":
                                # true_only + chimera positives + chimera negatives
                                if x_same_pos.numel() > 0: xs.append(x_same_pos); ys.append(y_same_pos)
                                if x_same_neg.numel() > 0: xs.append(x_same_neg); ys.append(y_same_neg)
                                if x_chi_pos.numel() > 0: xs.append(x_chi_pos); ys.append(y_chi_pos)
                                if x_chi_neg.numel() > 0: xs.append(x_chi_neg); ys.append(y_chi_neg)

                            elif negatives == "chimeras_only":
                                # ONLY chimeras (both positives and negatives)
                                if x_chi_pos.numel() > 0: xs.append(x_chi_pos); ys.append(y_chi_pos)
                                if x_chi_neg.numel() > 0: xs.append(x_chi_neg); ys.append(y_chi_neg)

                            else:
                                pass  # fall through to true_only code (won't happen here)

                            if len(xs) == 0:
                                continue

                            x_batch = torch.cat(xs, dim=0)  # NOTE: size != Bsz in general
                            y_batch = torch.cat(ys, dim=0)

                            # shuffle the mixed batch
                            idx = torch.randperm(x_batch.size(0), device=self.device)
                            x_batch = x_batch[idx]; y_batch = y_batch[idx]

                            # ---- optimize with your existing gate and optimizer ----
                            gate = node_gates[n]
                            gate.train()
                            parent_feat_train, pred_train = gate(x_batch)   # training batch, arbitrary size
                            loss = bce(pred_train, y_batch.float())
                            node_opts[n].zero_grad(); loss.backward(); node_opts[n].step()
                            batch_loss += float(loss.item())

                            # ---- write back SAME-IMAGE features/probs for *all Bsz* to feed higher nodes ----
                            with torch.no_grad():
                                # build same-image pairs for ALL images in this batch
                                flags_all = flag_vec.expand(Bsz, -1)
                                x_same_all = torch.cat([feat_left, feat_right, flags_all], dim=1)  # (Bsz, 2F+2)
                                gate.eval()
                                parent_feat_all, pred_all = gate(x_same_all)
                                node_feats[:, n, :] = parent_feat_all.detach()   # (Bsz, F)
                                node_probs[:,  n]   = pred_all.detach()          # (Bsz,)
                            continue  # skip true_only batching


                        # else: fall through to your existing true_only batching logic

                        gate = node_gates[n]
                        gate.train()
                        child_flat = self._gather_child_flat(n, node_feats)
                        target = truths[:, n]
                        par_feat, pred = gate(child_flat)
                        loss = bce(pred, target)
                        node_opts[n].zero_grad(); loss.backward(); node_opts[n].step()
                        batch_loss += float(loss.item())
                        node_feats[:, n, :] = par_feat.detach()
                        node_probs[:, n] = pred.detach()

                    total_loss += batch_loss / max(len(nodes_at_level), 1)
                    n_batches += 1
                    if use_tqdm:
                        avg = total_loss / max(n_batches, 1)
                        tqdm.write(f"[L{depth_idx} E{ep}] batch={n_batches} avg_loss={avg:.4f}") if verbose and n_batches % 50 == 0 else None

                if skip_level:
                    break

                if verbose and not use_tqdm:
                    print(f"  level {depth_idx} | epoch {ep}/{epochs_per_level} | avg_loss={total_loss/max(n_batches,1):.4f}")

            if skip_level:
                continue

            # persist + register
            saved = 0
            for n in nodes_at_level:
                gate_cpu = node_gates[n].cpu()
                self.runtime_gates[n] = gate_cpu
                can_cache_here = (not self.cache_leafroot_only) or self._children_are_leaves(n)
                if can_cache_here:
                    key_n = self.cache_key(n)
                    self.cache.save(key_n, gate_cpu); saved += 1
            if verbose:
                print(f"[Trainer] Saved {saved} gate(s) for level {depth_idx}.")

        if verbose:
            print("[Trainer] Training complete. Gates cached with lineage-aware keys; deeper gates kept in-memory too.")

    # ---- predict ----
    def predict_root(self, image: torch.Tensor, leaf_bank) -> float:
        g = self.g.to(self.device)
        if self.lineage_aware and self._enc_fprint is None:
            # infer fingerprint if user constructed trainer and went straight to predict_root
            self._enc_fprint = encoder_fingerprint(leaf_bank)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        leaf_feats, leaf_probs = self._leaf_feats_probs(image, leaf_bank)
        F = self.feat_dim or leaf_feats.shape[-1]
        B = image.shape[0]
        num_nodes = g.num_nodes()
        node_feats = torch.zeros(B, num_nodes, F, device=self.device)
        node_probs = torch.zeros(B, num_nodes, device=self.device)

        leaf_prob_nodes = fill_node_vector_from_concepts(g, leaf_probs)
        node_probs[:, g.ndata['mask'].bool()] = leaf_prob_nodes[:, g.ndata['mask'].bool()]

        cid = g.ndata['x'].long(); leaf_idx = torch.nonzero(cid > 0, as_tuple=False).flatten()
        if leaf_idx.numel() > 0:
            cids0 = (cid[leaf_idx] - 1).tolist()
            node_feats[:, leaf_idx, :] = leaf_feats[:, cids0, :]

        # propagate level-by-level
        for d in range(1, len(self.levels)):
            for n in self.levels[d]:
                gate = self.runtime_gates.get(n, None)
                if gate is None:
                    key = self.cache_key(n)
                    if self.cache.has(key):
                        gate = self.cache.load(key, arity=len(g.in_edges(n)[0]), feat_dim=F)
                        self.runtime_gates[n] = gate
                if gate is None:
                    raise RuntimeError(
                        f"No trained/cached gate available for node {n} at depth {d}. "
                        f"Train this tree or enable lineage-aware caching for reuse."
                    )
                gate = gate.to(self.device).eval()
                with torch.no_grad():
                    child_flat = self._gather_child_flat(n, node_feats)
                    par_feat, par_prob = gate(child_flat)
                    node_feats[:, n, :] = par_feat
                    node_probs[:, n] = par_prob

        out_deg = g.out_degrees()
        root = int(torch.nonzero(out_deg == 0, as_tuple=False).flatten()[0].item())
        return float(node_probs[0, root].item())

# ------------------------------
# Minimal direct run
# ------------------------------
if __name__ == "__main__":
    print("LevelwiseTrainer ready (lineage-aware caching). Import and use in your pipeline.")
