#!/usr/bin/env python3
# train_vae_probvalue.py
#
# VAE over postfix tree "programs" + predictor for full prob_value vector.
#
# Assumptions about dataset graphs (from your builder):
#   - g.ndata['mask'] (1 leaf, 0 internal)
#   - g.ndata['x'] (leaf concept id in {1..N}, internal 0)
#   - g.ndata['y'] (internal op code in {1..4}, leaf 0)
#       1=IFF, 2=IMPLIES, 3=AND, 4=OR
#   - g.edata['neg'] in {+1,-1}
#   - optional g.edata['pos'] for IMPLIES ordering (0 left, 1 right)
#   - g.ndata['prob_value'] float32 (V,) containing leaf+intermediate+root probs
#
# Tokenization:
#   - LEAF(c): token_id = c-1                           in [0, N-1]
#   - OP(op, negL, negR): token_id = N + (op-1)*4 + pat in [N, N+15]
#       pat = negL*2 + negR, neg bit 1 means edge negated (neg=-1)
#   - BOS = N+16, EOS = N+17, PAD = N+18
#
# Targets:
#   - token reconstruction via CE
#   - prob prediction via BCE on node tokens only (leaf/op); EOS/PAD masked out
#
# Generation:
#   - constrained greedy decode with stack-depth grammar (optional, provided)
#
import argparse
import math
import random
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import load_graphs


# -----------------------------
# Token helpers
# -----------------------------

OP_IFF, OP_IMPL, OP_AND, OP_OR = 1, 2, 3, 4

def is_leaf_token(tok: int, N: int) -> bool:
    return 0 <= tok < N

def is_op_token(tok: int, N: int) -> bool:
    return N <= tok < N + 16

def is_eos(tok: int, N: int) -> bool:
    return tok == N + 17

def is_pad(tok: int, N: int) -> bool:
    return tok == N + 18

def bos_id(N: int) -> int:
    return N + 16

def eos_id(N: int) -> int:
    return N + 17

def pad_id(N: int) -> int:
    return N + 18

def vocab_size(N: int) -> int:
    return N + 19  # N leaf + 16 op-pattern + BOS + EOS + PAD

def op_token_id(N: int, op: int, negL_bit: int, negR_bit: int) -> int:
    pat = int(negL_bit) * 2 + int(negR_bit)
    return N + (op - 1) * 4 + pat

def decode_op_token(N: int, tok: int) -> Tuple[int, int, int]:
    # returns (op, negL_bit, negR_bit)
    k = tok - N
    op = k // 4 + 1
    pat = k % 4
    negL = pat // 2
    negR = pat % 2
    return op, negL, negR


# -----------------------------
# Graph -> postfix tokens
# -----------------------------

def _ensure_child_to_parent(g: dgl.DGLGraph) -> dgl.DGLGraph:
    # Root convention: unique node with out_degree == 0
    out_deg = g.out_degrees()
    roots = torch.nonzero(out_deg == 0, as_tuple=False).flatten()
    if len(roots) == 1:
        return g
    gr = dgl.reverse(g, copy_ndata=True, copy_edata=True)
    out_deg = gr.out_degrees()
    roots = torch.nonzero(out_deg == 0, as_tuple=False).flatten()
    if len(roots) != 1:
        raise ValueError("Graph cannot be oriented to have unique root (out_degree==0).")
    return gr

def _root_id(g: dgl.DGLGraph) -> int:
    out_deg = g.out_degrees()
    roots = torch.nonzero(out_deg == 0, as_tuple=False).flatten()
    if len(roots) != 1:
        raise ValueError("Expected exactly one root (out_degree==0).")
    return int(roots[0].item())

def _ordered_children_with_eids(g: dgl.DGLGraph, parent: int) -> List[Tuple[int, int]]:
    """
    Returns list of (child_node_id, edge_id) ordered deterministically:
      - if parent is IMPLIES and 'pos' exists: order by pos 0 then 1
      - else: order by child node id ascending
    """
    src, _, eid = g.in_edges(parent, form="all")
    if len(src) == 0:
        return []
    src = src.long()
    eid = eid.long()

    op = int(g.ndata["y"][parent].item()) if "y" in g.ndata else 0
    if op == OP_IMPL and "pos" in g.edata:
        pos = g.edata["pos"][eid].long()
        perm = torch.argsort(pos)  # 0 then 1
        src = src[perm]
        eid = eid[perm]
    else:
        perm = torch.argsort(src)
        src = src[perm]
        eid = eid[perm]

    return [(int(s.item()), int(e.item())) for s, e in zip(src, eid)]

def graph_to_postfix_tokens_and_probs(g: dgl.DGLGraph, N: int) -> Tuple[List[int], List[float]]:
    """
    Produce postfix token list (no BOS; includes EOS at end) and aligned prob_value list
    (same length; EOS prob is dummy and masked later).
    """
    g = _ensure_child_to_parent(g)

    mask = g.ndata["mask"].long()
    x = g.ndata["x"].long()
    y = g.ndata["y"].long()
    prob_value = g.ndata["prob_value"].float()

    root = _root_id(g)

    tokens: List[int] = []
    probs: List[float] = []

    def dfs_post(node: int):
        children = _ordered_children_with_eids(g, node)
        # recurse
        for c, _eid in children:
            dfs_post(c)
        # emit this node
        if int(mask[node].item()) == 1:
            cid = int(x[node].item())
            if cid < 1 or cid > N:
                raise ValueError(f"Leaf concept id out of range: {cid}")
            tokens.append(cid - 1)  # leaf token
            probs.append(float(prob_value[node].item()))
        else:
            op = int(y[node].item())
            if op < 1 or op > 4:
                raise ValueError(f"Internal op out of range: {op}")

            # SST should be binary; expect 2 children. If not, you can either skip or coerce.
            if len(children) != 2:
                # Coerce: sort children, take first two (best-effort)
                if len(children) < 2:
                    # degenerate: treat as leaf-like zero-op (shouldn't happen)
                    tokens.append(op_token_id(N, op, 0, 0))
                    probs.append(float(prob_value[node].item()))
                    return
                children2 = children[:2]
            else:
                children2 = children

            # neg bits are on incoming edges (child -> node)
            (cL, eL), (cR, eR) = children2[0], children2[1]
            negL = 1 if int(g.edata["neg"][eL].item()) == -1 else 0
            negR = 1 if int(g.edata["neg"][eR].item()) == -1 else 0

            tokens.append(op_token_id(N, op, negL, negR))
            probs.append(float(prob_value[node].item()))

    dfs_post(root)

    # EOS
    tokens.append(eos_id(N))
    probs.append(0.0)  # dummy

    return tokens, probs


# -----------------------------
# Dataset + collate
# -----------------------------

class PostfixDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path: str, N: int, max_graphs: Optional[int] = None):
        graphs, _labels = load_graphs(bin_path)
        if max_graphs is not None:
            graphs = graphs[:max_graphs]
        self.N = N
        self.samples = []
        for g in graphs:
            # ensure required fields
            for k in ["mask", "x", "y", "prob_value"]:
                if k not in g.ndata:
                    raise ValueError(f"Graph missing ndata['{k}']")
            if "neg" not in g.edata:
                raise ValueError("Graph missing edata['neg']")
            toks, probs = graph_to_postfix_tokens_and_probs(g, N)
            self.samples.append((toks, probs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_batch(batch, N: int):
    """
    Returns:
      tokens_in:  (B, T) decoder input with BOS + tokens[:-1]
      tokens_tgt: (B, T) target tokens (original tokens including EOS, padded)
      attn_mask:  (B, T) 1 for non-pad in tokens_tgt, 0 for pad
      prob_tgt:   (B, T) target probs aligned to tokens_tgt (EOS/PAD are dummy)
      prob_mask:  (B, T) 1 for node tokens (leaf/op), 0 for EOS/PAD
      root_pos:   (B,)   position index of root node token (last node token before EOS)
    """
    toks_list, probs_list = zip(*batch)
    B = len(toks_list)
    lengths = [len(t) for t in toks_list]  # includes EOS
    T = max(lengths)
    PAD = pad_id(N)
    BOS = bos_id(N)

    tokens_tgt = torch.full((B, T), PAD, dtype=torch.long)
    prob_tgt = torch.zeros((B, T), dtype=torch.float32)
    attn_mask = torch.zeros((B, T), dtype=torch.float32)
    prob_mask = torch.zeros((B, T), dtype=torch.float32)
    root_pos = torch.zeros((B,), dtype=torch.long)

    for i, (toks, probs) in enumerate(zip(toks_list, probs_list)):
        L = len(toks)
        tokens_tgt[i, :L] = torch.tensor(toks, dtype=torch.long)
        prob_tgt[i, :L] = torch.tensor(probs, dtype=torch.float32)
        attn_mask[i, :L] = 1.0

        # prob_mask: 1 for node tokens (leaf/op), 0 for EOS/PAD
        for j in range(L):
            tok = toks[j]
            if is_leaf_token(tok, N) or is_op_token(tok, N):
                prob_mask[i, j] = 1.0

        # root position is last node token before EOS
        # (EOS is last token by construction)
        # find last j < L with prob_mask==1
        rp = None
        for j in range(L - 1, -1, -1):
            if prob_mask[i, j] == 1.0:
                rp = j
                break
        if rp is None:
            rp = 0
        root_pos[i] = rp

    # decoder input: BOS + tokens_tgt[:-1]
    tokens_in = torch.full((B, T), PAD, dtype=torch.long)
    tokens_in[:, 0] = BOS
    if T > 1:
        tokens_in[:, 1:] = tokens_tgt[:, :-1]

    return tokens_in, tokens_tgt, attn_mask, prob_tgt, prob_mask, root_pos


# -----------------------------
# VAE model
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        # x: (B,T,d)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class SyntaxVAE(nn.Module):
    def __init__(
        self,
        N_concepts: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        latent_dim: int = 64,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.N = N_concepts
        self.V = vocab_size(N_concepts)
        self.PAD = pad_id(N_concepts)
        self.BOS = bos_id(N_concepts)
        self.EOS = eos_id(N_concepts)

        self.tok_emb = nn.Embedding(self.V, d_model, padding_idx=self.PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        # Decoder: memory is a single latent token
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.z_proj = nn.Linear(latent_dim, d_model)

        self.lm_head = nn.Linear(d_model, self.V)       # token logits
        self.prob_head = nn.Linear(d_model, 1)          # per-position prob prediction (sigmoid later)

    def encode(self, tokens_tgt: torch.Tensor, attn_mask: torch.Tensor):
        # tokens_tgt: (B,T) including EOS/PAD
        x = self.tok_emb(tokens_tgt)
        x = self.drop(self.pos(x))

        # key_padding_mask expects True for PAD positions
        key_padding_mask = (attn_mask == 0.0)  # (B,T) bool
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,T,d)

        # mean-pool over non-pad
        denom = attn_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom  # (B,d)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, tokens_in: torch.Tensor, attn_mask_tgt: torch.Tensor):
        # tokens_in: (B,T) BOS + shifted tokens
        B, T = tokens_in.shape
        x = self.tok_emb(tokens_in)
        x = self.drop(self.pos(x))

        # causal mask for autoregressive decoding
        causal = torch.triu(torch.ones(T, T, device=tokens_in.device), diagonal=1).bool()

        # decoder key padding mask uses PAD positions in the *target* sequence (tokens_in shares PAD)
        tgt_key_padding_mask = (tokens_in == self.PAD)

        # memory is (B, 1, d)
        mem = self.z_proj(z).unsqueeze(1)

        out = self.decoder(
            tgt=x,
            memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B,T,d)

        logits = self.lm_head(out)  # (B,T,V)
        prob = torch.sigmoid(self.prob_head(out)).squeeze(-1)  # (B,T)

        return logits, prob

    def forward(self, tokens_in, tokens_tgt, attn_mask, prob_tgt, prob_mask):
        mu, logvar = self.encode(tokens_tgt, attn_mask)
        z = self.reparameterize(mu, logvar)
        logits, prob_pred = self.decode(z, tokens_in, attn_mask)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "logits": logits,
            "prob_pred": prob_pred,
        }


# -----------------------------
# Loss + training
# -----------------------------

def kl_div(mu, logvar):
    # KL(q||p) with p=N(0,I)
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))

def train_one_epoch(model, loader, optimizer, device, beta_kl, w_prob, w_root, N):
    model.train()
    total = 0.0
    pbar = tqdm(iterable=loader, unit=" batch")
    for (tokens_in, tokens_tgt, attn_mask, prob_tgt, prob_mask, root_pos) in pbar:
        tokens_in = tokens_in.to(device)
        tokens_tgt = tokens_tgt.to(device)
        attn_mask = attn_mask.to(device)
        prob_tgt = prob_tgt.to(device)
        prob_mask = prob_mask.to(device)
        root_pos = root_pos.to(device)

        out = model(tokens_in, tokens_tgt, attn_mask, prob_tgt, prob_mask)

        logits = out["logits"]      # (B,T,V)
        prob_pred = out["prob_pred"]

        # token reconstruction CE (ignore PAD)
        V = logits.size(-1)
        loss_rec = F.cross_entropy(
            logits.view(-1, V),
            tokens_tgt.view(-1),
            ignore_index=pad_id(N),
        )

        # prob prediction BCE on node tokens only (leaf/op)
        # prob_mask is 1 for node tokens, 0 for EOS/PAD
        denom = prob_mask.sum().clamp_min(1.0)
        loss_prob = F.binary_cross_entropy(prob_pred, prob_tgt, reduction="none")
        loss_prob = (loss_prob * prob_mask).sum() / denom

        # root extra term
        # root_pos indexes into token positions; take per-sample
        B = tokens_tgt.size(0)
        idx = torch.arange(B, device=device)
        root_pred = prob_pred[idx, root_pos]
        root_tgt = prob_tgt[idx, root_pos]
        loss_root = F.binary_cross_entropy(root_pred, root_tgt)

        loss_kl = kl_div(out["mu"], out["logvar"])

        loss = loss_rec + beta_kl * loss_kl + w_prob * loss_prob + w_root * loss_root

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device, beta_kl, w_prob, w_root, N):
    model.eval()
    total = 0.0
    pbar = tqdm(iterable=loader, unit=" batch")
    for (tokens_in, tokens_tgt, attn_mask, prob_tgt, prob_mask, root_pos) in pbar:
        tokens_in = tokens_in.to(device)
        tokens_tgt = tokens_tgt.to(device)
        attn_mask = attn_mask.to(device)
        prob_tgt = prob_tgt.to(device)
        prob_mask = prob_mask.to(device)
        root_pos = root_pos.to(device)

        out = model(tokens_in, tokens_tgt, attn_mask, prob_tgt, prob_mask)
        logits = out["logits"]
        prob_pred = out["prob_pred"]

        V = logits.size(-1)
        loss_rec = F.cross_entropy(
            logits.view(-1, V),
            tokens_tgt.view(-1),
            ignore_index=pad_id(N),
        )

        denom = prob_mask.sum().clamp_min(1.0)
        loss_prob = F.binary_cross_entropy(prob_pred, prob_tgt, reduction="none")
        loss_prob = (loss_prob * prob_mask).sum() / denom

        B = tokens_tgt.size(0)
        idx = torch.arange(B, device=device)
        root_pred = prob_pred[idx, root_pos]
        root_tgt = prob_tgt[idx, root_pos]
        loss_root = F.binary_cross_entropy(root_pred, root_tgt)

        loss_kl = kl_div(out["mu"], out["logvar"])
        loss = loss_rec + beta_kl * loss_kl + w_prob * loss_prob + w_root * loss_root

        total += float(loss.item())
    return total / max(1, len(loader))


# -----------------------------
# Optional: constrained greedy decode from z (postfix grammar)
# -----------------------------

@torch.no_grad()
def greedy_decode_constrained(model: SyntaxVAE, z: torch.Tensor, max_len: int = 256):
    """
    Decode a postfix token program with a simple stack-depth constraint:
      - leaf token increases stack depth by +1
      - op token requires depth>=2 and changes depth by -1 (pop2 push1)
      - EOS allowed only when depth==1 and length>0
    Returns list of tokens INCLUDING EOS.
    """
    device = z.device
    N = model.N
    BOS, EOS, PAD = model.BOS, model.EOS, model.PAD

    tokens_in = torch.full((1, max_len), PAD, dtype=torch.long, device=device)
    tokens_in[0, 0] = BOS

    depth = 0
    for t in range(max_len):
        # decode up to position t (tokens_in has BOS at pos0, predictions align to tokens_tgt positions)
        T = t + 1
        logits, prob_pred = model.decode(z, tokens_in[:, :T], attn_mask_tgt=None)
        # next token is at position t (since tokens_in includes BOS, target index t corresponds to position t)
        next_logits = logits[0, t]  # (V,)

        # build mask
        allow = torch.zeros_like(next_logits, dtype=torch.bool)

        # allow leaf tokens always (but avoid finishing too early unless you want)
        allow[:N] = True

        # allow op tokens only if depth >= 2
        if depth >= 2:
            allow[N:N+16] = True

        # allow EOS only if depth == 1 and we already emitted at least 1 node token
        if depth == 1 and t > 0:
            allow[EOS] = True

        # disallow PAD/BOS always
        allow[PAD] = False
        allow[BOS] = False

        masked = next_logits.clone()
        masked[~allow] = -1e9
        tok = int(torch.argmax(masked).item())

        tokens_in[0, t+1 if (t+1) < max_len else t] = tok  # write token for next step

        # update depth
        if is_leaf_token(tok, N):
            depth += 1
        elif is_op_token(tok, N):
            depth -= 1
        elif tok == EOS:
            return tokens_in[0, 1:t+1].tolist() + [EOS]  # exclude BOS
        else:
            # fallback: stop
            return tokens_in[0, 1:t+1].tolist() + [EOS]

    return tokens_in[0, 1:].tolist()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_bin", type=str, required=True)
    ap.add_argument("--N_concepts", type=int, required=True)
    ap.add_argument("--max_graphs", type=int, default=None)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta_kl", type=float, default=0.1)
    ap.add_argument("--w_prob", type=float, default=1.0)
    ap.add_argument("--w_root", type=float, default=2.0)

    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--save_path", type=str, default="vae_probvalue.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    ds = PostfixDataset(args.data_bin, N=args.N_concepts, max_graphs=args.max_graphs)
    n = len(ds)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(args.train_split * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:] if n_train < n else idx[: max(1, n - n_train)]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    collate = lambda batch: collate_batch(batch, N=args.N_concepts)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SyntaxVAE(
        N_concepts=args.N_concepts,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, args.beta_kl, args.w_prob, args.w_root, args.N_concepts)
        va = eval_one_epoch(model, val_loader, device, args.beta_kl, args.w_prob, args.w_root, args.N_concepts)
        print(f"epoch {ep:03d} | train {tr:.6f} | val {va:.6f}")

        if va < best:
            best = va
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                },
                args.save_path,
            )

    print(f"Saved best checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()