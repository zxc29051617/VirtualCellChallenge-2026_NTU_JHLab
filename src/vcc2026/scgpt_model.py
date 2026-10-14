"""scGPT's encoder, rebuilt from the checkpoint's tensors for CPU inference.

The `scgpt` package does not install on current Pythons (its pinned dependencies include
torchtext, IPython, scanpy, scvi...), and only the encoder is needed. Parameter names below
mirror the checkpoint's state dict one for one, and `load_scgpt` fails loudly if any tensor is
missing or unexpected, so a mismatch cannot pass silently. The architecture follows
`scgpt/model/model.py` (TransformerModel with the flash-attention layer, post-norm, ReLU).

Importing this module needs torch; nothing else in the package imports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PAD_VALUE = -2.0
MASK_VALUE = -1.0
N_BINS = 51
MAX_LEN = 1200  # genes per cell including <cls>, as in pretraining
TARGET_SUM = 1e4
IGNORED_PREFIXES = ("mvc_decoder",)  # a training-time head, not needed here


class _GeneEncoder(nn.Module):
    def __init__(self, ntoken: int, d: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(ntoken, d)
        self.enc_norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.enc_norm(self.embedding(ids))


class _ValueEncoder(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(1, d)
        self.linear2 = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(values.unsqueeze(-1), max=512)
        return self.norm(self.linear2(F.relu(self.linear1(x))))


class _Attention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.Wqkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        # Layout is (three, heads, head_dim), as in FlashMHA's rearrange of Wqkv.
        q, k, v = self.Wqkv(x).view(b, length, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        return self.out_proj(out.transpose(1, 2).reshape(b, length, d))


class _Layer(nn.Module):
    def __init__(self, d: int, heads: int, d_hid: int) -> None:
        super().__init__()
        self.self_attn = _Attention(d, heads)
        self.linear1 = nn.Linear(d, d_hid)
        self.linear2 = nn.Linear(d_hid, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, keep))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


class _Stack(nn.Module):
    def __init__(self, d: int, heads: int, d_hid: int, nlayers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer(d, heads, d_hid) for _ in range(nlayers))


class _ExprDecoder(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, 1)
        )


class ScGPT(nn.Module):
    def __init__(
        self, ntoken: int, d: int = 512, heads: int = 8, nlayers: int = 12, d_hid: int = 512
    ) -> None:
        super().__init__()
        self.encoder = _GeneEncoder(ntoken, d)
        self.value_encoder = _ValueEncoder(d)
        self.flag_encoder = nn.Embedding(2, d)
        self.transformer_encoder = _Stack(d, heads, d_hid, nlayers)
        self.decoder = _ExprDecoder(d)

    def forward(
        self,
        ids: torch.Tensor,
        values: torch.Tensor,
        keep: torch.Tensor,
        flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hidden states [batch, length, d]. `keep` is true for real tokens, false for padding.

        `flags` (0 = observed, 1 = to be generated) is the pretraining-time marker; downstream
        use of scGPT, and `encode_cells`, leaves it out.
        """
        x = self.encoder(ids) + self.value_encoder(values)
        if flags is not None:
            x = x + self.flag_encoder(flags)
        for layer in self.transformer_encoder.layers:
            x = layer(x, keep)
        return x

    def predict_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder.fc(hidden).squeeze(-1)


def load_scgpt(checkpoint_dir: str | Path) -> tuple[ScGPT, dict[str, int]]:
    root = Path(checkpoint_dir)
    vocab: dict[str, int] = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    state = torch.load(root / "best_model.pt", map_location="cpu", weights_only=True)
    model = ScGPT(len(vocab))
    result = model.load_state_dict(state, strict=False)
    unexpected = set(result.unexpected_keys) - {k for k in state if k.startswith(IGNORED_PREFIXES)}
    if result.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the architecture: missing {result.missing_keys}, "
            f"unexpected {sorted(unexpected)}"
        )
    return model.eval(), vocab


# --- input preparation, following scgpt.preprocess.binning and the tokenizer --------------------


def _digitize(x: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """np.digitize with ties spread uniformly across the tied bins, as scGPT does."""
    left = np.digitize(x, bins)
    right = np.digitize(x, bins, right=True)
    return np.ceil(rng.random(len(x)) * (right - left) + left).astype(np.int64)


def bin_cells(counts: np.ndarray, rng: np.random.Generator, n_bins: int = N_BINS) -> np.ndarray:
    """Per-cell expression bins: scale to 1e4 counts, then quantile-bin the non-zero genes.

    log1p is monotone, so skipping it changes no bin. Zero stays zero.
    """
    totals = counts.sum(axis=1, keepdims=True)
    scaled = counts / np.where(totals > 0, totals, 1.0) * TARGET_SUM
    out = np.zeros(counts.shape, dtype=np.int64)
    for i, row in enumerate(scaled):
        nonzero = np.flatnonzero(row)
        if nonzero.size == 0:
            continue
        bins = np.quantile(row[nonzero], np.linspace(0, 1, n_bins - 1))
        out[i, nonzero] = _digitize(row[nonzero], bins, rng)
    return out


def _tokenize(
    binned: np.ndarray, token_ids: np.ndarray, cls_id: int, rng: np.random.Generator, max_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(token ids, bin values, source column of each gene token) for one cell, <cls> first."""
    columns = np.flatnonzero(binned)
    if columns.size > max_len - 1:
        columns = rng.choice(columns, size=max_len - 1, replace=False)
    ids = np.concatenate(([cls_id], token_ids[columns]))
    values = np.concatenate(([0.0], binned[columns].astype(np.float32)))
    return ids, values, columns


def _batch(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], pad_id: int):
    length = max(len(r[0]) for r in rows)
    ids = np.full((len(rows), length), pad_id, dtype=np.int64)
    values = np.full((len(rows), length), PAD_VALUE, dtype=np.float32)
    for i, (token_ids, v, _) in enumerate(rows):
        ids[i, : len(token_ids)] = token_ids
        values[i, : len(v)] = v
    ids_t = torch.from_numpy(ids)
    return ids_t, torch.from_numpy(values), ids_t != pad_id


@torch.no_grad()
def encode_cells(
    model: ScGPT,
    vocab: dict[str, int],
    counts: np.ndarray,
    symbols: list[str],
    *,
    seed: int = 0,
    batch_size: int = 8,
    max_len: int = MAX_LEN,
    progress: bool = False,
) -> dict[str, np.ndarray | list[str]]:
    """Run scGPT over raw-count cells and return what the transformer made of them.

    Returns `cell` [n, d] (the <cls> state), `gene_sum` [n_genes, d] and `gene_count`
    [n_genes] -- each gene's hidden state summed over the cells that expressed it and sampled
    into the context -- and `genes`, the symbols those rows belong to. Genes outside scGPT's
    vocabulary are dropped before normalising, as scGPT's own pipeline does.
    """
    usable = np.array([s in vocab for s in symbols])
    genes = [s for s, ok in zip(symbols, usable) if ok]
    token_ids = np.array([vocab[g] for g in genes], dtype=np.int64)
    rng = np.random.default_rng(seed)
    binned = bin_cells(np.asarray(counts, dtype=np.float64)[:, usable], rng)

    d = model.encoder.embedding.embedding_dim
    cell = np.zeros((len(binned), d), dtype=np.float32)
    gene_sum = torch.zeros(len(genes), d, dtype=torch.float64)
    gene_count = torch.zeros(len(genes), dtype=torch.float64)

    for start in range(0, len(binned), batch_size):
        rows = [
            _tokenize(b, token_ids, vocab["<cls>"], rng, max_len)
            for b in binned[start : start + batch_size]
        ]
        ids, values, keep = _batch(rows, vocab["<pad>"])
        hidden = model(ids, values, keep)
        cell[start : start + len(rows)] = hidden[:, 0].numpy()
        for i, (_, _, columns) in enumerate(rows):
            gene_sum.index_add_(0, torch.from_numpy(columns), hidden[i, 1 : 1 + len(columns)].double())
            gene_count.index_add_(0, torch.from_numpy(columns), torch.ones(len(columns), dtype=torch.float64))
        if progress and (start // batch_size) % 10 == 0:
            print(f"  encoded {min(start + batch_size, len(binned))}/{len(binned)} cells", flush=True)

    return {
        "cell": cell,
        "gene_sum": gene_sum.numpy(),
        "gene_count": gene_count.numpy(),
        "genes": genes,
    }


@torch.no_grad()
def masked_value_check(
    model: ScGPT,
    vocab: dict[str, int],
    counts: np.ndarray,
    symbols: list[str],
    *,
    mask_ratio: float = 0.25,
    seed: int = 0,
    batch_size: int = 8,
) -> dict[str, float]:
    """Does the rebuilt model do what scGPT was pretrained to do?

    Hide a quarter of each cell's expression values and ask the decoder to recover them from
    the rest. Compared per cell, by rank correlation, against the best a model that has learned
    nothing about cells could do: each gene's average bin over the batch. A wrong layer, a wrong
    norm placement or a mistranscribed tensor drives the model's number to zero or below.
    """
    from scipy.stats import spearmanr

    usable = np.array([s in vocab for s in symbols])
    genes = [s for s, ok in zip(symbols, usable) if ok]
    token_ids = np.array([vocab[g] for g in genes], dtype=np.int64)
    rng = np.random.default_rng(seed)
    binned = bin_cells(np.asarray(counts, dtype=np.float64)[:, usable], rng)
    gene_mean = np.where((binned > 0).any(0), binned.sum(0) / np.maximum((binned > 0).sum(0), 1), 0.0)

    model_r, baseline_r = [], []
    for start in range(0, len(binned), batch_size):
        rows = [
            _tokenize(b, token_ids, vocab["<cls>"], rng, MAX_LEN)
            for b in binned[start : start + batch_size]
        ]
        ids, values, keep = _batch(rows, vocab["<pad>"])
        truth = values.clone()
        hide = (torch.rand(values.shape) < mask_ratio) & keep
        hide[:, 0] = False
        values[hide] = MASK_VALUE
        predicted = model.predict_values(model(ids, values, keep))
        for i, (_, _, columns) in enumerate(rows):
            sel = hide[i, 1 : 1 + len(columns)].numpy()
            if sel.sum() < 10:
                continue
            t = truth[i, 1 : 1 + len(columns)].numpy()[sel]
            model_r.append(spearmanr(predicted[i, 1 : 1 + len(columns)].numpy()[sel], t)[0])
            baseline_r.append(spearmanr(gene_mean[columns][sel], t)[0])
    return {
        "model_spearman": float(np.nanmean(model_r)),
        "gene_mean_baseline_spearman": float(np.nanmean(baseline_r)),
        "cells": len(model_r),
    }
