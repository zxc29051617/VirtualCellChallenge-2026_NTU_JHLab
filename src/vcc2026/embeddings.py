"""Gene features from a pretrained scGPT checkpoint.

scGPT learns one vector per gene token while pretraining on 33M human cells, and genes that
act in the same programs end up near each other. Only that token-embedding table is read
here. It is a deterministic lookup: no GPU, no `scgpt` package (whose pinned dependencies do
not install on current Pythons), and no forward pass over cells. Contextual embeddings from
the transformer would need all three, and are not what the neighbour search needs -- a
Challenge target arrives as a gene name, with nothing about the cell attached.

torch is imported lazily and only by `load_scgpt_gene_embeddings`, so everything downstream
works from the `.npz` cache that `cache_gene_embeddings` writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The table's name in the state dict. scGPT's GeneEncoder wraps nn.Embedding as `.embedding`.
EMBEDDING_KEYS = ("encoder.embedding.weight", "gene_encoder.embedding.weight")


def load_scgpt_gene_embeddings(checkpoint_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """(gene symbols, vectors[n_genes, d]) for every gene token in the checkpoint's vocab.

    Special tokens such as `<pad>` and `<cls>` carry no gene and are dropped.
    """

    import torch  # optional dependency, see module docstring

    root = Path(checkpoint_dir)
    vocab: dict[str, int] = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    state = torch.load(root / "best_model.pt", map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    key = next((k for k in EMBEDDING_KEYS if k in state), None)
    if key is None:
        candidates = [k for k in state if "embed" in k.lower()]
        raise KeyError(f"No gene embedding table in {root / 'best_model.pt'}; saw {candidates}")

    table = state[key].detach().float().numpy()
    if table.shape[0] < max(vocab.values()) + 1:
        raise ValueError(
            f"Embedding table has {table.shape[0]} rows but the vocab reaches id {max(vocab.values())}"
        )

    genes = [token for token in vocab if not (token.startswith("<") and token.endswith(">"))]
    vectors = table[[vocab[g] for g in genes]].astype(np.float32, copy=False)
    return genes, vectors


def cache_gene_embeddings(checkpoint_dir: str | Path, output: str | Path) -> Path:
    genes, vectors = load_scgpt_gene_embeddings(checkpoint_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, genes=np.asarray(genes), vectors=vectors)
    return output


def load_gene_embeddings(path: str | Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(Path(path), allow_pickle=False)
    return [str(g) for g in payload["genes"]], payload["vectors"]
