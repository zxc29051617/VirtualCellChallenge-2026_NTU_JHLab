"""Cache scGPT's gene embeddings so torch is only needed once.

    python scripts/extract_scgpt_embeddings.py

The checkpoint (best_model.pt, vocab.json, args.json -- the "whole-human" model) is on a
public bucket, no credentials needed:

    https://czi-scgenept-public.s3.amazonaws.com/models/pretrained/scgpt/<file>

Requires `pip install torch`. Writes external/profiles/scgpt_gene_embeddings.npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.embeddings import cache_gene_embeddings, load_gene_embeddings
from vcc2026.io import read_csv_column, read_name_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="external/scgpt")
    parser.add_argument("--output", default="external/profiles/scgpt_gene_embeddings.npz")
    parser.add_argument("--genes", default="dataset/gene_names.csv")
    parser.add_argument("--perts", default="dataset/pert_counts.csv")
    args = parser.parse_args()

    output = cache_gene_embeddings(args.checkpoint_dir, args.output)
    genes, vectors = load_gene_embeddings(output)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.0f} MB): "
          f"{len(genes):,} genes x {vectors.shape[1]} dims")

    known = set(genes)
    axis = read_name_list(args.genes)
    targets = read_csv_column(args.perts, expected_header="target_gene")
    print(f"Challenge gene axis: {sum(g in known for g in axis):,}/{len(axis):,} in the scGPT vocab")
    missing = [t for t in targets if t not in known]
    print(f"Challenge targets:   {len(targets) - len(missing)}/{len(targets)} in the scGPT vocab")
    if missing:
        print(f"  without an embedding: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
