"""Check the rebuilt scGPT encoder does what scGPT was pretrained to do, before trusting it.

Masks a quarter of each real control cell's expression values and asks the decoder to recover
them. Reported against the best a model with no knowledge of cells could do (each gene's
average bin). A wrong layer, norm placement or tensor drives the model's number to ~0.

    python scripts/validate_scgpt_forward.py --cells 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.cell_validation import read_rows
from vcc2026.io import read_name_list
from vcc2026.scgpt_model import load_scgpt, masked_value_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="external/scgpt")
    parser.add_argument("--control", default="dataset/context_A.h5ad")
    parser.add_argument("--genes", default="dataset/gene_names.csv")
    parser.add_argument("--cells", type=int, default=48)
    args = parser.parse_args()

    model, vocab = load_scgpt(args.checkpoint_dir)
    symbols = read_name_list(args.genes)
    rows = np.sort(np.random.default_rng(0).choice(18400, size=args.cells, replace=False))
    counts = read_rows(args.control, rows)

    started = time.perf_counter()
    result = masked_value_check(model, vocab, counts, symbols)
    elapsed = time.perf_counter() - started
    print(f"{result['cells']} cells, {elapsed:.0f}s ({elapsed / result['cells']:.1f}s/cell)")
    print(f"masked-value recovery, per-cell Spearman over hidden genes:")
    print(f"  rebuilt scGPT          {result['model_spearman']:+.3f}")
    print(f"  gene-average baseline  {result['gene_mean_baseline_spearman']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
