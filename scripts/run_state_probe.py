"""Run Arc's STATE on a Challenge context and see what it does with Challenge targets.

STATE (ST-HVG-Replogle, zero-shot K562 checkpoint) identifies a perturbation by position in a
fixed 2,024-gene one-hot map. None of the Challenge's 300 targets is in it, and STATE's own
inference code replaces an unmapped label with the *control* one-hot. This builds an input of
real control cells from a Challenge context, asks STATE to simulate some perturbations it knows
and some it does not, and compares the outputs.

    python scripts/run_state_probe.py prepare            # python 3.14 env (anndata)
    state tx infer ...                                   # python 3.12 env, printed by `prepare`
    python scripts/run_state_probe.py analyze
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.cell_validation import read_rows
from vcc2026.io import read_csv_column, read_name_list
from vcc2026.transfer import source_effect_table

MODEL = Path("external/state/hvg_k562")
WORK = Path("external/state/probe")
CONTEXT = "A"
N_CONTROLS = 600
N_KNOWN = N_UNKNOWN = 6
CELLS_PER_PERT = 64


def pick_perturbations() -> tuple[list[str], list[str]]:
    vocab = {str(k) for k in torch.load(MODEL / "pert_onehot_map.pt", weights_only=False)}
    vocab.discard("non-targeting")
    labels, effects, _ = source_effect_table("external/profiles/K562gw_perturbation_means.npz")
    strength = {l: float(np.sqrt((effects[i] ** 2).mean())) for i, l in enumerate(labels)}
    known = sorted((g for g in vocab if g in strength), key=lambda g: -strength[g])[:N_KNOWN]
    targets = read_csv_column("dataset/pert_counts.csv", expected_header="target_gene")
    unknown = [t for t in targets if t in strength and t not in vocab][:N_UNKNOWN]
    return known, unknown


def prepare() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    with open(MODEL / "var_dims.pkl", "rb") as f:
        hvg = list(pickle.load(f)["gene_names"])
    symbols = read_name_list("dataset/gene_names.csv")
    column = {g: i for i, g in enumerate(symbols)}
    rows = np.sort(np.random.default_rng(0).choice(18400, size=N_CONTROLS, replace=False))
    counts = read_rows(f"dataset/context_{CONTEXT}.h5ad", rows)
    scaled = counts / np.maximum(counts.sum(1, keepdims=True), 1) * 1e4
    logged = np.log1p(scaled)
    present = [g for g in hvg if g in column]
    x = np.zeros((N_CONTROLS, len(hvg)), dtype=np.float32)
    for j, g in enumerate(hvg):
        if g in column:
            x[:, j] = logged[:, column[g]]
    print(f"STATE's 2,000 HVGs: {len(present)} present on the Challenge gene axis")

    obs = pd.DataFrame(
        # STATE has a batch encoder trained on 56 Replogle gem groups. A Challenge context has no
        # such label, so inference falls back to training batch index 0, an arbitrary choice.
        {"gene": "non-targeting", "cell_line": CONTEXT, "gem_group": CONTEXT},
        index=[f"{CONTEXT}_ctrl_{i}" for i in range(N_CONTROLS)],
    )
    adata = ad.AnnData(X=x.copy(), obs=obs, var=pd.DataFrame(index=hvg))
    adata.obsm["X_hvg"] = x
    adata.write_h5ad(WORK / "input.h5ad")

    known, unknown = pick_perturbations()
    pd.DataFrame(
        {"perturbation": known + unknown, "num_cells": CELLS_PER_PERT}
    ).to_csv(WORK / "perts.tsv", sep="\t", index=False)
    print("known to STATE:  ", known)
    print("unknown to STATE:", unknown)
    print("\nnow run (python 3.12 env):")
    print("  external\\state\\venv\\Scripts\\state.exe tx infer --model-dir external/state/hvg_k562 "
          "--checkpoint external/state/hvg_k562/checkpoints/final.ckpt "
          f"--adata {WORK / 'input.h5ad'} --pert-col gene --celltype-col cell_line "
          f"--embed-key X_hvg --tsv {WORK / 'perts.tsv'} --output {WORK / 'output.h5ad'}")
    return 0


def analyze() -> int:
    out = ad.read_h5ad(WORK / "output.h5ad")
    print("output:", out.shape, "| obsm keys:", list(out.obsm.keys()), "| obs cols:", list(out.obs.columns))
    key = "X_hvg" if "X_hvg" in out.obsm else None
    pred = np.asarray(out.obsm[key] if key else out.X)
    labels = out.obs["gene"].astype(str).values
    control = pred[labels == "non-targeting"].mean(0)
    known, unknown = pick_perturbations()

    shift = {p: pred[labels == p].mean(0) - control for p in known + unknown if (labels == p).any()}
    print(f"\ncells per label: {[ (p, int((labels == p).sum())) for p in list(shift)[:3] ]} ...")
    print("\nsize of the predicted shift from the control prediction (RMS over 2,000 HVGs)")
    for p, s in shift.items():
        tag = "known to STATE  " if p in known else "UNKNOWN to STATE"
        print(f"  {p:<10}{tag}  rms {np.sqrt((s ** 2).mean()):.4f}")

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    print("\nhow different are the unknown targets' predictions from each other?")
    u = [shift[p] for p in unknown if p in shift]
    if len(u) > 1:
        pair = [corr(u[i], u[j]) for i in range(len(u)) for j in range(i + 1, len(u))]
        print(f"  mean pairwise correlation of unknown targets' shifts: {np.mean(pair):+.3f}")
    k = [shift[p] for p in known if p in shift]
    if len(k) > 1:
        pair = [corr(k[i], k[j]) for i in range(len(k)) for j in range(i + 1, len(k))]
        print(f"  mean pairwise correlation of known genes' shifts:     {np.mean(pair):+.3f}")

    # Sanity for the known genes: does STATE's shift resemble the measured K562 genome-wide effect?
    labels_k, effects, genes = source_effect_table("external/profiles/K562gw_perturbation_means.npz")
    with open(MODEL / "var_dims.pkl", "rb") as f:
        hvg = list(pickle.load(f)["gene_names"])
    gcol = {g: i for i, g in enumerate(genes)}
    shared = [j for j, g in enumerate(hvg) if g in gcol]
    cols = [gcol[hvg[j]] for j in shared]
    print(f"\nknown genes: correlation of STATE's predicted shift (in context {CONTEXT}) with the measured "
          f"K562 effect, over {len(shared)} shared HVGs")
    for p in known:
        if p in shift and p in labels_k:
            m = effects[labels_k.index(p)][cols]
            print(f"  {p:<10}{corr(shift[p][shared], m):+.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "analyze"])
    args = parser.parse_args()
    return prepare() if args.step == "prepare" else analyze()


if __name__ == "__main__":
    raise SystemExit(main())
