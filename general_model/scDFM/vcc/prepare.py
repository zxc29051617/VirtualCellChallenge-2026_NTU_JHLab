"""Build one training/evaluation fold for multi-cell-line scDFM.

    python -m vcc.prepare --train HCT116 --heldout HEK293T --out data/stage3/model/heldout_HEK293T
    python -m vcc.prepare --train HCT116 HEK293T K562 --heldout VCC --out data/stage3/model/vcc

Gene set G (the genes the model predicts): genes expressed >= 5 CPM in the held-out controls
(the only held-out information a method is given), ranked by perturbation-response variance in
the training lines, top --n-genes. Every other gene is copied from the paired control cell at
generation time. The vocab is G plus every gene perturbed in the training lines, in alphabetical
order after 4 special tokens, which is the layout scDFM's coexpression mask assumes.

Outputs in --out:
  genes.txt          G, in matrix column order
  vocab.json         scDFM GeneVocab (specials + sorted(G | perturbed genes))
  mask.pt            coexpression graph over the vocab (scDFM format, topk=30, negative edges)
  <line>.npz         X: float16 log1p(counts / library * 1e4) on G; target; is_control; lib; measured_G
  heldout_ctrl.npz   the same for the held-out controls (inputs at generation time)
  fold.json          provenance
"""
import argparse, json, os
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, torch

from vcc.stats import FILES, load, lfc_table, signal_variance

ROOT = "/home/zxc29051617/VCC/data"
SPECIALS = ["<pad>", "<cls>", "<mask>", "control"]
TARGET_SUM = 1e4


def lognorm_rows(X, lib, cols):
    """log1p(count / lib * 1e4) for the selected columns, dense float16."""
    X = sp.csr_matrix(X)[:, cols].toarray().astype(np.float32)
    return np.log1p(X / lib[:, None] * TARGET_SUM).astype(np.float16)


def read_line(line, cols, chunk=20000):
    A = ad.read_h5ad(f"{ROOT}/stage3/{FILES[line]}.h5ad", backed="r")
    out, libs = [], []
    for lo in range(0, A.n_obs, chunk):
        X = sp.csr_matrix(A.X[lo:min(A.n_obs, lo + chunk)])
        lib = np.asarray(X.sum(1)).ravel()
        out.append(lognorm_rows(X, lib, cols)); libs.append(lib)
    return A.obs.copy(), np.concatenate(out), np.concatenate(libs), A.var.measured.values


def heldout_controls(name):
    """Raw-count control matrix of the held-out context(s): a stage-3 line, or the VCC contexts."""
    if name == "VCC":
        mats = [sp.csr_matrix(ad.read_h5ad(f"{ROOT}/vcc2026/context_{c}.h5ad").X) for c in "ABC"]
        ctx = np.concatenate([[c] * m.shape[0] for c, m in zip("ABC", mats)])
        return sp.vstack(mats).tocsr(), ctx
    A = ad.read_h5ad(f"{ROOT}/stage3/{FILES[name]}.h5ad", backed="r")
    idx = np.flatnonzero(A.obs.is_control.values)
    return sp.csr_matrix(A.X[idx]), np.array([name] * len(idx))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-genes", type=int, default=5000)
    p.add_argument("--mask-cells", type=int, default=20000)
    p.add_argument("--holdout-targets", default="", help="file of perturbations never trained on in the held-out line "
                   "(used when the held-out line is also a training line)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    genes = pd.read_csv(f"{ROOT}/vcc2026/gene_names.csv").gene_name.astype(str).tolist()

    # 1. gene set
    Ch, ctx = heldout_controls(a.heldout)
    cpm = np.asarray(Ch.sum(0)).ravel(); cpm = cpm / cpm.sum() * 1e6
    expressed = cpm >= 5
    stats = {l: load(l) for l in a.train}
    if a.holdout_targets and a.heldout in stats:        # gene ranking must not see the held-out perturbations
        hold = set(open(a.holdout_targets).read().split()); st = stats[a.heldout]
        k = np.array([n not in hold for n in st["names"]])
        stats[a.heldout] = {**st, "names": st["names"][k], "sums": st["sums"][k], "ncell": st["ncell"][k]}
    ranks = []
    for l, st in stats.items():
        sv = np.nan_to_num(signal_variance(st), nan=-np.inf)
        r = np.full(len(genes), np.nan); r[np.argsort(-sv)] = np.arange(len(genes))
        r[~st["measured"]] = np.nan                    # unmeasured in this line: no vote
        ranks.append(r)
    score = np.nanmean(np.stack(ranks), 0)             # mean rank across the lines that measure it
    score[~expressed | np.isnan(score)] = np.inf
    G = np.argsort(score)[:a.n_genes]
    G = np.sort(G)
    G_names = [genes[i] for i in G]
    print(f"G: {len(G)} genes ({expressed.sum()} expressed >=5 CPM in held-out controls)")

    # 2. vocab: G plus all perturbed genes of the training lines
    perturbed = set()
    for st in stats.values():
        perturbed |= {n for n in st["names"] if n != "__ctrl__"}
    perturbed &= set(genes)
    vocab_genes = sorted(set(G_names) | perturbed)
    vocab = {t: i for i, t in enumerate(SPECIALS + vocab_genes)}
    json.dump(vocab, open(f"{a.out}/vocab.json", "w"))
    open(f"{a.out}/genes.txt", "w").write("\n".join(G_names) + "\n")
    print(f"vocab: {len(vocab)} tokens ({len(perturbed)} perturbed genes, {len(set(G_names) & perturbed)} also in G)")

    # 3. per-line matrices on G
    vcols = np.array([genes.index(g) for g in vocab_genes])
    mask_rows = []
    rng = np.random.default_rng(0)
    for l in a.train:
        obs, X, lib, measured = read_line(l, G)
        target = np.where(obs.is_control.values, "__ctrl__", obs.target_gene.astype(str).values)
        np.savez(f"{a.out}/{l}.npz", X=X, target=target, is_control=obs.is_control.values, lib=lib,
                 measured_G=measured[G])
        print(f"{l}: {X.shape[0]} cells x {X.shape[1]} genes, {X.nbytes/2**30:.2f} GiB")
        # cells for the coexpression graph (vocab genes, lines that measure ~all genes)
        if measured.mean() > 0.9:
            A = ad.read_h5ad(f"{ROOT}/stage3/{FILES[l]}.h5ad", backed="r")
            pick = np.sort(rng.choice(A.n_obs, min(a.mask_cells // len(a.train), A.n_obs), replace=False))
            Xs = sp.csr_matrix(A.X[pick]); lib_s = np.asarray(Xs.sum(1)).ravel()
            mask_rows.append(lognorm_rows(Xs, lib_s, vcols))
        del X

    # 4. coexpression mask over the vocab, padded in scDFM's layout
    from src.utils.utils import build_gene_coexpression_graph, sorted_pad_mask
    Xm = np.concatenate(mask_rows).astype(np.float32)
    m = build_gene_coexpression_graph(Xm, method="pearson", sparsify="topk", k=30, log1p=False,
                                      use_negative_edge=True)
    m = sorted_pad_mask(m, pad_size=len(SPECIALS), gene_names=vocab_genes)
    torch.save(m, f"{a.out}/mask.pt")
    print(f"mask {tuple(m.shape)} from {Xm.shape[0]} cells")

    # 5. held-out controls on G (generation inputs) + their raw library sizes
    lib_h = np.asarray(Ch.sum(1)).ravel()
    np.savez(f"{a.out}/heldout_ctrl.npz", X=lognorm_rows(Ch, lib_h, G), lib=lib_h, context=ctx)
    hold = sorted(set(open(a.holdout_targets).read().split())) if a.holdout_targets else []
    json.dump({"train": a.train, "heldout": a.heldout, "n_genes": len(G), "vocab": len(vocab),
               "target_sum": TARGET_SUM, "holdout_targets": hold}, open(f"{a.out}/fold.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
