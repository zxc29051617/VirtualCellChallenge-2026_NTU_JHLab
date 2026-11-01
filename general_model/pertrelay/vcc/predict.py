"""Predict held-out perturbations with a trained multi-line scDFM, as raw counts on the full
18,533-gene VCC axis, in the layout tools/loco.py scores.

    python -m vcc.predict --fold F --model runs/x/model.pt --heldout HEK293T --name x --method onestep

Each predicted cell is paired with one control cell of the held-out line (the same controls the
scorer uses, as in VCC). Genes in G come from the model, converted from log1p(count/lib*1e4) back
to counts with that control cell's library size and stochastic rounding (unbiased in the mean);
every other gene keeps the control cell's own counts.
"""
import argparse, json, os, time
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, torch

os.environ.setdefault("SCDFM_SDPA", "1")
from vcc.train import build_model, generate, TestCond, to_counts

LOCO = "/home/zxc29051617/VCC/data/stage3/loco"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", required=True); p.add_argument("--model", required=True)
    p.add_argument("--heldout", required=True); p.add_argument("--name", required=True)
    p.add_argument("--method", default="onestep"); p.add_argument("--steps", type=int, default=10)
    p.add_argument("--n-cells", type=int, default=200); p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--effect-scale", type=float, default=1.0,
                   help="shift each cell by (s-1) x the target's mean predicted effect (log space)")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda")

    vocab = json.load(open(f"{a.fold}/vocab.json"))
    G = [g.strip() for g in open(f"{a.fold}/genes.txt")]
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    cond_mode = ck["cfg"].get("cond", "token")
    vf = build_model(a.fold, vocab, ck["cfg"]["d_model"], cond_mode); vf.load_state_dict(ck["model"]); vf.to(dev).eval()
    data = TestCond(a.fold, vocab, ck["cfg"]["min_cells"], cond_mode)

    D = f"{LOCO}/{a.heldout}"
    R = ad.read_h5ad(f"{D}/real.h5ad")
    targets = json.load(open(f"{D}/real.json"))["targets"]
    gcol = np.array([list(R.var_names).index(g) for g in G])
    C = sp.csr_matrix(R[R.obs.target_gene == "non-targeting"].X)
    lib = np.asarray(C.sum(1)).ravel()
    gene_tok = torch.tensor([vocab[g] for g in G])
    missing = [t for t in targets if t not in vocab and cond_mode == "token"]
    print(f"{len(targets)} targets ({len(missing)} not in vocab -> predicted as unperturbed controls)")

    blocks, labels, t0 = [], [], time.time()
    for k, t in enumerate(targets):
        idx = rng.choice(C.shape[0], a.n_cells, replace=True)
        cells = C[idx].toarray().astype(np.float32)
        if t not in missing:
            src_all = np.log1p(cells[:, gcol] / lib[idx, None] * TARGET_SUM)
            c_full = data.test_cond(t, vocab)
            ys = []
            for lo in range(0, len(G), a.chunk):
                src = torch.from_numpy(src_all[:, lo:lo + a.chunk]).float().to(dev)
                genes = gene_tok[lo:lo + a.chunk].to(dev).unsqueeze(0).repeat(a.n_cells, 1)
                c = (c_full[lo:lo + a.chunk] if cond_mode == "effect" else c_full).unsqueeze(0).repeat(a.n_cells, 1).to(dev)
                ys.append(generate(vf, src, genes, c, a.method, a.steps).cpu().numpy())
            y = np.concatenate(ys, 1)
            if a.effect_scale != 1.0:
                y = y + (a.effect_scale - 1.0) * (y.mean(0) - src_all.mean(0))[None, :]
            cells[:, gcol] = to_counts(y, lib[idx])
        blocks.append(sp.csr_matrix(cells)); labels += [t] * a.n_cells
        if k % 25 == 0:
            print(f"  {k+1}/{len(targets)} targets, {time.time()-t0:.0f}s", flush=True)
    P = ad.AnnData(sp.vstack(blocks + [C]).tocsr(),
                   obs=pd.DataFrame({"target_gene": labels + ["non-targeting"] * C.shape[0]}),
                   var=pd.DataFrame(index=R.var_names))
    P.obs_names = [f"p{i}" for i in range(P.n_obs)]
    P.write_h5ad(f"{D}/pred_scdfm_{a.name}.h5ad")
    print(f"wrote pred_scdfm_{a.name}.h5ad: {P.n_obs} cells in {time.time()-t0:.0f}s; score with: "
          f"python tools/loco.py score --heldout {a.heldout} --method scdfm_{a.name}")


if __name__ == "__main__":
    main()
