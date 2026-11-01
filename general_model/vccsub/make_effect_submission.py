"""Build a VCC 2026 prediction h5ad by applying per-target log2 fold changes to each context's own
control cells -- the cell construction scored on held-out H1 by tools/loco.py (transfer / effectfile).

For every context and target, 400 control cells of that context are drawn without replacement and
each gene g with fold change fc_g = 2^lfc_g (lfc clipped to +-4) is changed:
  fc < 1: each count is binomially thinned with p = fc;
  fc > 1: Poisson(mean expression of g in this context's controls, per unit library,
          x the cell's library x (fc - 1)) counts are added.
Targets without an effect stay unperturbed control cells.

    python make_effect_submission.py --data ../data/vcc2026 --effects VCC_mean_x1.5.npz --out out/pred.h5ad
    python make_effect_submission.py ... --contexts A --limit-targets 3      # quick check
"""
import argparse, os, sys, time
import anndata as ad, numpy as np, scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from writer import PredictionWriter, read_gene_names, read_perts  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--effects", required=True, nargs="+",
               help="one npz for all contexts, or CONTEXT=npz pairs (e.g. A=a.npz B=bc.npz C=bc.npz)")
p.add_argument("--contexts", nargs="+", default=["A", "B", "C"])
p.add_argument("--limit-targets", type=int, default=0)
p.add_argument("--clip", type=float, default=4.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--render", choices=["current", "exact", "dual", "dual_lib"], default="current",
               help="cell construction, see pertrelay/vcc/render.py (current = the construction described above)")
p.add_argument("--top-frac", type=float, default=None,
               help="dual render: fraction of a target's cells (largest counts) that receive added counts (default 0.1)")
p.add_argument("--effects-bulk", nargs="+", default=None,
               help="dual only: effects at the bulk amplitude, same layout as --effects")
p.add_argument("--require-all", action="store_true",
               help="stop if any target has no effect or an effect file has non-finite values (default: warn and write"
                    " unperturbed control cells for missing targets)")
a = p.parse_args()
if a.render != "current":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pertrelay"))
    from vcc.render import render  # noqa: E402
    if a.top_frac is not None:
        import vcc.render as _VR; _VR.TOP_FRAC = a.top_frac
if a.render in ("dual", "dual_lib") and not a.effects_bulk:
    p.error("--render dual/dual_lib needs --effects-bulk")

rng = np.random.default_rng(a.seed)
genes = read_gene_names(f"{a.data}/gene_names.csv")
perts = read_perts(f"{a.data}/pert_counts.csv")
if a.limit_targets:
    perts = perts[:a.limit_targets]
def load_effects(path):
    z = np.load(path, allow_pickle=True)
    assert len(z["targets"]) == len(z["lfc"]), f"{path}: {len(z['targets'])} targets but {len(z['lfc'])} rows"
    bad = int((~np.isfinite(z["lfc"])).sum())
    if bad:
        msg = f"{path}: {bad:,} non-finite entries (treated as no change)"
        if a.require_all: raise SystemExit(msg)
        print("WARNING " + msg, flush=True)
    return {t: np.clip(np.nan_to_num(v), -a.clip, a.clip) for t, v in zip(z["targets"], z["lfc"]) if not np.isnan(v).all()}
def load_spec(spec):
    if len(spec) == 1 and "=" not in spec[0]:
        return {c: load_effects(spec[0]) for c in a.contexts}
    return {kv.split("=", 1)[0]: load_effects(kv.split("=", 1)[1]) for kv in spec}
eff = load_spec(a.effects)
bulk = load_spec(a.effects_bulk) if a.render in ("dual", "dual_lib") else None
for c in a.contexts:
    n_eff = sum(t in eff[c] for t in perts)
    print(f"context {c}: {n_eff}/{len(perts)} targets with an effect", flush=True)
    if n_eff < len(perts):
        msg = f"context {c}: {len(perts) - n_eff} targets have no effect and would be written as unperturbed control cells"
        if a.require_all: raise SystemExit(msg)
        print("WARNING " + msg, flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
w = PredictionWriter(a.out, genes, a.contexts, perts, cells_per_pert=400)
t0 = time.time()
for c in a.contexts:
    lfc = eff[c]
    rng = np.random.default_rng([a.seed, ord(c)])      # per-context stream: changing context A leaves B and C's draws unchanged
    ctrl = ad.read_h5ad(f"{a.data}/context_{c}.h5ad")
    assert list(ctrl.var_names) == genes and (ctrl.obs["context"] == c).all()
    C = sp.csr_matrix(ctrl.X); del ctrl
    lib = np.asarray(C.sum(1)).ravel()
    rate = np.asarray(C.sum(0)).ravel() / lib.sum()            # expected count per unit library, per gene
    ref_mean = np.asarray(C.sum(0)).ravel() / C.shape[0]      # mean count per cell over the whole control pool
    for k, t in enumerate(perts):
        idx = rng.choice(C.shape[0], 400, replace=False)
        cells = C[idx].tocsr(copy=True)
        if t in lfc and a.render != "current":
            lb = bulk[c].get(t) if bulk is not None else None
            if a.render in ("dual", "dual_lib") and lb is None:
                raise SystemExit(f"no bulk effect for {c}/{t}")
            cells = render(cells, lib[idx], rate, ref_mean, lfc[t], rng, mode=a.render, lfc_bulk=lb,
                           fix_seed=a.seed * 1_000_003 + ord(c) * 1_000 + k)
        elif t in lfc:
            fc = 2.0 ** lfc[t]
            f = fc[cells.indices]; down = f < 1
            d = cells.data.astype(np.int64)
            d[down] = rng.binomial(d[down], f[down])
            cells.data = d.astype(np.float32)
            up = np.flatnonzero(fc > 1)
            if len(up):
                mu = lib[idx, None] * rate[None, up] * (fc[up] - 1)[None, :]
                add = sp.csr_matrix(rng.poisson(mu).astype(np.float32))
                cells = cells + sp.csr_matrix((add.data, up[add.indices], add.indptr), shape=cells.shape)
        w.add(c, t, cells)
        if k % 50 == 0:
            print(f"  {c}: {k+1}/{len(perts)} targets, nnz so far {w.nnz:,}, {time.time()-t0:.0f}s", flush=True)
nnz = w.close()
n = len(a.contexts) * len(perts) * 400
print(f"wrote {a.out}: {n:,} cells, nnz {nnz:,} ({nnz/n:.0f}/cell), {time.time()-t0:.0f}s")
