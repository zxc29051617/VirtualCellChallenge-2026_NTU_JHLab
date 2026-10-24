"""Sign benchmark for the DE members (2026-10-23): on hidden cell types, how often is the predicted direction right for
the genes a method is most confident about?

    RPE1=1 .venv/bin/python -I repro/model_v2_20261022/bench_sign.py <method> [...]        (from scDFM/)

Same folds, targets and trans-gene masks as bench.py. Truth = the hidden context's full ln ratio L = Y + common mode
(what the scorer's DE compares: perturbed vs control cells), significant where |L| / sqrt(var) > 3.
Per target, genes are ranked by |p| x sqrt(control expression) (what a uniform shift turns into Wilcoxon power);
  precA@k  share of the top k with sign(p) = sign(L) over all usable genes (fid counts matches against every real lfc)
  precR@k  the same within the significant genes only (reach ranks those by our p-values)
averaged over targets, hidden contexts, then folds. Methods: consensus (n0 200), consensus x source agreement (the
cells input), adaptnet (beta 1, seed 0) and variants registered in METHODS.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B

KS = (10, 30, 100, 300, 1000)


def agreement(train, held, targets):
    """Per (target, gene): share of the training sources whose response has the sign of their mean (as cell_effects.py)."""
    D = B.data(); out = np.ones((len(targets), len(D[held]["ctrl"])), np.float32)
    for i, p in enumerate(targets):
        n = 0; s1 = 0; pos = 0
        for t in train:
            j = D[t]["row"].get(p)
            if j is None: continue
            y = D[t]["Y"][j].astype(np.float32); ok = np.isfinite(y); yz = np.where(ok, y, 0)
            n = n + ok; s1 = s1 + yz; pos = pos + (yz > 0)
        if np.isscalar(n): continue
        mean = s1 / np.maximum(n, 1); out[i] = np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))
    return out


def sign_scores(held, targets, P):
    d = B.data()[held]; base = d["measured"] & (np.expm1(d["ctrl"]) >= B.X_MIN)
    rows = [d["row"][p] for p in targets]
    L = d["Y"][rows].astype(np.float64) + (0 if os.environ.get("TRUTH") == "centered" else d["cm"][None].astype(np.float64))
    V = d["var"][rows].astype(np.float64)                      # TRUTH=centered: without the hidden context's common mode
    expr = np.expm1(d["ctrl"].astype(np.float64))
    out = {f"A{k}": [] for k in KS} | {f"R{k}": [] for k in KS}
    for i, p in enumerate(targets):
        m = B.trans_mask(p, base) & np.isfinite(L[i]) & np.isfinite(V[i]) & (L[i] != 0)
        idx = np.flatnonzero(m); pr = np.nan_to_num(P[i][idx].astype(np.float64)); l = L[i][idx]
        sig = np.abs(l) / np.sqrt(np.maximum(V[i][idx], 1e-12)) > 3
        score = np.abs(pr) * np.sqrt(expr[idx]); order = np.argsort(-score)
        good = np.sign(pr) == np.sign(l)
        for k in KS:
            top = order[:k]; out[f"A{k}"].append(good[top].mean())
            o_r = order[sig[order]][:k]
            if len(o_r) >= min(k, 5): out[f"R{k}"].append(good[o_r].mean())
    return {k: float(np.mean(v)) if v else np.nan for k, v in out.items()}


def run(name, fn):
    t0 = time.time(); res = {}
    for fold, held_list in B.FOLDS.items():
        if fold in B.SKIP_EVAL: continue
        train = [c for c in B.CONTEXTS if c not in held_list]
        for held in held_list:
            targets = B.eval_targets(held, train)
            if len(targets) < 20: continue
            res.setdefault(fold, {})[held] = sign_scores(held, targets, fn(train, held, targets))
    fm = {f: {k: float(np.nanmean([r[k] for r in v.values()])) for k in next(iter(v.values()))} for f, v in res.items()}
    ov = {k: float(np.nanmean([x[k] for x in fm.values()])) for k in next(iter(fm.values()))}
    os.makedirs(f"{B.HERE}/results", exist_ok=True)
    tag = "_centered" if os.environ.get("TRUTH") == "centered" else ""
    json.dump(dict(method=name, overall=ov, folds=fm, contexts=res), open(f"{B.HERE}/results/sign_{name}{tag}.json", "w"), indent=1)
    print(f"{name:22s} all genes " + " ".join(f"@{k} {ov[f'A{k}']:.3f}" for k in KS) + " | significant " +
          " ".join(f"@{k} {ov[f'R{k}']:.3f}" for k in KS) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return ov


_M = {}


def adaptnet(train, held, targets, **kw):
    import adaptnet as AN
    key = (tuple(sorted(train)), tuple(sorted(kw.items())))
    if key not in _M:
        _M.clear(); torch.cuda.empty_cache()
        _M[key] = AN.train(list(train), steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0, **kw)
    return AN.predict(_M[key], held, targets)


METHODS = {
    "consensus": lambda tr, h, t: B.consensus(tr, h, t, n0=200.0),
    "consensus_agree": lambda tr, h, t: B.consensus(tr, h, t, n0=200.0) * agreement(tr, h, t) / 0.75,
    "adaptnet": adaptnet,
    "adaptnet_agree": lambda tr, h, t: adaptnet(tr, h, t) * agreement(tr, h, t) / 0.75,
    "adaptnet_sign1": lambda tr, h, t: adaptnet(tr, h, t, sign_w=1.0),
    "adaptnet_sign3": lambda tr, h, t: adaptnet(tr, h, t, sign_w=3.0),
}

if __name__ == "__main__":
    for m in sys.argv[1:]:
        run(m, METHODS[m])
