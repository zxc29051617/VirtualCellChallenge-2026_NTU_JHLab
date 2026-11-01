"""Hidden-cell-type benchmark in the scorer's space (2026-10-23): the same folds and targets as bench.py, but every gene
weighted by (x / (1 + x))^2 (x = hidden context's control CP50k), since the scorer compares log1p(CP50k) pseudobulks
where a ln-ratio error moves the value by ~x/(1+x). Reports weighted dir / pds / skill and the unweighted ones.

    RPE1=1 .venv/bin/python -I model/bench_w.py <method> [...]        (from pertrelay/)
methods: adaptnet (ew 0), adaptnet_ew1, adaptnet_ew05 (seed 0, beta 1)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B

_score0 = B.score


def score_w(held, targets, P):
    d = B.data()[held]; x = np.expm1(d["ctrl"].astype(np.float64)); w = (x / (1 + x)) ** 2
    out = _score0(held, targets, P)
    dd = dict(d); dd["Y"] = (d["Y"].astype(np.float64) * np.sqrt(w)[None]).astype(np.float32); dd["var"] = (d["var"].astype(np.float64) * w[None]).astype(np.float32)
    B._D[held] = dd
    try:
        ow = _score0(held, targets, (np.nan_to_num(P.astype(np.float64)) * np.sqrt(w)[None]).astype(np.float32))
    finally:
        B._D[held] = d
    out.update({f"w_{k}": v for k, v in ow.items() if k in ("dir", "pds", "skill")})
    return out


_M = {}


def adaptnet(train, held, targets, ew=0.0):
    import adaptnet as AN
    key = (tuple(sorted(train)), ew)
    if key not in _M:
        _M.clear(); torch.cuda.empty_cache()
        _M[key] = AN.train(list(train), steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0, ew=ew)
    return AN.predict(_M[key], held, targets)


def run(name, fn):
    res = {}
    for fold, held_list in B.FOLDS.items():
        if fold in B.SKIP_EVAL: continue
        train = [c for c in B.CONTEXTS if c not in held_list]
        for held in held_list:
            targets = B.eval_targets(held, train)
            if len(targets) < 20: continue
            res.setdefault(fold, {})[held] = score_w(held, targets, fn(train, held, targets))
    keys = ("dir", "pds", "skill", "w_dir", "w_pds", "w_skill")
    fm = {f: {k: float(np.mean([r[k] for r in v.values()])) for k in keys} for f, v in res.items()}
    ov = {k: float(np.mean([x[k] for x in fm.values()])) for k in keys}
    print(f"{name:16s} weighted dir {ov['w_dir']:+.4f} pds {ov['w_pds']:.4f} skill {ov['w_skill']:+.4f} | unweighted dir {ov['dir']:+.4f} pds {ov['pds']:.4f} skill {ov['skill']:+.4f}", flush=True)
    for f, x in fm.items():
        print(f"    {f:7s} w_dir {x['w_dir']:+.4f} w_pds {x['w_pds']:.4f} | dir {x['dir']:+.4f} pds {x['pds']:.4f}", flush=True)


METHODS = {"adaptnet": lambda tr, h, t: adaptnet(tr, h, t, 0.0), "adaptnet_ew1": lambda tr, h, t: adaptnet(tr, h, t, 1.0),
           "adaptnet_ew05": lambda tr, h, t: adaptnet(tr, h, t, 0.5),
           "consensus": lambda tr, h, t: B.consensus(tr, h, t, n0=200.0)}
if __name__ == "__main__":
    for m in sys.argv[1:]:
        run(m, METHODS[m])
