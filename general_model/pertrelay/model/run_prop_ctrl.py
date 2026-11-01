"""Does propagating a prediction through the NEW cell type's own control-cell co-variation help? (2026-10-23)

    RPE1=1 .venv/bin/python -I model/run_prop_ctrl.py [consensus|adaptnet]        (from pertrelay/)

P' = (1 - a) P/|P| + a Q/|Q|, Q = P Lz^T Lz / n per target (Lz: the hidden context's 2,000 control cells, log1p CP10k,
standardised per gene; genes expressed > 0.05 CP10k), rescaled to |P|. Scored on the hidden-cell-type benchmark for the 9
contexts that have saved control cells (experiments/signature_20261013/controls); others keep P.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp, torch
import bench as B
CT = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013/controls"
HAVE = {"H1", "HCT116", "HEK293T", "HepG2", "Jurkat", "K562", "K562ess", "KOLF", "VIPK562"}
_L = {}


def ctrl_mat(h):
    if h not in _L:
        z = np.load(f"{CT}/{h}_counts.npz"); X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).astype(np.float64)
        lib = np.asarray(X.sum(1)).ravel(); N = sp.diags(1e4 / lib) @ X; mu = np.asarray(N.mean(0)).ravel(); ok = np.flatnonzero(mu > 0.05)
        L = N[:, ok].tocsr(); L.data = np.log1p(L.data); L = L.toarray(); L -= L.mean(0); L /= L.std(0) + 1e-9
        _L.clear(); _L[h] = (ok, L.astype(np.float32))
    return _L[h]


_P = {}
def base(train, held, targets):
    key = (tuple(sorted(train)), held)
    if key not in _P:
        if sys.argv[1] == "adaptnet":
            import adaptnet as AN
            tk = ("m", tuple(sorted(train)))
            if tk not in _P:
                for k in [k for k in _P if k[0] == "m"]: del _P[k]
                torch.cuda.empty_cache(); _P[tk] = AN.train(list(train), steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0)
            _P[key] = AN.predict(_P[tk], held, targets).astype(np.float64)
        else:
            _P[key] = B.consensus(train, held, targets, n0=200.0).astype(np.float64)
    return _P[key]


def prop(train, held, targets, a):
    P = base(train, held, targets)
    if held not in HAVE or a == 0: return P.astype(np.float32)
    ok, L = ctrl_mat(held)
    Q = np.zeros_like(P); Q[:, ok] = ((P[:, ok] @ L.T) @ L) / L.shape[0]
    nP = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12; nQ = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    return (((1 - a) * P / nP + a * Q / nQ) * nP).astype(np.float32)


for a in (0.0, 0.2, 0.4):
    B.run(f"propctrl_{sys.argv[1]}_a{a}", lambda tr, h, t, a=a: prop(tr, h, t, a))
