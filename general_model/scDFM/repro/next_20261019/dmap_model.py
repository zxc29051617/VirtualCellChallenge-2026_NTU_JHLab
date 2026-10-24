"""DepMap-informed response model (after the decomposition preprint, bioRxiv 2026.07.24.740459): a regression from a
perturbed gene's DepMap CRISPR gene-effect profile (PCA of 1,178+ cell lines) to its response, trained on every
perturbation of the training contexts; compared on the hold-outs with the model's measured-response transfer.

    python dmap_model.py [set ...]          (scDFM venv)

Descriptor: DepMap 24Q4 CRISPRGeneEffect (CC BY 4.0), genes x cell lines, NaN -> gene mean, PCA over genes (fitted on
all DepMap genes: no response data used). Target: cache_all responses on 5,000 response genes (common-mode removed,
own/cis masked -> 0). Each training context gets total weight 1. Ridge alpha chosen by inner leave-one-group-out CV
inside the training contexts (mean cosine); the outer hold-out never enters fitting or selection.
Screen on the hold-out: per-target cosine with the real bulk delta (5,000 genes, panel targets and own/cis removed)
for transfer, DepMap model and norm-matched blends; pds-like rank score of the full prediction (own + cis + trans, V6
shared-component removal applied as in the submission).
"""
import json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"
sys.path.insert(0, HERE); sys.path.insert(0, f"{ROOT}/scDFM"); sys.path.insert(0, f"{ROOT}/scDFM/repro/signature_20261013")
import sigexp as S, coexp_screen as CS, render_eval as E
from vcc.gm import Genome
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
NPC, ALPHAS = 50, [1.0, 10.0, 100.0, 1000.0]

# DepMap descriptors
DM = pd.read_csv(f"{ROOT}/data/external/depmap/CRISPRGeneEffect.csv", index_col=0)
DM.columns = [c.split(" (")[0] for c in DM.columns]; DM = DM.loc[:, ~DM.columns.duplicated()]
M = DM.values.T.astype(np.float64)                                    # genes x cell lines
mu = np.nanmean(M, 1, keepdims=True); M = np.where(np.isnan(M), mu, M); M = M - M.mean(0, keepdims=True)
U, sv, Vt = np.linalg.svd(M, full_matrices=False)          # rows = genes; columns centred (keeps mean essentiality)
Xall = U[:, :NPC] * sv[:NPC]; Xall = Xall / Xall.std(0, keepdims=True)
DX = {g: Xall[i] for i, g in enumerate(DM.columns)}
log(f"DepMap: {M.shape[0]} genes x {M.shape[1]} cell lines -> {NPC} PCs")

genes5k = np.load(f"{HERE}/cache_all/genes.npy")
CA = {c: np.load(f"{HERE}/cache_all/{c}.npz", allow_pickle=True) for c in S.CONTEXTS}


def table(ctxs):
    X, Y, W, grp = [], [], [], []
    for c in ctxs:
        z = CA[c]; keep = [i for i, p in enumerate(z["perts"]) if p in DX]
        if not keep: continue
        X.append(np.stack([DX[z["perts"][i]] for i in keep])); Y.append(np.nan_to_num(z["Y"][keep].astype(np.float32)))
        W.append(np.full(len(keep), 1.0 / len(keep))); grp += [S.GROUP_OF[c]] * len(keep)
    return np.vstack(X), np.vstack(Y), np.concatenate(W), np.array(grp)


def ridge(X, Y, w, alpha):
    Xa = np.hstack([X, np.ones((len(X), 1))]); A = Xa.T @ (Xa * w[:, None]) + alpha * np.diag([1.0] * X.shape[1] + [0.0])
    return np.linalg.solve(A, Xa.T @ (Y * w[:, None]))


def pred(B, X):
    if isinstance(B, tuple): return mlp_pred(B, X)
    return np.hstack([X, np.ones((len(X), 1))]) @ B


def mlp_fit(X, Y, w, epochs, seed=0):
    import torch
    torch.manual_seed(seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, dtype=torch.float32, device=dev); Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    wt = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    steps = epochs * ((len(X) + 255) // 256); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g).to(dev)
        for b in range(0, len(X), 256):
            idx = perm[b:b + 256]; opt.zero_grad()
            loss = (wt[idx, None] * (net(Xt[idx]) - Yt[idx]) ** 2).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    return (net.cpu().eval(),)


def mlp_pred(B, X):
    import torch
    with torch.no_grad(): return B[0](torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def cosrows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


res = {}
for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
    t0 = time.time(); o = S.OUTER[name]; train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    X, Y, w, grp = table(train)
    cv = {}
    MODEL = os.environ.get("DMAP_MODEL", "ridge")
    grid = ALPHAS if MODEL == "ridge" else [10, 30]
    for a in grid:                                     # inner leave-one-group-out
        sc = []
        for g in np.unique(grp):
            tr, te = grp != g, grp == g
            B = ridge(X[tr], Y[tr], w[tr] / w[tr].sum(), a) if MODEL == "ridge" else mlp_fit(X[tr], Y[tr], w[tr], a)
            P = pred(B, X[te])
            m = np.linalg.norm(Y[te], axis=1) > 0; sc.append(float(np.average(cosrows(P[m], Y[te][m]), weights=w[te][m])))
        cv[a] = float(np.mean(sc))
    alpha = max(cv, key=cv.get); B = ridge(X, Y, w / w.sum(), alpha) if MODEL == "ridge" else mlp_fit(X, Y, w, alpha)
    log(f"[{name}] train {len(X)} perturbations from {len(train)} contexts; inner CV cosine by alpha {cv}; chosen {alpha}")
    targets, D, rate, mu_c = CS.real_deltas(name); _, oc, full, shuf = E.components(name)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True
    gen = Genome(S.GENES)
    trans = (full - oc).astype(np.float64)                            # model's trans part, log2, x3
    dm = np.zeros_like(trans); has = np.zeros(len(targets), bool)
    for i, t in enumerate(targets):
        if t in DX: dm[i, genes5k] = pred(B, DX[t][None])[0] * 3 / np.log(2); has[i] = True
        g, _ = gen.cis(t); dm[i, g] = 0
        if t in S.GI: dm[i, S.GI[t]] = 0
    m5 = np.zeros(S.G, bool); m5[genes5k] = True; m5 &= ~panel & (rate > 0)
    masks = [m5.copy() for _ in targets]
    for i, t in enumerate(targets):
        g, _ = gen.cis(t); masks[i][g] = False
        if t in S.GI: masks[i][S.GI[t]] = False
    def med_cos(Pm):
        return float(np.median([np.dot(Pm[i][masks[i]], D[i][masks[i]]) / (np.linalg.norm(Pm[i][masks[i]]) * np.linalg.norm(D[i][masks[i]]) + 1e-12) for i in range(len(targets)) if has[i]]))
    def norm_match(a, b):                              # scale a's rows (on the 5k genes) to b's norms
        na = np.linalg.norm(a[:, genes5k], axis=1, keepdims=True); nb = np.linalg.norm(b[:, genes5k], axis=1, keepdims=True)
        return a * np.where(na > 0, nb / np.maximum(na, 1e-12), 0)
    dmn = norm_match(dm, trans); dmn[~has] = trans[~has]       # no DepMap profile -> keep the transfer
    row = {"n_targets": int(len(targets)), "with_depmap": int(has.sum()), "alpha": alpha,
           "cos_transfer": round(med_cos(trans), 4), "cos_depmap": round(med_cos(dm), 4)}
    for lam in [0.25, 0.5, 0.75]:
        bl = trans.copy(); bl[:, genes5k] = (1 - lam) * trans[:, genes5k] + lam * dmn[:, genes5k]
        row[f"cos_blend{lam}"] = round(med_cos(bl), 4)
    mpds = ~panel & (rate > 0)
    def pds_full(tr_part):
        lfc = oc + E.remove_shared(tr_part); return round(CS.pds_like(CS.pred_deltas(lfc, rate, mu_c), D, mpds), 4)
    row["pds_V6_transfer"] = pds_full(trans)
    row["pds_V6_depmap_only"] = pds_full(dmn)
    for lam in [0.25, 0.5]:
        bl = trans.copy(); bl[:, genes5k] = (1 - lam) * trans[:, genes5k] + lam * dmn[:, genes5k]
        row[f"pds_V6_blend{lam}"] = pds_full(bl)
    res[name] = row; log(f"[{name}] {json.dumps(row)} ({time.time() - t0:.0f}s)")
json.dump(res, open(f"{HERE}/dmap_model_{os.environ.get('DMAP_MODEL', 'ridge')}.json", "w"), indent=1)
