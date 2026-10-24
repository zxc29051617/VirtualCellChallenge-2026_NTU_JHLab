"""Leave-genes-out benchmark: can a model predict the knockdown of a gene that was never perturbed? (2026-10-23)

    .venv/bin/python -I repro/model_v2_20261022/bench_genes.py [method ...]        (from scDFM/)

Every perturbed gene goes to one of 5 gene folds (seed 0). For fold f, ALL knockdown data of f's genes are removed from
training; the task is to predict their responses
  S1  in cell types that are otherwise seen (the context's responses to other genes are known), and
  S2  in a hidden cell type as well (bench.FOLDS group hidden too; checked on the H1, XAtlas and iPSC groups).
Gene representations that never use a gene's own knockdown:
  resp   its loadings on the response programs (top-128 SVD of the training responses, its column as a RESPONDER)
  coexp  co-expression in control cells (top-64 SVD of log-normalised control cells of A-F and H1, centred per context)
Scoring as bench.score (trans genes; dir, pds among the fold's genes in that context, skill), up to 300 genes per
context and fold. Results: repro/model_v2_20261022/results/genes_<method>.json
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

HERE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda"; N_FOLD, N_EVAL, K_RESP, K_COEXP = 5, 300, 128, 64
S2_GROUPS = ["H1", "XAtlas", "iPSC"]


def gene_folds():
    allp = sorted(set().union(*(set(B.data()[c]["perts"]) for c in B.CONTEXTS if not B.is_extra(c))))   # same folds with or without extras
    perm = np.random.default_rng(0).permutation(len(allp))
    return {allp[i]: int(k % N_FOLD) for k, i in enumerate(perm)}


_COEXP = {}


def coexp_embedding():
    """genes x 64 co-expression loadings from control cells only (no perturbation data)."""
    if "e" in _COEXP: return _COEXP["e"]
    import anndata as ad, scipy.sparse as sp
    blocks = []; rng = np.random.default_rng(0)
    files = [f"/home/zxc29051617/VCC/data/vcc2026/context_{c}.h5ad" for c in "ABC"] + \
            [f"/home/zxc29051617/VCC/data/vcc2026_final/context_{c}.h5ad" for c in "DEF"] + ["/home/zxc29051617/VCC/data/mock_h1/context_H.h5ad"]
    for f in files:
        X = sp.csr_matrix(ad.read_h5ad(f).X).astype(np.float32); idx = rng.choice(X.shape[0], min(2000, X.shape[0]), replace=False); X = X[idx]
        X = sp.diags(1e4 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X; X.data = np.log1p(X.data)
        D = torch.tensor(X.toarray(), device=DEV); blocks.append(D - D.mean(0, keepdim=True))
    M = torch.cat(blocks); _, S, V = torch.svd_lowrank(M, q=K_COEXP, niter=4)
    E = (V * S).cpu().numpy(); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    _COEXP["e"] = E; return E


def resp_programs(train_rows):
    """genes x K_RESP response programs (right singular vectors) from training rows {context: row indices}."""
    rng = np.random.default_rng(0); X = []
    for c, rows in train_rows.items():
        r = rng.choice(rows, min(1500, len(rows)), replace=False); X.append(np.nan_to_num(B.data()[c]["Y"][r].astype(np.float32)))
    _, S, V = torch.svd_lowrank(torch.tensor(np.vstack(X), device=DEV), q=K_RESP, niter=4)
    return V.cpu().numpy(), S.cpu().numpy()


def setup(fold, hidden_group=None):
    """training rows per context (genes outside the fold; hidden group's contexts dropped) and test genes per context."""
    gf = gene_folds(); D = B.data(); hidden = set(B.FOLDS[hidden_group]) if hidden_group else set()
    train_rows = {c: np.array([j for j, p in enumerate(D[c]["perts"]) if gf.get(p, -1) != fold]) for c in B.CONTEXTS if c not in hidden}   # ELEM-only genes: always training
    test_ctx = list(B.FOLDS[hidden_group]) if hidden_group else [c for c in B.CONTEXTS if not B.is_extra(c)]
    rng = np.random.default_rng(fold); tests = {}
    for c in test_ctx:
        cand = [p for j, p in enumerate(D[c]["perts"]) if gf.get(p, -1) == fold and D[c]["ncell"][j] >= 50]
        if len(cand) >= 20: tests[c] = sorted(rng.choice(cand, min(N_EVAL, len(cand)), replace=False).tolist())
    return train_rows, tests


def gene_index():
    return B.genome().idx


def knn(emb, train_rows, c, targets, k=20, own_context=True):
    """mean response of the k most similar training genes (cosine of emb), from context c (S1) or the training consensus (S2)."""
    D = B.data(); gi = gene_index(); out = np.zeros((len(targets), len(D[c]["ctrl"])), np.float32)
    if own_context:
        pool = [(D[c]["perts"][j], np.nan_to_num(D[c]["Y"][j].astype(np.float32))) for j in train_rows[c]]
    else:                                                     # consensus of each training gene over the training contexts
        acc = {}
        for cc, rows in train_rows.items():
            for j in rows:
                p = D[cc]["perts"][j]; y = np.nan_to_num(D[cc]["Y"][j].astype(np.float32))
                s, n = acc.get(p, (0, 0)); acc[p] = (s + y, n + 1)
        pool = [(p, s / (n + 1.0)) for p, (s, n) in acc.items()]
    pool = [(p, y) for p, y in pool if p in gi]
    if not pool: return out
    Pe = emb[[gi[p] for p, _ in pool]]; Yp = np.stack([y for _, y in pool])
    for i, t in enumerate(targets):
        if t not in gi: continue
        sim = Pe @ emb[gi[t]]; top = np.argsort(-sim)[:k]; w = np.clip(sim[top], 0, None)
        if w.sum() > 0: out[i] = (w[:, None] * Yp[top]).sum(0) / w.sum()
    return out


def run(name, fn, settings=("S1", "S2"), quiet=False):
    t0 = time.time(); res = {}
    for st in settings:
        for fold in range(N_FOLD):
            for hg in ([None] if st == "S1" else S2_GROUPS):
                train_rows, tests = setup(fold, hg)
                for c, tg in tests.items():
                    P = fn(train_rows, c, tg, st)
                    res.setdefault(st, []).append(dict(fold=fold, group=hg, context=c, **B.score(c, tg, P)))
    summ = {st: {k: float(np.mean([r[k] for r in rs])) for k in ("dir", "pds", "skill", "scale")} for st, rs in res.items()}
    json.dump(dict(method=name, summary=summ, rows=res), open(f"{HERE}/results/genes_{name}.json", "w"), indent=1)
    if not quiet:
        for st, sm in summ.items():
            print(f"{name:14s} {st}: dir {sm['dir']:+.4f}  pds {sm['pds']:.4f}  skill {sm['skill']:+.4f}  scale {sm['scale']:.2f}  ({len(res[st])} context-folds, {time.time() - t0:.0f}s)", flush=True)
    return summ


def m_zero(train_rows, c, tg, st): return np.zeros((len(tg), len(B.data()[c]["ctrl"])), np.float32)
def m_knn_coexp(train_rows, c, tg, st): return knn(coexp_embedding(), train_rows, c, tg, own_context=(st == "S1"))
_RP = {}
def m_knn_resp(train_rows, c, tg, st):
    key = tuple(sorted((k, len(v)) for k, v in train_rows.items()))
    if key not in _RP: _RP.clear(); _RP[key] = resp_programs(train_rows)
    V, S = _RP[key]; E = V * S; E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return knn(E, train_rows, c, tg, own_context=(st == "S1"))


_KN = {}


def knowledge():
    """STRING and GO embeddings (gene_knowledge.py), rows unit-normalised (zeros where a gene has none)."""
    if not _KN:
        z = np.load("/home/zxc29051617/VCC/data/stage3/model_v2/gene_knowledge.npz", allow_pickle=True)
        for k in ("string", "go") + (("protein",) if "protein" in z.files else ()):
            E = z[k].astype(np.float32); _KN[k] = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return _KN


def gene_features(train_rows, parts=("resp", "coexp", "string", "go")):
    """Concatenated gene representations, standardised: response-program loadings x singular values, co-expression,
    STRING network, GO. None of them uses the gene's own knockdown."""
    key = tuple(sorted((k, len(v)) for k, v in train_rows.items()))
    if key not in _RP: _RP.clear(); _RP[key] = resp_programs(train_rows)
    V, S = _RP[key]; blocks = {"resp": V * S, "coexp": coexp_embedding(), **knowledge()}
    F = np.hstack([blocks[p] for p in parts])
    return V, (F - F.mean(0)) / (F.std(0) + 1e-9)


def m_ridge(train_rows, c, tg, st, lam=30.0, parts=("resp", "coexp", "string", "go")):
    """Trained linear map from gene features to the response's program coefficients (A = y V): per context (S1) or
    pooled over the training contexts (S2), ridge-regularised."""
    D = B.data(); gi = gene_index(); V, F = gene_features(train_rows, parts)
    ctxs = [c] if st == "S1" else list(train_rows)
    X, A = [], []
    for cc in ctxs:
        for j in train_rows[cc]:
            p = D[cc]["perts"][j]
            if p not in gi: continue
            X.append(F[gi[p]]); A.append(np.nan_to_num(D[cc]["Y"][j].astype(np.float32)) @ V)
    X = np.array(X); A = np.array(A)
    W = np.linalg.solve(X.T @ X + lam * len(X) / 100 * np.eye(X.shape[1]), X.T @ A)
    out = np.zeros((len(tg), len(D[c]["ctrl"])), np.float32)
    for i, t in enumerate(tg):
        if t in gi: out[i] = (F[gi[t]] @ W) @ V.T
    return out


def m_knn_string(train_rows, c, tg, st): return knn(knowledge()["string"], train_rows, c, tg, own_context=(st == "S1"))
def m_knn_go(train_rows, c, tg, st): return knn(knowledge()["go"], train_rows, c, tg, own_context=(st == "S1"))
def m_ridge_old(train_rows, c, tg, st): return m_ridge(train_rows, c, tg, st, parts=("resp", "coexp"))


def m_knn_combo(train_rows, c, tg, st):
    """neighbours by the mean of STRING and GO cosine similarity (concatenated unit vectors / sqrt 2)."""
    K = knowledge(); E = np.hstack([K["string"], K["go"]]) / np.sqrt(2)
    return knn(E, train_rows, c, tg, own_context=(st == "S1"))


def m_blend(train_rows, c, tg, st):
    """half STRING neighbours, half the knowledge ridge, each scaled to the same overall size first."""
    a = m_knn_string(train_rows, c, tg, st).astype(np.float64); b = m_ridge(train_rows, c, tg, st).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return (a + b).astype(np.float32)
    return (0.5 * (a / na + b / nb) * (na + nb) / 2).astype(np.float32)


def m_knn_protein(train_rows, c, tg, st): return knn(knowledge()["protein"], train_rows, c, tg, own_context=(st == "S1"))
def m_knn_combo3(train_rows, c, tg, st):
    K = knowledge(); E = np.hstack([K["string"], K["go"], K["protein"]]) / np.sqrt(3)
    return knn(E, train_rows, c, tg, own_context=(st == "S1"))
def m_ridge_p(train_rows, c, tg, st): return m_ridge(train_rows, c, tg, st, parts=("resp", "coexp", "string", "go", "protein"))


_ADJ = {}


def string_walk():
    """row-normalised STRING adjacency (combined score >= 400, weights score / 1000) on the VCC gene axis."""
    if "P" in _ADJ: return _ADJ["P"]
    import gzip, scipy.sparse as sp
    f = "/home/zxc29051617/VCC/data/stage3/model_v2/string_adj.npz"
    if os.path.exists(f):
        A = sp.load_npz(f)
    else:
        import pandas as pd
        E = "/home/zxc29051617/VCC/data/external/string"; gi = gene_index(); G = len(B.genome().genes)
        info = pd.read_csv(f"{E}/9606.protein.info.v12.0.txt.gz", sep="\t", usecols=[0, 1])
        p2g = {pp: gi[g] for pp, g in zip(info.iloc[:, 0], info.iloc[:, 1]) if g in gi}
        r, c_, v = [], [], []
        with gzip.open(f"{E}/9606.protein.links.v12.0.txt.gz", "rt") as fh:
            next(fh)
            for line in fh:
                x, y, sc = line.split(); sc = int(sc)
                if sc < 400: continue
                i, j = p2g.get(x), p2g.get(y)
                if i is None or j is None or i == j: continue
                r.append(i); c_.append(j); v.append(sc / 1000.0)
        A = sp.csr_matrix((v, (r, c_)), shape=(G, G)); A = A.maximum(A.T).tocsr(); sp.save_npz(f, A)
    d = np.asarray(A.sum(1)).ravel(); P = sp.diags(np.where(d > 0, 1 / np.maximum(d, 1e-12), 0)) @ A
    _ADJ["P"] = P.tocsr(); return _ADJ["P"]


def m_prop(train_rows, c, tg, st, alpha=0.6, hops=3):
    """Diffuse the measured genes' responses (in program coefficients) over the STRING network: for a never-perturbed
    gene, the walk-weighted average of measured genes within `hops` steps (step t weighted alpha^t)."""
    D = B.data(); gi = gene_index(); P = string_walk(); G = P.shape[0]
    key = tuple(sorted((k, len(v)) for k, v in train_rows.items()))
    if key not in _RP: _RP.clear(); _RP[key] = resp_programs(train_rows)
    V, _ = _RP[key]; k = V.shape[1]
    R0 = np.zeros((G, k)); m0 = np.zeros(G)
    ctxs = [c] if st == "S1" else list(train_rows)
    for cc in ctxs:
        for j in train_rows[cc]:
            p = D[cc]["perts"][j]
            if p in gi: R0[gi[p]] += np.nan_to_num(D[cc]["Y"][j].astype(np.float32)) @ V; m0[gi[p]] += 1
    R0[m0 > 0] /= m0[m0 > 0, None]; m0 = (m0 > 0).astype(float)
    H = np.zeros_like(R0); M = np.zeros(G); h, m = R0, m0
    for t in range(1, hops + 1):
        h = P @ h; m = P @ m; H += alpha ** t * h; M += alpha ** t * m
    out = np.zeros((len(tg), len(D[c]["ctrl"])), np.float32)
    for i, t in enumerate(tg):
        if t in gi and M[gi[t]] > 1e-6: out[i] = (H[gi[t]] / M[gi[t]]) @ V.T
    return out


def m_prop9(train_rows, c, tg, st): return m_prop(train_rows, c, tg, st, alpha=0.9, hops=3)


METHODS = {"prop": m_prop, "prop9": m_prop9, "knn_protein": m_knn_protein, "knn_combo3": m_knn_combo3, "ridge_p": m_ridge_p, "knn_combo": m_knn_combo, "blend": m_blend, "zero": m_zero, "knn_coexp": m_knn_coexp, "knn_resp": m_knn_resp, "ridge_old": m_ridge_old, "ridge": m_ridge,
           "knn_string": m_knn_string, "knn_go": m_knn_go}

if __name__ == "__main__":
    for m in sys.argv[1:] or list(METHODS):
        run(m, METHODS[m])
