"""ResponseNet: a trained neural model of CRISPRi responses in a cell type it has never seen perturbed (2026-10-22).

Training task (episodes), the same situation as the final exam: predict cell type c's response to knocking down gene p
(on the trans genes) from
  (a) p's measured responses in OTHER cell types: a set of 1-13 vectors over 18,533 genes, each with its depth,
      knockdown strength and the source's control profile, and
  (b) c's own control profile (its cell state).
The sources never include c's cell type or its same-screen siblings (bench.FOLDS groups).

Model:
  gene embeddings E_in, E_out (18,533 x d), initialised from the top-d gene programs (SVD) of the training responses
  context encoder z(c) = MLP(PCA_8 of the control profile); the PCA basis comes from control profiles only
  source encoder  [y_s . E_in / sqrt(n_s), z(s), z(c), depth, knockdown] -> MLP -> latent h_s and a weight logit
  target summary  r = sum_s softmax(logit)_s h_s, modulated by the new cell: r * (1 + tanh(W z(c)))
  output          yhat = a * consensus + r . E_out      (consensus = the current model's depth-weighted ridge mean)
Loss: mean squared error over the query's usable trans genes. One model per benchmark fold (bench.py), trained only
on that fold's training contexts.
"""
import os, time
import numpy as np, torch, torch.nn as nn
import bench as B

DEV = "cuda"
D_LAT, K_CTX, LAM, N0 = 64, 8, 3.0, 200.0


class Store:
    """All contexts on the GPU: responses (fp16, NaN -> 0), usable mask, row metadata, target occurrences."""
    def __init__(self, contexts):
        D = B.data(); self.ctx_names = list(contexts); self.cidx = {n: i for i, n in enumerate(self.ctx_names)}
        rows_Y, rows_M, meta = [], [], []
        self.targets = sorted(set().union(*(set(D[n]["perts"]) for n in contexts))); self.tid = {t: i for i, t in enumerate(self.targets)}
        for n in contexts:
            d = D[n]; Y = d["Y"]; M = np.isfinite(Y)
            rows_Y.append(torch.from_numpy(np.where(M, Y, 0).astype(np.float16))); rows_M.append(torch.from_numpy(M))
            for j, p in enumerate(d["perts"]):
                meta.append((self.cidx[n], self.tid[p], d["ncell"][j], d["own_obs"][j]))
        self.Y = torch.cat(rows_Y).to(DEV); self.M = torch.cat(rows_M).to(DEV)
        m = np.array(meta, dtype=np.float64)
        self.row_ctx = torch.tensor(m[:, 0], dtype=torch.long, device=DEV); self.row_tid = m[:, 1].astype(int)
        own = m[:, 3]; ks = np.where(np.isfinite(own) & (own < -0.3), np.clip(B.OWN / np.where(np.isfinite(own), own, -1), 0.5, B.KCAP), 1.0)
        self.row_feat = torch.tensor(np.stack([np.log1p(m[:, 2]) / 6, np.nan_to_num(np.clip(own, -4, 1)), ks, np.isfinite(own)], 1), dtype=torch.float32, device=DEV)
        self.row_w = torch.tensor(m[:, 2] / (m[:, 2] + N0), dtype=torch.float32, device=DEV)
        self.row_ks = torch.tensor(ks, dtype=torch.float32, device=DEV)
        self.occ = {}
        for r, t in enumerate(self.row_tid): self.occ.setdefault(t, []).append(r)
        group = {c: g for g, cs in B.FOLDS.items() for c in cs}
        self.row_group = np.array([list(B.FOLDS).index(group[self.ctx_names[int(c)]]) for c in m[:, 0]])
        G = B.genome(); self.G = len(G.genes)
        ex = [np.concatenate([G.cis(t)[0], [G.idx[t]] if t in G.idx else []]).astype(int) for t in self.targets]
        L = max(1, max(len(e) for e in ex)); E = np.full((len(ex), L), -1, int)
        for i, e in enumerate(ex): E[i, :len(e)] = e
        self.excl = torch.tensor(E, device=DEV)

    def excl_mask(self, tids):
        e = self.excl[tids]; m = torch.zeros(len(tids), self.G + 1, dtype=torch.bool, device=DEV)
        m.scatter_(1, torch.where(e < 0, self.G, e), True); return m[:, :self.G]


_PCA = {}


def ctx_pca(contexts):
    """8-d context coordinates from control profiles only (the 14 sources + challenge contexts A-F as extra points).
    STRICT_PCA=1 (2026-10-24, Codex audit): fit on the given training contexts only, without any held-out or A-F controls."""
    if os.environ.get("STRICT_PCA") == "1": return _ctx_pca_strict(contexts)
    if "f" in _PCA: return _PCA["f"]
    import anndata as ad, scipy.sparse as sp
    D = B.data(); prof = {n: D[n]["ctrl"].astype(np.float64) for n in contexts if not B.is_extra(n)}   # basis from the original contexts only
    extra = []
    for d, cs in (("vcc2026", "ABC"), ("vcc2026_final", "DEF")):
        for c in cs:
            X = sp.csr_matrix(ad.read_h5ad(f"/home/zxc29051617/VCC/data/{d}/context_{c}.h5ad").X); s = np.asarray(X.sum(0)).ravel()
            extra.append(np.log1p(5e4 * s / s.sum()))
    P = np.stack(list(prof.values()) + extra); mu = P.mean(0); U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
    basis = Vt[:K_CTX]; sd = (P - mu) @ basis.T; sd = sd.std(0) + 1e-6
    _PCA["f"] = lambda x: ((x - mu) @ basis.T) / sd
    return _PCA["f"]


def _ctx_pca_strict(contexts):
    key = ("strict",) + tuple(sorted(n for n in contexts if not B.is_extra(n)))
    if key in _PCA: return _PCA[key]
    D = B.data(); P = np.stack([D[n]["ctrl"].astype(np.float64) for n in key[1:]])
    mu = P.mean(0); U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
    basis = Vt[:K_CTX]; sd = ((P - mu) @ basis.T).std(0) + 1e-6
    _PCA[key] = lambda x: ((x - mu) @ basis.T) / sd
    return _PCA[key]


class Net(nn.Module):
    def __init__(self, G, E0):
        super().__init__()
        self.E_in = nn.Parameter(torch.tensor(E0, dtype=torch.float32)); self.E_out = nn.Parameter(torch.tensor(E0, dtype=torch.float32) * 0.1)
        self.ctx = nn.Sequential(nn.Linear(K_CTX, 32), nn.GELU(), nn.Linear(32, D_LAT))
        self.src = nn.Sequential(nn.Linear(3 * D_LAT + 4, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, D_LAT + 1))
        self.mod = nn.Linear(D_LAT, D_LAT); nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.a = nn.Parameter(torch.tensor(0.4))

    def forward(self, Ys, Ms, smask, feats, zs_in, zq_in, cons):
        n = Ms.sum(-1, keepdim=True).clamp(min=1).float()
        proj = torch.einsum("bsg,gd->bsd", Ys * Ms, self.E_in) / n.sqrt()
        zq = self.ctx(zq_in); zs = self.ctx(zs_in)
        h = self.src(torch.cat([proj, zs, zq[:, None].expand_as(zs), feats], -1))
        lat, logit = h[..., :-1], h[..., -1].masked_fill(~smask, -1e4)
        r = (torch.softmax(logit, -1)[..., None] * lat).sum(1) * (1 + torch.tanh(self.mod(zq)))
        return self.a * cons + r @ self.E_out.T


_CUT_NOTED = [False]


def batch(S, zfun_t, q_rows, src_lists, tids, smax=13):
    """At most smax sources per target (the first smax are used; v1-v5 were built this way)."""
    if not _CUT_NOTED[0] and any(len(s) > smax for s in src_lists):
        _CUT_NOTED[0] = True
        print(f"note: some targets have more than {smax} sources; only the first {smax} are used", flush=True)
    Bn = len(q_rows); R = torch.full((Bn, smax), -1, dtype=torch.long, device=DEV)
    for i, s in enumerate(src_lists): R[i, :len(s)] = torch.tensor(s[:smax], device=DEV)
    smask = R >= 0; Rc = R.clamp(min=0)
    ex = S.excl_mask(torch.tensor(tids, device=DEV))
    Ys = S.Y[Rc].float(); Ms = (S.M[Rc] & smask[..., None] & ~ex[:, None]).float()
    w = (S.row_w[Rc] * smask)[..., None]; ks = S.row_ks[Rc][..., None]
    cons = (w * ks * Ys * Ms).sum(1) / ((w * Ms).sum(1) + LAM)
    feats = S.row_feat[Rc] * smask[..., None]
    zs = zfun_t[S.row_ctx[Rc]]
    return Ys, Ms, smask, feats, zs, cons, ex


def train(train_ctx, steps=4000, bs=96, lr=2e-3, seed=0, log=True, nce=0.0, tau=0.1, within=True):
    """nce > 0 adds a contrastive term: within one context's batch, each prediction must be closer (cosine) to its own
    target's measured response than to the other targets' (what the pds metric rewards). within: one context per batch."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    S = Store(train_ctx); zf = ctx_pca(list(B.data()))
    Z = torch.tensor(np.stack([zf(B.data()[n]["ctrl"].astype(np.float64)) for n in S.ctx_names]), dtype=torch.float32, device=DEV)
    # gene programs for initialisation: top-d right singular vectors of a sample of training responses
    samp = torch.tensor(rng.choice(S.Y.shape[0], min(6000, S.Y.shape[0]), replace=False), device=DEV)
    _, _, V = torch.svd_lowrank(S.Y[samp].float(), q=D_LAT, niter=4)
    net = Net(S.G, V.cpu().numpy()).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    # eligible query rows: the target was perturbed in another cell-type group among the training contexts
    elig = {}
    for r in range(len(S.row_tid)):
        o = [s for s in S.occ[S.row_tid[r]] if S.row_group[s] != S.row_group[r]]
        if o: elig.setdefault(int(S.row_ctx[r]), []).append((r, o))
    ctxs = sorted(elig); hist = []
    for step in range(steps):
        qs, srcs, tids = [], [], []
        cs = [rng.choice(ctxs)] * bs if within else rng.choice(ctxs, bs)
        pick = {c: iter(rng.permutation(len(elig[c]))) for c in set(cs)}
        for c in cs:
            k = next(pick[c], None); k = rng.integers(len(elig[c])) if k is None else k
            r, o = elig[c][k]
            keep = [s for s in o if rng.random() < 0.8] or [o[rng.integers(len(o))]]   # source dropout
            qs.append(r); srcs.append(keep); tids.append(S.row_tid[r])
        Ys, Ms, smask, feats, zs, cons, ex = batch(S, Z, qs, srcs, tids)
        q = torch.tensor(qs, device=DEV)
        Yq = S.Y[q].float(); Mq = (S.M[q] & ~ex).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(Ys, Ms, smask, feats, zs, Z[S.row_ctx[q]], cons)
        mse = ((pred.float() - Yq) ** 2 * Mq).sum() / Mq.sum()
        base = ((0.4 * cons - Yq) ** 2 * Mq).sum() / Mq.sum()
        loss = mse / base.detach()
        if nce > 0:
            pn = torch.nn.functional.normalize(pred.float() * Mq, dim=1); yn = torch.nn.functional.normalize(Yq * Mq, dim=1)
            logits = pn @ yn.T / tau
            loss = loss + nce * torch.nn.functional.cross_entropy(logits, torch.arange(len(qs), device=DEV)) / np.log(len(qs))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        hist.append((mse.item(), base.item()))
        if log and (step + 1) % 500 == 0:
            l, b = np.mean(hist[-500:], 0)
            print(f"    step {step + 1}: loss {l:.5f} (0.4 x consensus {b:.5f}), a {net.a.item():.2f}, {time.time() - t0:.0f}s", flush=True)
    net.eval(); return net, S, Z


@torch.no_grad()
def predict(model, held, targets):
    net, S, Z = model; D = B.data()[held]
    out = np.zeros((len(targets), S.G), np.float32)
    zf = ctx_pca(list(B.data())); zq = torch.tensor(zf(D["ctrl"].astype(np.float64)), dtype=torch.float32, device=DEV)
    for lo in range(0, len(targets), 64):
        tl = targets[lo:lo + 64]; src = [S.occ.get(S.tid.get(p, -1), []) for p in tl]
        ok = [i for i, s in enumerate(src) if s]
        if not ok: continue
        tids = [S.tid[tl[i]] for i in ok]
        Ys, Ms, smask, feats, zs, cons, ex = batch(S, Z, [0] * len(ok), [src[i] for i in ok], tids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(Ys, Ms, smask, feats, zs, zq[None].expand(len(ok), -1), cons)
        out[[lo + i for i in ok]] = pred.float().cpu().numpy()
    return out


_CACHE = {}


def method(train_ctx, held, targets, **kw):
    key = tuple(sorted(train_ctx))
    if key not in _CACHE:
        print(f"  training for fold without {sorted(set(B.CONTEXTS) - set(train_ctx))}", flush=True)
        _CACHE.clear(); torch.cuda.empty_cache(); _CACHE[key] = train(list(train_ctx), **kw)
    return predict(_CACHE[key], held, targets)
