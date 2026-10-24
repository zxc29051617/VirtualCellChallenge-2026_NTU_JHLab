"""AdaptNet: a small trained network that decides, per target and per new cell type, how to combine other cell types'
measurements (2026-10-22).

Lesson from responsenet.py and the probes: free gene embeddings overfit the 7 training cell types, and projecting onto
shared gene programs raises accuracy (direction, error) but lowers target discrimination, with a fixed trade-off.
AdaptNet learns that trade-off per target instead of fixing it:
  programs V (genes x k): top-k SVD directions of the training responses (fixed, per fold)
  each source s (target p measured in another cell type): C_s = kscale x y_s; program part P_s = C_s V V^T; detail R_s = C_s - P_s
  source weights   w_s = softmax(MLP_w([depth, knockdown, kscale, agreement with the other sources, size, z(s), z(c), z(s) - z(c)]))
  program gains    g = 1 + 0.5 tanh(MLP_g(z(c)))           (the new cell's state strengthens or weakens programs)
  detail weight    beta = sigmoid(MLP_b(target features)),  overall size alpha = softplus(MLP_a(target features))
                   target features: number of sources, mean agreement, mean knockdown, program and detail size
  prediction       yhat = alpha * ((sum_s w_s A_s) * g) V^T + alpha * beta * sum_s w_s R_s   (A_s = program coefficients)
About 8.4k trainable weights (8,363; the program matrix V is a fixed buffer). Episodes and loss as responsenet.py (MSE relative to 0.4 x consensus, plus the within-context
contrastive term).
"""
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import bench as B, responsenet as RN

DEV = "cuda"


def mlp(i, o, h=32):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class AdaptNet(nn.Module):
    def __init__(self, V, beta_fix=None):
        super().__init__(); self.beta_fix = beta_fix                  # beta_fix: keep this share of the detail (no learned shrinkage)
        self.register_buffer("V", torch.tensor(V, dtype=torch.float32)); k = V.shape[1]
        self.w = mlp(6 + 3 * RN.K_CTX, 1); self.g = mlp(RN.K_CTX, k); self.b = mlp(5, 1); self.a = mlp(5, 1)
        for m in (self.g, self.b, self.a): nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        with torch.no_grad(): self.a[-1].bias.fill_(np.log(np.expm1(0.4)))      # alpha starts at 0.4

    def forward(self, Ys, Ms, smask, feats, zs, zq, ks):
        C = ks[..., None] * Ys * Ms                                   # B S G
        A = C @ self.V                                                # B S k
        P = (A @ self.V.T) * Ms; R = C - P
        n = smask.sum(1).clamp(min=1).float()
        mean_other = (C.sum(1, keepdim=True) - C) / (n[:, None, None] - 1).clamp(min=1)
        agree = F.cosine_similarity(C, mean_other, dim=-1) * (n[:, None] > 1)           # B S
        size = torch.log1p(C.norm(dim=-1))
        f = torch.cat([feats, agree[..., None], size[..., None], zs, zq[:, None].expand_as(zs), zs - zq[:, None]], -1)
        w = torch.softmax(self.w(f).squeeze(-1).masked_fill(~smask, -1e4), -1)            # B S
        Ahat = (w[..., None] * A).sum(1)                                                  # B k
        Rw = (w[..., None] * Ms).sum(1)
        Rhat = (w[..., None] * R).sum(1) / Rw.clamp(min=1e-3)                             # B G
        t = torch.stack([torch.log(n), (agree * w).sum(1), (feats[..., 1] * w).sum(1), torch.log1p(Ahat.norm(dim=-1)), torch.log1p(Rhat.norm(dim=-1))], -1)
        g = 1 + 0.5 * torch.tanh(self.g(zq)); alpha = F.softplus(self.a(t))
        beta = torch.full_like(alpha, self.beta_fix) if self.beta_fix is not None else torch.sigmoid(self.b(t))
        return alpha * ((Ahat * g) @ self.V.T + beta * Rhat), beta.squeeze(-1), alpha.squeeze(-1)


def programs(S, k, seed=0):
    rng = np.random.default_rng(seed)
    samp = torch.tensor(rng.choice(S.Y.shape[0], min(12000, S.Y.shape[0]), replace=False), device=DEV)
    _, _, V = torch.svd_lowrank(S.Y[samp].float(), q=k, niter=4); return V.cpu().numpy()


def shared_components(S, k_sh):
    """Per training context: its top-k_sh response directions across all of its perturbations (genes x k_sh), the part
    the pipeline removes from a panel's totals before scoring (shared components carry no target identity)."""
    out = []
    for ci in range(len(S.ctx_names)):
        rows = torch.nonzero(S.row_ctx == ci).squeeze(1)
        X = S.Y[rows].float(); X = X - X.mean(0, keepdim=True)
        _, _, Vc = torch.svd_lowrank(X, q=k_sh, niter=4); out.append(Vc)
    return torch.stack(out)                                            # contexts x G x k_sh


def train(train_ctx, steps=1500, bs=96, lr=3e-3, seed=0, k=200, nce=1.0, tau=0.1, log=True, orth=0, beta_fix=None, store=None,
          sign_w=0.0, sign_tau=10.0, sign_z=3.0, ew=0.0):
    """orth > 0: the loss sees prediction and truth with the query context's top-orth shared components removed, so the
    model spends its capacity on what tells targets apart (what survives the pipeline's shared removal).
    sign_w > 0 (2026-10-23): adds a direction loss on the query's significant entries (|y| / sqrt(var) > sign_z):
    softplus(-sign(y) x sign_tau x prediction), i.e. a logistic loss on getting the sign right where the truth is clear
    (the DE members score signs of significant genes).
    ew > 0 (2026-10-23): weights every gene's error by (x / (1 + x))^(2 ew), x = the query cell type's control expression
    (CP50k): the scorer compares log1p(CP50k) pseudobulks, where a ln-ratio error on gene g moves the value by ~x/(1+x)
    times that error, so barely expressed genes hardly count (ew 1 = exactly that weighting)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    S = store if store is not None else RN.Store(train_ctx); zf = RN.ctx_pca(list(B.data()))   # store: reuse (e.g. genes dropped)
    SH = shared_components(S, orth) if orth > 0 else None
    if ew > 0:                                                       # per-context gene weights from the control profiles
        X = np.expm1(np.stack([B.data()[n]["ctrl"].astype(np.float64) for n in S.ctx_names]))
        EW = torch.tensor((X / (1 + X)) ** (2 * ew), dtype=torch.float32, device=DEV)
    if sign_w > 0 and not hasattr(S, "V"):                         # sampling variances, row order as Store
        S.V = torch.cat([torch.from_numpy(np.nan_to_num(B.data()[n]["var"].astype(np.float16), nan=1e4)) for n in S.ctx_names]).to(DEV)
    Z = torch.tensor(np.stack([zf(B.data()[n]["ctrl"].astype(np.float64)) for n in S.ctx_names]), dtype=torch.float32, device=DEV)
    net = AdaptNet(programs(S, k, seed), beta_fix=beta_fix).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    elig = {}
    for r in range(len(S.row_tid)):
        o = [s for s in S.occ[S.row_tid[r]] if S.row_group[s] != S.row_group[r]]
        if o: elig.setdefault(int(S.row_ctx[r]), []).append((r, o))
    ctxs = sorted(elig); hist = []
    for step in range(steps):
        c = rng.choice(ctxs); idx = rng.permutation(len(elig[c]))[:bs]
        qs = [elig[c][i][0] for i in idx]; srcs = []
        for i in idx:
            o = elig[c][i][1]; srcs.append([s for s in o if rng.random() < 0.8] or [o[rng.integers(len(o))]])
        tids = [S.row_tid[r] for r in qs]
        Ys, Ms, smask, feats, zs, cons, ex = RN.batch(S, Z, qs, srcs, tids)
        R = torch.full((len(qs), 13), -1, dtype=torch.long, device=DEV)
        for i, s in enumerate(srcs): R[i, :len(s)] = torch.tensor(s[:13], device=DEV)
        ks = S.row_ks[R.clamp(min=0)] * (R >= 0)
        q = torch.tensor(qs, device=DEV); Yq = S.Y[q].float(); Mq = (S.M[q] & ~ex).float()
        pred, beta, alpha = net(Ys, Ms, smask, feats, zs, Z[S.row_ctx[q]], ks)
        if SH is not None:                                             # one query context per batch: its shared directions
            Sc = SH[int(S.row_ctx[q[0]])]
            orth_ = lambda v: v - (v @ Sc) @ Sc.T
            pred = orth_(pred * Mq) * Mq; Yq = orth_(Yq * Mq) * Mq; cons = orth_(cons * Mq) * Mq
        if ew > 0: Mq = Mq * EW[S.row_ctx[q]]                          # expression weights enter every loss term
        mse = ((pred - Yq) ** 2 * Mq).sum() / Mq.sum(); base = ((0.4 * cons - Yq) ** 2 * Mq).sum() / Mq.sum()
        loss = mse / base.detach()
        if sign_w > 0:
            sig = (Yq.abs() > sign_z * S.V[q].float().clamp(min=1e-8).sqrt()) & (Mq > 0)
            if sig.any():
                loss = loss + sign_w * F.softplus(-torch.sign(Yq[sig]) * sign_tau * pred[sig]).mean()
        if nce > 0 and len(qs) > 1:
            Mc = Mq.sqrt()                                             # 0/1 masks unchanged; weights -> cosine in the weighted space
            logits = F.normalize(pred * Mc, dim=1) @ F.normalize(Yq * Mc, dim=1).T / tau
            loss = loss + nce * F.cross_entropy(logits, torch.arange(len(qs), device=DEV)) / np.log(len(qs))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        hist.append((mse.item(), base.item(), beta.mean().item(), alpha.mean().item()))
        if log and (step + 1) % 500 == 0:
            m = np.mean(hist[-500:], 0)
            print(f"    step {step + 1}: mse {m[0]:.5f} (0.4 x consensus {m[1]:.5f}), beta {m[2]:.2f}, alpha {m[3]:.2f}, {time.time() - t0:.0f}s", flush=True)
    net.eval(); return net, S, Z


@torch.no_grad()
def predict(model, held, targets):
    return predict_ctrl(model, B.data()[held]["ctrl"], targets)


@torch.no_grad()
def predict_ctrl(model, ctrl, targets):
    """Trans response (ln ratio, targets x genes) for a new cell type given only its control profile log1p(CP50k)."""
    net, S, Z = model; out = np.zeros((len(targets), S.G), np.float32)
    zq = torch.tensor(RN.ctx_pca(list(B.data()))(np.asarray(ctrl, np.float64)), dtype=torch.float32, device=DEV)
    for lo in range(0, len(targets), 48):
        tl = targets[lo:lo + 48]; src = [S.occ.get(S.tid.get(p, -1), []) for p in tl]; ok = [i for i, s in enumerate(src) if s]
        if not ok: continue
        srcs = [src[i] for i in ok]; tids = [S.tid[tl[i]] for i in ok]
        Ys, Ms, smask, feats, zs, cons, ex = RN.batch(S, Z, [0] * len(ok), srcs, tids)
        R = torch.full((len(ok), 13), -1, dtype=torch.long, device=DEV)
        for i, s in enumerate(srcs): R[i, :len(s)] = torch.tensor(s[:13], device=DEV)
        ks = S.row_ks[R.clamp(min=0)] * (R >= 0)
        pred, _, _ = net(Ys, Ms, smask, feats, zs, zq[None].expand(len(ok), -1), ks)
        out[[lo + i for i in ok]] = pred.cpu().numpy()
    return out


_CACHE = {}


def method(train_ctx, held, targets, **kw):
    key = tuple(sorted(train_ctx))
    if key not in _CACHE:
        print(f"  training for fold without {sorted(set(B.CONTEXTS) - set(train_ctx))}", flush=True)
        _CACHE.clear(); torch.cuda.empty_cache(); _CACHE[key] = train(list(train_ctx), **kw)
    return predict(_CACHE[key], held, targets)
