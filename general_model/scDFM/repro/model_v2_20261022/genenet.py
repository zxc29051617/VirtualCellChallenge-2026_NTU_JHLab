"""GeneNet: one trained network for any gene in any cell type (2026-10-23).

For a query (cell type c, gene p) it combines three paths, with learned gates:
  self      p's own measured responses in OTHER cell types (AdaptNet, full detail): empty for a never-perturbed gene
  neighbour learned attention over the 32 genes most similar to p in STRING + GO; each neighbour contributes its
            consensus response over the available cell types (c's group excluded)
  feature   p's gene description (response-program loadings, control co-expression, STRING, GO) -> program
            coefficients, scaled per program by c's cell state
Episodes: one training context per batch; c's same-screen group is never a source; with probability 0.5 the query
gene's own measurements are dropped, so the gates learn the never-perturbed case. Loss: MSE (relative to the query's
signal) + within-batch contrastive term. A Store built without some genes / cell types gives the leave-genes-out and
leave-cell-type-out versions (bench_genes.py S2, bench.py).
"""
import os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import bench as B, responsenet as RN, adaptnet as AN, bench_genes as BG

DEV = "cuda"; K_NB, P_DIM = 32, 32


class Store2(RN.Store):
    """RN.Store restricted to some rows, plus per-group sums for neighbour consensus without the query's group."""
    def __init__(self, contexts, drop_targets=()):
        super().__init__(contexts)
        drop = set(drop_targets)
        if drop:                                    # remove all rows of the held-out genes
            keep = np.array([self.targets[t] not in drop for t in self.row_tid])
            idx = torch.tensor(np.flatnonzero(keep), device=DEV)
            for a in ("Y", "M", "row_ctx", "row_feat", "row_w", "row_ks"): setattr(self, a, getattr(self, a)[idx])
            self.row_tid = self.row_tid[keep]; self.row_group = self.row_group[keep]
            self.occ = {}
            for r, t in enumerate(self.row_tid): self.occ.setdefault(t, []).append(r)
        self.pool = sorted(self.occ); self.pidx = {t: i for i, t in enumerate(self.pool)}
        # group sums stored only for the (gene, group) pairs that exist: total sum per pool gene, plus per group the
        # rows of the genes that group measured (most genes are measured in few groups)
        ng = len(B.FOLDS); Pn = len(self.pool)
        rt = torch.tensor([self.pidx[t] for t in self.row_tid], device=DEV); rg = torch.tensor(self.row_group, device=DEV)
        self.tsum = torch.zeros(Pn, self.G, dtype=torch.float16, device=DEV); self.tcnt = torch.zeros(Pn, device=DEV)
        self.gpos, self.gsum, self.gcnt = [], [], []
        for g in range(ng):
            sel = torch.nonzero(rg == g).squeeze(1)
            present = torch.unique(rt[sel]) if len(sel) else torch.zeros(0, dtype=torch.long, device=DEV)
            pos = torch.full((Pn,), -1, dtype=torch.long, device=DEV); pos[present] = torch.arange(len(present), device=DEV)
            gs = torch.zeros(len(present), self.G, dtype=torch.float16, device=DEV); gc = torch.zeros(len(present), device=DEV)
            for lo in range(0, len(sel), 2000):
                sl = sel[lo:lo + 2000]; li = pos[rt[sl]]
                acc = torch.zeros(len(present), self.G, device=DEV)
                acc.index_add_(0, li, (self.Y[sl].float() * self.M[sl]) * self.row_ks[sl, None])
                gs += acc.half(); gc.index_add_(0, li, torch.ones(len(sl), device=DEV)); del acc
            self.gpos.append(pos); self.gsum.append(gs); self.gcnt.append(gc)
            if len(present):
                self.tsum.index_add_(0, present, gs); self.tcnt.index_add_(0, present, gc)

    def neighbour_consensus(self, pool_idx, group):
        """pool_idx (B x K) consensus responses over the cell types outside the (single) query group, and their counts."""
        g = int(group[0]); assert all(int(x) == g for x in group), "one query group per call"
        s = self.tsum[pool_idx].float(); n = self.tcnt[pool_idx].clone()
        pos = self.gpos[g][pool_idx]; has = pos >= 0
        if has.any():
            s[has] -= self.gsum[g][pos[has]].float(); n[has] -= self.gcnt[g][pos[has]]
        return s / (n[..., None] + 1.0), n


def gene_tables(train_rows_for_resp):
    """Gene features (genes x d, standardised), unit-normalised STRING+GO vectors, and the four similarity bases."""
    V, Fz = BG.gene_features(train_rows_for_resp)
    K = BG.knowledge(); combo = np.hstack([K["string"], K["go"]]) / np.sqrt(2)
    Vs = BG._RP[next(iter(BG._RP))]; R = Vs[0] * Vs[1]; R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    return V, Fz, combo, K["string"], K["go"], R, BG.coexp_embedding()


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class GeneNet(nn.Module):
    def __init__(self, V, f_dim):
        super().__init__()
        self.self_net = AN.AdaptNet(V, beta_fix=1.0)
        self.register_buffer("V", torch.tensor(V, dtype=torch.float32)); k = V.shape[1]
        self.proj = nn.Linear(f_dim, P_DIM); self.att = mlp(6 + P_DIM, 1)
        self.feat = mlp(f_dim, k, h=256); self.gfeat = mlp(RN.K_CTX, k)
        nn.init.zeros_(self.gfeat[-1].weight); nn.init.zeros_(self.gfeat[-1].bias)
        self.register_buffer("unmeasured_scale", torch.tensor(1.0))     # set by calibration on held-out genes
        self.gate = mlp(6, 3, h=32); self.gate0 = mlp(6, 2, h=32)   # gates when the gene has / has no measurements
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate0[-1].weight)
        with torch.no_grad():                                          # start: measured genes follow their own data
            self.gate[-1].bias.copy_(torch.tensor([np.log(np.expm1(1.0)), np.log(np.expm1(0.05)), np.log(np.expm1(0.05))]))
            self.gate0[-1].bias.copy_(torch.tensor([np.log(np.expm1(0.5)), np.log(np.expm1(0.5))]))

    def forward(self, sb, nbY, nbf, Fp, Fq, zq, has_self):
        y_self = self.self_net(*sb)[0] if sb is not None else torch.zeros(len(Fp), self.V.shape[0], device=DEV)
        y_self = y_self * has_self[:, None]
        pf = self.proj(Fp)[:, None] * self.proj(Fq)                                       # B K P
        a = torch.softmax(self.att(torch.cat([nbf, pf], -1)).squeeze(-1), -1)             # B K
        y_nb = (a[..., None] * nbY).sum(1)
        y_feat = (self.feat(Fp) * (1 + 0.5 * torch.tanh(self.gfeat(zq)))) @ self.V.T
        t = torch.stack([has_self, nbf[..., 0].max(1).values, nbf[..., 0].mean(1), torch.log1p(y_nb.norm(dim=1)),
                         torch.log1p(y_feat.norm(dim=1)), torch.log1p(y_self.norm(dim=1))], 1)
        g1 = F.softplus(self.gate(t)); g0 = F.softplus(self.gate0(t))
        hs = has_self[:, None]
        # measured genes: own data + neighbours only (the feature path can memorise training genes, so it serves the
        # never-perturbed case alone)
        pred = hs * y_self + (1 - hs) * (g0[:, :1] * y_nb + g0[:, 1:2] * y_feat) * self.unmeasured_scale
        g = torch.cat([g1 * hs + 0 * g1, g0 * (1 - hs)], 1)
        return pred, g


class Tables:
    def __init__(self, S, train_rows):
        V, Fz, combo, Es, Eg, Er, Ec = gene_tables(train_rows)
        self.V = V; gi = B.genome().idx; self.gi = gi
        self.F = torch.tensor(Fz, dtype=torch.float32, device=DEV)
        pool_g = np.array([gi.get(S.targets[t], -1) for t in S.pool]); ok = pool_g >= 0          # pool holds target ids
        self.pool_ok = torch.tensor(np.flatnonzero(ok), device=DEV); self.pool_gene = torch.tensor(pool_g, device=DEV)
        Cp = torch.tensor(combo[pool_g[ok]], dtype=torch.float32, device=DEV)
        self.combo = torch.tensor(combo, dtype=torch.float32, device=DEV); self.bases = [torch.tensor(x, dtype=torch.float32, device=DEV) for x in (Es, Eg, Er, Ec)]
        self.Cp = Cp

    def neighbours(self, genes, exclude_pool):
        """top-K pool genes by STRING+GO similarity for gene indices `genes`; exclude_pool: pool index of the gene itself (or -1)."""
        sim = self.combo[genes] @ self.Cp.T                                                # B x pool_ok
        if exclude_pool is not None:
            ex = torch.tensor(exclude_pool, device=DEV); m = (self.pool_ok[None] == ex[:, None]); sim = sim.masked_fill(m, -1e4)
        top = sim.topk(K_NB, dim=1)
        pidx = self.pool_ok[top.indices]; qg = self.pool_gene[pidx]
        feats = [top.values] + [(b[genes][:, None] * b[qg]).sum(-1) for b in self.bases]   # combo, string, go, resp, coexp sims
        return pidx, qg, torch.stack(feats, -1)


def make_batch(S, T, Z, genes, groups, self_lists, zq, tids=None):
    G = B.genome()
    pidx, qg, sims = T.neighbours(torch.tensor(genes, device=DEV), [S.pidx.get(S.tid.get(G.genes[g], -1), -1) for g in genes])
    nbY, n = S.neighbour_consensus(pidx, groups)
    nbf = torch.cat([sims, torch.log1p(n)[..., None]], -1)                               # B K 6: 5 similarities + log contexts
    has_self = torch.tensor([1.0 if s else 0.0 for s in self_lists], device=DEV)
    sb = None
    if any(self_lists):
        filled = [s if s else [0] for s in self_lists]
        tids = tids if tids is not None else [S.tid.get(B.genome().genes[g], 0) for g in genes]
        Ys, Ms, smask, feats, zs, cons, ex = RN.batch(S, Z, [0] * len(genes), filled, tids)
        R = torch.full((len(genes), 13), -1, dtype=torch.long, device=DEV)
        for i, s in enumerate(filled): R[i, :len(s)] = torch.tensor(s[:13], device=DEV)
        ks = S.row_ks[R.clamp(min=0)] * (R >= 0)
        sb = (Ys, Ms, smask, feats, zs, zq, ks)
    return sb, nbY, nbf, T.F[torch.tensor(genes, device=DEV)], T.F[qg], has_self


def train(train_ctx, drop_targets=(), calibrate=True, seed=0, log=True, **kw):
    """Train; with calibrate, first train once more with 20 % of the training genes held out, predict those genes in the
    training contexts (as never-perturbed, their cell-type group excluded as always) and set the never-perturbed
    branch's size to the least-squares factor on them (a model that memorised training genes is otherwise overconfident)."""
    scale = 1.0
    if calibrate:
        D = B.data(); drop = set(drop_targets); rng = np.random.default_rng(seed + 100)
        genes = sorted(set().union(*(set(D[c]["perts"]) for c in train_ctx)) - drop)
        inner = set(rng.choice(genes, len(genes) // 5, replace=False).tolist())
        clean = os.environ.get("CLEAN_CENTRING") == "1"           # 2026-10-24: inner held-out genes stay out of the centring
        rows_in = {c: np.array([j for j, p in enumerate(D[c]["perts"]) if p not in drop and p not in inner]) for c in train_ctx}
        cc = B.centred_on(rows_in) if clean else None
        if cc: cc.__enter__()
        m_in = _train(train_ctx, drop | inner, seed=seed, log=False, train_self=False, **kw)
        num = den = 0.0
        for c in train_ctx:
            tg = [p for j, p in enumerate(D[c]["perts"]) if p in inner and D[c]["ncell"][j] >= 50][:150]
            if len(tg) < 10: continue
            P = predict(m_in, c, tg, use_self=False).astype(np.float64)
            base = D[c]["measured"] & (np.expm1(D[c]["ctrl"]) >= B.X_MIN)
            for i, p in enumerate(tg):
                y = D[c]["Y"][D[c]["row"][p]].astype(np.float64); m = B.trans_mask(p, base) & np.isfinite(y)
                num += float((P[i][m] * y[m]).sum()); den += float((P[i][m] ** 2).sum())
        scale = num / max(den, 1e-12)
        if cc: cc.__exit__(None, None, None)
        del m_in; torch.cuda.empty_cache()
        if log: print(f"    calibration on {len(inner)} held-out training genes: never-perturbed size x{scale:.3f}", flush=True)
    model = _train(train_ctx, drop_targets, seed=seed, log=log, **kw)
    model[0].unmeasured_scale.fill_(max(scale, 0.0))
    return model


def _train(train_ctx, drop_targets=(), steps=1500, bs=64, lr=2e-3, seed=0, nce=1.0, tau=0.1, p_drop_self=1.0, log=True, train_self=True):
    """Measured genes: an AdaptNet (beta 1) trained first on the same data and then frozen (the best model for genes with
    measurements). The rest of the network learns the never-perturbed branch (p_drop_self 1: every episode hides the
    query gene's own data). train_self False skips the AdaptNet (calibration models never use it)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    S = Store2(train_ctx, drop_targets); zf = RN.ctx_pca(list(B.data()))
    Z = torch.tensor(np.stack([zf(B.data()[n]["ctrl"].astype(np.float64)) for n in S.ctx_names]), dtype=torch.float32, device=DEV)
    D = B.data(); drop = set(drop_targets)
    train_rows = {c: np.array([j for j, p in enumerate(D[c]["perts"]) if p not in drop]) for c in train_ctx}
    T = Tables(S, train_rows); gi = T.gi
    net = GeneNet(T.V, T.F.shape[1]).to(DEV)
    if train_self:
        an = AN.train(list(train_ctx), steps=1500, k=200, nce=1.0, seed=seed, log=False, beta_fix=1.0, store=S)
        net.self_net = an[0]
        if log: print(f"    measured-gene expert (AdaptNet) trained, {time.time() - t0:.0f}s", flush=True)
    for p_ in net.self_net.parameters(): p_.requires_grad_(False)
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    byctx = {}
    for r in range(len(S.row_tid)):
        t = S.targets[S.row_tid[r]]
        if t in gi: byctx.setdefault(int(S.row_ctx[r]), []).append(r)
    ctxs = sorted(byctx); hist = []
    for step in range(steps):
        c = rng.choice(ctxs); rows = [byctx[c][i] for i in rng.permutation(len(byctx[c]))[:bs]]
        grp = [int(S.row_group[r]) for r in rows]; genes = [gi[S.targets[S.row_tid[r]]] for r in rows]
        selfs = []
        for r in rows:
            o = [s for s in S.occ[S.row_tid[r]] if S.row_group[s] != S.row_group[r]]
            o = [s for s in o if rng.random() < 0.8]
            selfs.append(o if (o and rng.random() > p_drop_self) else [])
        zq = Z[c][None].expand(len(rows), -1)
        sb, nbY, nbf, Fp, Fq, hs = make_batch(S, T, Z, genes, grp, selfs, zq)
        q = torch.tensor(rows, device=DEV); Yq = S.Y[q].float()
        ex = S.excl_mask(torch.tensor([S.row_tid[r] for r in rows], device=DEV)); Mq = (S.M[q] & ~ex).float()
        pred, g = net(sb, nbY, nbf, Fp, Fq, zq, hs)
        mse = ((pred - Yq) ** 2 * Mq).sum() / Mq.sum(); sig = (Yq ** 2 * Mq).sum() / Mq.sum()
        loss = mse / sig.detach()
        if nce > 0:
            logits = F.normalize(pred * Mq, dim=1) @ F.normalize(Yq * Mq, dim=1).T / tau
            loss = loss + nce * F.cross_entropy(logits, torch.arange(len(rows), device=DEV)) / np.log(len(rows))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        hist.append((mse.item() / sig.item(), g.mean(0).tolist()))
        if log and (step + 1) % 500 == 0:
            m = np.mean([h[0] for h in hist[-500:]]); gm = np.mean([h[1] for h in hist[-500:]], 0) * 2      # each half is active ~50 %
            print(f"    step {step + 1}: relative error {m:.4f}, measured gene: self/nb/feat {gm[0]:.2f}/{gm[1]:.2f}/{gm[2]:.2f}; unmeasured: nb/feat {gm[3]:.2f}/{gm[4]:.2f}, {time.time() - t0:.0f}s", flush=True)
    net.eval(); return net, S, Z, T


@torch.no_grad()
def predict(model, held, targets, use_self=True):
    net, S, Z, T = model; D = B.data()[held]; G = B.genome(); out = np.zeros((len(targets), S.G), np.float32)
    zq1 = torch.tensor(RN.ctx_pca(list(B.data()))(D["ctrl"].astype(np.float64)), dtype=torch.float32, device=DEV)
    hgroup = list(B.FOLDS).index(next(g for g, cs in B.FOLDS.items() if held in cs))
    for lo in range(0, len(targets), 32):
        tl = [t for t in targets[lo:lo + 32]]
        selfs_all = [(S.occ.get(S.tid.get(t, -1), []) if use_self else []) for t in tl]
        # genes off the VCC axis have no knowledge description, but with measurements the AdaptNet expert still applies
        ok = [i for i, t in enumerate(tl) if t in G.idx or selfs_all[i]]
        if not ok: continue
        genes = [G.idx.get(tl[i], 0) for i in ok]
        selfs = [selfs_all[i] for i in ok]
        zq = zq1[None].expand(len(ok), -1)
        sb, nbY, nbf, Fp, Fq, hs = make_batch(S, T, Z, genes, [hgroup] * len(ok), selfs, zq, tids=[S.tid.get(tl[i], 0) for i in ok])
        pred, _ = net(sb, nbY, nbf, Fp, Fq, zq, hs)
        out[[lo + i for i in ok]] = pred.cpu().numpy()
    return out
