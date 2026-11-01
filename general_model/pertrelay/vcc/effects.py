"""Stage 1 of the two-stage model: predict a perturbation's pseudobulk effect in an unseen line
from its effects in other lines.

Effects are log2((CPM_pert + 1) / (CPM_ctrl + 1)) over the VCC gene axis (NaN where a line did not
measure the gene), the same quantity tools/loco.py's transfer baseline applies to control cells.

Predictors (fit on source lines only; tuned leave-one-line-out among the training lines):
  mean       mean over source lines                                  (= the transfer baseline)
  lowrank    each source line's (targets x genes) effect matrix denoised by truncated SVD, then mean
  geneweight lowrank, times a per-gene transfer coefficient learned on source-line pairs:
             w_g = sum_t a_tg b_tg / sum_t a_tg^2 for every ordered pair of source lines (a -> b),
             shrunk toward the global coefficient

    python -m vcc.effects cv            # leave-one-line-out among K562 / HCT116 / HEK293T
"""
import sys
import numpy as np, pandas as pd

from vcc.stats import load

TRAIN = ["K562", "HCT116", "HEK293T"]


def effect_matrix(line, min_cells=30):
    st = load(line); names = list(st["names"])
    cpm = st["sums"] / st["sums"].sum(1, keepdims=True) * 1e6
    c = cpm[names.index("__ctrl__")]
    keep = [i for i, n in enumerate(names) if n != "__ctrl__" and st["ncell"][i] >= min_cells]
    E = np.log2((cpm[keep] + 1) / (c + 1)).astype(np.float32)
    E[:, ~st["measured"]] = np.nan
    return pd.Index([names[i] for i in keep]), E, np.log2(c + 1), st["measured"]


def lowrank(E, k):
    """Truncated-SVD reconstruction of a (targets x genes) effect matrix, NaN genes left NaN."""
    if not k:
        return E
    meas = ~np.isnan(E).all(0)
    X = np.nan_to_num(E[:, meas])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    out = np.full_like(E, np.nan); out[:, meas] = (U[:, :k] * S[:k]) @ Vt[:k]
    return out


class Predictor:
    def __init__(self, sources, k=0, geneweight=False, shrink=50.0, scale=1.0):
        self.mats = {l: effect_matrix(l) for l in sources}
        self.den = {l: (idx, lowrank(E, k)) for l, (idx, E, _, _) in self.mats.items()}
        self.scale = scale
        self.w = None
        if geneweight and len(sources) >= 2:
            num = 0; den = 0
            for a in sources:
                for b in sources:
                    if a == b:
                        continue
                    ia, Ea = self.den[a]; ib, Eb = self.mats[b][0], self.mats[b][1]
                    shared = ia.intersection(ib)
                    A = Ea[ia.get_indexer(shared)]; B = Eb[ib.get_indexer(shared)]
                    ok = ~np.isnan(A) & ~np.isnan(B)
                    num = num + np.where(ok, A * B, 0).sum(0); den = den + np.where(ok, A * A, 0).sum(0)
            g = np.nansum(num) / np.nansum(den)                     # global transfer coefficient
            self.w = (num + shrink * g) / (den + shrink)            # per gene, shrunk toward it
            self.w = np.where(np.isfinite(self.w), self.w, g)

    def predict(self, targets):
        out = np.full((len(targets), self.mats[next(iter(self.mats))][1].shape[1]), np.nan, np.float32)
        acc = np.zeros_like(out); n = np.zeros_like(out)
        for l, (idx, E) in self.den.items():
            ii = idx.get_indexer(targets); ok = ii >= 0
            v = E[ii[ok]]; m = ~np.isnan(v)
            acc[ok] += np.where(m, v, 0); n[ok] += m
        out = np.where(n > 0, acc / np.maximum(n, 1), np.nan)
        if self.w is not None:
            out = out * self.w[None, :]
        return self.scale * out


def score(P, R, expr):
    """pearson of effects and PDS-like rank over expressed genes measured on both sides."""
    ok = expr & ~np.isnan(P).any(0) & ~np.isnan(R).any(0)
    P, R = P[:, ok], R[:, ok]
    pear = np.mean([np.corrcoef(p, r)[0, 1] for p, r in zip(P, R) if p.std() > 0])
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12); Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    C = Pn @ Rn.T
    mae = np.mean(np.abs(P - R)) / np.mean(np.abs(R))
    return pear, 1 - np.mean((C > np.diag(C)[:, None]).sum(1) / (len(P) - 1)), mae


def strong_targets(line, idx, k=10):
    s = pd.read_csv(f"/home/zxc29051617/VCC/data/stage3/stats/{line}_responsive.csv", index_col=0).iloc[:, 0]
    return [t for t in s.index[s >= k] if t in idx]


def cv(configs):
    rows = []
    for held in TRAIN + ["H1"]:
        srcs = [l for l in TRAIN if l != held]
        idx, R_all, c, meas = effect_matrix(held)
        targets = strong_targets(held, idx)
        R = R_all[idx.get_indexer(targets)]
        expr = (2 ** c - 1 >= 5) & meas
        for name, kw in configs.items():
            P = Predictor(srcs, **kw).predict(targets)
            have = ~np.isnan(P).all(1)
            pear, pds, mae = score(P[have], R[have], expr)
            rows.append(dict(heldout=held, config=name, n=int(have.sum()), pearson=pear, pds_like=pds, nmae_like=mae))
    df = pd.DataFrame(rows)
    print(df.pivot(index="config", columns="heldout", values="pds_like").round(3).to_string())
    print(df.pivot(index="config", columns="heldout", values="pearson").round(3).to_string())
    print(df.pivot(index="config", columns="heldout", values="nmae_like").round(3).to_string())
    return df


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv({"mean": {}, "mean_x1.5": dict(scale=1.5),
            "lowrank5": dict(k=5), "lowrank10": dict(k=10), "lowrank20": dict(k=20), "lowrank50": dict(k=50),
            "geneweight": dict(geneweight=True), "lowrank10+gw": dict(k=10, geneweight=True),
            "lowrank20+gw": dict(k=20, geneweight=True)})


def export(targets, out, sources=TRAIN, direction="geneweight", magnitude_scale=1.5):
    """Write predicted effects for `targets`: direction from `direction`, per-target L2 norm matched
    to magnitude_scale x the plain mean transfer (the magnitude the official scale rewarded)."""
    base = Predictor(sources).predict(targets)
    if direction == "mean":
        P = base * magnitude_scale
    else:
        D = Predictor(sources, geneweight=True).predict(targets)
        nb = np.sqrt(np.nansum(base ** 2, 1, keepdims=True)); nd = np.sqrt(np.nansum(D ** 2, 1, keepdims=True))
        P = D * (magnitude_scale * nb / np.maximum(nd, 1e-12))
    np.savez(out, targets=np.array(targets), lfc=P.astype(np.float32))
    print(f"wrote {out}: {len(targets)} targets, {int((~np.isnan(P).all(1)).sum())} with a prediction")


if __name__ == "__main__" and sys.argv[1] == "export":
    import json
    held = sys.argv[2]
    targets = json.load(open(f"/home/zxc29051617/VCC/data/stage3/loco/{held}/real.json"))["targets"]
    d = f"/home/zxc29051617/VCC/data/stage3/effects"; import os; os.makedirs(d, exist_ok=True)
    export(targets, f"{d}/{held}_mean_x1.5.npz", direction="mean")
    export(targets, f"{d}/{held}_gw_x1.5.npz", direction="geneweight")


if __name__ == "__main__" and sys.argv[1] == "export_vcc":
    import os
    targets = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").target_gene.astype(str).tolist()
    d = "/home/zxc29051617/VCC/data/stage3/effects"; os.makedirs(d, exist_ok=True)
    export(targets, f"{d}/VCC_mean_x1.5.npz", direction="mean")


class Wiener:
    """Reliability-weighted transfer with entrywise shrinkage.
    Per source line: effect e_l (log2) with sampling variance s2_l = phi/ln2^2 (1/(N_t+1) + 1/(N_c+1))
    (N = summed counts, phi = overdispersion). Lines are combined by precision with a transfer variance
    tau2 added to every line (cross-line disagreement that more counts do not remove). Each entry is then
    shrunk by w = S/(S + v), S = per-gene signal variance across targets, v = combined variance.
    Optionally the across-target mean (each line's generic response) is removed first."""

    def __init__(self, sources, tau2=0.05, phi=4.0, amp=1.5, remove_shared=False, min_cells=30, weights=None):
        self.amp = amp; self.lines = {}
        for l in sources:
            st = load(l); names = list(st["names"]); S = st["sums"]
            cpm = S / S.sum(1, keepdims=True) * 1e6; c = names.index("__ctrl__")
            keep = [i for i, n in enumerate(names) if n != "__ctrl__" and st["ncell"][i] >= min_cells]
            E = np.log2((cpm[keep] + 1) / (cpm[c] + 1))
            # summed counts rescaled to the line's control depth, so a big group is not "less noisy" by depth alone
            V = phi / np.log(2) ** 2 * (1 / (S[keep] + 1) + 1 / (S[c] + 1))
            E[:, ~st["measured"]] = np.nan; V[:, ~st["measured"]] = np.nan
            if remove_shared:
                E = E - np.nanmean(E, 0, keepdims=True)
            w = 1.0 if weights is None else weights.get(l, 1.0)
            self.lines[l] = (pd.Index([names[i] for i in keep]), E.astype(np.float32), (V + tau2).astype(np.float32), w)

    def predict(self, targets):
        G = next(iter(self.lines.values()))[1].shape[1]
        num = np.zeros((len(targets), G)); den = np.zeros((len(targets), G))
        for idx, E, V, w in self.lines.values():
            ii = idx.get_indexer(targets); ok = ii >= 0
            e, v = E[ii[ok]], V[ii[ok]]; m = ~np.isnan(e)
            num[ok] += np.where(m, w * e / v, 0); den[ok] += np.where(m, w / v, 0)
        mu = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan); var = np.where(den > 0, 1 / np.maximum(den, 1e-12), np.nan)
        S = np.clip(np.nanvar(mu, 0) - np.nanmean(var, 0), 1e-6, None)          # per-gene signal variance
        return self.amp * mu * (S / (S + var))


if __name__ == "__main__" and sys.argv[1] == "export_wiener":
    import json, os
    held = sys.argv[2]; tag = sys.argv[3]; kw = json.loads(sys.argv[4])
    srcs = kw.pop("sources", [l for l in TRAIN if l != held])
    if held == "VCC":
        targets = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").target_gene.astype(str).tolist()
    else:
        targets = json.load(open(f"/home/zxc29051617/VCC/data/stage3/loco/{held}/real.json"))["targets"]
    P = Wiener(srcs, **kw).predict(targets)
    d = "/home/zxc29051617/VCC/data/stage3/effects"; os.makedirs(d, exist_ok=True)
    np.savez(f"{d}/{held}_{tag}.npz", targets=np.array(targets), lfc=P.astype(np.float32))
    print(f"wrote {held}_{tag}: {len(targets)} targets, median |lfc| {np.nanmedian(np.abs(P)):.4f}, max {np.nanmax(np.abs(P)):.2f}")


if __name__ == "__main__" and sys.argv[1] == "export_mean":
    import json, os
    held, tag, scale = sys.argv[2], sys.argv[3], float(sys.argv[4]); srcs = sys.argv[5].split(",")
    w = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None
    if held == "VCC":
        targets = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").target_gene.astype(str).tolist()
    else:
        targets = json.load(open(f"/home/zxc29051617/VCC/data/stage3/loco/{held}/real.json"))["targets"]
    mats = {l: effect_matrix(l) for l in srcs}
    acc = 0; n = 0
    for l, (idx, E, _, _) in mats.items():
        ii = idx.get_indexer(targets); v = np.full((len(targets), E.shape[1]), np.nan, np.float32); v[ii >= 0] = E[ii[ii >= 0]]
        wl = 1.0 if w is None else w.get(l, 1.0); m = ~np.isnan(v)
        acc = acc + np.where(m, wl * v, 0); n = n + np.where(m, wl, 0)
    P = scale * np.where(n > 0, acc / np.maximum(n, 1e-12), np.nan)
    d = "/home/zxc29051617/VCC/data/stage3/effects"; os.makedirs(d, exist_ok=True)
    np.savez(f"{d}/{held}_{tag}.npz", targets=np.array(targets), lfc=P.astype(np.float32))
    print(f"wrote {held}_{tag}: {int((~np.isnan(P).all(1)).sum())}/{len(targets)} targets with an effect")


if __name__ == "__main__" and sys.argv[1] == "reshape":
    """Reshape a base effect file: genes with the largest predicted |lfc| per target (the DE signal,
    top-K) get amplitude a_hi, every other gene a_lo (the bulk shift).
        reshape <in.npz> <out.npz> <a_hi> <a_lo> <K>        (K = 0: one amplitude a_hi everywhere)"""
    z = np.load(sys.argv[2], allow_pickle=True); L = z["lfc"].copy()
    a_hi, a_lo, K = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    if K == 0:
        out = a_hi * L
    else:
        A = np.nan_to_num(np.abs(L)); thr = -np.sort(-A, axis=1)[:, K - 1:K]
        out = np.where(A >= thr, a_hi * L, a_lo * L)
    np.savez(sys.argv[3], targets=z["targets"], lfc=out.astype(np.float32))
    print(f"wrote {sys.argv[3]}")
