"""Per-context cache for the source-weight experiment (memory-light re-implementation of gm.GeneralModel.fit pieces).

    python cache_contexts.py            (pertrelay venv)

For each of the 14 training contexts, streams its pseudobulk stats in chunks and stores, for the target set T
(union of the three hold-out evaluation target lists and the 300 challenge targets):
  Ysrc[|T_c|, G]  = kscale x (log ratio - context common mode), NaN at the target's own + cis genes and where unmeasured
                    (exactly the y that gm.GeneralModel.fit accumulates; kscale = clip(-1.9/own_obs, 0.5, 1))
  kscale, ncell per row, cis least-squares sums (num, den per TSS bin), control profile, measured mask, source quality.
Same formulas as vcc/gm.py (EPS, X_MIN, BULK, clip); equivalence is asserted in verify_equivalence.py.
"""
import json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"
sys.path.insert(0, f"{ROOT}/pertrelay")
from vcc.contexts import CONTEXTS, ROOT as STAGE3, build
from vcc.gm import _ratios, Genome, CIS_BINS
OWN, KCAP, CHUNK = -1.9, 1.0, 800
OUT = f"{HERE}/cache"; os.makedirs(OUT, exist_ok=True)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

genes = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); G = len(genes)
vcc300 = list(dict.fromkeys(pd.read_csv(f"{ROOT}/data/vcc2026/pert_counts.csv").target_gene.astype(str)))
FINAL = f"{ROOT}/data/vcc2026_final/pert_counts.csv"          # final panel D/E/F (2026-10-22): its targets join T
final300 = list(dict.fromkeys(pd.read_csv(FINAL).target_gene.astype(str))) if os.path.exists(FINAL) else []
sets = {"H1": json.load(open(f"{ROOT}/data/stage3/loco/H1/real.json"))["targets"],
        "HCT116": json.load(open(f"{ROOT}/data/stage3/loco/HCT116/real.json"))["targets"],
        "KOLF200": json.load(open(f"{ROOT}/data/stage3/loco/KOLF200/real.json"))["targets"]}
T = sorted(set(vcc300) | set(final300) | set().union(*map(set, sets.values()))); Tset = set(T)
json.dump({"T": T, "sets": {k: len(v) for k, v in sets.items()}, "n_T": len(T), "n_vcc300": len(vcc300), "n_final300": len(final300)}, open(f"{OUT}/targets.json", "w"))
log(f"target set T: {len(T)} genes (VCC300 {len(vcc300)} + eval sets {sets and {k: len(v) for k, v in sets.items()}})")
Gn = Genome(genes); gi = Gn.idx
avail = [n for n, (p, _) in CONTEXTS.items() if os.path.exists(f"{STAGE3}/{p}.npz")]
for name in avail:
    if os.path.exists(f"{OUT}/{name}.npz"): log(f"{name}: exists"); continue
    t0 = time.time()
    z = np.load(f"{STAGE3}/{CONTEXTS[name][0]}.npz", allow_pickle=True)
    names = list(z["names"]); sums = z["sums"]; ncell = z["ncell"]; measured = np.asarray(z["measured"], bool)
    i_c = names.index("__ctrl__")
    rows = [i for i, n in enumerate(names) if i != i_c and not n.startswith("__") and ncell[i] >= 20]
    st_of = lambda idx: {"names": np.array(["__ctrl__"] + [names[i] for i in idx]),
                         "sums": np.vstack([sums[i_c][None], sums[idx]]), "ncell": np.concatenate([[ncell[i_c]], ncell[idx]]),
                         "genes": z["genes"], "measured": z["measured"]}
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    colsum = np.zeros(G, np.float64); colcnt = np.zeros(G, np.int64); kd = []
    for k, idx in enumerate(chunks):                                  # pass 1: common mode (nanmean over all perts)
        ch = build(f"{name}#{k}", st_of(idx), 20, keep_var=False); L, xc = _ratios(ch)
        f = np.isfinite(L); colsum += np.where(f, L, 0).sum(0); colcnt += f.sum(0); kd.append(ch.kd)
        if k == 0: ctrl = ch.ctrl.copy()
    cm = np.where(colcnt > 0, colsum / np.maximum(colcnt, 1), np.nan).astype(np.float32)
    num = np.zeros(len(CIS_BINS)); den = np.zeros(len(CIS_BINS))
    out_t, out_Y, out_ks, out_nc, out_own = [], [], [], [], []
    for k, idx in enumerate(chunks):                                  # pass 2: cis sums + target rows
        ch = build(f"{name}#{k}", st_of(idx), 20, keep_var=False); L, xc = _ratios(ch); Y = L - cm[None]
        for j, p in enumerate(ch.perts):
            kk = gi.get(p); own_obs = L[j, kk] if kk is not None else np.nan
            g, b = Gn.cis(p)
            if np.isfinite(own_obs) and own_obs < -0.5 and len(g):
                y = Y[j, g]; ok = np.isfinite(y); np.add.at(num, b[ok], y[ok] * own_obs); np.add.at(den, b[ok], own_obs ** 2)
            if p in Tset:
                ks = np.clip(OWN / own_obs, 0.5, KCAP) if np.isfinite(own_obs) and own_obs < -0.3 else 1.0
                y = Y[j].copy(); y[g] = np.nan
                if kk is not None: y[kk] = np.nan
                out_t.append(p); out_Y.append((ks * y).astype(np.float32)); out_ks.append(ks); out_nc.append(int(ch.ncell[j])); out_own.append(own_obs)
    kdv = np.concatenate(kd)
    np.savez(f"{OUT}/{name}.npz", targets=np.array(out_t), Ysrc=np.stack(out_Y) if out_Y else np.zeros((0, G), np.float32),
             kscale=np.array(out_ks), ncell=np.array(out_nc), own_obs=np.array(out_own, np.float32), cis_num=num, cis_den=den,
             ctrl=ctrl.astype(np.float32), measured=measured, quality=float(-np.nanmedian(kdv)), n_perts=len(rows),
             n_targets_T=len(out_t))
    log(f"{name}: {len(rows)} perts, {len(out_t)} in T, quality {-np.nanmedian(kdv):.2f}, {time.time() - t0:.0f}s")

# control profiles of the three hold-out contexts from their real-set NTC cells (what a method is given for a new context)
import anndata as ad, scipy.sparse as sp
for ctx, loco in [("H1", "H1"), ("HCT116", "HCT116"), ("KOLF", "KOLF200")]:
    f = f"{OUT}/heldout_ctrl_{ctx}.npz"
    if os.path.exists(f): continue
    A = ad.read_h5ad(f"{STAGE3}/loco/{loco}/real.h5ad"); m = A.obs.target_gene.astype(str).values == "non-targeting"
    s = np.asarray(sp.csr_matrix(A.X[np.flatnonzero(m)]).sum(0)).ravel().astype(np.float64)
    np.savez(f, ctrl=np.log1p(50000.0 * s / s.sum()).astype(np.float32), measured=s > 0, n_cells=int(m.sum()))
    log(f"heldout ctrl {ctx}: {int(m.sum())} NTC cells")
