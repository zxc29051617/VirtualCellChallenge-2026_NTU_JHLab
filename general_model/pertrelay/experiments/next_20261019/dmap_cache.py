"""Training table for a DepMap-informed response model: every perturbation (>= 20 cells) of every source context,
common-mode-removed log-ratio response (same construction as the model's signatures: kscale-normalised, own gene and
cis window masked), on a fixed subset of response genes.

    python dmap_cache.py            (pertrelay venv)  -> cache_all/<ctx>.npz (perts, Y [n_pert, n_genes] float16), genes.npy

Gene subset: the 5,000 genes with the largest pooled response variance in the existing target cache, computed only from
contexts that are never held out (K562, K562ess, VIPK562, CD4 x3, HepG2, Jurkat), before any hold-out evaluation.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"
sys.path.insert(0, f"{ROOT}/pertrelay"); sys.path.insert(0, f"{ROOT}/pertrelay/experiments/signature_20261013")
import sigexp as S
from vcc.contexts import CONTEXTS, ROOT as STAGE3, build
from vcc.gm import _ratios, Genome
OWN, KCAP, CHUNK, NG = -1.9, 1.0, 800, 5000
OUT = f"{HERE}/cache_all"; os.makedirs(OUT, exist_ok=True)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

if not os.path.exists(f"{OUT}/genes.npy"):
    NEVER_HELD = ["K562", "K562ess", "VIPK562", "CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr", "HepG2", "Jurkat"]
    C = S.load_cache(NEVER_HELD); ss = np.zeros(S.G); n = np.zeros(S.G)
    for c in NEVER_HELD:
        Y = C[c]["Ysrc"]; ok = np.isfinite(Y); ss += (np.where(ok, Y, 0) ** 2).sum(0); n += ok.sum(0)
    genes = np.sort(np.argsort(-(ss / np.maximum(n, 1)))[:NG]); np.save(f"{OUT}/genes.npy", genes); del C
genes = np.load(f"{OUT}/genes.npy"); log(f"{len(genes)} response genes")
gen = Genome(S.GENES); gi = gen.idx
for name in S.CONTEXTS:
    if os.path.exists(f"{OUT}/{name}.npz"): log(f"{name}: exists"); continue
    t0 = time.time(); z = np.load(f"{STAGE3}/{CONTEXTS[name][0]}.npz", allow_pickle=True)
    names = list(z["names"]); sums = z["sums"]; ncell = z["ncell"]; i_c = names.index("__ctrl__")
    rows = [i for i, nm in enumerate(names) if i != i_c and not nm.startswith("__") and ncell[i] >= 20]
    st_of = lambda idx: {"names": np.array(["__ctrl__"] + [names[i] for i in idx]), "sums": np.vstack([sums[i_c][None], sums[idx]]),
                         "ncell": np.concatenate([[ncell[i_c]], ncell[idx]]), "genes": z["genes"], "measured": z["measured"]}
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    colsum = np.zeros(S.G); colcnt = np.zeros(S.G)
    for k, idx in enumerate(chunks):
        ch = build(f"{name}#{k}", st_of(idx), 20, keep_var=False); L, _ = _ratios(ch); f = np.isfinite(L)
        colsum += np.where(f, L, 0).sum(0); colcnt += f.sum(0)
    cm = np.where(colcnt > 0, colsum / np.maximum(colcnt, 1), np.nan)
    perts, out, nc = [], [], []
    for k, idx in enumerate(chunks):
        ch = build(f"{name}#{k}", st_of(idx), 20, keep_var=False); L, _ = _ratios(ch); Y = L - cm[None]
        for j, p in enumerate(ch.perts):
            kk = gi.get(p); own_obs = L[j, kk] if kk is not None else np.nan
            ks = np.clip(OWN / own_obs, 0.5, KCAP) if np.isfinite(own_obs) and own_obs < -0.3 else 1.0
            y = Y[j].copy(); g, _ = gen.cis(p); y[g] = np.nan
            if kk is not None: y[kk] = np.nan
            perts.append(p); out.append((ks * y[genes]).astype(np.float16)); nc.append(int(ch.ncell[j]))
    np.savez(f"{OUT}/{name}.npz", perts=np.array(perts), Y=np.stack(out), ncell=np.array(nc))
    log(f"{name}: {len(perts)} perturbations, {time.time() - t0:.0f}s")
