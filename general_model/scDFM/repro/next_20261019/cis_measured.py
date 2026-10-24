"""Screen: predict each target's neighbour (cis) genes from their MEASURED responses in the source contexts instead of
the distance-bin kernel alone. CRISPRi spreading depends on promoter architecture, which the cell lines share.

    python cis_measured.py            (scDFM venv)

1. cache: for every source context and every target in the target set, the kscale-normalised, common-mode-removed log
   ratio on the target's cis genes (the entries the trans signatures mask) -> cache_cis/<ctx>.npz
2. per hold-out (training contexts only): measured cis m = (sum_c y_c + lam_cis * kernel) / (n_c + lam_cis), kernel =
   own x beta[bin]; compared with the kernel alone by (a) median cosine on cis genes with the real delta and (b) the
   pds-like score of the full V6-style prediction. Uses hold-out truth (screen).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"
sys.path.insert(0, HERE); sys.path.insert(0, f"{ROOT}/scDFM"); sys.path.insert(0, f"{ROOT}/scDFM/repro/signature_20261013")
import sigexp as S, coexp_screen as CS, render_eval as E
from vcc.contexts import CONTEXTS, ROOT as STAGE3, build
from vcc.gm import _ratios, Genome
OWN, KCAP, CHUNK = -1.9, 1.0, 800
OUT = f"{HERE}/cache_cis"; os.makedirs(OUT, exist_ok=True)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
gen = Genome(S.GENES); gi = gen.idx; T = set(S.target_list())
for name in S.CONTEXTS:
    if os.path.exists(f"{OUT}/{name}.npz"): continue
    z = np.load(f"{STAGE3}/{CONTEXTS[name][0]}.npz", allow_pickle=True)
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
    rec = {}
    for k, idx in enumerate(chunks):
        sub = [i for i in idx if names[i] in T]
        if not sub: continue
        ch = build(f"{name}#{k}", st_of(idx), 20, keep_var=False); L, _ = _ratios(ch); Y = L - cm[None]
        for j, p in enumerate(ch.perts):
            if p not in T: continue
            kk = gi.get(p); own_obs = L[j, kk] if kk is not None else np.nan
            ks = np.clip(OWN / own_obs, 0.5, KCAP) if np.isfinite(own_obs) and own_obs < -0.3 else 1.0
            g, _ = gen.cis(p); rec[p] = (ks * Y[j, g]).astype(np.float32)
    np.savez(f"{OUT}/{name}.npz", targets=np.array(list(rec)), vals=np.array(list(rec.values()), dtype=object), allow_pickle=True)
    log(f"{name}: cis responses for {len(rec)} targets")
CC = {c: dict(zip(np.load(f"{OUT}/{c}.npz", allow_pickle=True)["targets"], np.load(f"{OUT}/{c}.npz", allow_pickle=True)["vals"])) for c in S.CONTEXTS}
Cc = S.load_cache(); res = {}
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); o = S.OUTER[name]; train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    _, oc, full, shuf = E.components(name); trans = full - oc
    beta = S.cis_beta(Cc, train)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True; mp = ~panel & (rate > 0)
    row = {}
    for lam in [None, 4.0, 1.0, 0.25]:
        oc2 = oc.copy(); cos = []
        for i, t in enumerate(targets):
            g, b = gen.cis(t)
            if not len(g): continue
            kern = OWN * beta[b]
            if lam is not None:
                num = lam * kern.copy(); den = np.full(len(g), lam)
                for c in train:
                    v = CC[c].get(t)
                    if v is None: continue
                    ok = np.isfinite(v); num[ok] += v[ok]; den[ok] += 1
                m = num / den
            else:
                m = kern
            oc2[i, g] = m / np.log(2)
            sel = mp[g] & (np.abs(D[i, g]) > 0)
            if sel.sum() >= 2:
                a = oc2[i, g][sel]; d = D[i, g][sel]; cos.append(float(a @ d / (np.linalg.norm(a) * np.linalg.norm(d) + 1e-12)))
        lfc = oc2 + E.remove_shared(trans)
        key = "kernel" if lam is None else f"measured_lam{lam}"
        row[key] = {"median_cos_cis": round(float(np.median(cos)), 4), "pds_like_V6": round(CS.pds_like(CS.pred_deltas(lfc, rate, mu_c), D, mp), 4)}
    res[name] = row; log(f"{name}: {json.dumps(row)}")
json.dump(res, open(f"{HERE}/cis_measured.json", "w"), indent=1)
