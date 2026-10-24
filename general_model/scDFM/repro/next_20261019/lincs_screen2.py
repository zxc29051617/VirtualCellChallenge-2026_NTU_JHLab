"""Screen (hold-out truth): LINCS 2020 consensus profiles (extracted in the cloud: cloud/lincs_extract.py ->
lincs_profiles.npz) as extra source contexts in the trans average.

    python lincs_screen2.py           (scDFM venv)

Per (target, LINCS cell line): profile minus that line's common mode (mean over all signatures of the same type), on the
L1000 genes mapped by symbol; put on the model's log-ratio scale gene by gene (sd of the never-held-out contexts'
responses / sd across LINCS profiles of that type). Each (target, line) joins the trans average with weight W.
Sources: xpr (CRISPR KO, qc_pass), sh (shRNA consensus cgs), both. Reported per hold-out: number of targets with a LINCS
profile, median cosine with the real bulk delta on L1000 genes for those targets, pds-like score of the full V6 prediction.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"
sys.path.insert(0, HERE); sys.path.insert(0, f"{ROOT}/scDFM"); sys.path.insert(0, f"{ROOT}/scDFM/repro/signature_20261013")
import sigexp as S, coexp_screen as CS, render_eval as E
from vcc.gm import Genome
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
z = np.load(f"{HERE}/cloud/lincs_profiles.npz", allow_pickle=True)
sym = z["symbols"]; gpos = np.array([S.GI.get(str(s), -1) for s in sym]); okg = gpos >= 0
NEVER = ["K562", "K562ess", "VIPK562", "CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr", "HepG2", "Jurkat"]
Cc = S.load_cache(); sd_model = np.nanstd(np.vstack([Cc[c]["Ysrc"] for c in NEVER]), 0)
src = {}
for key, pt in [("xpr", "trt_xpr"), ("sh", "trt_sh.cgs")]:
    pairs = z[f"{pt}__pairs"]; prof = z[f"{pt}__prof"].astype(np.float64); lines = list(z[f"{pt}__cm_lines"]); cm = z[f"{pt}__cm"].astype(np.float64)
    prof = prof - cm[[lines.index(cl) for _, cl in pairs]]
    sd = prof.std(0); scale = np.nan_to_num(np.where(sd > 0, sd_model[np.maximum(gpos, 0)] / np.maximum(sd, 1e-9), 0)) * okg
    d = {}
    for (t, cl), v in zip(pairs, prof):
        a = np.zeros(S.G); a[gpos[okg]] = (v * scale)[okg]; d.setdefault(str(t), []).append(a)
    src[key] = d; log(f"{key}: {len(pairs)} profiles, {len(d)} targets, {len(lines)} lines")
gen = Genome(S.GENES); res = {}
lm = np.zeros(S.G, bool); lm[gpos[okg]] = True
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); o = S.OUTER[name]; train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    _, oc, full, shuf = E.components(name)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True; mp = ~panel & (rate > 0)
    masks = []
    for t in targets:
        m = lm & mp; g, _ = gen.cis(t); m[g] = False
        if t in S.GI: m[S.GI[t]] = False
        masks.append(m)
    row = {}
    for use in [(), ("xpr",), ("sh",), ("xpr", "sh")]:
        for W in ([0.0] if not use else [0.5, 1.0]):
            tr = np.zeros((len(targets), S.G)); has = []
            for i, t in enumerate(targets):
                num = np.zeros(S.G); den = np.zeros(S.G)
                for c in train:
                    j = Cc[c]["tidx"].get(t)
                    if j is None: continue
                    y = Cc[c]["Ysrc"][j]; ok = np.isfinite(y); num[ok] += y[ok]; den[ok] += 1
                extra = [a for k in use for a in src[k].get(t, [])]
                if extra: has.append(i)
                for a in extra:
                    ok = a != 0; num[ok] += W * a[ok]; den[ok] += W
                tr[i] = num / (den + S.LAM); g, _ = gen.cis(t); tr[i, g] = 0
                if t in S.GI: tr[i, S.GI[t]] = 0
            tr3 = 3 * tr / np.log(2)
            ref = has if has else [i for i, t in enumerate(targets) if any(t in src[k] for k in ("xpr", "sh"))]
            cos = [np.dot(tr3[i][masks[i]], D[i][masks[i]]) / (np.linalg.norm(tr3[i][masks[i]]) * np.linalg.norm(D[i][masks[i]]) + 1e-12) for i in ref]
            lfc = oc + E.remove_shared(tr3)
            row[f"{'+'.join(use) or 'none'}_W{W}"] = {"n_with_lincs": len(has), "median_cos": round(float(np.median(cos)), 4),
                                                       "pds_like_V6": round(CS.pds_like(CS.pred_deltas(lfc, rate, mu_c), D, mp), 4)}
    res[name] = row; log(f"{name}: {json.dumps(row)}")
json.dump(res, open(f"{HERE}/lincs_screen2.json", "w"), indent=1)
