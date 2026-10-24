"""Screen: do LINCS 2020 CRISPR-knockout consensus signatures (L1000, ~20 cancer cell lines) add to the measured-response
transfer on the hold-outs?

    python lincs_screen.py            (scDFM venv; needs level5_beta_trt_xpr gctx + siginfo/geneinfo)

Per (target, LINCS cell line): mean of the target's trt_xpr level-5 signatures (qc_pass), minus that cell line's mean
over all its trt_xpr signatures (common mode), on the 12,328 L1000 genes (landmark + inferred) mapped by symbol.
Units: L1000 robust z-scores are put on the model's log-ratio scale gene by gene (ratio of the gene's standard
deviation across the never-held-out contexts' responses to its standard deviation across LINCS signatures).
Each LINCS cell line then joins the trans average as one more source with weight W (0 = current model).
Evaluated like the other screens: median cosine with the real bulk delta (L1000 genes; panel targets, own and cis removed)
and the pds-like score of the full V6-style prediction. Uses hold-out truth (screen).
"""
import json, os, sys, time
import numpy as np, pandas as pd, h5py
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"; L = f"{ROOT}/data/external/lincs2020"
sys.path.insert(0, HERE); sys.path.insert(0, f"{ROOT}/scDFM"); sys.path.insert(0, f"{ROOT}/scDFM/repro/signature_20261013")
import sigexp as S, coexp_screen as CS, render_eval as E
from vcc.gm import Genome
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

sig = pd.read_csv(f"{L}/siginfo_beta.txt", sep="\t", low_memory=False)
gin = pd.read_csv(f"{L}/geneinfo_beta.txt", sep="\t")
x = sig[(sig.pert_type == "trt_xpr") & (sig.qc_pass == 1)] if "qc_pass" in sig.columns else sig[sig.pert_type == "trt_xpr"]
gcol = "cmap_name" if "cmap_name" in x.columns else "pert_iname"
need = set(pd.read_csv(f"{ROOT}/data/vcc2026/pert_counts.csv").target_gene.astype(str))
for s in ["H1", "HCT116", "KOLF200"]: need |= set(json.load(open(f"{ROOT}/data/stage3/loco/{s}/real.json"))["targets"])
f = h5py.File(f"{L}/level5_beta_trt_xpr_n142901x12328.gctx", "r")
cid = f["0/META/COL/id"][:].astype(str); rid = f["0/META/ROW/id"][:].astype(str); mat = f["0/DATA/0/matrix"]
log(f"gctx matrix {mat.shape}, cols {len(cid)}, rows {len(rid)}")
col_pos = {c: i for i, c in enumerate(cid)}
sym = dict(zip(gin.gene_id.astype(str), gin.gene_symbol.astype(str)))
gpos = np.array([S.GI.get(sym.get(r, ""), -1) for r in rid]); okg = gpos >= 0
log(f"L1000 genes on the challenge axis: {int(okg.sum())} of {len(rid)}")
def read_cols(ids):
    idx = np.sort([col_pos[s] for s in ids if s in col_pos])
    if mat.shape[0] == len(cid): return mat[idx, :]
    return mat[:, idx].T
# per cell line common mode (mean over all its trt_xpr signatures; sampled to at most 2000 for memory)
cm, rng = {}, np.random.default_rng(0)
for cl, grp in x.groupby("cell_iname"):
    ids = grp.sig_id.values; ids = rng.choice(ids, min(2000, len(ids)), replace=False)
    cm[cl] = read_cols(ids).mean(0)
sel = x[x[gcol].isin(need)]
prof = {}
for (t, cl), grp in sel.groupby([gcol, "cell_iname"]):
    prof[(t, cl)] = read_cols(grp.sig_id.values).mean(0) - cm[cl]
log(f"{len(prof)} (target, cell line) profiles for {len(set(t for t, _ in prof))} targets in {len(set(c for _, c in prof))} lines")
# scale: per gene, sd of model responses (never-held-out contexts) / sd of LINCS profiles
NEVER = ["K562", "K562ess", "VIPK562", "CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr", "HepG2", "Jurkat"]
Cc = S.load_cache(); sd_model = np.nanstd(np.vstack([Cc[c]["Ysrc"] for c in NEVER]), 0)
P = np.stack(list(prof.values())); sd_l = P.std(0)
scale = np.where((sd_l > 0), sd_model[np.maximum(gpos, 0)] / np.maximum(sd_l, 1e-9), 0) * okg
lines = sorted(set(c for _, c in prof))
def lincs_full(t):
    """per-line LINCS profiles for target t on the challenge axis (ln units), list of arrays"""
    out = []
    for cl in lines:
        v = prof.get((t, cl))
        if v is None: continue
        a = np.zeros(S.G); a[gpos[okg]] = (v * scale)[okg]; out.append(a)
    return out
gen = Genome(S.GENES); res = {}
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); o = S.OUTER[name]; train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    _, oc, full, shuf = E.components(name)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True
    lm = np.zeros(S.G, bool); lm[gpos[okg]] = True
    masks = []
    for t in targets:
        m = lm & ~panel & (rate > 0); g, _ = gen.cis(t); m[g] = False
        if t in S.GI: m[S.GI[t]] = False
        masks.append(m)
    w = {c: 1.0 for c in train}; row = {"targets_with_lincs": int(sum(1 for t in targets if lincs_full(t)))}
    for W in [0.0, 0.5, 1.0, 2.0]:
        tr = np.zeros((len(targets), S.G))
        for i, t in enumerate(targets):
            num = np.zeros(S.G); den = np.zeros(S.G)
            for c in train:
                j = Cc[c]["tidx"].get(t)
                if j is None: continue
                y = Cc[c]["Ysrc"][j]; ok = np.isfinite(y); num[ok] += y[ok]; den[ok] += 1
            for a in (lincs_full(t) if W > 0 else []):
                ok = a != 0; num[ok] += W * a[ok]; den[ok] += W
            tr[i] = num / (den + S.LAM)
            g, _ = gen.cis(t); tr[i, g] = 0
            if t in S.GI: tr[i, S.GI[t]] = 0
        tr3 = 3 * tr / np.log(2)
        has = [i for i, t in enumerate(targets) if lincs_full(t)]
        cos = [np.dot(tr3[i][masks[i]], D[i][masks[i]]) / (np.linalg.norm(tr3[i][masks[i]]) * np.linalg.norm(D[i][masks[i]]) + 1e-12) for i in has]
        lfc = oc + E.remove_shared(tr3)
        row[f"W{W}"] = {"median_cos_on_lincs_targets": round(float(np.median(cos)), 4),
                        "pds_like_V6": round(CS.pds_like(CS.pred_deltas(lfc, rate, mu_c), D, ~panel & (rate > 0)), 4)}
    res[name] = row; log(f"{name}: {json.dumps(row)}")
json.dump(res, open(f"{HERE}/lincs_screen.json", "w"), indent=1)
