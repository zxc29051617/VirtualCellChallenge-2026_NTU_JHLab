"""Does a cell line's own CRISPR dependency on a gene predict how strongly the transcriptome responds to knocking the
gene down there? (2026-10-25)

    .venv/bin/python -I model/dep_magnitude.py <CRISPRGeneEffect.csv>        (from pertrelay/)

DepMap 24Q4 Chronos gene effects (more negative = cells need the gene more). For every training context that is a
DepMap line (K562 x3 screens, HepG2, Jurkat, HCT116), the response energy of each target = noise-corrected sum of
squares of its trans response (bench.score genes and noise, log1p-scale noise as in the benchmark tables). Spearman
correlation of energy with (a) the line's own gene effect, (b) the mean gene effect over all DepMap lines, and (c) the
part of (a) not explained by (b) (residual of a linear fit). Then the D/E/F targets that only MOLM-13 (D) or U-87 MG
(E) depend on. Writes results/dep_magnitude.json.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

HERE = os.path.dirname(os.path.abspath(__file__))
G = pd.read_csv(sys.argv[1], index_col=0)
G.columns = [c.split(" (")[0] for c in G.columns]
G = G.loc[:, ~G.columns.duplicated()]
mean_eff = G.mean(0)
LINE = {"K562": "ACH-000551", "K562ess": "ACH-000551", "VIPK562": "ACH-000551", "HepG2": "ACH-000739",
        "Jurkat": "ACH-000995", "HCT116": "ACH-000971"}
D = B.data(); out = {"contexts": {}}
for c, mid in LINE.items():
    d = D[c]; base = d["measured"] & (np.expm1(d["ctrl"]) >= B.X_MIN)
    rows = [(p, j) for j, p in enumerate(d["perts"]) if d["ncell"][j] >= 50 and p in G.columns]
    en = []
    for p, j in rows:
        y = d["Y"][j].astype(np.float64); v = d["var"][j].astype(np.float64)
        m = B.trans_mask(p, base) & np.isfinite(y) & np.isfinite(v)
        en.append((y[m] ** 2).sum() - v[m].sum())
    en = np.array(en); own = G.loc[mid, [p for p, _ in rows]].to_numpy(); pan = mean_eff[[p for p, _ in rows]].to_numpy()
    ok = np.isfinite(own) & np.isfinite(pan); en, own, pan = en[ok], own[ok], pan[ok]
    resid = own - np.polyval(np.polyfit(pan, own, 1), pan)
    r = dict(n=int(ok.sum()), own=float(spearmanr(-own, en)[0]), pan=float(spearmanr(-pan, en)[0]),
             own_beyond_pan=float(spearmanr(-resid, en)[0]))
    out["contexts"][c] = r
    print(f"{c:8s} n={r['n']:5d}  energy vs own dependency {r['own']:+.3f}   vs mean over lines {r['pan']:+.3f}   "
          f"own beyond mean {r['own_beyond_pan']:+.3f}", flush=True)

T = [l.strip() for l in open("/home/zxc29051617/VCC/data/vcc2026_final/pert_counts.csv")][1:]
T = [t.split(",")[0] for t in T if t and not t.startswith("non-targeting")]
for name, mid in (("D (MOLM-13)", "ACH-000362"), ("E (U-87 MG)", "ACH-000075")):
    rows = [(t, G.at[mid, t], mean_eff[t], (G[t] < -0.5).mean()) for t in T if t in G.columns]
    ess = sorted([r for r in rows if r[1] < -0.5], key=lambda r: r[1])
    spec = [r for r in ess if r[3] < 0.2]
    print(f"\n{name}: {len(rows)}/{len(T)} targets screened; {len(ess)} with gene effect < -0.5, of which {len(spec)} "
          f"essential in < 20 % of DepMap lines (line-specific):")
    print("   " + ", ".join(f"{t} {e:+.2f} (mean {m:+.2f}, {f:.0%} of lines)" for t, e, m, f in spec))
    out[name] = dict(screened=len(rows), essential=[r[0] for r in ess], specific=[r[0] for r in spec])
json.dump(out, open(f"{HERE}/results/dep_magnitude.json", "w"), indent=1)
