"""Does TF -> target knowledge (CollecTRI via OmniPath) help predict knockdowns of never-perturbed TFs? (2026-10-23)

    .venv/bin/python -I model/tf_eval.py        (from pertrelay/)

prior_p[g] = +1 if TF p activates g, -1 if it represses g (consensus signs), / sqrt(number of targets). Knocking down p
should move g by -prior_p[g]. Scored on bench_genes.py's S1 and S2 test sets restricted to genes that are CollecTRI TFs
with >= 5 targets on the gene axis: tf prior alone, the blend (kNN STRING + ridge), and blend + tf prior (equal size).
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import bench as B, bench_genes as BG

T = pd.read_csv("/home/zxc29051617/VCC/data/external/collectri/collectri.tsv", sep="\t")
gi = B.genome().idx; G = len(B.genome().genes)
prior = {}
for tf, d in T.groupby("source_genesymbol"):
    v = np.zeros(G, np.float32)
    for tg, st, inh in zip(d.target_genesymbol, d.consensus_stimulation, d.consensus_inhibition):
        k = gi.get(tg)
        if k is None: continue
        v[k] = 1.0 if st and not inh else (-1.0 if inh and not st else 0.0)
    n = int((v != 0).sum())
    if n >= 5 and tf in gi: prior[tf] = v / np.sqrt(n)
print(f"{len(prior)} TFs with >= 5 signed targets on the gene axis", flush=True)


def run():
    res = {}
    for st in ("S1", "S2"):
        for fold in range(BG.N_FOLD):
            for hg in ([None] if st == "S1" else BG.S2_GROUPS):
                train_rows, tests = BG.setup(fold, hg)
                for c, tg in tests.items():
                    tfs = [p for p in tg if p in prior]
                    if len(tfs) < 8: continue
                    P_tf = -np.stack([prior[p] for p in tfs])
                    P_bl = BG.m_blend(train_rows, c, tfs, st).astype(np.float64)
                    nb, nt = np.linalg.norm(P_bl), np.linalg.norm(P_tf)
                    P_mix = P_bl + P_tf * (nb / max(nt, 1e-9))
                    for nm, P in (("tf_prior", P_tf), ("blend", P_bl), ("blend+tf", P_mix)):
                        r = B.score(c, tfs, P.astype(np.float32)); res.setdefault((st, nm), []).append(r)
    for (st, nm), rs in sorted(res.items()):
        print(f"{st} {nm:9s} dir {np.mean([r['dir'] for r in rs]):+.4f}  pds {np.mean([r['pds'] for r in rs]):.4f}  "
              f"skill {np.mean([r['skill'] for r in rs]):+.4f}  ({len(rs)} context-folds, {int(np.mean([r['n'] for r in rs]))} TFs each)", flush=True)


run()
