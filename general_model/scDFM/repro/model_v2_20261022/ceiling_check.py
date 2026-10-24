"""How much of a hidden cell type's response could be carried over from the training cell types at all? (2026-10-24)

    .venv/bin/python -I repro/model_v2_20261022/ceiling_check.py        (from scDFM/; RPE1=1 adds RPE1 as a source)

Same hidden contexts, targets and genes as the 7-fold benchmark (bench.score). The sources of a target are the training
contexts' measured responses to the same target. Sampling noise on the ln-ratio scale (var_scale_check.py) is removed
from every sum of squares, so the numbers estimate what infinitely deep sequencing of the same experiments would give:
  mean    noise-corrected cosine between the hidden response and the plain mean of its sources' TRUE responses, i.e.
          the dir of a model that averages the sources perfectly and does not adapt to the cell type
  best1   the same for the single source context that agrees best (picked with the answer: an optimistic reference)
  noisy   dir of the plain mean as measured (sources' noise left in), the counterpart of consensus_d200
Fold and overall averages as bench.run. Results: results/ceiling{_rpe1}{_oldvar}.json.
VAR=old keeps bench.score's log1p-scale variance instead. The delta method on the ln-ratio scale overstates the noise
where a pseudobulk has only a few counts (ln(x + 0.5) is bounded at x = 0) and the log1p scale understates it for
moderately expressed genes, so the two bracket the truth (TianNeuron2021, shallow: noise share 0.71 vs 1.20 of sum y^2).
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

HERE = os.path.dirname(os.path.abspath(__file__)); t0 = time.time(); OLD = os.environ.get("VAR") == "old"
D = B.data()


def ln_var(d, j):
    """row j's response and its sampling variance on the ln-ratio scale (NaN where unusable)."""
    y = d["Y"][j].astype(np.float64); v = d["var"][j].astype(np.float64)
    if OLD: return y, v
    xc = np.expm1(d["ctrl"].astype(np.float64)); xp = (xc + 0.5) * np.exp(y + d["cm"].astype(np.float64)) - 0.5
    return y, v * ((1 + xp) / (0.5 + xp)) ** 2


res = {}
for fold, held_list in B.FOLDS.items():
    if fold in B.SKIP_EVAL or (B.ONLY_FOLDS and fold not in B.ONLY_FOLDS): continue
    train = [c for c in B.CONTEXTS if c not in held_list]
    for h in held_list:
        targets = B.eval_targets(h, train)
        if len(targets) < 20: continue
        d = D[h]; base = d["measured"] & (np.expm1(d["ctrl"]) >= B.X_MIN)
        hh = cross = mm = noisy_pp = 0.0; per_src = {t: [0.0, 0.0] for t in train}      # source -> [cross, tt]
        for p in targets:
            y, v = ln_var(d, d["row"][p]); M = B.trans_mask(p, base) & np.isfinite(y) & np.isfinite(v)
            y = np.where(M, y, 0.0); hh += (y ** 2).sum() - v[M].sum()
            s = np.zeros_like(y); sv = np.zeros_like(y); n = np.zeros_like(y)
            for t in train:
                j = D[t]["row"].get(p)
                if j is None: continue
                yt, vt = ln_var(D[t], j); ok = M & np.isfinite(yt) & np.isfinite(vt)
                s[ok] += yt[ok]; sv[ok] += vt[ok]; n[ok] += 1
                per_src[t][0] += (y[ok] * yt[ok]).sum(); per_src[t][1] += (yt[ok] ** 2).sum() - vt[ok].sum()
            k = n > 0; m = np.zeros_like(y); m[k] = s[k] / n[k]; vm = np.zeros_like(y); vm[k] = sv[k] / n[k] ** 2
            cross += (y * m).sum(); mm += (m ** 2).sum() - vm.sum(); noisy_pp += (m ** 2).sum()
        if hh <= 0: hh = np.nan                          # the estimated noise exceeds the response: undefined
        best = max(((c / np.sqrt(hh * tt), t) for t, (c, tt) in per_src.items() if tt > 0 and hh > 0), default=(np.nan, None))
        r = dict(mean=cross / np.sqrt(hh * mm) if mm > 0 else np.nan, best1=float(best[0]), best1_source=best[1],
                 noisy=cross / np.sqrt(hh * max(noisy_pp, 1e-9)), n=len(targets))
        res.setdefault(fold, {})[h] = r
        print(f"{fold:7s} {h:15s} mean {r['mean']:.3f}  best1 {r['best1']:.3f} ({best[1]})  noisy {r['noisy']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

fm = {f: {k: float(np.mean([r[k] for r in v.values()])) for k in ("mean", "best1", "noisy")} for f, v in res.items()}
overall = {k: float(np.mean([x[k] for x in fm.values()])) for k in ("mean", "best1", "noisy")}
tag = ("_rpe1" if os.environ.get("RPE1") == "1" else "") + ("_" + os.environ["NEURON"] if os.environ.get("NEURON") else "") + ("_oldvar" if OLD else "")
json.dump(dict(overall=overall, folds=fm, contexts=res), open(f"{HERE}/results/ceiling{tag}.json", "w"), indent=1)
print(f"\noverall: mean {overall['mean']:.4f}  best1 {overall['best1']:.4f}  noisy {overall['noisy']:.4f}")
for f, x in fm.items(): print(f"  {f:7s} mean {x['mean']:.3f}  best1 {x['best1']:.3f}  noisy {x['noisy']:.3f}")
