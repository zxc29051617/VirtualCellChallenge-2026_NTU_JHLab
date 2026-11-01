"""Noise term of the 7-fold benchmark on the right scale (Codex audit M2, 2026-10-24).

    .venv/bin/python -I model/var_scale_check.py        (from pertrelay/; set NEURON=outer for the neuron runs)

bench.score subtracts sampling noise with the variance vcc.contexts.build computes for log1p(CP50k) (pseudo-count 1, both
sides), but the scored response is ln((x_p + 0.5)/(x_c + 0.5)) minus the context's common mode. By the delta method the
perturbed side's variance on that scale is larger by ((1 + x_p)/(0.5 + x_p))^2 (x_p in CP50k), up to 4x for barely
expressed genes, and the control side cancels with the common mode (it is <~5 % of the total anyway and is kept here).
Predictions, pds and scale do not involve the noise term, so every saved result is re-scored exactly from its JSON:
  dir' = dir x sqrt(signal / signal'),  skill' = 1 - (SSE - sum V') / signal',  SSE = sum V + (1 - skill) x signal.
The factor is the same for every method on a hidden context (same targets and genes), so comparisons within a context are
unchanged; only fold weights in the averages move. Writes results/var_scale.json (per context: old and new noise
share; per saved result: dir and skill both ways). The result files themselves are not modified.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

HERE = os.path.dirname(os.path.abspath(__file__))
for n in B.CONTEXTS:                      # light cache: perts only, the full arrays are read one hidden context at a time
    z = np.load(f"{B.DATA}/{n}.npz", allow_pickle=True)
    B._D[n] = dict(perts=z["perts"].astype(str), ncell=z["ncell"], genes=z["genes"])
    B._D[n]["row"] = {p: j for j, p in enumerate(B._D[n]["perts"])}

terms = {}
for fold, held_list in B.FOLDS.items():
    if fold in B.SKIP_EVAL or (B.ONLY_FOLDS and fold not in B.ONLY_FOLDS): continue
    train = [c for c in B.CONTEXTS if c not in held_list]
    for held in held_list:
        targets = B.eval_targets(held, train)
        if len(targets) < 20: continue
        z = np.load(f"{B.DATA}/{held}.npz", allow_pickle=True); d = B._D[held]
        rows = [d["row"][p] for p in targets]
        ctrl = z["ctrl"].astype(np.float64); base = z["measured"] & (np.expm1(ctrl) >= B.X_MIN)
        Y = z["Y"][rows].astype(np.float64); V = z["var"][rows].astype(np.float64)
        M = np.stack([B.trans_mask(p, base) for p in targets]) & np.isfinite(Y) & np.isfinite(V)
        xc = np.expm1(ctrl); xp = (xc + 0.5) * np.exp(Y + z["cm"].astype(np.float64)) - 0.5   # perturbed CP50k
        V2 = V * ((1 + xp) / (0.5 + xp)) ** 2
        yy = float((Y[M] ** 2).sum()); v1 = float(V[M].sum()); v2 = float(V2[M].sum())
        terms[held] = dict(n=len(targets), yy=yy, v_old=v1, v_new=v2, noise_share_old=v1 / yy, noise_share_new=v2 / yy)
        print(f"{fold:7s} {held:15s} {len(targets):3d} targets  noise share of sum y^2: {v1 / yy:.3f} -> {v2 / yy:.3f}", flush=True)
        del z, Y, V, V2, M

tag = "_" + os.environ["NEURON"] if os.environ.get("NEURON") else ""
out = dict(note="noise term on the ln-ratio scale; see var_scale_check.py", contexts=terms, methods={})
print("\nre-scored saved results (dir, skill: old -> noise on the ln-ratio scale; pds unchanged):")
for f in sorted(glob.glob(f"{HERE}/results/*.json")):
    r = json.load(open(f))
    if not isinstance(r, dict) or "contexts" not in r or "overall" not in r or "var_scale" in f: continue
    ctx = {h: v for fm in r["contexts"].values() for h, v in fm.items()}
    if not ctx or any(h not in terms or not {"dir", "skill"} <= set(v) or v.get("n") != terms[h]["n"] for h, v in ctx.items()):
        continue
    new = {}
    for fold, fm in r["contexts"].items():
        for h, v in fm.items():
            t = terms[h]; s1 = t["yy"] - t["v_old"]; s2 = t["yy"] - t["v_new"]
            if s1 <= 0 or s2 <= 0:                       # the estimated noise exceeds the response: undefined
                new.setdefault(fold, []).append((np.nan, np.nan)); continue
            sse = t["v_old"] + (1 - v["skill"]) * s1
            new.setdefault(fold, []).append((v["dir"] * np.sqrt(s1 / s2), 1 - (sse - t["v_new"]) / s2))
    d2 = float(np.mean([np.mean([a for a, _ in x]) for x in new.values()]))
    k2 = float(np.mean([np.mean([b for _, b in x]) for x in new.values()]))
    name = os.path.basename(f)[:-5]
    out["methods"][name] = dict(dir_old=r["overall"]["dir"], dir=d2, skill_old=r["overall"]["skill"], skill=k2,
                                pds=r["overall"]["pds"], folds={k: float(np.mean([a for a, _ in x])) for k, x in new.items()})
    print(f"  {name:32s} dir {r['overall']['dir']:+.4f} -> {d2:+.4f}   skill {r['overall']['skill']:+.4f} -> {k2:+.4f}"
          f"   pds {r['overall']['pds']:.4f}")
json.dump(out, open(f"{HERE}/results/var_scale{tag}.json", "w"), indent=1)
