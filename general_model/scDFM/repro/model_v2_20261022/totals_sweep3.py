"""Totals variants scored by BOTH proxies (mse and pds, both reproduce the official values from the totals alone) on all
three mock sets (2026-10-23 evening): does v4's processing (no control axes, target weights, no shared removal) cost pds on
some cell types?

    .venv/bin/python -I repro/model_v2_20261022/totals_sweep3.py [sets, default h1,hct116,kolf200]        (from scDFM/)

Inputs per set and subset: the v4 mock run's d200 / oc files (AdaptNet 3-seed totals input, consensus own + cis) and the
old pipeline's totals ("current", consensus G3c config) for reference.
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mse_proxy as MP

ROOT = "/home/zxc29051617/VCC"; EFF = f"{ROOT}/data/stage3/effects"; OUT = f"{EFF}/sweep3"; os.makedirs(OUT, exist_ok=True)
SETS = {"h1": ("H1", "mock_hyb_ax0_tw_ens3", "mock_current", 5), "hct116": ("HCT116,HEK293T", "mock_hct116_v4", "mock_hct116_current", 4),
        "kolf200": ("FengNF,FengF,KOLF", "mock_kolf200_v4", "mock_kolf200_current", 4)}
if os.environ.get("SWEEP") == "axes":                 # round 2: number of control axes only
    VARIANTS_ = {"v4 (ax0 tw0.5 k0)": ["--ctrl-axes", "0", "TW"], "ax5 tw0.5": ["--ctrl-axes", "5", "TW"], "ax10 tw0.5": ["--ctrl-axes", "10", "TW"],
                 "ax20 tw0.5": ["--ctrl-axes", "20", "TW"], "ax30 tw0.5": ["--ctrl-axes", "30", "TW"]}
VARIANTS = {"v4 (ax0 tw0.5 k0)": ["--ctrl-axes", "0", "TW"], "ax50 tw0.5": ["--ctrl-axes", "50", "TW"], "ax20 tw0.5": ["--ctrl-axes", "20", "TW"],
            "ax0 tw0.5 shared3": ["--ctrl-axes", "0", "TW", "--shared-k", "3"], "ax0 tw0.5 shared10": ["--ctrl-axes", "0", "TW", "--shared-k", "10"],
            "ax0 no tw": ["--ctrl-axes", "0"], "ax50 tw0.5 shared10": ["--ctrl-axes", "50", "TW", "--shared-k", "10"]}
if os.environ.get("SWEEP") == "axes": VARIANTS = VARIANTS_
for st in (sys.argv[1] if len(sys.argv) > 1 else "h1,hct116,kolf200").split(","):
    excl, pre_v, pre_a, nk = SETS[st]
    MP.MOCK = f"{ROOT}/data/mock_{st}"
    res = {"old pipeline (current)": {"mse": [], "pds": []}} | {v: {"mse": [], "pds": []} for v in VARIANTS}
    for k in range(nk):
        if not os.path.exists(f"{EFF}/{pre_v}_s{k}_d200.npz"): continue
        P = MP.Proxy(k); pre = f"{EFF}/{pre_v}_s{k}"
        A = P.load(f"{EFF}/{pre_a}_s{k}_bulk_g1_H.npz"); res["old pipeline (current)"]["mse"].append(P.raw(A)); res["old pipeline (current)"]["pds"].append(P.pds(A))
        for name, extra in VARIANTS.items():
            flags = []; it = iter(extra); args = {"--shared-k": "0", "--gamma": "1"}
            for a in it:
                if a == "TW": flags += ["--target-weights", f"{pre}_tw.npz"]
                elif a in args: args[a] = next(it)
                else: flags += [a, next(it)]
            out = f"{OUT}/{st}_s{k}_" + name.split(" (")[0].replace(" ", "_")
            r = subprocess.run([f"{ROOT}/scDFM/.venv/bin/python", "-I", f"{ROOT}/tools/make_bulk_effects.py", "--full", f"{pre}_d200.npz", "--oc", f"{pre}_oc.npz",
                                "--data", f"{ROOT}/data/mock_{st}/s{k}", "--contexts", "H", "--match-claim", "--out", out] + [x for kv in args.items() for x in kv] + flags,
                               env={**os.environ, "VCC_EXCLUDE": excl}, cwd=f"{ROOT}/scDFM", capture_output=True, text=True)
            if r.returncode: print(r.stderr[-800:]); continue
            L = P.load(f"{out}_H.npz"); res[name]["mse"].append(P.raw(L)); res[name]["pds"].append(P.pds(L))
        print(f"{st} s{k} done", flush=True)
    print(f"\n{st}: mean over subsets (mse lower = better, pds higher = better)")
    for name, d in res.items():
        if d["mse"]: print(f"  {name:26s} mse {np.mean(d['mse']):.4f}  pds {np.mean(d['pds']):.4f}", flush=True)
