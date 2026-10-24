"""Totals post-processing sweep for the trained-model totals on the mock subsets, scored with mse_proxy (2026-10-23).

    .venv/bin/python -I repro/model_v2_20261022/totals_sweep.py [subsets]        (from scDFM/)

Runs tools/make_bulk_effects.py on each subset's hybrid inputs (mock_hybrid_rpe1_s<k>_d200 / _oc) with the variants
below and prints the proxy raw mse (lower is better; official = proxy - ~0.05). pds is NOT covered: candidates that
gain here must still pass the official mock exam.
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mse_proxy import Proxy
ROOT = "/home/zxc29051617/VCC"; EFF = f"{ROOT}/data/stage3/effects"; OUT = f"{EFF}/sweep"
VARIANTS = {"current (k0 ax50 g1)": [], "ctrl axes 0": ["--ctrl-axes", "0"], "ctrl axes 20": ["--ctrl-axes", "20"],
            "ctrl axes 100": ["--ctrl-axes", "100"], "shared-k 3": ["--shared-k", "3"], "gamma 1.5": ["--gamma", "1.5"],
            "entry agree 0.5": ["--entry-agree", "0.5"], "entry agree 1": ["--entry-agree", "1"], "target weights 0.5": ["TW", "0.5"]}
if os.environ.get("SWEEP") == "tw":                    # round 2 (after v4): around CTRLAXES=0 + target weights
    VARIANTS = {"current (k0 ax50 g1)": [], "ax0 tw0.5 (v4)": ["--ctrl-axes", "0", "TW", "0.5"], "ax0 tw0.25": ["--ctrl-axes", "0", "TW", "0.25"],
                "ax0 tw0.75": ["--ctrl-axes", "0", "TW", "0.75"], "ax0 tw1.0": ["--ctrl-axes", "0", "TW", "1.0"],
                "ax0 tw0.5 gamma1.5": ["--ctrl-axes", "0", "TW", "0.5", "--gamma", "1.5"], "ax10 tw0.5": ["--ctrl-axes", "10", "TW", "0.5"]}
PREFIX = os.environ.get("PREFIX", "mock_hybrid_rpe1")
subsets = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4").split(",")]
env = {**os.environ, "VCC_EXCLUDE": "H1"}
res = {v: [] for v in VARIANTS}
for k in subsets:
    P = Proxy(k); pre = f"{EFF}/{PREFIX}_s{k}"; d = f"{ROOT}/data/mock_h1/s{k}"
    for name, extra in VARIANTS.items():
        args = {"--shared-k": "0", "--ctrl-axes": "50", "--gamma": "1"}
        flags = []
        it = iter(extra)
        for a in it:
            if a == "TW":
                pw = next(it); tw = f"{OUT}/tw_s{k}_p{pw}.npz"
                if not os.path.exists(tw):
                    subprocess.run([f"{ROOT}/scDFM/.venv/bin/python", "-I", f"{ROOT}/tools/target_weights.py", f"{pre}.npz", tw, pw], check=True, env=env, cwd=f"{ROOT}/scDFM", capture_output=True)
                flags += ["--target-weights", tw]
            elif a in args: args[a] = next(it)
            else: flags += [a, next(it)]
        out = f"{OUT}/{PREFIX}_s{k}_" + name.split(" (")[0].replace(" ", "_")
        cmd = [f"{ROOT}/scDFM/.venv/bin/python", "-I", f"{ROOT}/tools/make_bulk_effects.py", "--full", f"{pre}_d200.npz", "--oc", f"{pre}_oc.npz",
               "--data", d, "--contexts", "H", "--match-claim", "--out", out] + [x for kv in args.items() for x in kv] + flags
        r = subprocess.run(cmd, env=env, cwd=f"{ROOT}/scDFM", capture_output=True, text=True)
        if r.returncode: print(r.stderr[-1500:]); continue
        v = P.raw(P.load(f"{out}_H.npz")); res[name].append(v)
        print(f"s{k} {name:22s} proxy {v:.4f}", flush=True)
print("\nmean over subsets " + ",".join(map(str, subsets)))
base = np.mean(res["current (k0 ax50 g1)"])
for name, v in res.items():
    if v: print(f"  {name:22s} {np.mean(v):.4f} ({np.mean(v) - base:+.4f})")
