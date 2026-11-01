"""Do the totals choices found on H1 (no control-axis removal, target weights) also hold on other hidden cell types?
(2026-10-23)

    RPE1=1 .venv/bin/python -I model/totals_xcheck.py [HCT116,KOLF200]        (from pertrelay/)

For each hold-out with real single cells (data/stage3/loco/<X>/real.h5ad: 4,000 controls + perturbed cells), AdaptNet is
trained without the hold-out's cell-type group; its trans prediction (no own / cis part, so the comparison isolates the
trans processing) goes through tools/make_bulk_effects.py with each variant and is scored by mse_proxy on that hold-out.
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
import bench as B, adaptnet as AN, mse_proxy as MP

ROOT = "/home/zxc29051617/VCC"; OUT = f"{ROOT}/data/stage3/effects/xcheck"; os.makedirs(OUT, exist_ok=True)
GROUP = {"HCT116": ("XAtlas", "HCT116,HEK293T"), "KOLF200": ("iPSC", "KOLF")}      # benchmark fold, signature-cache contexts to exclude
VARIANTS = {"ctrl axes 50 (current)": ["--ctrl-axes", "50"], "ctrl axes 0": ["--ctrl-axes", "0"],
            "ctrl axes 50 + target weights": ["--ctrl-axes", "50", "TW"], "ctrl axes 0 + target weights": ["--ctrl-axes", "0", "TW"]}
genes = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist()
for held in (sys.argv[1] if len(sys.argv) > 1 else "HCT116,KOLF200").split(","):
    fold, excl = GROUP[held]
    d = f"{ROOT}/data/mock_loco/{held}"; os.makedirs(d, exist_ok=True)
    R = ad.read_h5ad(f"{ROOT}/data/stage3/loco/{held}/real.h5ad"); lab = R.obs["target_gene"].astype(str).values
    train = [c for c in B.CONTEXTS if c not in B.FOLDS[fold]]
    have = set().union(*(set(B.data()[t]["perts"]) for t in train))
    targets = sorted(t for t in set(lab) if t != "non-targeting" and t in have and t in set(genes))
    if not os.path.exists(f"{d}/context_H.h5ad"):
        C = R[lab == "non-targeting"].copy(); C.obs = pd.DataFrame({"target_gene": "non-targeting", "context": "H"}, index=C.obs_names)
        C.write_h5ad(f"{d}/context_H.h5ad")
        keep = np.isin(lab, targets + ["non-targeting"]); R[keep].write_h5ad(f"{d}/real.h5ad")
        pd.DataFrame({"target_gene": targets}).to_csv(f"{d}/pert_counts.csv", index=False)
        pd.DataFrame({"gene_name": genes}).to_csv(f"{d}/gene_names.csv", index=False)
    del R
    Cx = sp.csr_matrix(ad.read_h5ad(f"{d}/context_H.h5ad").X); s = np.asarray(Cx.sum(0)).ravel(); ctrl = np.log1p(5e4 * s / s.sum())
    torch.cuda.empty_cache()
    model = AN.train(train, steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0)
    T = AN.predict_ctrl(model, ctrl, targets).astype(np.float64) / np.log(2)
    G = B.genome()
    for i, p in enumerate(targets):
        g, _ = G.cis(p); T[i, g] = 0.0
        if p in G.idx: T[i, G.idx[p]] = 0.0
    full, oc = f"{OUT}/{held}_full.npz", f"{OUT}/{held}_oc.npz"
    np.savez_compressed(full, targets=np.array(targets), lfc=T.astype(np.float32)); np.savez_compressed(oc, targets=np.array(targets), lfc=np.zeros_like(T, np.float32))
    del model; torch.cuda.empty_cache()
    env = {**os.environ, "VCC_EXCLUDE": excl}
    tw = f"{OUT}/{held}_tw.npz"
    subprocess.run([f"{ROOT}/pertrelay/.venv/bin/python", "-I", f"{ROOT}/tools/target_weights.py", f"{d}/pert_counts.csv", tw, "0.5"], check=True, env=env, cwd=f"{ROOT}/pertrelay", capture_output=True)
    MP.MOCK = f"{ROOT}/data/mock_loco"                                  # Proxy reads <MOCK>/s<k>: link the hold-out there
    os.makedirs(f"{ROOT}/data/mock_loco/s_{held}", exist_ok=True)
    for f in ("real.h5ad", "context_H.h5ad"):
        if not os.path.lexists(f"{ROOT}/data/mock_loco/s_{held}/{f}"): os.symlink(f"{d}/{f}", f"{ROOT}/data/mock_loco/s_{held}/{f}")
    P = MP.Proxy(f"_{held}")
    for name, extra in VARIANTS.items():
        flags = [x for x in extra if x != "TW"] + (["--target-weights", tw] if "TW" in extra else [])
        out = f"{OUT}/{held}_" + name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
        r = subprocess.run([f"{ROOT}/pertrelay/.venv/bin/python", "-I", f"{ROOT}/tools/make_bulk_effects.py", "--full", full, "--oc", oc, "--data", d,
                            "--contexts", "H", "--match-claim", "--shared-k", "0", "--gamma", "1", "--out", out] + flags,
                           env=env, cwd=f"{ROOT}/pertrelay", capture_output=True, text=True)
        if r.returncode: print(r.stderr[-1500:]); continue
        print(f"{held:8s} {name:32s} proxy {P.raw(P.load(f'{out}_H.npz')):.4f}", flush=True)
