"""Effect files for groups A (uniform), B, C, D and A_shuf (uniform weights, targets paired with another target's
trans signature), for each outer hold-out. Everything but the source weights is identical across groups.

    python export_effects.py [outer ...]      (scDFM venv)
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM")
from vcc.gm import Genome
import sigexp as S

os.makedirs(f"{HERE}/effects", exist_ok=True)
C = S.load_cache(); gen = Genome(S.GENES); man = {}
for name in sys.argv[1:] or list(S.OUTER):
    o = S.OUTER[name]; W = json.load(open(f"{HERE}/weights_{name}.json")); train = W["train"]
    targets = json.load(open(f"{S.ROOT}/data/stage3/loco/{o['loco']}/real.json"))["targets"]
    perm = S.derangement(len(targets)); donors = [targets[j] for j in perm]
    man[name] = {"train": train, "excluded": o["exclude"], "n_targets": len(targets)}
    for group in ["A", "B", "C", "D", "A_shuf"]:
        w = W["A" if group == "A_shuf" else group]["weights"]
        lfc = S.build_effects(C, train, w, targets, gen, shuffle_donor=donors if group == "A_shuf" else None)
        np.savez_compressed(f"{HERE}/effects/{name}_{group}.npz", targets=np.array(targets), lfc=lfc.astype(np.float32))
        wv = np.array([w[c] for c in train]); eff_n = float(wv.sum() ** 2 / (wv ** 2).sum())
        man[name][group] = {"effective_n_sources": round(eff_n, 2), "max_weight": round(float(wv.max()), 2), "mean_abs_lfc_trans": float(np.abs(lfc).mean())}
        print(name, group, "eff_n", round(eff_n, 2), "max w", round(float(wv.max()), 2), flush=True)
json.dump(man, open(f"{HERE}/effects_manifest.json", "w"), indent=1)
