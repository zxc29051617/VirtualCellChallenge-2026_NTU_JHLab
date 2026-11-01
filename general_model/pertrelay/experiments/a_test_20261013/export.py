"""Option A test: effects for base vs sparse-trans (top-200 at alpha 3, rest at 0.5), each with a shuffled control,
from ONE fit per held-out set (all training contexts except the held-out cell type). Copies per generation seed."""
import json, shutil, sys
sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
import numpy as np
from vcc.contexts import load_all
from vcc.gm import GeneralModel, GROUPS
E = "/home/zxc29051617/VCC/data/stage3/effects"
SETS = {"H1": ("H1", "/home/zxc29051617/VCC/data/stage3/loco/H1/real.json"),
        "KOLFbig": ("KOLF", "/home/zxc29051617/VCC/data/stage3/loco/KOLFbig/real.json")}
VARIANTS = {"base": dict(topk=0), "A": dict(topk=200, alpha_rest=0.5)}
C = load_all(keep_var=False)
man = {}
for set_, (held, rj) in SETS.items():
    grp = next(g for g in GROUPS if held in g); train = {k: v for k, v in C.items() if k not in grp}
    targets = json.load(open(rj))["targets"]
    m = GeneralModel(per_celltype=False, lam=3.0, kcap=1.0, alpha=3.0).fit(train, targets=targets)
    man[set_] = {"held_out_group": grp, "training": sorted(train), "n_targets": len(targets)}
    for v, kw in VARIANTS.items():
        for shuf in (False, True):
            m.topk, m.alpha_rest = kw.get("topk", 0), kw.get("alpha_rest")
            m.shuffle_trans = shuf
            name = f"{set_}_at_{v}_{'shuf' if shuf else 'plain'}"
            lfc = m.predict(list(targets)) / np.log(2)
            np.savez_compressed(f"{E}/{name}_s0.npz", targets=np.array(targets), lfc=lfc.astype(np.float32))
            for s in (1, 2): shutil.copy(f"{E}/{name}_s0.npz", f"{E}/{name}_s{s}.npz")
            print("wrote", name, flush=True)
json.dump(man, open("/home/zxc29051617/VCC/pertrelay/experiments/a_test_20261013/manifest.json", "w"), indent=1)
