"""Check that the cache-based effect construction equals vcc.gm.GeneralModel on a small subset of contexts.

    python verify_equivalence.py          (pertrelay venv)
Uniform weights, contexts {CD4_Rest, CD4_Stim8hr, H1, HepG2, Jurkat, K562ess}, 120 H1-set targets, plain and shuffled.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
from vcc.contexts import load_all
from vcc.gm import GeneralModel, Genome
import sigexp as S

sub = ["CD4_Rest", "CD4_Stim8hr", "H1", "HepG2", "Jurkat", "K562ess"]
targets = json.load(open(f"{S.ROOT}/data/stage3/loco/H1/real.json"))["targets"][:120]
Cx = load_all(sub, keep_var=False)
m = GeneralModel(per_celltype=False, lam=3.0, kcap=1.0, alpha=3.0).fit(Cx, targets=targets)
ref = m.predict(targets) / np.log(2)
cache = S.load_cache(sub); gen = Genome(S.GENES)
mine = S.build_effects(cache, sub, {c: 1.0 for c in sub}, targets, gen)
d = np.abs(ref - mine); rep = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                              "ref_max_abs": float(np.abs(ref).max()), "allclose_1e-4": bool(np.allclose(ref, mine, atol=1e-4))}
# shuffled pairing: GeneralModel derangement is drawn over the requested targets
m.shuffle_trans = True; ref_s = m.predict(targets) / np.log(2)
perm = S.derangement(len(targets)); mine_s = S.build_effects(cache, sub, {c: 1.0 for c in sub}, targets, gen, shuffle_donor=[targets[j] for j in perm])
rep["shuffled_max_abs_diff"] = float(np.abs(ref_s - mine_s).max()); rep["shuffled_allclose_1e-4"] = bool(np.allclose(ref_s, mine_s, atol=1e-4))
# cis beta
rep["beta_max_abs_diff"] = float(np.abs(m.beta - S.cis_beta(cache, sub)).max())
json.dump(rep, open(f"{HERE}/verify_equivalence.json", "w"), indent=1); print(json.dumps(rep, indent=1))
assert rep["allclose_1e-4"] and rep["shuffled_allclose_1e-4"], "cache-based construction differs from GeneralModel"
