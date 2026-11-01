"""Validation clean-up after the Codex audit (2026-10-24).

    .venv/bin/python -I model/run_clean_checks.py genes  <method> [...]   (CLEAN_CENTRING=1: leave-genes-out without leakage)
    .venv/bin/python -I model/run_clean_checks.py adapt  <name>            (7-fold hidden cell types, AdaptNet beta 1, seed 0)
    .venv/bin/python -I model/run_clean_checks.py consensus <name>         (7-fold, consensus_d200)
                                                                                                (from pertrelay/)
Environment switches: STRICT_PCA=1 (context PCA from the training contexts only), RPE1=1, NEURON=outer ONLY_FOLDS=Neuron
(hidden neurons as an untouched outer test). Results go to results/<name>.json or results/genes_<method>_clean.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import bench as B

mode, names = sys.argv[1], sys.argv[2:]
if mode == "genes":
    import bench_genes as BG
    assert BG.CLEAN, "set CLEAN_CENTRING=1"
    for m in names: BG.run(f"{m}_clean", BG.METHODS[m])
elif mode == "adapt":
    import adaptnet as AN
    _C = {}

    def get(train):
        key = tuple(sorted(train))
        if key not in _C:
            _C.clear(); torch.cuda.empty_cache(); _C[key] = AN.train(list(train), steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0)
        return _C[key]
    B.run(names[0], lambda tr, h, t: AN.predict(get(tr), h, t))
elif mode == "consensus":
    B.run(names[0], B.METHODS["consensus_d200"])
else:
    raise SystemExit(__doc__)
