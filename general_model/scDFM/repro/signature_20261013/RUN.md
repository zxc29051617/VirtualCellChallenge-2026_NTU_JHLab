# Re-run commands (from /home/zxc29051617/VCC/scDFM/repro/signature_20261013)
1. env/ = `uv venv -p 3.11 env`; `uv pip install --no-deps captum==0.8.0`; a .pth file points to scDFM/.venv site-packages (torch 2.7.1+cu126, scanpy, numpy)
2. `../../.venv/bin/python -I sample_controls.py`                 # 2000 control cells per context (needs data/stage3, data/external, VIPerturb skeleton pickle)
3. `env/bin/python compute_attributions.py`                       # SIGnature IG attributions, SCimilarity (models/scimilarity from Zenodo 15729925)
4. `../../.venv/bin/python -I cache_contexts.py`                  # per-context source tensors
5. `../../.venv/bin/python -I verify_equivalence.py`              # cache path == vcc.gm.GeneralModel
6. `../../.venv/bin/python -I weights.py`                         # groups B/C/D weights, inner CV
7. `../../.venv/bin/python -I export_effects.py`                  # A, B, C, D, A_shuf effect files
8. `../../.venv/bin/python -I evaluate_runner.py`                 # generation + official scoring (resumable, serial)
9. `../../.venv/bin/python -I analyze.py`                         # statistics + tables
