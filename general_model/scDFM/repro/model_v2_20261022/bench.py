"""Leave-one-cell-type-out benchmark for response models (2026-10-22). Fixed before any neural model is trained.

    .venv/bin/python -I repro/model_v2_20261022/bench.py [method ...]        (from scDFM/; default: zero consensus consensus_d200)

Folds: hide one cell type at a time together with its same-screen siblings (7 folds: K562 screens, X-Atlas HCT116 +
HEK293T, H1, CD4 states, HepG2, Jurkat, iPSC screens). For each hidden context, up to 500 targets (>= 50 cells, seed 0)
that at least one training context perturbed. Only trans genes are scored: the target's own gene and its cis neighbours
are excluded, as are genes unmeasured or unexpressed in the hidden context.

Metrics per hidden context (truth = the context's measured response y with sampling variance v; prediction p):
  dir    noise-corrected cosine  sum(p y) / sqrt(sum(p^2) * (sum(y^2) - sum(v)))      direction, 1 = perfect
  pds    cosine-rank discrimination among the context's evaluated targets (vcc/gm.py metrics), 0.5 = chance
  skill  1 - (sum((y - p)^2) - sum(v)) / (sum(y^2) - sum(v))                          size as given, 0 = no change
  scale  the least-squares factor that would best rescale p (1 = right size; < 1 = predicted too large)
A fold's score is the mean over its hidden contexts; the overall score is the mean over folds.
Results go to repro/model_v2_20261022/results/<method>.json.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "/home/zxc29051617/VCC/scDFM")
from vcc.gm import Genome

DATA = "/home/zxc29051617/VCC/data/stage3/model_v2"
HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = {"K562": ["K562", "K562ess", "VIPK562"], "XAtlas": ["HCT116", "HEK293T"], "H1": ["H1"],
         "CD4": ["CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr"], "HepG2": ["HepG2"], "Jurkat": ["Jurkat"],
         "iPSC": ["FengNF", "FengF", "KOLF"]}
CONTEXTS = [c for f in FOLDS.values() for c in f]
SKIP_EVAL = set(); EXTRA = set()                     # extra training-only contexts: never scored, never define test sets
if os.environ.get("ELEM") == "1":                    # ELEM-1200RS (elem_build.py)
    FOLDS["ELEM"] = ["ELEM_HAP1", "ELEM_lines"]; CONTEXTS += FOLDS["ELEM"]; SKIP_EVAL.add("ELEM"); EXTRA |= set(FOLDS["ELEM"])
if os.environ.get("RPE1") == "1":                    # Replogle RPE1 essential screen from its bulk summary (rpe1_build.py)
    FOLDS["RPE1"] = ["RPE1"]; CONTEXTS += ["RPE1"]; SKIP_EVAL.add("RPE1"); EXTRA.add("RPE1")
if os.environ.get("NEURON") == "1":                  # Tian 2021 CRISPRi in iPSC-induced neurons (sc_build.py, scPerturb)
    FOLDS["Neuron"] = ["TianNeuron2021"]; CONTEXTS += ["TianNeuron2021"]; SKIP_EVAL.add("Neuron"); EXTRA.add("TianNeuron2021")
is_extra = lambda c: c in EXTRA or c.startswith("ELEM")
OWN, KCAP, LAM, X_MIN, N_EVAL = -1.9, 1.0, 3.0, 0.25, 500

_D = {}


def data():
    """name -> dict of arrays (loaded once)."""
    if not _D:
        for n in CONTEXTS:
            z = np.load(f"{DATA}/{n}.npz", allow_pickle=True)
            d = {k: z[k] for k in z.files}; d["perts"] = d["perts"].astype(str); d["row"] = {p: j for j, p in enumerate(d["perts"])}
            _D[n] = d
    return _D


_G = {}


def genome():
    if "g" not in _G: _G["g"] = Genome(data()[CONTEXTS[0]]["genes"])
    return _G["g"]


def trans_mask(p, base):
    """base (genes usable in the hidden context) minus the target's own gene and cis neighbours."""
    G = genome(); m = base.copy(); g, _ = G.cis(p); m[g] = False
    k = G.idx.get(p)
    if k is not None: m[k] = False
    return m


def eval_targets(held, train):
    d = data()[held]; have = set().union(*(set(data()[t]["perts"]) for t in train if not is_extra(t)))   # same targets with or without extras
    cand = [p for j, p in enumerate(d["perts"]) if d["ncell"][j] >= 50 and p in have]
    rng = np.random.default_rng(0)
    return sorted(rng.choice(cand, min(N_EVAL, len(cand)), replace=False).tolist()) if cand else []


def kscale(own_obs):
    return float(np.clip(OWN / own_obs, 0.5, KCAP)) if np.isfinite(own_obs) and own_obs < -0.3 else 1.0


def consensus(train, held, targets, n0=0.0, lam=LAM):
    """The current model's trans signature: ridge mean over training contexts of kscale x y (own + cis masked)."""
    D = data(); out = np.zeros((len(targets), len(D[held]["ctrl"])), np.float32); G = genome()
    for i, p in enumerate(targets):
        s = np.zeros(out.shape[1]); w = np.zeros(out.shape[1])
        for t in train:
            j = D[t]["row"].get(p)
            if j is None: continue
            y = D[t]["Y"][j].astype(np.float64); ok = np.isfinite(y)
            wt = D[t]["ncell"][j] / (D[t]["ncell"][j] + n0) if n0 > 0 else 1.0
            s[ok] += wt * kscale(D[t]["own_obs"][j]) * y[ok]; w[ok] += wt
        out[i] = s / (w + lam)
    return out


METHODS = {"zero": lambda train, held, targets: np.zeros((len(targets), len(data()[held]["ctrl"])), np.float32),
           "consensus": lambda train, held, targets: consensus(train, held, targets),
           "consensus_d200": lambda train, held, targets: consensus(train, held, targets, n0=200.0)}


def score(held, targets, P):
    d = data()[held]; base = d["measured"] & (np.expm1(d["ctrl"]) >= X_MIN)
    rows = [d["row"][p] for p in targets]
    Y = d["Y"][rows].astype(np.float64); V = d["var"][rows].astype(np.float64); P = np.nan_to_num(P.astype(np.float64))
    M = np.stack([trans_mask(p, base) for p in targets]) & np.isfinite(Y) & np.isfinite(V)
    Y = np.where(M, Y, 0.0); V = np.where(M, V, 0.0); P = np.where(M, P, 0.0)
    signal = max((Y ** 2).sum() - V.sum(), 1e-9)
    pp, py = (P ** 2).sum(), (P * Y).sum()
    dirc = py / np.sqrt(max(pp, 1e-12) * signal)
    skill = 1 - (((Y - P) ** 2).sum() - V.sum()) / signal
    scale = py / max(pp, 1e-12)
    Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9); Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    Dm = 1 - Pn @ Yn.T; own = np.diag(Dm)[:, None]
    rank = (Dm < own).sum(1) + 0.5 * ((Dm == own).sum(1) - 1)
    pds = float(np.mean(1 - rank / max(len(targets) - 1, 1)))
    return dict(dir=float(dirc), pds=pds, skill=float(skill), scale=float(scale), n=len(targets))


def post_shared(P, k=10):
    """The pipeline's panel step: remove the panel mean and its top-k shared components from the predictions."""
    from vcc.gm import remove_shared
    return remove_shared(np.nan_to_num(P).astype(np.float32), k)


def run(name, fn, quiet=False, post=None):
    """post: optional function applied to each hidden context's prediction panel before scoring (e.g. post_shared)."""
    t0 = time.time(); res = {}
    for fold, held_list in FOLDS.items():
        if fold in SKIP_EVAL: continue
        train = [c for c in CONTEXTS if c not in held_list]
        for held in held_list:
            targets = eval_targets(held, train)
            if len(targets) < 20: continue
            P = fn(train, held, targets)
            res.setdefault(fold, {})[held] = score(held, targets, post(P) if post else P)
    fold_mean = {f: {k: float(np.mean([r[k] for r in v.values()])) for k in ("dir", "pds", "skill", "scale")} for f, v in res.items()}
    overall = {k: float(np.mean([fm[k] for fm in fold_mean.values()])) for k in ("dir", "pds", "skill", "scale")}
    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump(dict(method=name, overall=overall, folds=fold_mean, contexts=res), open(f"{HERE}/results/{name}.json", "w"), indent=1)
    if not quiet:
        print(f"{name}: overall dir {overall['dir']:+.4f}  pds {overall['pds']:.4f}  skill {overall['skill']:+.4f}  scale {overall['scale']:.2f}  ({time.time() - t0:.0f}s)")
        for f, fm in fold_mean.items():
            print(f"    {f:7s} dir {fm['dir']:+.4f}  pds {fm['pds']:.4f}  skill {fm['skill']:+.4f}  scale {fm['scale']:.2f}  "
                  + " ".join(f"{h}:{r['n']}" for h, r in res[f].items()), flush=True)
    return overall


if __name__ == "__main__":
    for m in sys.argv[1:] or ["zero", "consensus", "consensus_d200"]:
        run(m, METHODS[m])
