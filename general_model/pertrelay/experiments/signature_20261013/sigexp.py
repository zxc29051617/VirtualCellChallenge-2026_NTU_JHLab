"""Shared pieces of the source-weight experiment: cache access, effect construction, features, weight learning.

Effect construction mirrors vcc/gm.py (GeneralModel with per_celltype=False, lam=3, kcap=1, alpha=3, own=-1.9,
use_cis, use_trans) but takes per-context weights; equivalence with GeneralModel is asserted in verify_equivalence.py.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"
CACHE = f"{HERE}/cache"
OWN, ALPHA, LAM = -1.9, 3.0, 3.0
GENES = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); G = len(GENES)
GI = {g: i for i, g in enumerate(GENES)}
X_MIN = 0.25                                           # CP50k, as in vcc/gm.py
CONTEXTS = ["K562", "HCT116", "HEK293T", "H1", "CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr", "K562ess", "HepG2", "Jurkat",
            "FengNF", "FengF", "KOLF", "VIPK562"]
# exclusion groups: same cell line / same screen / related donor background never serve each other
GROUPS = [["K562", "K562ess", "VIPK562"], ["HCT116", "HEK293T"], ["H1"], ["CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr"],
          ["HepG2"], ["Jurkat"], ["FengNF", "FengF", "KOLF"]]
GROUP_OF = {c: i for i, g in enumerate(GROUPS) for c in g}
SC_CONTEXTS = ["H1", "HCT116", "KOLF", "HEK293T", "K562", "K562ess", "HepG2", "Jurkat", "VIPK562"]   # have single-cell controls
OUTER = {"H1": {"held": ["H1"], "exclude": ["H1"], "eval_set": "H1", "loco": "H1"},
         "HCT116": {"held": ["HCT116"], "exclude": ["HCT116", "HEK293T"], "eval_set": "HCT116", "loco": "HCT116"},
         "KOLF200": {"held": ["KOLF"], "exclude": ["KOLF", "FengNF", "FengF"], "eval_set": "KOLF200", "loco": "KOLF200"}}


def load_cache(names=None):
    out = {}
    for n in names or CONTEXTS:
        z = np.load(f"{CACHE}/{n}.npz", allow_pickle=True)
        out[n] = {k: z[k] for k in z.files}
        out[n]["tidx"] = {t: i for i, t in enumerate(out[n]["targets"])}
    return out


def target_list():
    return json.load(open(f"{CACHE}/targets.json"))["T"]


def cis_beta(C, train):
    num = sum(C[c]["cis_num"] for c in train); den = sum(C[c]["cis_den"] for c in train)
    return num / np.maximum(den, 1e-9)


def trans_signature(C, train, w, p, lam=LAM):
    """sig[g] = sum_c w_c Ysrc[c,p,g] / (sum_c w_c [finite] + lam) over training contexts that have target p."""
    num = np.zeros(G); den = np.zeros(G)
    for c in train:
        i = C[c]["tidx"].get(p)
        if i is None: continue
        y = C[c]["Ysrc"][i]; ok = np.isfinite(y)
        num[ok] += w[c] * y[ok]; den[ok] += w[c]
    return (num / (den + lam)).astype(np.float32)


def build_effects(C, train, w, targets, genome, shuffle_donor=None, alpha=ALPHA, use_trans=True):
    """log2 fold changes [len(targets), G]. `genome` = vcc.gm.Genome. shuffle_donor: list of donor targets (pairing control)."""
    from vcc.gm import CIS_BINS
    beta = cis_beta(C, train); out = np.zeros((len(targets), G), np.float32)
    for i, p in enumerate(targets):
        if use_trans:
            donor = shuffle_donor[i] if shuffle_donor is not None else p
            out[i] = alpha * trans_signature(C, train, w, donor)
        g, b = genome.cis(p)
        out[i, g] += OWN * beta[b]
        k = GI.get(p)
        if k is not None: out[i, k] = OWN
    return out / np.log(2)


def derangement(n, seed=0):
    """Same construction as vcc/gm.py shuffle_trans (default_rng(0), re-drawn until no fixed point)."""
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    while (perm == np.arange(n)).any(): perm = rng.permutation(n)
    return perm
