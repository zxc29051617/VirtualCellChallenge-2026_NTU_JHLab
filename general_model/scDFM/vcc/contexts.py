"""All training contexts (cell line / condition) as pseudobulk responses in the scorer's expression space.

The vcc2026 scorer compares groups in `log1p(50,000 * group gene count sum / group total count)` (cell-eval2
bulk_lognorm comparator). For every context and perturbation group with enough cells this module gives
  delta = e(perturbed group) - e(controls)      on the VCC gene axis (NaN where the source did not measure),
  var   = its Poisson sampling variance          (delta method: 1/N_pert,g + 1/N_ctrl,g in counts, damped),
  kd    = the target gene's own delta (knockdown), when the target is a measured VCC gene.

    from vcc.contexts import load_all
    C = load_all()            # dict name -> Context
"""
import os
from dataclasses import dataclass
import numpy as np

from vcc.stats import load as load_stats

ROOT = "/home/zxc29051617/VCC/data/stage3"
BULK = 50_000.0
# name -> (stats file, kind). kind: 'vcc' = targets + auxiliary genes only, 'all' = every perturbation of the screen
CONTEXTS = {
    "K562": ("stats/K562", "vcc"), "HCT116": ("stats/HCT116", "vcc"), "HEK293T": ("stats/HEK293T", "vcc"),
    "H1": ("stats/H1", "vcc"),
    "CD4_Rest": ("stats/CD4_Rest", "vcc"), "CD4_Stim8hr": ("stats/CD4_Stim8hr", "vcc"),
    "CD4_Stim48hr": ("stats/CD4_Stim48hr", "vcc"),
    "K562ess": ("stats_ext/K562ess", "all"), "HepG2": ("stats_ext/HepG2", "all"), "Jurkat": ("stats_ext/Jurkat", "all"),
    "FengNF": ("stats_ext/FengNF", "all"), "FengF": ("stats_ext/FengF", "all"),
    "KOLF": ("stats_ext/KOLF", "all"), "TeloHAEC": ("stats_ext/TeloHAEC", "all"),
    "VIPK562": ("stats_ext/VIPK562", "all"),          # VIPerturb-seq K562, 10x Flex
}
OPT_IN = {"TeloHAEC": "TELO"}   # sources added after gm_final_v5: used only with their flag = 1, so v5 rebuilds unchanged


@dataclass
class Context:
    name: str
    genes: np.ndarray          # VCC gene axis
    measured: np.ndarray       # bool per gene
    ctrl: np.ndarray           # control expression e(ctrl), per gene
    ctrl_counts: np.ndarray    # summed control counts per gene
    ctrl_cells: int
    perts: np.ndarray          # perturbation names (target gene symbols)
    ncell: np.ndarray          # cells per perturbation
    delta: np.ndarray          # perts x genes, float32, NaN where unmeasured
    var: np.ndarray            # perts x genes, float32
    kd: np.ndarray             # own-gene delta per perturbation (NaN if the target is not a measured VCC gene)


def lognorm(sums):
    tot = sums.sum(-1, keepdims=True)
    return np.log1p(BULK * sums / np.maximum(tot, 1))


def build(name, st, min_cells=20, keep_var=True):
    names = list(st["names"]); i_c = names.index("__ctrl__")
    genes = np.asarray(st["genes"]).astype(str); meas = np.asarray(st["measured"], bool)
    sums = np.asarray(st["sums"], np.float64); ncell = np.asarray(st["ncell"])
    keep = [i for i, n in enumerate(names) if i != i_c and not n.startswith("__") and ncell[i] >= min_cells]
    e = lognorm(sums[keep]); c = lognorm(sums[i_c][None])[0]
    delta = (e - c).astype(np.float32); delta[:, ~meas] = np.nan
    # sampling variance of log1p(B x/T) for Poisson counts x: ((B/T)/(1 + B x/T))^2 * x, per side
    def v(s):
        tot = np.maximum(s.sum(-1, keepdims=True), 1); return (BULK / tot) ** 2 * s / (1 + BULK * s / tot) ** 2
    var = (v(sums[keep]) + v(sums[i_c][None])).astype(np.float32) if keep_var else None
    gi = {g: k for k, g in enumerate(genes)}
    perts = np.array([names[i] for i in keep])
    kd = np.array([delta[j, gi[p]] if p in gi and meas[gi[p]] else np.nan for j, p in enumerate(perts)], np.float32)
    return Context(name, genes, meas, c.astype(np.float32), sums[i_c].astype(np.float32), int(ncell[i_c]),
                   perts, ncell[keep], delta, var, kd)


def load(name, min_cells=20, keep_var=True):
    path, _ = CONTEXTS[name]
    os.environ.setdefault("STATS_DIR", f"{ROOT}/stats")
    z = np.load(f"{ROOT}/{path}.npz", allow_pickle=True)
    return build(name, {k: z[k] for k in z.files}, min_cells, keep_var)


def available():
    return [n for n, (p, _) in CONTEXTS.items() if os.path.exists(f"{ROOT}/{p}.npz") and (n not in OPT_IN or os.environ.get(OPT_IN[n]) == "1")]


def load_all(names=None, min_cells=20, keep_var=True):
    return {n: load(n, min_cells, keep_var) for n in (names or available())}
