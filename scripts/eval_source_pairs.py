"""Agreement between every pair of perturbation screens, and whether a screen helps a blend.

Generalises eval_third_source.py to any number of screens. Each comparison uses the genes and
targets only the screens involved share, so a screen with a small gene axis does not shrink the
others' comparisons. Scored as the Challenge scores discrimination: is each target's effect in one
screen closest, by cosine, to the same target's effect in the other, among all targets, with every
target's own gene removed (0.5 is chance). `r` compares matched targets with mismatched ones.

    python scripts/eval_source_pairs.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.io import read_csv_column
from vcc2026.local_eval import _own_gene_columns, perturbation_discrimination
from vcc2026.transfer import secondary_fold_changes, source_effect_table

P = Path("external/profiles")
SCREENS = {
    "K562": P / "K562gw_perturbation_means.npz",
    "CD4": P / "CD4_target_lfc.npz",
    "HCT116": P / "HCT116_target_lfc.npz",
    "hPSC": P / "hPSC_target_lfc.npz",
}
TARGETS = read_csv_column("dataset/pert_counts.csv", expected_header="target_gene")


def genes_of(name: str) -> list[str]:
    if name == "K562":
        return source_effect_table(SCREENS[name])[2]
    return [str(g) for g in np.load(SCREENS[name], allow_pickle=True)["genes"]]


_k562 = None


def effects(name: str, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    global _k562
    if name != "K562":
        fc, covered, _ = secondary_fold_changes(SCREENS[name], TARGETS, genes)
        return fc, covered
    if _k562 is None:
        _k562 = source_effect_table(SCREENS["K562"])
    labels, table, source_genes = _k562
    last = {g: i for i, g in enumerate(source_genes)}
    row = {t: i for i, t in enumerate(labels)}
    fc = np.zeros((len(TARGETS), len(genes)), dtype=np.float32)
    covered = np.array([t in row for t in TARGETS])
    cols = [last[g] for g in genes]
    for i, t in enumerate(TARGETS):
        if covered[i]:
            fc[i] = table[row[t]][cols]
    return fc, covered


def shared_axis(names: list[str]) -> list[str]:
    sets = [set(genes_of(n)) for n in names]
    return [g for g in dict.fromkeys(genes_of(names[0])) if all(g in s for s in sets)]


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a @ b.T) / np.maximum(np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :], 1e-12)


def main() -> int:
    print("coverage of the 300 targets:")
    for n in SCREENS:
        print(f"  {n:<8}{int(effects(n, genes_of(n)[:50])[1].sum()):>4}")

    print(f"\n{'pair (A predicts B)':<22}{'genes':>7}{'targets':>9}{'PDS':>8}{'r matched':>11}{'r mismatched':>14}")
    for a, b in itertools.permutations(SCREENS, 2):
        if a > b:
            continue
        axis = shared_axis([a, b])
        (fa, ca), (fb, cb) = effects(a, axis), effects(b, axis)
        rows = np.flatnonzero(ca & cb)
        own = _own_gene_columns([TARGETS[i] for i in rows], axis)
        pds = perturbation_discrimination(fa[rows], fb[rows], own_columns=own)
        c = corr_rows(fa[rows], fb[rows])
        print(f"{a + ' ~ ' + b:<22}{len(axis):>7,}{len(rows):>9}{pds:>8.4f}"
              f"{np.diag(c).mean():>+11.4f}{(c.sum() - np.trace(c)) / (c.size - len(rows)):>+14.4f}")

    rng = np.random.default_rng(0)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt((x**2).mean()))

    print("\nadding a screen to a base, predicting a held-out third (PDS; shuffled = extra attached to wrong targets)")
    print(f"{'truth':<8}{'base':<8}{'extra':<8}{'targets':>8}{'w=0':>8}{'w=.1':>8}{'w=.25':>8}{'w=.5':>8}{'shuf@.25':>10}")
    for extra in ("hPSC", "HCT116"):
        for truth, base in itertools.permutations([n for n in SCREENS if n != extra], 2):
            axis = shared_axis([truth, base, extra])
            (ft, ct), (fb, cb), (fe, ce) = (effects(n, axis) for n in (truth, base, extra))
            rows = np.flatnonzero(ct & cb & ce)
            own = _own_gene_columns([TARGETS[i] for i in rows], axis)
            y, a = ft[rows], fb[rows]
            b = fe[rows] * (rms(a) / rms(fe[rows]))
            vals = [perturbation_discrimination((1 - w) * a + w * b, y, own_columns=own) for w in (0, 0.1, 0.25, 0.5)]
            sh = np.mean([perturbation_discrimination(0.75 * a + 0.25 * b[rng.permutation(len(b))], y, own_columns=own)
                          for _ in range(20)])
            print(f"{truth:<8}{base:<8}{extra:<8}{len(rows):>8}" + "".join(f"{v:>8.4f}" for v in vals) + f"{sh:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
