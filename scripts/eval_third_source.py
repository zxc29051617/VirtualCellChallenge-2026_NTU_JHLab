"""Does HCT116 (X-Atlas/Orion) carry target-specific signal that agrees with K562 and CD4?

Three independent measurements of the same knockdowns in three cell types (K562 leukemia,
primary CD4 T cells, HCT116 colorectal). For each pair, over targets both measured:

  PDS     is each target's effect in one screen closest, by cosine, to that same target's effect
          in the other, among all targets (own genes of every target removed; 0.5 is chance)
  r       mean correlation of matched targets against mismatched ones

Then the question that matters for the pipeline: does adding HCT116 to K562 improve prediction of
a third, held-out screen, against a control that attaches HCT116's numbers to the wrong targets.

    python scripts/eval_third_source.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.io import read_csv_column
from vcc2026.local_eval import _own_gene_columns, perturbation_discrimination
from vcc2026.transfer import secondary_fold_changes, source_effect_table

P = Path("external/profiles")
K562 = P / "K562gw_perturbation_means.npz"
CD4 = P / "CD4_target_lfc.npz"
HCT = P / "HCT116_target_lfc.npz"


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a @ b.T) / np.maximum(np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :], 1e-12)


def main() -> int:
    targets_all = read_csv_column("dataset/pert_counts.csv", expected_header="target_gene")
    labels, effects, k_genes = source_effect_table(K562)
    cd4_genes = [str(g) for g in np.load(CD4, allow_pickle=True)["genes"]]
    hct_genes = [str(g) for g in np.load(HCT, allow_pickle=True)["genes"]]
    shared = [g for g in dict.fromkeys(k_genes) if g in set(cd4_genes) and g in set(hct_genes)]
    last = {g: i for i, g in enumerate(k_genes)}
    row_k = {t: i for i, t in enumerate(labels)}

    cd4_fc, cd4_cov, _ = secondary_fold_changes(CD4, targets_all, shared)
    hct_fc, hct_cov, _ = secondary_fold_changes(HCT, targets_all, shared)
    k_cov = np.array([t in row_k for t in targets_all])
    k_fc = np.zeros_like(cd4_fc)
    for i, t in enumerate(targets_all):
        if k_cov[i]:
            k_fc[i] = effects[row_k[t]][[last[g] for g in shared]]
    print(f"shared gene axis: {len(shared):,} genes | targets covered: K562gw {k_cov.sum()}, "
          f"CD4 {cd4_cov.sum()}, HCT116 {hct_cov.sum()}, all three {(k_cov & cd4_cov & hct_cov).sum()}\n")

    sources = {"K562": (k_fc, k_cov), "CD4": (cd4_fc, cd4_cov), "HCT116": (hct_fc, hct_cov)}

    def pds_pair(a: str, b: str) -> tuple[int, float, float, float]:
        (fa, ca), (fb, cb) = sources[a], sources[b]
        rows = np.flatnonzero(ca & cb)
        tg = [targets_all[i] for i in rows]
        own = _own_gene_columns(tg, shared)
        pds = perturbation_discrimination(fa[rows], fb[rows], own_columns=own)
        c = corr_rows(fa[rows], fb[rows])
        matched = float(np.diag(c).mean())
        mismatched = float((c.sum() - np.trace(c)) / (c.size - len(rows)))
        return len(rows), pds, matched, mismatched

    print("pairwise agreement between screens (A predicts B)")
    print(f"{'pair':<16}{'targets':>8}{'PDS':>8}{'r matched':>11}{'r mismatched':>14}")
    for a, b in [("HCT116", "K562"), ("HCT116", "CD4"), ("K562", "CD4"), ("CD4", "K562")]:
        n, pds, m, mm = pds_pair(a, b)
        print(f"{a + ' -> ' + b:<16}{n:>8}{pds:>8.4f}{m:>+11.4f}{mm:>+14.4f}")

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt((x**2).mean()))

    rng = np.random.default_rng(0)

    def blend_test(truth: str, base: str, extra: str) -> None:
        both = np.flatnonzero(sources[truth][1] & sources[base][1] & sources[extra][1])
        tg = [targets_all[i] for i in both]
        own = _own_gene_columns(tg, shared)
        y = sources[truth][0][both]
        a = sources[base][0][both]
        b = sources[extra][0][both] * (rms(a) / rms(sources[extra][0][both]))
        print(f"\nheld-out truth {truth} ({len(both)} targets): {base} alone, plus {extra} at weight w")
        print(f"{'w':>6}{'PDS':>9}{'PDS if '+extra+' is shuffled':>34}")
        for w in (0.0, 0.1, 0.2, 0.35, 0.5):
            real = perturbation_discrimination((1 - w) * a + w * b, y, own_columns=own)
            sh = np.mean([perturbation_discrimination((1 - w) * a + w * b[rng.permutation(len(b))], y, own_columns=own)
                          for _ in range(20)])
            print(f"{w:>6.2f}{real:>9.4f}{sh:>34.4f}")

    blend_test("CD4", "K562", "HCT116")
    blend_test("K562", "CD4", "HCT116")
    blend_test("HCT116", "K562", "CD4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
