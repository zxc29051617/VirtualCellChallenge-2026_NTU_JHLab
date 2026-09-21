"""Price the gene-selection rule, and check the virtual-dataset source path resolves."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vcc2026.io import control_gene_profile, select_within_budget  # noqa: E402
from vcc2026.baseline import generic_perturbation_response  # noqa: E402
from vcc2026.io import read_csv_column, read_name_list  # noqa: E402
from vcc2026.transfer import source_fold_changes  # noqa: E402

TS = 5e4


def spec_effect(profiles, ctrl):
    p = profiles / profiles.sum(axis=1, keepdims=True)
    c = ctrl / ctrl.sum()
    return np.log1p(TS * p) - np.log1p(TS * c)[None, :]


def quantize(v, k):
    fl = np.floor(v)
    return fl + np.minimum(np.rint((v - fl) * k), k) / k


def main() -> None:
    genes = read_name_list(ROOT / "dataset/gene_names.csv")
    targets = read_csv_column(ROOT / "dataset/pert_counts.csv", expected_header="target_gene")
    ctrl, lib, _ = control_gene_profile(ROOT / "dataset/context_A.h5ad")
    ctrl = ctrl.astype(np.float64)
    lfc, _ = source_fold_changes(
        ROOT / "external/profiles/K562gw_perturbation_means.npz", targets, genes
    )
    lfc = lfc.astype(np.float64)
    fb = generic_perturbation_response(
        [ROOT / "external/profiles/K562_perturbation_means.npz",
         ROOT / "external/profiles/RPE1_perturbation_means.npz"], genes)
    lfc[~lfc.any(axis=1)] = fb
    full = ctrl[None, :] * np.exp(lfc * 0.3)

    cpm = ctrl * (1e6 / ctrl.sum())
    de_set = np.flatnonzero(cpm >= 5.0)
    expressed = np.flatnonzero(ctrl > 0)

    print(f"mean library {lib:,.0f}   DE-scored {de_set.size:,}   control-expressed {expressed.size:,}")
    print(f"{'selection':<26}{'scale':>6}{'K':>5}{'genes':>8}{'nonzero':>9}"
          f"{'entries/cell':>14}{'prep GB':>9}{'|shared|':>10}{'ratio':>8}")
    for label, cols in (("5 CPM DE set (HEAD)", de_set), ("every expressed gene", expressed)):
        for ps in (0.15, 0.30, 0.50):
            for k in (200, 400):
                sub = np.zeros_like(full)
                sub[:, cols] = quantize(full[:, cols] * ps, k)
                nz = int(np.count_nonzero(sub.sum(axis=0)))
                per_cell = float(np.minimum(full[0, cols] * ps, 1.0).sum())
                eff = spec_effect(sub, ctrl)
                s = eff.mean(axis=0)
                sn = float(np.linalg.norm(s))
                pn = float(np.linalg.norm(eff - s, axis=1).mean())
                print(f"{label:<26}{ps:>6.2f}{k:>5}{cols.size:>8,}{nz:>9,}"
                      f"{per_cell:>14,.0f}{per_cell * 360_000 * 12 / 1e9:>9.1f}"
                      f"{sn:>10.3f}{sn / pn:>8.2f}")

    print("\nvirtual-dataset source path")
    path = ROOT / "artifacts/transfer_prediction.h5ad"
    if not path.exists():
        print("  artifacts/transfer_prediction.h5ad missing")
        return
    with h5py.File(path, "r") as h:
        x = h["X"]
        print(f"  X/data is virtual: {x['data'].is_virtual}")
        if x["data"].is_virtual:
            for vs in x["data"].virtual_sources():
                print(f"  source file recorded as: {vs.file_name!r}")
                break
    for cwd in (str(ROOT), str(Path(os.environ.get('TEMP', 'C:/'))), "C:/"):
        try:
            os.chdir(cwd)
            with h5py.File(path, "r") as h:
                head = h["X/data"][0:200_000]
            print(f"  read from cwd={cwd!r}: first 200k values sum to {head.sum():,.1f},"
                  f" {np.count_nonzero(head):,} nonzero")
        except Exception as exc:  # noqa: BLE001
            print(f"  read from cwd={cwd!r}: {type(exc).__name__}: {exc}")
    os.chdir(str(ROOT))


if __name__ == "__main__":
    main()
