"""Print metadata for each .h5ad dataset in ./data/ without loading .X into memory."""

import glob
import os

import anndata as ad


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PERT_COL_CANDIDATES = ["target_gene", "perturbation", "gene", "guide_target"]


def find_pert_col(obs_columns):
    for candidate in PERT_COL_CANDIDATES:
        if candidate in obs_columns:
            return candidate
    return None


def main():
    h5ad_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5ad")))
    if not h5ad_paths:
        print(f"No .h5ad files found in {DATA_DIR!r}")
        return

    for path in h5ad_paths:
        print("=" * 80)
        print(f"File: {path}")
        adata = ad.read_h5ad(path, backed="r")

        print(f"n_obs (total cells): {adata.n_obs}")
        print(f"n_vars (total genes): {adata.n_vars}")

        obs_columns = list(adata.obs.columns)
        print(f"obs columns ({len(obs_columns)}): {obs_columns}")

        pert_col = find_pert_col(obs_columns)
        if pert_col is None:
            print("Could not find a perturbation column among candidates:", PERT_COL_CANDIDATES)
        else:
            unique_perts = adata.obs[pert_col].unique()
            print(f"Perturbation column: {pert_col!r}")
            print(f"Number of unique perturbation labels: {len(unique_perts)}")
            print(f"Unique perturbation labels: {sorted(map(str, unique_perts))}")

        adata.file.close()
        print()


if __name__ == "__main__":
    main()
