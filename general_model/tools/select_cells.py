"""Fix the stage-3 gene panels and the X-Atlas cells to extract.
Outputs (data/coverage/): aux_genes.txt, xatlas_keep_{HCT116,HEK293T}.csv (cell_barcode, role, gene_target)."""
import numpy as np, pandas as pd, anndata as ad, sys
root = "/home/zxc29051617/VCC/data"
rng = np.random.default_rng(0)
vcc = set(pd.read_csv(f"{root}/vcc2026/pert_counts.csv").target_gene.astype(str))
h1 = set(pd.concat([pd.read_csv(f"{root}/external/meta/h1_pert_counts_{s}.csv") for s in ["Training", "Validation", "Test"]])
         .target_gene.astype(str)) - vcc                    # VCC 2025 (H1) targets: the held-out-line validation panel
bulk = ad.read_h5ad(f"{root}/external/replogle2022/K562_gwps_raw_bulk_01.h5ad", backed="r").obs
k562 = bulk.groupby(bulk.index.str.split("_").str[1])["num_cells_filtered"].sum()
calls = {l: pd.read_csv(f"{root}/external/meta/{l}_guide_calls.csv.gz",
                        usecols=["cell_barcode", "gene_target", "pass_guide_filter"]).query("pass_guide_filter")
         for l in ["HCT116", "HEK293T"]}
cnt = {l: c.gene_target.value_counts() for l, c in calls.items()}
pool = sorted(g for g in k562.index[k562 >= 100]
              if g not in vcc and g not in h1 and cnt["HCT116"].get(g, 0) >= 64 and cnt["HEK293T"].get(g, 0) >= 64)
aux = sorted(rng.choice(pool, 1000, replace=False))
open(f"{root}/coverage/aux_genes.txt", "w").write("\n".join(aux) + "\n")
print(f"aux pool {len(pool)} genes (K562 gwps >=100 cells, X-Atlas >=64 in both lines, not VCC targets) -> 1000 chosen")
for l, c in calls.items():
    c = c.sample(frac=1, random_state=0)
    keep = [c[c.gene_target.isin(vcc)].assign(role="vcc_target"), c[c.gene_target.isin(h1)].assign(role="h1_target")]
    a = c[c.gene_target.isin(aux)]
    keep.append(a.groupby("gene_target").head(64).assign(role="aux"))
    keep.append(c[c.gene_target == "Non-Targeting"].head(20000).assign(role="control"))
    k = pd.concat(keep)[["cell_barcode", "role", "gene_target"]]
    k.to_csv(f"{root}/coverage/xatlas_keep_{l}.csv", index=False)
    print(l, k.role.value_counts().to_dict(), "| controls available", int((c.gene_target == "Non-Targeting").sum()))
