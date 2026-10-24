"""Cells per VCC-2026 target in each candidate CRISPRi dataset (from small metadata files)."""
import pandas as pd, sys
meta, vccdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
T = pd.read_csv(f"{vccdir}/pert_counts.csv")["target_gene"].astype(str)
cols = {}
rep = pd.read_csv(f"{out}/replogle_cells_per_target.csv", index_col=0)
cols["K562_gwps (Replogle)"] = rep["K562_gwps"]
h1 = pd.concat([pd.read_csv(f"{meta}/h1_pert_counts_{s}.csv").assign(split=s) for s in ["Training", "Validation", "Test"]])
cols["H1 (VCC2025)"] = h1.set_index("target_gene")["n_cells"]
for line in ["HCT116", "HEK293T"]:
    g = pd.read_csv(f"{meta}/{line}_guide_calls.csv.gz", usecols=["gene_target", "pass_guide_filter"])
    cols[f"{line} (X-Atlas)"] = g[g.pass_guide_filter].groupby("gene_target").size()
df = pd.DataFrame({k: v.reindex(T).values for k, v in cols.items()}, index=T).fillna(0).astype(int)
df.to_csv(f"{out}/vcc_target_coverage.csv")
print(f"H1 2025 panel overlap with VCC 2026 targets: {len(set(h1.target_gene) & set(T))}/300")
for c in df.columns:
    print(f"{c:22s} covered {int((df[c]>0).sum()):3d}/300   >=50 cells {int((df[c]>=50).sum()):3d}   median cells (covered) {df[c][df[c]>0].median():.0f}")
n = (df >= 50).sum(1)
print("targets with >=50 cells in how many lines:", n.value_counts().sort_index().to_dict())
print("not covered anywhere (>=50):", sorted(df.index[n == 0]))
