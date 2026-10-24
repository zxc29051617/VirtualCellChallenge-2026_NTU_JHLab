"""LINCS 2020 (CMap, public): which challenge / hold-out targets have CRISPR-KO (trt_xpr) or shRNA (trt_sh) consensus
signatures, in which cell lines, and how many high-quality signatures. Metadata only (siginfo_beta.txt)."""
import json, sys
import pandas as pd
ROOT = "/home/zxc29051617/VCC"; L = f"{ROOT}/data/external/lincs2020"
sig = pd.read_csv(f"{L}/siginfo_beta.txt", sep="\t", low_memory=False)
print("columns:", list(sig.columns)[:40]); print(sig.pert_type.value_counts().head(10).to_string())
vcc = list(dict.fromkeys(pd.read_csv(f"{ROOT}/data/vcc2026/pert_counts.csv").target_gene.astype(str)))
sets = {"VCC300": vcc}
for s in ["H1", "HCT116", "KOLF200"]: sets[s] = json.load(open(f"{ROOT}/data/stage3/loco/{s}/real.json"))["targets"]
out = {}
for pt in ["trt_xpr", "trt_sh", "trt_sh.cgs"]:
    sub = sig[sig.pert_type == pt]
    hq = sub[sub.get("is_hiq", 1) == 1] if "is_hiq" in sub.columns else sub
    for name, tg in sets.items():
        cov = sub[sub.cmap_name.isin(tg)] if "cmap_name" in sub.columns else sub[sub.pert_iname.isin(tg)]
        gcol = "cmap_name" if "cmap_name" in sub.columns else "pert_iname"
        out[f"{pt}|{name}"] = {"targets_covered": int(cov[gcol].nunique()), "of": len(tg), "signatures": int(len(cov)),
                               "cell_lines": int(cov.cell_iname.nunique()),
                               "top_cell_lines": cov.cell_iname.value_counts().head(12).to_dict()}
for k, v in out.items(): print(k, json.dumps(v)[:400])
json.dump(out, open(f"{ROOT}/scDFM/repro/next_20261019/lincs_coverage.json", "w"), indent=1)
