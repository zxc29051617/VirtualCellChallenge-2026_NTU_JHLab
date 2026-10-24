"""Decision-level simulator for the four DE members (2026-10-23): score a hypothetical prediction DE table (which genes
we call significant, in what order, with what sign and log2 fold change) against a mock subset's real DE table with the
scorer's own metric functions, without building cells. Scaled like the leaderboard: (raw - baseline) / (replicate - baseline)
from the subset's bundle.

    import de_sim as DS
    ref = DS.Ref("hybrid_rpe1_de", 0)                       # real DE table saved by mock_exam.py WRITE_DE=1
    DS.score(ref, table)                                    # table: pandas target, feature, log2_fold_change, p_value, p_adj
"""
import os
import numpy as np, pandas as pd, polars as pl
from cell_eval2.de import prepare_de
from cell_eval2.metrics.de import de_sig_jaccard, de_lfc_nmae
from cell_eval2.metrics.direction import de_direction_fidelity_yield_raw, de_direction_reach

HERE = os.path.dirname(os.path.abspath(__file__)); MOCK = "/home/zxc29051617/VCC/data/mock_h1"
MEMBERS = {"nmae": "de_wilcoxon_lfc_nmae", "fid": "de_wilcoxon_direction_fidelity_yield_raw",
           "reach": "de_wilcoxon_direction_reach_raw", "jac": "de_wilcoxon_sig_jaccard"}


class Ref:
    def __init__(self, variant, k):
        d = f"{HERE}/mock/mock_{variant}_s{k}_ce2"
        self.real = pl.read_parquet(f"{d}/de_real.parquet"); self.pred = pl.read_parquet(f"{d}/de_pred.parquet")
        base = pd.read_csv(f"{MOCK}/s{k}/bundle/baseline_agg.csv").set_index("statistic").loc["mean"]
        rep = pl.read_parquet(f"{MOCK}/s{k}/bundle/anchor_agg.parquet").to_pandas().set_index("metric")["replicate"]
        self.base = {m: float(base[c]) for m, c in MEMBERS.items()}; self.rep = {m: float(rep[c]) for m, c in MEMBERS.items()}


def raw(ref, table):
    t = table if isinstance(table, pl.DataFrame) else pl.from_pandas(table)
    P = prepare_de(t, ref.real, control="non-targeting")
    out = {"nmae": de_lfc_nmae(P), "fid": de_direction_fidelity_yield_raw(P),
           "reach": de_direction_reach(P, universe="adjudicated", corrected=False), "jac": de_sig_jaccard(P)}
    return {m: float(np.nanmean([v for v in d.values() if v is not None])) for m, d in out.items()}


def score(ref, table):
    r = raw(ref, table)
    sc = {m: (ref.base[m] - r[m]) / (ref.base[m] - ref.rep[m]) if m == "nmae" else (r[m] - ref.base[m]) / (ref.rep[m] - ref.base[m]) for m in r}
    return r, sc
