"""Where does the trained model lose on the mock exam? Subset s0 (60 H1 targets): bench metrics (trans genes, H1 truth from
the stats) for the raw inputs (consensus d200 vs AdaptNet) and for the pipeline's processed totals of A and B."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import bench as B
E = "/home/zxc29051617/VCC/data/stage3/effects"
targets = pd.read_csv("/home/zxc29051617/VCC/data/mock_h1/s0/pert_counts.csv")["target_gene"].astype(str).tolist()
d = B.data()["H1"]; tg = [t for t in targets if t in d["row"]]
def load(path, minus=None):
    z = np.load(path, allow_pickle=True); T = [str(t) for t in z["targets"]]; L = z["lfc"].astype(np.float64) * np.log(2)
    if minus is not None:
        zo = np.load(minus, allow_pickle=True); L = L - zo["lfc"].astype(np.float64) * np.log(2)
    return L[[T.index(t) for t in tg]]
rows = {
 "A input (consensus d200, x3)": load(f"{E}/mock_current_s0_d200.npz", f"{E}/mock_current_s0_oc.npz"),
 "B input (AdaptNet, norm-matched)": load(f"{E}/mock_anpre_s0_d200.npz", f"{E}/mock_anpre_s0_oc.npz"),
 "A totals after processing": load(f"{E}/mock_current_s0_bulk_g1_H.npz", f"{E}/mock_current_s0_oc.npz"),
 "B totals after processing": load(f"{E}/mock_trained_s0_bulk_g1_H.npz", f"{E}/mock_trained_s0_oc.npz"),
 "A cells input": load(f"{E}/mock_current_s0_cells_agree.npz", f"{E}/mock_current_s0_oc.npz"),
 "B cells input": load(f"{E}/mock_trained_s0_cells_agree.npz", f"{E}/mock_trained_s0_oc.npz"),
}
for k, P in rows.items():
    r = B.score("H1", tg, P.astype(np.float32))
    print(f"{k:34s} dir {r['dir']:+.4f}  pds {r['pds']:.4f}  skill {r['skill']:+.4f}  scale {r['scale']:.2f}  | norm {np.linalg.norm(P):.1f}", flush=True)
