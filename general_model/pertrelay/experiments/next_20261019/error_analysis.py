"""Where does target discrimination fail? Per target: pds-like rank score (V6 prediction) vs the size of the real
effect (norm of the real bulk delta outside the panel targets) and the number of source contexts that measured it."""
import sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay"); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013")
import sigexp as S, coexp_screen as CS, render_eval as E
Cc = S.load_cache()
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); _, oc, full, shuf = E.components(name); o = S.OUTER[name]
    train = [c for c in S.CONTEXTS if c not in o["exclude"]]
    lfc = oc + E.remove_shared(full - oc); P = CS.pred_deltas(lfc, rate, mu_c)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True; m = ~panel & (rate > 0)
    a = P[:, m]; b = D[:, m]; a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12); bn = np.linalg.norm(b, axis=1); b = b / (bn[:, None] + 1e-12)
    Cm = a @ b.T; d = np.diag(Cm)[:, None]; n = len(targets); score = ((Cm < d).sum(1) + 0.5 * ((Cm == d).sum(1) - 1)) / (n - 1)
    nsrc = np.array([sum(1 for c in train if t in Cc[c]["tidx"]) for t in targets])
    q = np.quantile(bn, [0.25, 0.5, 0.75]); bins = np.digitize(bn, q)
    print(name, "pds by real-effect quartile (weak -> strong):", [round(float(score[bins == k].mean()), 3) for k in range(4)],
          "| by #sources 0-3/4-7/8+:", [round(float(score[(nsrc >= lo) & (nsrc <= hi)].mean()), 3) if ((nsrc >= lo) & (nsrc <= hi)).any() else None for lo, hi in [(0, 3), (4, 7), (8, 99)]],
          "| counts:", [int(((nsrc >= lo) & (nsrc <= hi)).sum()) for lo, hi in [(0, 3), (4, 7), (8, 99)]], flush=True)
