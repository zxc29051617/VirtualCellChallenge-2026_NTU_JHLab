import os, json, numpy as np, pandas as pd, torch
os.environ["SCDFM_SDPA"] = "1"
from vcc.train import build_model, generate
from vcc.stats import load
F = "/home/zxc29051617/VCC/data/stage3/model/heldout_H1"
vocab = json.load(open(f"{F}/vocab.json")); G = [g.strip() for g in open(f"{F}/genes.txt")]
ck = torch.load("runs/h1_local_b16/model.pt", map_location="cpu", weights_only=False)
vf = build_model(F, vocab, 512); vf.load_state_dict(ck["model"]); vf.cuda().eval()
z = np.load(f"{F}/K562.npz", allow_pickle=True); X = z["X"]; ctrl = np.flatnonzero(z["target"] == "__ctrl__")
meas = np.flatnonzero(z["measured_G"])[:1000]; gt = torch.tensor([vocab[G[i]] for i in meas]).cuda()
st = load("K562"); names = list(st["names"]); gi = [list(st["genes"]).index(G[i]) for i in meas]
cpm = st["sums"][:, gi] / st["sums"].sum(1, keepdims=True) * 1e6; c0 = np.log1p(cpm[names.index("__ctrl__")] / 100)
rr = pd.read_csv("/home/zxc29051617/VCC/data/stage3/stats/K562_responsive.csv", index_col=0).iloc[:, 0]
tt = [t for t in rr.sort_values(ascending=False).index if t in vocab][:20]
rng = np.random.default_rng(0); n = 128
src = torch.from_numpy(X[rng.choice(ctrl, n)][:, meas]).float().cuda(); genes = gt.unsqueeze(0).repeat(n, 1)
def out(tok, m):
    torch.manual_seed(0); p = torch.tensor([[vocab[tok], vocab["control"]]]).repeat(n, 1).cuda()
    return generate(vf, src, genes, p, m, 10).cpu().numpy().mean(0)
for m in ("onestep", "euler"):
    base = out("control", m)
    P = np.stack([out(t, m) - base for t in tt]); R = np.stack([np.log1p(cpm[names.index(t)] / 100) - c0 for t in tt])
    pear = np.mean([np.corrcoef(p, r)[0, 1] for p, r in zip(P, R)])
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True); Rn = R / np.linalg.norm(R, axis=1, keepdims=True); C = Pn @ Rn.T
    pds = 1 - np.mean((C > np.diag(C)[:, None]).sum(1) / (len(tt) - 1))
    print(f"{m:8s}: pert-minus-control-token effect vs real (K562 top-20, 1000 measured genes): pearson {pear:.3f}, pds_like {pds:.3f}, "
          f"|pred| {np.linalg.norm(P,axis=1).mean():.3f} |real| {np.linalg.norm(R,axis=1).mean():.3f}")
