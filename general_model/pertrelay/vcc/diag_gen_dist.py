import os, json, numpy as np, torch
os.environ["SCDFM_SDPA"] = "1"
from vcc.train import build_model, generate
F = "/home/zxc29051617/VCC/data/stage3/model/heldout_H1"
vocab = json.load(open(f"{F}/vocab.json")); G = [g.strip() for g in open(f"{F}/genes.txt")]
ck = torch.load("runs/h1_local_b16/model.pt", map_location="cpu", weights_only=False)
vf = build_model(F, vocab, 512); vf.load_state_dict(ck["model"]); vf.cuda().eval()
z = np.load(f"{F}/K562.npz", allow_pickle=True); X = z["X"]; tg = z["target"]
ctrl = np.flatnonzero(tg == "__ctrl__"); rng = np.random.default_rng(0)
cols = np.arange(1000); gt = torch.tensor([vocab[G[i]] for i in cols]).cuda()
src = torch.from_numpy(X[rng.choice(ctrl, 64)][:, cols]).float().cuda()
real = X[rng.choice(ctrl, 64)][:, cols].astype(np.float32)
pert = torch.tensor([[vocab["control"], vocab["control"]]]).repeat(64, 1).cuda()
for m in ("onestep", "euler"):
    torch.manual_seed(0)
    y = generate(vf, src, gt.unsqueeze(0).repeat(64, 1), pert, m, 10).cpu().numpy()
    print(f"{m:8s} gen: zeros {np.mean(y==0):.3f} mean {y.mean():.3f} | real ctrl: zeros {np.mean(real==0):.3f} mean {real.mean():.3f} | "
          f"corr of gene means {np.corrcoef(y.mean(0), real.mean(0))[0,1]:.3f} | frac 0<y<0.3 {np.mean((y>0)&(y<0.3)):.3f} vs real {np.mean((real>0)&(real<0.3)):.3f}")
print("measured genes in K562 among first 1000 of G:", int(z["measured_G"][:1000].sum()))
