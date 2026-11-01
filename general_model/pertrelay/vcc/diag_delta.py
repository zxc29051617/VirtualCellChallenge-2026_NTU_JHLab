"""Proxy with delta generation: cell = paired control + (gen(target) - gen(zero effect)), same noise."""
import os, sys, json, numpy as np, torch
os.environ["SCDFM_SDPA"] = "1"
from vcc.train import build_model, generate, TestCond, Proxy, to_counts
F, M = sys.argv[1], sys.argv[2]; method = sys.argv[3] if len(sys.argv) > 3 else "onestep"
vocab = json.load(open(f"{F}/vocab.json")); ck = torch.load(M, map_location="cpu", weights_only=False)
vf = build_model(F, vocab, ck["cfg"]["d_model"], ck["cfg"]["cond"]); vf.load_state_dict(ck["model"]); vf.cuda().eval()
data = TestCond(F, vocab, ck["cfg"]["min_cells"], ck["cfg"]["cond"])
px = Proxy(F, data.G, 60, np.random.default_rng(123)); rng = np.random.default_rng(1); n = int(os.environ.get("NCELLS", 64))
P = {"direct": [], "delta": []}
for t in px.targets:
    idx = rng.choice(len(px.src), n, replace=False); c_full = data.test_cond(t, vocab)
    yt, y0 = [], []
    for lo in range(0, len(data.G), 1000):
        src = px.src[idx][:, lo:lo + 1000].float().cuda()
        genes = data.gene_tok[lo:lo + 1000].cuda().unsqueeze(0).repeat(n, 1)
        c = c_full[lo:lo + 1000].unsqueeze(0).repeat(n, 1).cuda()
        torch.manual_seed(lo); yt.append(generate(vf, src, genes, c, method).cpu())
        torch.manual_seed(lo); y0.append(generate(vf, src, genes, torch.zeros_like(c), method).cpu())
    yt, y0 = torch.cat(yt, 1).numpy(), torch.cat(y0, 1).numpy()
    s = px.src[idx].float().numpy()
    for k, y in (("direct", yt), ("delta", np.clip(s + yt - y0, 0, None))):
        m = to_counts(y, px.lib[idx]).sum(0); P[k].append(np.log1p(m / m.sum() * 1e6 / 100) - px.ctrl)
R = px.real
for k, v in P.items():
    v = np.stack(v); pear = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(v, R)])
    C = (v / np.linalg.norm(v, axis=1, keepdims=True)) @ (R / np.linalg.norm(R, axis=1, keepdims=True)).T
    print(f"[{k:6s}] it={ck.get('it')} {method}: pearson_delta={pear:.4f} pds_like={1-np.mean((C > np.diag(C)[:,None]).sum(1)/(len(v)-1)):.4f} "
          f"effect_ratio={np.linalg.norm(v,axis=1).mean()/np.linalg.norm(R,axis=1).mean():.3f}")
