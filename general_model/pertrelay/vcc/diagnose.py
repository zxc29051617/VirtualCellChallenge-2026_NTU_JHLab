"""Why does the H1 proxy stay at chance? Three checks on a trained run.

    python -m vcc.diagnose --fold F --model runs/x/model.pt
1. proxy sanity: the same proxy metric applied to the transfer baseline (mean training-line LFC)
2. perturbation sensitivity: same source cells and noise, different targets -> how different?
3. in-distribution: does the model reproduce strong perturbations of a TRAINING line (K562)?
"""
import argparse, json, os
import numpy as np, torch

os.environ.setdefault("SCDFM_SDPA", "1")
from vcc.stats import load, lfc_table
from vcc.train import build_model, generate, Lines


def proxy_scores(P, R):
    pear = np.mean([np.corrcoef(p, r)[0, 1] for p, r in zip(P, R)])
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True); Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    C = Pn @ Rn.T
    return pear, 1 - np.mean((C > np.diag(C)[:, None]).sum(1) / (len(P) - 1))


def effects_from_stats(st, G, targets):
    names = list(st["names"]); gi = [list(st["genes"]).index(g) for g in G]
    cpm = st["sums"][:, gi] / st["sums"].sum(1, keepdims=True) * 1e6
    ctrl = np.log1p(cpm[names.index("__ctrl__")] / 100)
    return np.stack([np.log1p(cpm[names.index(t)] / 100) - ctrl for t in targets]), ctrl


@torch.no_grad()
def model_effects(vf, src_X, targets, vocab, gene_tok, dev, n=64, seed=1, chunk=1000):
    rng = np.random.default_rng(seed); out = []
    for t in targets:
        idx = torch.from_numpy(rng.choice(len(src_X), n, replace=False))
        pert = torch.tensor([[vocab[t], vocab["control"]]], device=dev).repeat(n, 1)
        ys = []
        for lo in range(0, len(gene_tok), chunk):
            src = src_X[idx][:, lo:lo + chunk].float().to(dev)
            genes = gene_tok[lo:lo + chunk].to(dev).unsqueeze(0).repeat(n, 1)
            torch.manual_seed(seed)
            ys.append(generate(vf, src, genes, pert, os.environ.get("GEN", "onestep"), 10).cpu())
        y = torch.cat(ys, 1).numpy(); m = np.expm1(y).mean(0)
        out.append(m / m.sum() * 1e6)
    return np.stack(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--fold", required=True); p.add_argument("--model", required=True)
    a = p.parse_args(); dev = torch.device("cuda")
    info = json.load(open(f"{a.fold}/fold.json")); vocab = json.load(open(f"{a.fold}/vocab.json"))
    G = [g.strip() for g in open(f"{a.fold}/genes.txt")]; gene_tok = torch.tensor([vocab[g] for g in G])
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    vf = build_model(a.fold, vocab, ck["cfg"]["d_model"]); vf.load_state_dict(ck["model"]); vf.to(dev).eval()

    # 1. proxy on the transfer baseline
    H = load(info["heldout"]); resp = __import__("pandas").read_csv(
        f"/home/zxc29051617/VCC/data/stage3/stats/{info['heldout']}_responsive.csv", index_col=0).iloc[:, 0]
    targets = [t for t in resp.index[resp >= 30] if t in vocab][:60]
    R, ctrlH = effects_from_stats(H, G, targets)
    T = []
    for l in info["train"]:
        st = load(l); names = set(st["names"])
        E, _ = effects_from_stats(st, G, [t if t in names else "__ctrl__" for t in targets])
        T.append(E)
    T = np.nanmean(np.stack(T), 0)
    print(f"[1] proxy on transfer baseline ({len(targets)} H1 targets, >=30 responsive genes): "
          f"pearson_delta={proxy_scores(T, R)[0]:.3f} pds_like={proxy_scores(T, R)[1]:.3f}")

    # 2. sensitivity: same cells, same noise, different perturbations
    zh = np.load(f"{a.fold}/heldout_ctrl.npz"); srcH = torch.from_numpy(zh["X"])
    M = model_effects(vf, srcH, targets[:10], vocab, gene_tok, dev)        # same idx/noise for all targets
    Mrep = model_effects(vf, srcH, targets[:2], vocab, gene_tok, dev, seed=2)
    L = np.log1p(M / 100)
    between = np.mean([np.linalg.norm(L[i] - L[j]) for i in range(10) for j in range(i + 1, 10)])
    within = np.linalg.norm(np.log1p(Mrep[0] / 100) - L[0])
    print(f"[2] predicted pseudobulk distance between targets {between:.4f}  vs  resampling noise {within:.4f}  "
          f"vs  real between-target distance {np.mean([np.linalg.norm(R[i]-R[j]) for i in range(10) for j in range(i+1,10)]):.4f}")
    P_eff = L - ctrlH[None, :]
    print(f"    predicted effect vs real (10 targets): pearson_delta={proxy_scores(P_eff, R[:10])[0]:.3f}; "
          f"|pred effect|={np.linalg.norm(P_eff,axis=1).mean():.3f} |real effect|={np.linalg.norm(R[:10],axis=1).mean():.3f}")

    # 3. in-distribution: strongest responders of a training line
    l = info["train"][0]; st = load(l)
    rr = __import__("pandas").read_csv(f"/home/zxc29051617/VCC/data/stage3/stats/{l}_responsive.csv", index_col=0).iloc[:, 0]
    tt = [t for t in rr.sort_values(ascending=False).index if t in vocab][:20]
    Rt, ctrlT = effects_from_stats(st, G, tt)
    z = np.load(f"{a.fold}/{l}.npz", allow_pickle=True); srcT = torch.from_numpy(z["X"][z["is_control"]])
    Pt = np.log1p(model_effects(vf, srcT, tt, vocab, gene_tok, dev) / 100) - ctrlT[None, :]
    pear, pds = proxy_scores(Pt, Rt)
    print(f"[3] training line {l}, its 20 strongest perturbations: pearson_delta={pear:.3f} pds_like={pds:.3f}; "
          f"|pred effect|={np.linalg.norm(Pt,axis=1).mean():.3f} |real effect|={np.linalg.norm(Rt,axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
