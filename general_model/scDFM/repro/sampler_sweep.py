"""Which sampler settings reproduce the authors' released predictions?
Regenerates 128 cells for a few test perturbations under several ODE settings, and compares
each to the real cells and to the authors' pred.h5ad."""
import os, sys, time, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "repro")
os.environ["SCDFM_SDPA"] = "1"          # exact same function, faster (checked in bench.py)
import numpy as np, torch, anndata as ad
import stage1 as S
import src.script.run as R

PERTS = sys.argv[1].split(",")
cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = S.setup(1)
cfg.d_model = 512
vf = S.build_model(cfg, mask_path, acc)
vf.load_state_dict(torch.load("ckpts/additive/fold1/checkpoint.pt", map_location="cpu", weights_only=False)["model_state_dict"])
vf.eval(); S.cache_gene_encoder(vf)
gid = torch.tensor(vocab.encode(list(va.adata.var_names)), dtype=torch.long, device=acc.device)

real = ad.read_h5ad("ckpts/additive/fold1/real.h5ad"); auth = ad.read_h5ad("ckpts/additive/fold1/pred.h5ad")
ours = ad.read_h5ad("repro/out/eval_fold1/pred.h5ad")
assert list(real.var_names) == list(va.adata.var_names) or real.n_vars == va.adata.n_vars
dense = lambda x: np.asarray(x.toarray() if hasattr(x, "toarray") else x)
ctrl_mean = dense(real[real.obs.perturbation == "control"].X).mean(0)

def stats(X, p):
    X = dense(X)
    t = dense(real[real.obs.perturbation == p].X).mean(0) - ctrl_mean
    a = dense(auth[auth.obs.perturbation == p].X).mean(0) - ctrl_mean
    d = X.mean(0) - ctrl_mean
    return dict(pearson_delta=np.corrcoef(d, t)[0, 1], zeros=float((X == 0).mean()),
                delta_ratio=float(np.linalg.norm(d) / np.linalg.norm(t)), corr_to_authors=np.corrcoef(d, a)[0, 1],
                mse_to_authors_mean=float(((X.mean(0) - dense(auth[auth.obs.perturbation == p].X).mean(0)) ** 2).mean()))

def gen(src, pid, method, steps):
    torch.manual_seed(7)
    if method == "onestep":       # the commented-out alternative in run.py
        x0 = torch.randn(src.shape[0], cfg.infer_top_gene, device=src.device)
        t = torch.zeros(src.shape[0], device=src.device)
        return torch.clamp(x0 + R.wrapped_vf(x0, t, src, pid, vf, gid, gid), min=0)
    return R.generate_sample(R.wrapped_vf, src, pid, vf, gene_ids=gid, gene_all=gid, steps=steps, method=method)

SETTINGS = [("rk4", 20), ("euler", 20), ("midpoint", 20), ("euler", 100), ("dopri5", 20), ("onestep", 1)]
out = {}
ctrl_all = va.get_control_data()["src_cell_data"]
for p in PERTS:
    a_ = stats(auth[auth.obs.perturbation == p].X, p); o_ = stats(ours[ours.obs.perturbation == p].X, p)
    print(f"\n[{p}] authors : pd={a_['pearson_delta']:.3f} zeros={a_['zeros']:.3f} ratio={a_['delta_ratio']:.2f}")
    print(f"[{p}] full run rk4-20 (our eval): pd={o_['pearson_delta']:.3f} zeros={o_['zeros']:.3f} ratio={o_['delta_ratio']:.2f} corr_auth={o_['corr_to_authors']:.3f}", flush=True)
    torch.manual_seed(0)
    src = ctrl_all[torch.randperm(ctrl_all.shape[0])[:128]].to(acc.device)
    pid = S.crisper_ids(va.get_perturbation_data(p)["condition_id"].to(acc.device), vocab, 128)
    for method, steps in SETTINGS:
        t0 = time.time()
        with torch.inference_mode():
            X = torch.cat([gen(src[i:i + 64], pid[i:i + 64], method, steps) for i in (0, 64)]).cpu().numpy()
        s = stats(X, p); out[f"{p}|{method}{steps}"] = s
        print(f"[{p}] {method:8s}{steps:4d}: pd={s['pearson_delta']:.3f} zeros={s['zeros']:.3f} ratio={s['delta_ratio']:.2f} "
              f"corr_auth={s['corr_to_authors']:.3f} mse_auth={s['mse_to_authors_mean']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open("repro/out/sampler_sweep.json", "w"), indent=1, default=float)
