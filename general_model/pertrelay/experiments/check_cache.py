import sys, os, time, copy
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "repro")
import torch
import stage1 as S
import src.script.run as R

cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = S.setup(1)
cfg.d_model = 512
vf = S.build_model(cfg, mask_path, acc)
vf.load_state_dict(torch.load("ckpts/additive/fold1/checkpoint.pt", map_location="cpu", weights_only=False)["model_state_dict"])
vf.eval()
gid = torch.tensor(vocab.encode(list(va.adata.var_names)), dtype=torch.long, device=acc.device)
ctrl = va.get_control_data()["src_cell_data"][:48].to(acc.device)
pname = va._perturbation_covariates[0]
pid = va.get_perturbation_data(pname)["condition_id"].to(acc.device)
pid = S.crisper_ids(pid, vocab, ctrl.shape[0])

def gen():
    torch.manual_seed(123)
    torch.cuda.synchronize(); t = time.time()
    with torch.inference_mode():
        y = R.generate_sample(R.wrapped_vf, ctrl, pid, vf, gene_ids=gid, gene_all=gid)
    torch.cuda.synchronize()
    return y, time.time() - t

y0, t0 = gen()
S.cache_gene_encoder(vf)
y1, t1 = gen()
print(f"[cache] perturbation {pname}: original {t0:.1f}s  cached {t1:.1f}s  speedup {t0/t1:.1f}x")
print(f"[cache] max |diff| {(y0-y1).abs().max().item():.3e}  allclose(atol=1e-5): {torch.allclose(y0, y1, atol=1e-5)}")
