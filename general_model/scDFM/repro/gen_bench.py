import sys, os, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "repro"); os.environ["SCDFM_SDPA"] = "1"
import torch, stage1 as S, src.script.run as R
cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = S.setup(1)
cfg.d_model = 512
vf = S.build_model(cfg, mask_path, acc).eval(); S.cache_gene_encoder(vf)
gid = torch.tensor(vocab.encode(list(va.adata.var_names)), dtype=torch.long, device=acc.device)
ctrl = va.get_control_data()["src_cell_data"]
pid1 = S.crisper_ids(va.get_perturbation_data(va._perturbation_covariates[0])["condition_id"].to(acc.device), vocab, 1)
for B in (128, 400):
    src = ctrl[torch.randperm(ctrl.shape[0])[:B]].to(acc.device); pid = pid1.repeat(B, 1)
    for method, steps in (("rk4", 20), ("euler", 10), ("onestep", 1)):
        torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            if method == "onestep":
                x0 = torch.randn(B, 1000, device=acc.device); tt = torch.zeros(B, device=acc.device)
                R.wrapped_vf(x0, tt, src, pid, vf, gid, gid)
            else:
                R.generate_sample(R.wrapped_vf, src, pid, vf, gene_ids=gid, gene_all=gid, steps=steps, method=method)
        torch.cuda.synchronize(); dt = time.time() - t
        print(f"[gen] {B} cells x 1000 genes, {method}-{steps}: {dt:.1f}s  peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
