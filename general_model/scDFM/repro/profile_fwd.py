import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "repro")
import torch
from torch.profiler import profile, ProfilerActivity
import stage1 as S
import src.script.run as R

cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = S.setup(1)
cfg.d_model = 512
vf = S.build_model(cfg, mask_path, acc).eval()
gid = torch.tensor(vocab.encode(list(va.adata.var_names)), dtype=torch.long, device=acc.device)
B = 48
ctrl = va.get_control_data()["src_cell_data"][:B].to(acc.device)
pid = S.crisper_ids(va.get_perturbation_data(va._perturbation_covariates[0])["condition_id"].to(acc.device), vocab, B)
x = torch.randn(B, 1000, device=acc.device); t = torch.full((B,), 0.5, device=acc.device)
with torch.inference_mode():
    R.wrapped_vf(x, t, ctrl, pid, vf, gid, gid)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        R.wrapped_vf(x, t, ctrl, pid, vf, gid, gid); torch.cuda.synchronize()
print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", row_limit=14))
