import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "repro")
import torch
import stage1 as S
import src.script.run as R
from src.data_process.data import PerturbationDataset

cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = S.setup(1)
for d in (128, 512):
    cfg.d_model = d
    vf = S.build_model(cfg, mask_path, acc).train()
    for B in (2, 4, 8):
        b = PerturbationDataset(tr, B)[0]
        src, tgt = b["src_cell_data"].cuda(), b["tgt_cell_data"].cuda()
        pid = S.crisper_ids(b["condition_id"], vocab, B)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        loss = R.train_step(src, tgt, pid, vf, None, acc, noise_type=cfg.noise_type, mode=cfg.mode)
        fwd = torch.cuda.max_memory_allocated() / 2**30
        loss.backward()
        print(f"[mem] d_model={d} B={B}: fwd peak {fwd:.2f} GiB, fwd+bwd peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
        vf.zero_grad(set_to_none=True); del loss
    del vf
