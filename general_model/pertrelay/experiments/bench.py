"""Equivalence check + training benchmark for the SDPA path of MultiheadDiffAttn.

    python experiments/bench.py --equiv-batch 2 --batches 8 16 48 --d-model 512 --steps 10
Prints, per batch size and attention path, peak GPU memory and seconds per training step.
"""
import argparse, os, sys, time, platform
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "repro")
import torch
import stage1 as S
import src.script.run as R
import src.models.origin.blocks as blocks
from src.data_process.data import PerturbationDataset

p = argparse.ArgumentParser()
p.add_argument("--equiv-batch", type=int, default=2)
p.add_argument("--batches", type=int, nargs="*", default=[8])
p.add_argument("--d-model", type=int, default=512)
p.add_argument("--steps", type=int, default=10)
p.add_argument("--modes", nargs="*", default=["sdpa", "orig"])
a = p.parse_args()

print(f"[env] {platform.machine()} torch {torch.__version__} {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = S.setup(1)
cfg.d_model = a.d_model
torch.manual_seed(0)
vf = S.build_model(cfg, mask_path, acc).train()


def batch(B, seed):
    import numpy as np
    np.random.seed(seed)
    b = PerturbationDataset(tr, B)[0]
    return (b["src_cell_data"].cuda(), b["tgt_cell_data"].cuda(),
            S.crisper_ids(b["condition_id"], vocab, B))


def loss_with(mode, src, tgt, pid, seed):
    blocks.USE_SDPA = (mode == "sdpa")
    torch.manual_seed(seed)                      # same gene subset, t, noise and dropout masks
    return R.train_step(src, tgt, pid, vf, None, acc, noise_type=cfg.noise_type, mode=cfg.mode)


# 1) equivalence: same weights, inputs and RNG -> loss and gradients
if a.equiv_batch:
    src, tgt, pid = batch(a.equiv_batch, 1)
    grads = {}
    for mode in ("orig", "sdpa"):
        vf.zero_grad(set_to_none=True)
        loss = loss_with(mode, src, tgt, pid, 123)
        loss.backward()
        grads[mode] = (loss.item(), torch.cat([q.grad.flatten() for q in vf.parameters() if q.grad is not None]))
    (lo, go), (ls, gs) = grads["orig"], grads["sdpa"]
    rel = ((go - gs).norm() / go.norm()).item()
    print(f"[equiv] loss orig {lo:.6f} sdpa {ls:.6f} |diff| {abs(lo-ls):.2e}; grad rel. diff {rel:.2e}")

# 2) memory + speed
opt = torch.optim.Adam(vf.parameters(), lr=cfg.lr)
for B in a.batches:
    for mode in a.modes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            src, tgt, pid = batch(B, 2)
            for i in range(a.steps + 2):
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                loss = loss_with(mode, src, tgt, pid, i)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize()
            print(f"[bench] B={B:3d} {mode:4s}: {(time.time()-t0)/a.steps:.3f} s/step  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        except torch.OutOfMemoryError:
            print(f"[bench] B={B:3d} {mode:4s}: OOM", flush=True)
        finally:
            for n in ("src", "tgt", "pid", "loss"):
                globals().pop(n, None)
