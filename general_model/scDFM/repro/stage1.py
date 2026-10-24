"""Stage 1 reproduction harness for scDFM on Norman.

Reuses the original train_step / test / generate_sample from src/script/run.py by
injecting the module-level globals that run.py's __main__ would have created, so what
is measured here is the original code path, not a reimplementation.

    python repro/stage1.py data
    python repro/stage1.py forward --ckpt ckpts/additive/fold1/checkpoint.pt
    python repro/stage1.py train --steps 300
    python repro/stage1.py eval  --ckpt ckpts/additive/fold1/checkpoint.pt
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import torch

import src.script.run as R
from config.config_flow import FlowConfig
from src.data_process.data import Data, PerturbationDataset
from src.models.instantiate_model import instantiate_model
from src.utils.utils import process_vocab, set_seed


class _Acc:
    """Stand-in for accelerate.Accelerator on a single GPU."""
    is_main_process = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_sh_config(fold=1):
    # Exactly the flags in run.sh
    return FlowConfig(batch_size=48, devices="0", model_type="origin", lr=5e-5, steps=200000,
                      data_name="norman", d_model=128, eta_min=1e-6,
                      fusion_method="differential_perceiver", infer_top_gene=1000, n_top_genes=5000,
                      result_path="./result/additive", perturbation_function="crisper",
                      noise_type="Gaussian", mode="predict_y", gamma=0.5, split_method="additive",
                      use_mmd_loss=True, fold=fold, topk=30, use_negative_edge=True)


def setup(fold=1, batch_size=None):
    cfg = run_sh_config(fold)
    if batch_size:
        cfg.batch_size = batch_size
    acc = _Acc()
    t0 = time.time()
    dm = Data("./data")
    dm.load_data(cfg.data_name)
    dm.process_data(n_top_genes=cfg.n_top_genes, split_method=cfg.split_method, fold=cfg.fold,
                    use_negative_edge=cfg.use_negative_edge, k=cfg.topk)
    train_sampler, valid_sampler, _ = dm.load_flow_data(batch_size=cfg.batch_size)
    vocab = process_vocab(dm, cfg)
    gene_ids = torch.tensor(vocab.encode(list(dm.adata.var_names)), dtype=torch.long, device=acc.device)
    tag = "_negative_edge" if cfg.use_negative_edge else ""
    mask_path = os.path.join(dm.data_path, dm.data_name,
                             f"mask_fold_{cfg.fold}topk_{cfg.topk}{cfg.split_method}{tag}.pt")
    # inject the globals run.py's functions read
    R.config, R.device, R.gene_ids = cfg, acc.device, gene_ids
    R.inverse_dict = {v: str(k) for k, v in dm.perturbation_dict.items()}
    print(f"[setup] {time.time()-t0:.0f}s  genes={dm.adata.n_vars}  train_cells={dm.adata_train.n_obs}"
          f"  test_cells={dm.adata_test.n_obs} x {dm.adata_test.n_vars} genes"
          f"  train_perts={len(train_sampler._perturbation_covariates)}"
          f"  test_perts={len(valid_sampler._perturbation_covariates)}")
    return cfg, acc, dm, train_sampler, valid_sampler, vocab, gene_ids, mask_path


def build_model(cfg, mask_path, acc):
    vf = instantiate_model(cfg.model_type, ntoken=cfg.ntoken, d_model=cfg.d_model,
                           d_perturbation=cfg.d_model, fusion_method=cfg.fusion_method,
                           perturbation_function=cfg.perturbation_function, mask_path=mask_path)
    return vf.to(acc.device)


def cache_gene_encoder(vf):
    """Memoize GeneEncoder in eval mode. Its inputs (gene ids, perturbation ids) are the
    same at every ODE step of a batch and it does not see t or the cells, so recomputing it
    76 times per batch (19 rk4 intervals x 4 stages) changes nothing but the runtime."""
    enc = vf.encoder
    orig = enc.forward
    cache = {}

    def forward(x):
        if enc.training or torch.is_grad_enabled():
            return orig(x)
        key = (tuple(x.shape), x.device, x.cpu().numpy().tobytes())
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[key] = orig(x)
        return cache[key]

    enc.forward = forward


def crisper_ids(pert_ids, vocab, batch_rows):
    names = [R.inverse_dict[int(p)] for p in pert_ids[0].cpu().numpy()]
    return torch.tensor(vocab.encode(names), dtype=torch.long, device=R.device).repeat(batch_rows, 1)


def cmd_data(a):
    cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = setup(a.fold)
    var = list(dm.adata.var_names)
    print("[data] gene set == cached vocab (ids 4..N, mask row order):", sorted(gene_ids.tolist()) == list(range(4, 4 + len(var))),
          " test genes ⊂ train genes:", set(va.adata.var_names) <= set(var))
    print("[data] mask", tuple(torch.load(mask_path).shape), " test perts:", list(va._perturbation_covariates)[:5], "...")
    X = tr.adata.X[:500]
    X = X.toarray() if hasattr(X, "toarray") else X
    print(f"[data] train X log-scale: min={X.min():.2f} max={X.max():.2f} mean={X.mean():.3f}")


def cmd_forward(a):
    cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = setup(a.fold)
    if a.ckpt:
        cfg.d_model = 512  # the published checkpoints are 512-wide
    vf = build_model(cfg, mask_path, acc)
    if a.ckpt:
        vf.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    print("[forward] params %.1fM  d_model(effective)=%d" % (sum(p.numel() for p in vf.parameters()) / 1e6,
                                                         vf.t_embedder.mlp[2].weight.shape[0]))
    b = PerturbationDataset(tr, cfg.batch_size)[0]
    src, tgt = b["src_cell_data"].to(acc.device), b["tgt_cell_data"].to(acc.device)
    pid = crisper_ids(b["condition_id"], vocab, src.shape[0])
    vf.eval()
    with torch.no_grad():
        loss = R.train_step(src, tgt, pid, vf, None, acc, noise_type=cfg.noise_type, mode=cfg.mode)
    print(f"[forward] loss on one batch: {loss.item():.4f}  peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


def cmd_train(a):
    set_seed(0)
    cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = setup(a.fold, a.batch)
    if a.d_model:
        cfg.d_model = a.d_model
    vf = build_model(cfg, mask_path, acc)
    print(f"[train] batch={cfg.batch_size}")
    print("[train] params %.1fM  d_model(effective)=%d" % (
        sum(p.numel() for p in vf.parameters()) / 1e6, vf.t_embedder.mlp[2].weight.shape[0]))
    opt = torch.optim.Adam(vf.parameters(), lr=cfg.lr)
    ds = PerturbationDataset(tr, cfg.batch_size)
    log, t0 = [], time.time()
    vf.train()
    for it in range(a.steps):
        b = ds[it]
        src, tgt = b["src_cell_data"].to(acc.device), b["tgt_cell_data"].to(acc.device)
        pid = crisper_ids(b["condition_id"], vocab, src.shape[0])
        loss = R.train_step(src, tgt, pid, vf, None, acc, noise_type=cfg.noise_type, mode=cfg.mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.append(loss.item())
        if it % 50 == 0 or it == a.steps - 1:
            print(f"[train] it={it:5d} loss={loss.item():.4f} avg50={np.mean(log[-50:]):.4f} "
                  f"{(time.time()-t0)/(it+1):.2f}s/it  GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    os.makedirs("repro/out", exist_ok=True)
    json.dump(log, open(f"repro/out/train_loss_fold{a.fold}_{a.steps}_d{cfg.d_model}_b{cfg.batch_size}.json", "w"))
    q = max(1, len(log) // 10)
    print(f"[train] first-10% mean {np.mean(log[:q]):.4f}  last-10% mean {np.mean(log[-q:]):.4f}")


def cmd_eval(a):
    set_seed(0)
    cfg, acc, dm, tr, va, vocab, gene_ids, mask_path = setup(a.fold)
    cfg.d_model = 512
    vf = build_model(cfg, mask_path, acc)
    vf.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    vf.eval()
    if not a.no_cache:
        cache_gene_encoder(vf)
    out = f"repro/out/eval_fold{a.fold}" + ("_nocache" if a.no_cache else "")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    R.test(va, vf, acc, batch_size=cfg.batch_size, path=out, vocab=vocab, scheme="mse")
    print(f"[eval] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["data", "forward", "train", "eval"])
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--ckpt", default="")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=0, help="override run.sh batch size (48)")
    p.add_argument("--d-model", dest="d_model", type=int, default=0, help="override run.sh d_model (128)")
    p.add_argument("--no-cache", action="store_true", help="eval without GeneEncoder memoization")
    a = p.parse_args()
    {"data": cmd_data, "forward": cmd_forward, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)
