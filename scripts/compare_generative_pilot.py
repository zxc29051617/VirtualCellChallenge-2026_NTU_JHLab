"""Source-only, bidirectional single-cell pilot. Small models, NOT PRiMeFlow reproduction.

Run with a Python having torch, sklearn, scipy, h5py. Target perturbation cells are
used only in final evaluation. PCA, model checkpoints and hyperparameters see source only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcc2026.cell_validation import (
    cpm,
    differential_expression,
    read_groups,
    read_rows,
    write_cells,
)
from vcc2026.io import read_categorical, read_var_names
from vcc2026.local_eval import fold_change, load_profiles, score

FILES = {"K562": "ReplogleWeissman2022_K562_essential.h5ad",
         "RPE1": "ReplogleWeissman2022_rpe1.h5ad"}


def diffusion_schedule():
    # Cosine schedule reaches approximately pure Gaussian noise at the terminal step.
    # A 100-step linear beta<=.02 schedule would leave ~36% signal at inference start.
    t = torch.linspace(0, 1, 101)
    cumulative = torch.cos((t + .008) / 1.008 * np.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=.999)
    return beta, torch.cumprod(1 - beta, 0)


class Field(nn.Module):
    def __init__(self, dim, targets):
        super().__init__()
        self.embedding = nn.Embedding(targets, 16)
        self.net = nn.Sequential(nn.Linear(dim + 20, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, dim))
        self.register_buffer("lower", torch.full((dim,), -float("inf")))
        self.register_buffer("upper", torch.full((dim,), float("inf")))

    def forward(self, x, t, y):
        time_features = torch.cat([t, t**2, torch.sin(t * np.pi), torch.cos(t * np.pi)], 1)
        return self.net(torch.cat([x, time_features, self.embedding(y)], 1))


def train(kind, z, labels, zv, yv, seed, steps, output):
    torch.manual_seed(seed)
    model = Field(z.shape[1], int(labels.max()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, y = torch.tensor(z), torch.tensor(labels)
    model.lower.copy_(x.min(0).values)
    model.upper.copy_(x.max(0).values)
    vx, vy = torch.tensor(zv), torch.tensor(yv)
    _, abar = diffusion_schedule()
    validation_rng = torch.Generator().manual_seed(123456)
    vn = torch.randn(vx.shape, generator=validation_rng)
    vt = torch.rand((len(vx), 1), generator=validation_rng)

    def objective(data, label, noise, t):
        if kind == "flow":
            state = (1 - t) * noise + t * data
            target = data - noise
        else:
            idx = (t[:, 0] * 100).long().clamp(max=99)
            alpha = abar[idx, None]
            state = alpha.sqrt() * data + (1 - alpha).sqrt() * noise
            target = noise
            t = idx[:, None] / 99
        return (model(state, t, label) - target).square().mean()

    best = float("inf")
    history = []
    start = time.perf_counter()
    for step in range(steps):
        rows = torch.randint(len(x), (128,))
        data, label = x[rows], y[rows]
        loss = objective(data, label, torch.randn_like(data), torch.rand((128, 1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                validation = float(objective(vx, vy, vn, vt))
            history.append({"step": step + 1, "train": float(loss.detach()),
                            "source_validation": validation})
            if validation < best:
                best = validation
                torch.save(model.state_dict(), output)
    model.load_state_dict(torch.load(output, weights_only=True))
    return model, {"history": history, "seconds": time.perf_counter() - start,
                   "parameters": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def sample(model, kind, target, n, dim, seed):
    torch.manual_seed(seed)
    z = torch.randn(n, dim)
    y = torch.full((n,), target, dtype=torch.long)
    if kind == "flow":
        for k in range(100):
            z = z + model(z, torch.full((n, 1), k / 100), y) / 100
    else:
        beta, abar = diffusion_schedule()
        alpha = 1 - beta
        for k in reversed(range(100)):
            noise = model(z, torch.full((n, 1), k / 99), y)
            # Standard DDPM posterior with x0 clipping, using SOURCE training bounds.
            clean = ((z - (1 - abar[k]).sqrt() * noise) / abar[k].sqrt())
            clean = torch.maximum(torch.minimum(clean, model.upper), model.lower)
            previous = abar[k - 1] if k else torch.tensor(1.)
            z = (previous.sqrt() * beta[k] / (1 - abar[k]) * clean
                 + alpha[k].sqrt() * (1 - previous) / (1 - abar[k]) * z)
            if k:
                posterior = beta[k] * (1 - abar[k - 1]) / (1 - abar[k])
                z += posterior.sqrt() * torch.randn_like(z)
    # Apply the same source-support limit to both generative families.
    return torch.maximum(torch.minimum(z, model.upper), model.lower).numpy()


def prepare(source, target, out, n_targets):
    cache = out / f"pilot_cells_{source}_{target}_{n_targets}.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=False))
    sp, tp = [load_profiles(ROOT / f"external/profiles/{c}_perturbation_means.npz")
              for c in (source, target)]
    genes = sorted(set(sp.genes) & set(tp.genes))
    eligible = sorted(p for p in set(sp.perturbations) & set(tp.perturbations) - {"control"}
                      if min(sp.counts[sp.index_of(p)], tp.counts[tp.index_of(p)]) >= 128)
    perts = sorted(np.random.default_rng(20260921).choice(eligible, n_targets, False).tolist())
    payload = {"genes": np.array(genes), "perts": np.array(perts)}
    for prefix, cell in [("source", source), ("target", target)]:
        path = ROOT / "external" / FILES[cell]
        axis = read_var_names(path)
        cols = [axis.index(g) for g in genes]
        codes, categories = read_categorical(path, "perturbation")
        rng = np.random.default_rng(9384)
        rows = {p: rng.choice(np.flatnonzero(codes == categories.index(p)), 128, False)
                for p in perts}
        ctrl_rows = rng.choice(np.flatnonzero(codes == categories.index("control")), 800, False)
        indices = np.concatenate([ctrl_rows, *rows.values()])
        matrix = read_rows(path, indices, cols)
        payload[prefix + "_controls"] = matrix[:800]
        payload[prefix + "_cells"] = matrix[800:].reshape(n_targets, 128, len(genes))
        payload[prefix + "_rows"] = indices
        print("prepared", cell, matrix.shape, flush=True)
    np.savez_compressed(cache, **payload)
    return payload


def run(args):
    torch.set_num_threads(4)
    all_results = []
    for source, target in [("K562", "RPE1"), ("RPE1", "K562")]:
        data = prepare(source, target, args.out, args.targets)
        genes, perts = data["genes"].tolist(), data["perts"].tolist()
        n, dim = args.targets, 16
        sc = data["source_controls"][:400]
        tc, ref = data["target_controls"][:400], data["target_controls"][400:]
        source_baseline = np.log1p(cpm(sc).mean(0))
        source_cells, truth = data["source_cells"], data["target_cells"]
        # Explicit control condition lets source-only calibration remove decoder bias.
        training_groups = np.concatenate([source_cells,
            data["source_controls"][400:528][None]], axis=0)
        residual = np.stack([np.log1p(cpm(x)) - source_baseline for x in training_groups])
        train_x = residual[:, :96].reshape(-1, len(genes)).astype(np.float32)
        valid_x = residual[:, 96:].reshape(-1, len(genes)).astype(np.float32)
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=0)
        z = pca.fit_transform(train_x).astype(np.float32)
        zmean, zstd = z.mean(0), np.maximum(z.std(0), .01)
        z = (z - zmean) / zstd
        zv = ((pca.transform(valid_x) - zmean) / zstd).astype(np.float32)
        labels, yv = np.repeat(np.arange(n + 1), 96), np.repeat(np.arange(n + 1), 32)
        np.savez_compressed(args.out / f"pilot_transform_{source}_{target}.npz",
                            components=pca.components_, mean=pca.mean_, zmean=zmean, zstd=zstd)
        target_means = truth.mean(1)
        ctrl = ref.mean(0)
        truth_de = [differential_expression(truth[i], ref,
                    [genes.index(p)] if p in genes else []) for i, p in enumerate(perts)]
        fc = fold_change(source_cells[:, :96].mean(1), sc.mean(0))
        for seed in range(args.seeds):
            for method in ("control", "transfer", "transfer_resample",
                           "latent_mean", "flow", "diffusion"):
                prefix = f"pilot_{source}_{target}_{seed}_{method}"
                training = {}
                if method in ("flow", "diffusion"):
                    model, training = train(method, z, labels, zv, yv, seed, args.steps,
                                            args.out / (prefix + ".pt"))
                if method in ("latent_mean", "flow", "diffusion"):
                    control_latent = (np.tile(z[labels == n].mean(0), (1024, 1))
                        if method == "latent_mean" else sample(model, method, n, 1024, dim, seed))
                    control_decoded = pca.inverse_transform(control_latent * zstd + zmean)
                    generated_control = cpm(np.expm1(np.clip(
                        control_decoded + source_baseline, 0, 20))).mean(0)
                groups = {}
                clipping = []
                for i, pert in enumerate(perts):
                    rng = np.random.default_rng(seed * 1000 + i)
                    if method == "control":
                        cells = tc[rng.choice(len(tc), 128, replace=True)]
                    elif method in ("transfer", "transfer_resample"):
                        means = tc.mean(0) * np.exp(.3 * fc[i])
                        # Same rounding observation operator as the latent models.
                        if method == "transfer":
                            cells = np.broadcast_to(means, (128, len(genes))).copy()
                        else:
                            cells = tc[rng.choice(len(tc), 128, replace=True)] * np.exp(.3 * fc[i])
                    else:
                        latent = (np.tile(z[labels == i].mean(0), (128, 1))
                                  if method == "latent_mean" else
                                  sample(model, method, i, 128, dim, seed * 1000 + i))
                        decoded = pca.inverse_transform(latent * zstd + zmean) + source_baseline
                        clipping.append(float(((decoded < 0) | (decoded > 20)).mean()))
                        generated = cpm(np.expm1(np.clip(decoded, 0, 20)))
                        average = generated.mean(0)
                        effect = np.log((average + 1) / (generated_control + 1))
                        means = tc.mean(0) * np.exp(.3 * effect)
                        cells = generated * (means / np.maximum(average, 1e-9))
                    groups[pert] = (np.floor(cells) + (rng.random(cells.shape)
                                                    < cells - np.floor(cells))).astype(np.float32)
                filename = write_cells(args.out / (prefix + ".h5ad"), groups, genes, target)
                actual = read_groups(filename, genes)
                measured = np.stack([actual[p].mean(0) for p in perts])
                metrics = score(measured, target_means, ctrl, perturbations=perts, genes=genes)
                de_metrics = []
                for i, p in enumerate(perts):
                    pred = differential_expression(actual[p], ref,
                        [genes.index(p)] if p in genes else [])
                    a, b = pred["called"], truth_de[i]["called"]
                    de_metrics.append({"predicted_de": int(a.sum()), "true_de": int(b.sum()),
                        "intersection": int((a & b).sum()),
                        "jac": float((a & b).sum() / max((a | b).sum(), 1))})
                result = {"source": source, "target": target, "seed": seed, "method": method,
                    "targets": perts, "gene_count": len(genes), "cells_per_group": 128,
                    "training": training, "scores": metrics, "de_diagnostics": de_metrics,
                    "mean_de_jac": float(np.mean([x["jac"] for x in de_metrics])),
                    "clipped_fraction": float(np.mean(clipping)) if clipping else 0.,
                    "output": str(filename), "steps": args.steps, "latent_dim": dim,
                    "adapter": "source_control_calibrated_ratio; effect_scale=0.3; pseudocount=1CPM",
                    "diffusion_schedule": "cosine_100_s008_maxbeta0999"}
                result["sampling_support"] = "per-component min/max of source training latent"
                all_results.append(result)
                (args.out / "generative_pilot.json").write_text(json.dumps(all_results, indent=2))
                print("PILOT", source, target, seed, method, metrics,
                      "DE-jac", result["mean_de_jac"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/offline_20260921")
    parser.add_argument("--targets", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args)
