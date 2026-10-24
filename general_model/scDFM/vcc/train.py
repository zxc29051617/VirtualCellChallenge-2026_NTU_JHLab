"""Train scDFM across cell lines (one fold built by vcc.prepare).

    python -m vcc.train --fold F --out runs/x --cond effect --steps 20000 --batch 16

Same flow path and loss as src/script/run.py (affine CondOT path, Gaussian source, velocity MSE
+ gamma * multi-sigma MMD on the one-step x1 estimate). Batches are (controls of line L, cells of
line L perturbed by g), L and g sampled jointly; the genes of each step are drawn from G restricted
to the genes L measured. Two ways to tell the model which perturbation:
  --cond token   the paper's: the perturbed gene's learned token embedding (pair (g, 'control'));
  --cond effect  vcc.effect_model: g's measured effect in the OTHER training lines, per gene.
                 Leave-one-line-out during training, all training lines at test time.
"""
import argparse, json, os, time
from dataclasses import dataclass, asdict

import numpy as np
import torch

os.environ.setdefault("SCDFM_SDPA", "1")          # fused attention: same function, far less memory
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.scheduler import CondOTScheduler
from src.models.origin.model import model as OriginModel
from src.script.run import median_sigmas, mmd2_unbiased_multi_sigma
from vcc.effect_model import EffectModel
from vcc.prepare import TARGET_SUM

PATH = AffineProbPath(scheduler=CondOTScheduler())


@dataclass
class Cfg:
    fold: str
    out: str
    cond: str = "effect"
    steps: int = 2000
    batch: int = 8
    lr: float = 5e-5
    eta_min: float = 1e-6
    d_model: int = 512
    genes_per_step: int = 1000
    gamma: float = 0.5                 # MMD weight (run.sh: 0.5 with --use_mmd_loss)
    min_cells: int = 20                # perturbations with fewer cells in a line are not trained on
    amp: str = "none"                  # "tf32": TF32 matmuls; "bf16": autocast the velocity net (loss/MMD stay fp32)
    line_weights: str = ""             # e.g. "H1=10": sampling weight of each (line, perturbation) pair by line
    eval_every: int = 500
    eval_targets: int = 60
    eval_cells: int = 64
    seed: int = 0


def effect_table(line, G, min_cells, drop=()):
    """{perturbation: log1p(CPM/100) effect on G (NaN where the line did not measure the gene)}."""
    from vcc.stats import load
    st = load(line); names = list(st["names"])
    gi = np.array([list(st["genes"]).index(g) for g in G])
    cpm = st["sums"][:, gi] / st["sums"].sum(1, keepdims=True) * 1e6
    lc = np.log1p(cpm / 100); c = lc[names.index("__ctrl__")]
    meas = st["measured"][gi]
    out = {}
    for i, n in enumerate(names):
        if n != "__ctrl__" and st["ncell"][i] >= min_cells and n not in drop:
            e = lc[i] - c; e[~meas] = np.nan; out[n] = e.astype(np.float32)
    return out


def mean_effect(tables, t):
    v = [tab[t] for tab in tables if t in tab]
    return None if not v else np.nan_to_num(np.nanmean(np.stack(v), 0))


class TestCond:
    """Test-time conditioning only (no training matrices): effect tables of every training line."""

    def __init__(self, fold, vocab, min_cells, cond):
        info = json.load(open(f"{fold}/fold.json"))
        self.cond, self.G = cond, [g.strip() for g in open(f"{fold}/genes.txt")]
        self.gene_tok = torch.tensor([vocab[g] for g in self.G])
        hold = set(info.get("holdout_targets", []))
        self.tables = ([effect_table(l, self.G, min_cells) for l in info["train"] if not (hold and l == info["heldout"])]
                       if cond == "effect" else [])

    def test_cond(self, t, vocab):
        if self.cond == "effect":
            e = mean_effect(self.tables, t)
            return torch.zeros(len(self.G)) if e is None else torch.from_numpy(e)
        return torch.tensor([vocab[t], vocab["control"]])


class Lines:
    """All training lines of a fold, in memory (float16 log-normalized on G)."""

    def __init__(self, fold, vocab, min_cells, cond, line_weights=""):
        info = json.load(open(f"{fold}/fold.json"))
        self.cond = cond
        self.lw = {k: float(v) for k, v in (kv.split("=") for kv in line_weights.split(",") if kv)}
        self.G = [g.strip() for g in open(f"{fold}/genes.txt")]
        self.gene_tok = torch.tensor([vocab[g] for g in self.G])
        self.heldout, self.hold = info["heldout"], set(info.get("holdout_targets", []))
        # held-out perturbations of a training line are invisible everywhere: not trained on, and
        # their measured effect is not available as an input either
        tables = ({l: effect_table(l, self.G, min_cells, self.hold if l == self.heldout else ())
                   for l in info["train"]} if cond == "effect" else {})
        self.lines, self.pairs, self.eff = {}, [], {}
        for l in info["train"]:
            z = np.load(f"{fold}/{l}.npz", allow_pickle=True)
            groups = {}
            for i, t in enumerate(z["target"]):
                groups.setdefault(t, []).append(i)
            groups = {t: np.array(v) for t, v in groups.items()}
            ctrl = groups.pop("__ctrl__")
            ok = [t for t, v in groups.items() if len(v) >= min_cells and t in vocab
                  and not (l == self.heldout and t in self.hold)]
            if cond == "effect":                       # need the effect measured in some OTHER line
                others = [tables[o] for o in info["train"] if o != l]
                for t in ok:
                    e = mean_effect(others, t)
                    if e is not None:
                        self.eff[(l, t)] = torch.from_numpy(e)
                ok = [t for t in ok if (l, t) in self.eff]
            self.lines[l] = dict(X=torch.from_numpy(z["X"]), ctrl=ctrl, groups=groups,
                                 genes=np.flatnonzero(z["measured_G"]))
            self.pairs += [(l, t) for t in ok]
            print(f"[data] {l}: {len(z['X'])} cells, {len(ok)} trainable perturbations, "
                  f"{len(self.lines[l]['genes'])}/{len(self.G)} genes measured")
        # test-time effects use every training line except the held-out one (its held-out targets' truth)
        self.tables = [tab for l, tab in tables.items() if not (self.hold and l == self.heldout)]
        w = np.array([self.lw.get(l, 1.0) for l, _ in self.pairs]); self.p = w / w.sum()
        print(f"[data] {len(self.pairs)} (line, perturbation) pairs, conditioning: {cond}, line weights {self.lw or 'uniform'}")

    def test_cond(self, t, vocab):
        """Conditioning for target t at test time: full-G effect vector, or token ids."""
        if self.cond == "effect":
            e = mean_effect(self.tables, t)
            return torch.zeros(len(self.G)) if e is None else torch.from_numpy(e)
        return torch.tensor([vocab[t], vocab["control"]])

    def batch(self, rng, B, n_genes, vocab):
        l, t = self.pairs[rng.choice(len(self.pairs), p=self.p)]
        d = self.lines[l]
        cols = torch.from_numpy(rng.choice(d["genes"], min(n_genes, len(d["genes"])), replace=False))
        src = d["X"][torch.from_numpy(rng.choice(d["ctrl"], B))][:, cols].float()
        tgt = d["X"][torch.from_numpy(rng.choice(d["groups"][t], B))][:, cols].float()
        if self.cond == "effect":
            cond = self.eff[(l, t)][cols].unsqueeze(0).repeat(B, 1)
        else:
            cond = torch.tensor([[vocab[t], vocab["control"]]]).repeat(B, 1)
        return src, tgt, self.gene_tok[cols].unsqueeze(0).repeat(B, 1), cond


def build_model(fold, vocab, d_model, cond="token"):
    cls = EffectModel if cond == "effect" else OriginModel
    return cls(ntoken=len(vocab), d_model=d_model, nlayers=4, fusion_method="differential_perceiver",
               perturbation_function="crisper", mask_path=f"{fold}/mask.pt")


AMP = {"dtype": None}


def velocity(vf, genes, x, t, src, cond):
    with torch.autocast("cuda", dtype=AMP["dtype"], enabled=AMP["dtype"] is not None):
        v = vf(genes, x, t, src, cond) if isinstance(vf, EffectModel) else vf(genes, x, t, src, cond, genes)
    return v.float()


def loss_fn(vf, src, tgt, genes, cond, gamma):
    t = torch.rand(src.shape[0], device=src.device)
    ps = PATH.sample(t=t, x_0=torch.randn_like(tgt), x_1=tgt)
    v = velocity(vf, genes, ps.x_t, ps.t, src, cond)
    loss = ((v - ps.dx_t) ** 2).mean()
    if gamma:
        x1_hat = ps.x_t + v * (1 - t).unsqueeze(-1)
        loss = loss + gamma * mmd2_unbiased_multi_sigma(x1_hat, tgt, median_sigmas(tgt, (0.5, 1.0, 2.0, 4.0)))
    return loss


@torch.no_grad()
def generate(vf, src, genes, cond, method="euler", steps=10):
    """Cells on the given genes. method: 'onestep' (x0 + v(x0, 0)), or a torchdiffeq solver."""
    x0 = torch.randn_like(src)
    f = lambda t, x: velocity(vf, genes, x, t.expand(x.shape[0]), src, cond)
    if method == "onestep":
        x1 = x0 + f(torch.zeros((), device=src.device), x0)
    else:
        import torchdiffeq
        x1 = torchdiffeq.odeint(f, x0, torch.linspace(0, 1, steps, device=src.device), method=method)[-1]
    return x1.clamp(min=0)


def to_counts(y, lib):
    """log1p(count/lib*1e4) -> counts, rounded to the nearest integer: values worth under half a
    count become 0, which removes the near-zero smear a continuous flow puts on silent genes."""
    return np.round(np.expm1(np.clip(y, 0, None)) * lib[:, None] / TARGET_SUM)


class Proxy:
    """Held-out pseudobulk check, on what would be submitted: generated values are turned into
    counts with each paired control cell's library size, pseudobulked, and compared as effects
    (log1p(CPM/100) of G, perturbed - control) with the held-out line's measured effects: pearson
    of effects and a PDS-like rank (is each predicted effect closest, by cosine, to its own?)."""

    def __init__(self, fold, G, n_targets, rng):
        from vcc.stats import load, FILES
        info = json.load(open(f"{fold}/fold.json"))
        self.ok = info["heldout"] in FILES           # a stage-3 line with measured perturbations (not "VCC")
        if not self.ok:
            return
        st = load(info["heldout"]); names = list(st["names"])
        gi = [list(st["genes"]).index(g) for g in G]
        cpm = st["sums"][:, gi] / st["sums"].sum(1, keepdims=True) * 1e6
        hold = set(info.get("holdout_targets", []))
        cand = [n for n, c in zip(names, st["ncell"]) if n != "__ctrl__" and c >= 50 and (not hold or n in hold)]
        self.targets = sorted(rng.choice(cand, min(n_targets, len(cand)), replace=False))
        self.ctrl = np.log1p(cpm[names.index("__ctrl__")] / 100)
        self.real = np.stack([np.log1p(cpm[names.index(t)] / 100) - self.ctrl for t in self.targets])
        z = np.load(f"{fold}/heldout_ctrl.npz")
        self.src, self.lib = torch.from_numpy(z["X"]), z["lib"]

    def __call__(self, vf, data, vocab, device, n_cells, chunk=1000, method="onestep"):
        vf.eval(); rng = np.random.default_rng(1); pred = []
        for t in self.targets:
            idx = rng.choice(len(self.src), n_cells, replace=False)
            c_full = data.test_cond(t, vocab)
            out = []
            for lo in range(0, len(data.G), chunk):
                cols = slice(lo, lo + chunk)
                src = self.src[idx][:, cols].float().to(device)
                genes = data.gene_tok[cols].to(device).unsqueeze(0).repeat(n_cells, 1)
                cond = (c_full[cols] if data.cond == "effect" else c_full).unsqueeze(0).repeat(n_cells, 1).to(device)
                out.append(generate(vf, src, genes, cond, method).cpu())
            counts = to_counts(torch.cat(out, 1).numpy(), self.lib[idx])
            m = counts.sum(0); cpm = m / max(m.sum(), 1) * 1e6
            pred.append(np.log1p(cpm / 100) - self.ctrl)
        vf.train()
        P, R = np.stack(pred), self.real
        pear = np.mean([np.corrcoef(p, r)[0, 1] for p, r in zip(P, R)])
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True); Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
        C = Pn @ Rn.T
        pds = 1 - np.mean((C > np.diag(C)[:, None]).sum(1) / (len(P) - 1))
        return dict(pearson_delta=float(pear), pds_like=float(pds),
                    effect_ratio=float(np.linalg.norm(P, axis=1).mean() / np.linalg.norm(R, axis=1).mean()))


def main():
    p = argparse.ArgumentParser()
    for k, v in Cfg.__dataclass_fields__.items():
        if v.default is v.default_factory:  # required
            p.add_argument(f"--{k}", required=True)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(v.default), default=v.default)
    cfg = Cfg(**vars(p.parse_args()))
    os.makedirs(cfg.out, exist_ok=True)
    json.dump(asdict(cfg), open(f"{cfg.out}/cfg.json", "w"), indent=1)
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    dev = torch.device("cuda")
    AMP["dtype"] = torch.bfloat16 if cfg.amp == "bf16" else None
    if cfg.amp == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    vocab = json.load(open(f"{cfg.fold}/vocab.json"))
    data = Lines(cfg.fold, vocab, cfg.min_cells, cfg.cond, cfg.line_weights)
    proxy = Proxy(cfg.fold, data.G, cfg.eval_targets, np.random.default_rng(123))
    if proxy.ok and cfg.cond == "effect":                 # reference: transfer itself, same proxy targets
        P = np.stack([data.test_cond(t, vocab).numpy() for t in proxy.targets]); R = proxy.real
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12); Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
        C = Pn @ Rn.T
        print(f"[ref] transfer on proxy targets: pearson_delta={np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(P, R)]):.4f} "
              f"pds_like={1 - np.mean((C > np.diag(C)[:, None]).sum(1) / (len(P) - 1)):.4f}", flush=True)
    vf = build_model(cfg.fold, vocab, cfg.d_model, cfg.cond).to(dev)
    print(f"[model] {type(vf).__name__} {sum(q.numel() for q in vf.parameters())/1e6:.1f}M params, "
          f"vocab {len(vocab)}, G {len(data.G)}")
    opt = torch.optim.Adam(vf.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.eta_min)
    log, t0 = [], time.time()
    for it in range(cfg.steps + 1):
        if it % cfg.eval_every == 0 and proxy.ok:
            r = proxy(vf, data, vocab, dev, cfg.eval_cells)
            print(f"[eval] it={it} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
            log.append(dict(it=it, **r))
            torch.save({"model": vf.state_dict(), "cfg": asdict(cfg), "it": it}, f"{cfg.out}/model.pt")
        if it == cfg.steps:
            break
        src, tgt, genes, cond = (x.to(dev) for x in data.batch(rng, cfg.batch, cfg.genes_per_step, vocab))
        loss = loss_fn(vf, src, tgt, genes, cond, cfg.gamma)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        log.append(dict(it=it, loss=loss.item()))
        if it % 50 == 0:
            recent = [x["loss"] for x in log[-50:] if "loss" in x]
            print(f"[train] it={it} loss={np.mean(recent):.4f} {(time.time()-t0)/(it+1):.2f}s/it "
                  f"GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    torch.save({"model": vf.state_dict(), "cfg": asdict(cfg), "it": cfg.steps}, f"{cfg.out}/model.pt")
    json.dump(log, open(f"{cfg.out}/log.json", "w"))
    print(f"[done] {time.time()-t0:.0f}s -> {cfg.out}")


if __name__ == "__main__":
    main()
