"""Serial generation + official scoring for the source-weight experiment (resumable).

    python evaluate_runner.py [outer ...]        (pertrelay venv; scoring runs evalenv/bin/cell-eval2, one at a time)

For every outer hold-out (H1, HCT116, KOLF200), generation seed (0,1,2) and group (A, B, C, D, A_shuf): build 200
predicted cells per target from the hold-out's NTC cells with per-target independent random streams
([seed, target index, 0] for the control indices, shared by all groups; [seed, target index, 1] for the thinning /
Poisson counts), original-library up-regulation formula, lfc clipped to +-4; score against the fixed local bundle
(cell-eval2 0.18.0, vcc2026 preset, de.backend=pdex; real set and bundle are the existing, read-only files under
data/stage3/loco/<set>/). The prediction file is deleted after scoring; score directories are kept.
"""
import hashlib, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"; sys.path.insert(0, HERE)
import sigexp as S
CE2 = f"{ROOT}/evalenv/bin/cell-eval2"; N_PRED = 200; SEEDS = (0, 1, 2); GROUPS = ["A", "B", "C", "D", "A_shuf"]
COMMON = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting", "--input-type", "counts", "--set", "de.backend=pdex"]
ENV = {**os.environ, **{k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS")}}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"; print(line, flush=True); open(f"{HERE}/run.log", "a").write(line + "\n")


def status(**kw):
    p = f"{HERE}/status.json"; s = json.load(open(p)); s.update(kw); s["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); json.dump(s, open(p, "w"), indent=1)


def avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 2**20


def generate(name, seed, group, R, C, lib, rate):
    o = S.OUTER[name]; z = np.load(f"{HERE}/effects/{name}_{group}.npz", allow_pickle=True)
    targets = list(z["targets"]); lfc = {t: np.clip(np.nan_to_num(v), -4, 4) for t, v in zip(targets, z["lfc"])}
    blocks, labels = [], []
    for i, t in enumerate(targets):
        draw = np.random.default_rng([seed, i, 0]).choice(C.shape[0], N_PRED, replace=False); rng = np.random.default_rng([seed, i, 1])
        cells = C[draw].tocsr(copy=True); fc = 2.0 ** lfc[t]; f = fc[cells.indices]; down = f < 1
        d = cells.data.astype(np.int64); d[down] = rng.binomial(d[down], f[down]); cells.data = d.astype(np.float32)
        up = np.flatnonzero(fc > 1)
        if len(up):
            mu = lib[draw][:, None] * rate[None, up] * (fc[up] - 1)[None, :]
            add = sp.csr_matrix(rng.poisson(mu).astype(np.float32)); cells = cells + sp.csr_matrix((add.data, up[add.indices], add.indptr), shape=cells.shape)
        cells.eliminate_zeros(); blocks.append(cells); labels += [t] * N_PRED
    P = ad.AnnData(sp.vstack(blocks + [C]).tocsr(), obs=pd.DataFrame({"target_gene": labels + ["non-targeting"] * C.shape[0]}), var=pd.DataFrame(index=R.var_names))
    P.obs_names = [f"p{i}" for i in range(P.n_obs)]
    return P


def main(names):
    total = len(names) * len(SEEDS) * len(GROUPS); done = 0
    for name in names:
        o = S.OUTER[name]; D = f"{ROOT}/data/stage3/loco/{o['loco']}"
        R = ad.read_h5ad(f"{D}/real.h5ad"); C = sp.csr_matrix(R[R.obs.target_gene == "non-targeting"].X)
        lib = np.asarray(C.sum(1)).ravel(); rate = np.asarray(C.sum(0)).ravel() / lib.sum()
        for seed in SEEDS:
            for group in GROUPS:
                tag = f"{name}_{group}_seed{seed}"; out = f"{HERE}/scores/{tag}"
                if os.path.exists(f"{out}/score.csv"): done += 1; continue
                while avail_gb() < 5: log(f"waiting for memory ({avail_gb():.1f} GB available)"); time.sleep(60)
                t0 = time.time(); os.makedirs(f"{HERE}/pred", exist_ok=True); pf = f"{HERE}/pred/{tag}.h5ad"
                P = generate(name, seed, group, R, C, lib, rate); P.write_h5ad(pf)
                shutil.rmtree(out, ignore_errors=True)
                subprocess.run([CE2, "run", "-ap", pf, "-ar", f"{D}/real.h5ad", "-o", out, *COMMON], check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                agg = [f for f in os.listdir(out) if f.endswith("agg_results.csv")][0]
                subprocess.run([CE2, "score", "--user-agg", f"{out}/{agg}", "--real-bundle", f"{D}/bundle", "-o", f"{out}/score.csv"], check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(pf); done += 1
                avg = pd.read_csv(f"{out}/score.csv").set_index("metric").loc["avg_score", "from_replicate"]
                log(f"score {done}/{total}: {tag} avg {avg:.4f} ({time.time() - t0:.0f}s, {P.n_obs} cells, {P.X.nnz / P.n_obs:.0f} nnz/cell)")
                status(step=f"7 scoring {done}/{total}", last_scored=tag)
    log("scoring queue finished")


if __name__ == "__main__":
    main(sys.argv[1:] or list(S.OUTER))
