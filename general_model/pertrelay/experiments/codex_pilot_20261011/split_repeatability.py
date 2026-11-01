import json, pathlib
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
p = pathlib.Path('/home/zxc29051617/VCC/pertrelay/experiments/codex_pilot_20261011')
a = ad.read_h5ad(p / 'data/real.h5ad')
x = sp.csr_matrix(a.X)
labels = a.obs.target_gene.astype(str).to_numpy()
targets = json.loads((p/'manifest.json').read_text())['targets']
ctrl = np.flatnonzero(labels == 'non-targeting')
cs = np.asarray(x[ctrl].sum(0)).ravel()
mask = (cs / cs.sum() * 50000 >= 0.25) & ~np.isin(a.var_names, targets)
def profile(rows):
    z = np.asarray(x[rows].sum(0)).ravel().astype(float)
    return np.log1p(50000*z/z.sum())
rows = []
for seed in range(10):
    rng = np.random.default_rng([20261011, seed, 971])
    ci = rng.permutation(ctrl)
    ca, cb = profile(ci[:len(ci)//2]), profile(ci[len(ci)//2:])
    for target in targets:
        ti = rng.permutation(np.flatnonzero(labels == target))
        da = (profile(ti[:len(ti)//2])-ca)[mask]
        db = (profile(ti[len(ti)//2:])-cb)[mask]
        rows.append({'target':target, 'split_seed':seed,
                     'effect_pearson':float(np.corrcoef(da,db)[0,1]),
                     'effect_cosine':float(np.dot(da,db)/(np.linalg.norm(da)*np.linalg.norm(db)))})
d = pd.DataFrame(rows)
d.to_csv(p/'split_half_repeatability.csv',index=False)
s = d.groupby('target')[['effect_pearson','effect_cosine']].mean()
s.to_csv(p/'split_half_repeatability_by_target.csv')
result = {'median_target_mean_pearson':float(s.effect_pearson.median()),
          'median_target_mean_cosine':float(s.effect_cosine.median()),
          'genes_used':int(mask.sum()), 'targets':len(targets), 'random_splits':10,
          'description':'Independent half-controls and half-perturbed cells; log1p CP50000 effects; all 12 target genes excluded; control-expressed genes only',
          'limitation':'Exploratory within-screen split-cell repeatability, not biological replication or an official score ceiling'}
(p/'split_half_repeatability_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
