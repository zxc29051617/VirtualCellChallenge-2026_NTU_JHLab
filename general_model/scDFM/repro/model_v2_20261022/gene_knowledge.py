"""Gene knowledge from public databases, for genes never perturbed (2026-10-23).

    .venv/bin/python -I repro/model_v2_20261022/gene_knowledge.py        (from scDFM/)

STRING v12 human (data/external/string): protein-protein association network, combined score >= 400, mapped to gene
symbols (protein.info preferred_name) on the VCC gene axis; embedding = top-128 eigenvectors of the normalised adjacency
D^-1/2 A D^-1/2 scaled by sqrt(eigenvalue) (genes linked in the network get similar vectors).
Gene Ontology (data/external/go/goa_human.gaf.gz): gene x GO term (NOT qualifiers dropped; terms with 3-1000 genes on
the axis), TF-IDF weighted, rows normalised, truncated SVD to 128.
Writes data/stage3/model_v2/gene_knowledge.npz (genes, string, go, has_string, has_go). Neither source uses any
perturbation data.
"""
import gzip, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from sklearn.utils.extmath import randomized_svd

EXT = "/home/zxc29051617/VCC/data/external"; OUT = "/home/zxc29051617/VCC/data/stage3/model_v2/gene_knowledge.npz"
genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); G = len(genes)
gi = {g: i for i, g in enumerate(genes)}

info = pd.read_csv(f"{EXT}/string/9606.protein.info.v12.0.txt.gz", sep="\t", usecols=[0, 1])
p2g = {p: gi[g] for p, g in zip(info.iloc[:, 0], info.iloc[:, 1]) if g in gi}
rows, cols, vals = [], [], []
with gzip.open(f"{EXT}/string/9606.protein.links.v12.0.txt.gz", "rt") as fh:
    next(fh)
    for line in fh:
        a, b, s = line.split(); s = int(s)
        if s < 400: continue
        i, j = p2g.get(a), p2g.get(b)
        if i is None or j is None or i == j: continue
        rows.append(i); cols.append(j); vals.append(s / 1000.0)
A = sp.csr_matrix((vals, (rows, cols)), shape=(G, G)); A = A.maximum(A.T)
deg = np.asarray(A.sum(1)).ravel(); has_string = deg > 0
dinv = sp.diags(np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1e-12)), 0))
N = (dinv @ A @ dinv).tocsr()
U, S, _ = randomized_svd(N, 128, n_iter=6, random_state=0)
string_emb = (U * np.sqrt(S)).astype(np.float32)
print(f"STRING: {A.nnz // 2:,} links (score >= 400) among {has_string.sum():,} of {G:,} genes", flush=True)

ann = pd.read_csv(f"{EXT}/go/goa_human.gaf.gz", sep="\t", comment="!", header=None, usecols=[2, 3, 4], dtype=str)
ann = ann[~ann[3].str.contains("NOT", na=False) & ann[2].isin(gi)].drop_duplicates([2, 4])
terms = ann[4].value_counts(); terms = terms[(terms >= 3) & (terms <= 1000)].index; ti = {t: k for k, t in enumerate(terms)}
ann = ann[ann[4].isin(ti)]
M = sp.csr_matrix((np.ones(len(ann)), ([gi[g] for g in ann[2]], [ti[t] for t in ann[4]])), shape=(G, len(terms)))
idf = np.log(G / np.maximum(np.asarray((M > 0).sum(0)).ravel(), 1)); M = M @ sp.diags(idf)
nrm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); has_go = nrm > 0
M = sp.diags(np.where(nrm > 0, 1 / np.maximum(nrm, 1e-12), 0)) @ M
U, S, _ = randomized_svd(M.tocsr(), 128, n_iter=6, random_state=0)
go_emb = (U * S).astype(np.float32)
print(f"GO: {len(terms):,} terms, {has_go.sum():,} of {G:,} genes annotated", flush=True)

np.savez_compressed(OUT, genes=np.array(genes), string=string_emb, go=go_emb, has_string=has_string, has_go=has_go)
print(f"wrote {OUT}")
