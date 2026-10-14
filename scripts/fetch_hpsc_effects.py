"""Stream the hPSC genome-wide CRISPRi LFC table and keep the Challenge's targets.

Source: Feng et al., Cell Genomics 2025, "A genome-scale single-cell CRISPRi map of trans gene
regulation across human pluripotent stem cell lines" (figshare 26819743, MIT license). The
genome-wide screen has fitness-gene and non-fitness-gene arms, so unlike the essential-gene
screens it includes targets like ours. `GenomeWideScreen_LFC_byGene.tsv.gz` is a long table:
Target, Expressed_Gene_Symbol, Expressed_Gene_Ens_ID, lfc, pval_adj.

Streams the 0.8 GB file once without storing it, reports coverage of the Challenge targets, and
writes the same .npz layout the other screens use.

    python scripts/fetch_hpsc_effects.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

FILE = "GenomeWideScreen_LFC_byGene.tsv.gz"
ARTICLE = "https://api.figshare.com/v2/articles/26819743"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perts", default="dataset/pert_counts.csv")
    parser.add_argument("--output", default="external/profiles/hPSC_target_lfc.npz")
    parser.add_argument("--local", default=None,
                        help="a downloaded copy of the .tsv.gz; avoids a fragile 0.8 GB single stream "
                             "(curl -L -C - --retry 20 --retry-all-errors resumes)")
    parser.add_argument("--min-tested", type=float, default=0.9)
    args = parser.parse_args()

    wanted = {l.strip() for l in Path(args.perts).read_text().splitlines()[1:] if l.strip()}
    meta = json.load(urllib.request.urlopen(
        urllib.request.Request(ARTICLE, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))
    url = next(f["download_url"] for f in meta["files"] if f["name"] == FILE)

    started = time.perf_counter()
    all_targets: set[str] = set()
    rows: dict[str, dict[str, float]] = {}
    n = 0
    if args.local:
        print(f"reading {args.local}", flush=True)
        source = open(args.local, "rb")
    else:
        source = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300)
    with source as response:
        stream = gzip.open(response, "rt", encoding="utf-8")
        header = stream.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        print("columns:", header, flush=True)
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            target = parts[col["Target"]]
            all_targets.add(target)
            if target in wanted:
                try:
                    rows.setdefault(target, {})[parts[col["Expressed_Gene_Symbol"]]] = float(parts[col["lfc"]])
                except ValueError:
                    pass
            n += 1
            if n % 20_000_000 == 0:
                print(f"  {n / 1e6:.0f}M rows, {len(all_targets):,} targets, "
                      f"{len(rows)} of ours, {time.perf_counter() - started:.0f}s", flush=True)

    print(f"\n{n:,} rows, {len(all_targets):,} distinct targets in {time.perf_counter() - started:.0f}s")
    print(f"Challenge targets present: {len(rows)}/{len(wanted)}")
    if not rows:
        return 0

    targets = sorted(rows)
    genes_all = sorted({g for r in rows.values() for g in r})
    index = {g: i for i, g in enumerate(genes_all)}
    lfc = np.full((len(targets), len(genes_all)), np.nan, dtype=np.float32)
    for i, t in enumerate(targets):
        for g, v in rows[t].items():
            lfc[i, index[g]] = v
    keep = (~np.isnan(lfc)).mean(0) >= args.min_tested
    effects = np.nan_to_num(lfc[:, keep])  # natural log fold change as published
    np.savez_compressed(
        args.output,
        log_fc=effects.astype(np.float32),
        targets=np.asarray(targets),
        conditions=np.asarray(["hPSC"] * len(targets)),
        genes=np.asarray([g for g, k in zip(genes_all, keep) if k]),
    )
    print(f"wrote {args.output}: {effects.shape[0]} targets x {effects.shape[1]:,} genes, "
          f"median |lfc| {np.median(np.abs(effects)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
