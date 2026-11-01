"""CRISPRi cis effects: does knocking down a target also lower genes next to it on the chromosome?

For each context, every perturbation with a positioned target, and every other expressed gene on the same
chromosome within 200 kb of the target's TSS: the common-mode-removed delta, averaged per distance bin of the
neighbour's TSS (and separately for head-to-head 'bidirectional' partners: opposite strand, TSS within 2 kb).
Background = the same average over genes 1-5 Mb away.

    python -m vcc.cis K562ess HepG2 Jurkat K562 H1 CD4_Rest
"""
import sys
import numpy as np
from vcc.contexts import load_all
from vcc.genome import on_axis

BINS = [0, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5]


def cis_profile(c, min_e=1.0):
    chr_, tss, strand = on_axis(c.genes)
    D = c.delta - np.nanmean(c.delta, 0, keepdims=True)
    gi = {g: k for k, g in enumerate(c.genes)}
    expr = c.measured & (c.ctrl > min_e) & (tss >= 0)
    sums = np.zeros(len(BINS) - 1); cnt = np.zeros(len(BINS) - 1); bid = []; bg = []
    for j, p in enumerate(c.perts):
        k = gi.get(p)
        if k is None or tss[k] < 0: continue
        same = expr & (chr_ == chr_[k]); same[k] = False
        d = np.abs(tss - tss[k]); row = D[j]
        for b in range(len(BINS) - 1):
            m = same & (d >= BINS[b]) & (d < BINS[b + 1]) & ~np.isnan(row)
            sums[b] += row[m].sum(); cnt[b] += m.sum()
        m = same & (d < 2e3) & (strand == -strand[k]) & ~np.isnan(row); bid += list(row[m])
        m = same & (d > 1e6) & (d < 5e6) & ~np.isnan(row); bg.append(row[m].mean() if m.any() else np.nan)
    return sums / np.maximum(cnt, 1), cnt, np.array(bid), np.nanmean(bg)


if __name__ == "__main__":
    C = load_all(sys.argv[1:])
    hdr = " ".join(f"{int(BINS[b] / 1e3):>3d}-{int(BINS[b + 1] / 1e3):<4d}" for b in range(len(BINS) - 1))
    print(f"{'context':12s} kb: {hdr}  bidir(n)        bg")
    for n, c in C.items():
        m, cnt, bid, bg = cis_profile(c)
        kd = np.nanmedian(c.kd)
        print(f"{n:12s}     " + " ".join(f"{x:8.3f}" for x in m) + f"  {np.mean(bid) if len(bid) else np.nan:6.3f}({len(bid)})  {bg:7.4f}  own {kd:.2f}")
        print(f"{'':12s} n:  " + " ".join(f"{int(x):8d}" for x in cnt))
