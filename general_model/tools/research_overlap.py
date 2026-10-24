#!/usr/bin/env python3
"""Overlap of a perturbation gene list with the 300 VCC 2026 target genes.

Usage:
  python3 -I overlap.py GENELIST [--col NAME] [--sep ,|\t] [--show]
  GENELIST: text file, one symbol (or Ensembl ID) per line; or a CSV/TSV with --col.
           Use '-' to read from stdin. Entries may contain '+' or '_' combos (split) - see --split.
Reports: exact symbol matches, plus matches rescued via HGNC previous/alias symbols or Ensembl IDs.
"""
import sys, csv, gzip, argparse, re
TARGETS = '/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv'
HGNC = '/tmp/vcc_research/dl/hgnc/hgnc_complete_set.txt'

def load_targets():
    with open(TARGETS) as f:
        r = csv.DictReader(f)
        return [row['target_gene'].strip() for row in r if row['target_gene'].strip()]

def load_hgnc():
    approved, alias, ens = {}, {}, {}
    amb = set()
    with open(HGNC, encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            s = row['symbol']
            approved[s.upper()] = s
            if row.get('ensembl_gene_id'):
                ens[row['ensembl_gene_id']] = s
            for fld in ('prev_symbol', 'alias_symbol'):
                for a in (row.get(fld) or '').replace('"', '').split('|'):
                    a = a.strip().upper()
                    if not a: continue
                    if a in alias and alias[a] != s: amb.add(a)
                    alias[a] = s
    for a in amb: alias.pop(a, None)
    return approved, alias, ens

def opener(p):
    if p == '-': return sys.stdin
    return gzip.open(p, 'rt', encoding='utf-8', errors='replace') if p.endswith('.gz') else open(p, encoding='utf-8', errors='replace')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('genelist'); ap.add_argument('--col'); ap.add_argument('--sep')
    ap.add_argument('--split', default=r'[+;,|]', help='regex to split combo entries (default + ; , |)')
    ap.add_argument('--show', action='store_true', help='print matched and missing targets')
    a = ap.parse_args()
    targets = load_targets(); tset = set(targets)
    approved, alias, ens = load_hgnc()
    raw = []
    with opener(a.genelist) as f:
        if a.col:
            sep = a.sep or ('\t' if a.genelist.rstrip('.gz').endswith(('.tsv', '.txt')) else ',')
            for row in csv.DictReader(f, delimiter=sep):
                raw.append(row.get(a.col, '') or '')
        else:
            raw = [l.strip() for l in f]
    genes = set()
    for x in raw:
        for g in re.split(a.split, x):
            g = g.strip().strip('"')
            if g: genes.add(g)
    exact = tset & genes
    up = {g.upper(): g for g in genes}
    rescued = {}
    for t in targets:
        if t in exact: continue
        if t.upper() in up: rescued[t] = up[t.upper()]; continue
    # map dataset genes to approved symbols via alias/prev/ensembl
    for g in genes:
        gg = g.split('.')[0]
        sym = None
        if gg.startswith('ENSG'): sym = ens.get(gg)
        elif g.upper() in approved: sym = None  # approved symbol, exact handled above
        else: sym = alias.get(g.upper())
        if sym and sym in tset and sym not in exact and sym not in rescued:
            rescued[sym] = g
    nonapproved_targets = [t for t in targets if t.upper() not in approved]
    # targets that are outdated symbols (e.g. TMEM104 -> SLC38A12): match dataset genes on the new symbol
    for t in nonapproved_targets:
        newsym = alias.get(t.upper())
        if t in exact or t in rescued or not newsym: continue
        for g in genes:
            gg = g.split('.')[0]
            if g.upper() == newsym.upper() or (gg.startswith('ENSG') and ens.get(gg) == newsym):
                rescued[t] = g; break
    print(f'dataset genes: {len(genes)} | targets: {len(tset)} | exact match: {len(exact)} | '
          f'+rescued (case/alias/prev/Ensembl): {len(rescued)} | total overlap: {len(exact)+len(rescued)}')
    if rescued: print('rescued:', ', '.join(f'{k}<-{v}' for k, v in sorted(rescued.items())))
    if nonapproved_targets: print('note: targets not current HGNC symbols:', nonapproved_targets)
    if a.show:
        hit = sorted(exact | set(rescued)); miss = sorted(tset - set(hit))
        print('MATCHED:', ' '.join(hit)); print('MISSING:', ' '.join(miss))
main()
