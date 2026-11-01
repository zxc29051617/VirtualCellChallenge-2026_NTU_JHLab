"""Gene positions (GENCODE v46 basic) on the VCC gene axis: chromosome, TSS and strand per gene symbol."""
import gzip, os, re
import numpy as np, pandas as pd

GTF = "/home/zxc29051617/VCC/data/external/gencode/gencode.v46.basic.annotation.gtf.gz"
CACHE = "/home/zxc29051617/VCC/data/external/gencode/genes_v46.tsv"


def gene_table():
    if not os.path.exists(CACHE):
        rows = []
        with gzip.open(GTF, "rt") as f:
            for line in f:
                if line.startswith("#"): continue
                x = line.split("\t", 9)
                if x[2] != "gene": continue
                name = re.search(r'gene_name "([^"]+)"', x[8]).group(1)
                gtype = re.search(r'gene_type "([^"]+)"', x[8]).group(1)
                s, e = int(x[3]), int(x[4])
                rows.append((name, x[0], s if x[6] == "+" else e, x[6], s, e, gtype))
        df = pd.DataFrame(rows, columns=["gene", "chr", "tss", "strand", "start", "end", "type"])
        df = df.drop_duplicates("gene")          # first (usually the canonical) entry per symbol
        df.to_csv(CACHE, sep="\t", index=False)
    return pd.read_csv(CACHE, sep="\t")


def on_axis(genes):
    """chr (str, '' if unknown), tss (int, -1), strand (+1/-1/0) arrays aligned with `genes`."""
    t = gene_table().set_index("gene")
    chr_ = np.array([t.chr.get(g, "") for g in genes]); tss = np.array([t.tss.get(g, -1) for g in genes])
    strand = np.array([{"+": 1, "-": -1}.get(t.strand.get(g, ""), 0) for g in genes])
    return chr_, tss, strand
