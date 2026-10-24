"""Download one large file over many parallel HTTP range requests, resumable (2026-10-22).

    python par_download.py <url> <out> [--workers 128] [--part-mb 8]

S3 from this machine gives ~0.1 MB/s per connection but scales with connections (32 -> 3 MB/s). Parts are written
in place into a preallocated <out>.part; finished part numbers are appended to <out>.done so a rerun continues.
The file is renamed to <out> when every part is in.
"""
import argparse, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

p = argparse.ArgumentParser(); p.add_argument("url"); p.add_argument("out")
p.add_argument("--workers", type=int, default=128); p.add_argument("--part-mb", type=int, default=8)
a = p.parse_args()
size = int(urllib.request.urlopen(urllib.request.Request(a.url, method="HEAD"), timeout=60).headers["Content-Length"])
P = a.part_mb << 20; nparts = -(-size // P); tmp, done_path, meta_path = a.out + ".part", a.out + ".done", a.out + ".meta"
meta = f"{a.url} {size} {P}\n"                          # progress is only reused for this exact download
if not os.path.exists(tmp) or not os.path.exists(meta_path) or open(meta_path).read() != meta:
    for f_ in (done_path, tmp):
        if os.path.exists(f_): os.remove(f_)
    with open(tmp, "wb") as fh: fh.truncate(size)
    open(meta_path, "w").write(meta)
done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
todo = [k for k in range(nparts) if k not in done]
lock = threading.Lock(); fd = os.open(tmp, os.O_WRONLY); got = [0]; t0 = time.time(); last = [t0]
dlog = open(done_path, "a")
print(f"{size/1e9:.2f} GB in {nparts} parts of {a.part_mb} MB; {len(done)} already done, {len(todo)} to fetch, {a.workers} workers", flush=True)


def fetch(k):
    lo = k * P; hi = min(size, lo + P) - 1
    for att in range(20):
        try:
            req = urllib.request.Request(a.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = r.read()
            if len(d) != hi - lo + 1: raise IOError(f"short read {len(d)}")
            os.pwrite(fd, d, lo)
            with lock:
                dlog.write(f"{k}\n"); dlog.flush(); got[0] += len(d)
                now = time.time()
                if now - last[0] > 60:
                    last[0] = now; rate = got[0] / 1e6 / (now - t0); left = (size - (len(done) * P) - got[0]) / 1e6
                    print(f"{time.strftime('%H:%M:%S')} {got[0]/1e9:.2f} GB this run, {rate:.1f} MB/s, ~{left / max(rate, 1e-6) / 60:.0f} min left", flush=True)
            return k
        except Exception as e:
            time.sleep(min(60, 3 * (att + 1)))
    raise IOError(f"part {k} failed")


with ThreadPoolExecutor(a.workers) as ex:
    for fut in as_completed([ex.submit(fetch, k) for k in todo]): fut.result()
os.close(fd); dlog.close()
done = set(int(x) for x in open(done_path).read().split())
assert len(done) == nparts, f"{nparts - len(done)} parts missing; rerun to continue"
os.replace(tmp, a.out); os.remove(done_path); os.remove(meta_path)
print(f"done: {a.out} ({size/1e9:.2f} GB) in {(time.time()-t0)/60:.1f} min", flush=True)
