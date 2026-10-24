"""Parallel HTTP range downloader (resumable). Re-resolves redirects per chunk so signed
S3 URLs that expire mid-download are refreshed.
    python pget.py <url> <out> [--conns 16] [--chunk-mb 32] [--range START:END]
--range fetches only bytes [START, END) of the remote file into <out> (offset 0 of <out> = START).
"""
import argparse, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
p.add_argument("url"); p.add_argument("out")
p.add_argument("--conns", type=int, default=16); p.add_argument("--chunk-mb", type=int, default=32)
p.add_argument("--range", default="")
a = p.parse_args()

def resolve():
    req = urllib.request.Request(a.url, method="GET", headers={"Range": "bytes=0-0", "User-Agent": "pget"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Range"].split("/")[1]); return r.geturl(), total

url, total = resolve()
base = 0
if a.range:
    lo_, hi_ = (int(x) for x in a.range.split(":"))
    base, total = lo_, min(hi_, total) - lo_
CH = a.chunk_mb << 20
n = (total + CH - 1) // CH
done_path = a.out + ".done"
done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
if not os.path.exists(a.out + ".part"):
    with open(a.out + ".part", "wb") as f: f.truncate(total)
lock = threading.Lock(); t0 = time.time(); got = [0]
state = {"url": url, "t": time.time()}

def fetch(i):
    lo, hi = i * CH, min(total, (i + 1) * CH) - 1
    for attempt in range(8):
        try:
            if time.time() - state["t"] > 300:          # refresh signed URL every 5 min
                with lock:
                    if time.time() - state["t"] > 300: state["url"], _ = resolve(); state["t"] = time.time()
            req = urllib.request.Request(state["url"], headers={"Range": f"bytes={base + lo}-{base + hi}", "User-Agent": "pget"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != hi - lo + 1: raise IOError(f"short read {len(data)}")
            with open(a.out + ".part", "r+b") as f: f.seek(lo); f.write(data)
            with lock:
                done.add(i); got[0] += len(data)
                with open(done_path, "a") as f: f.write(f"{i}\n")
            return
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            if attempt >= 2:
                with lock: state["url"], _ = resolve(); state["t"] = time.time()
    raise RuntimeError(f"chunk {i} failed")

todo = [i for i in range(n) if i not in done]
print(f"{os.path.basename(a.out)}: {total/2**30:.2f} GiB, {len(todo)}/{n} chunks to fetch", flush=True)
with ThreadPoolExecutor(a.conns) as ex:
    for k, _ in enumerate(ex.map(fetch, todo)):
        if k % 20 == 0:
            el = time.time() - t0
            print(f"  {len(done)}/{n} chunks  {got[0]/el/2**20:.1f} MiB/s", flush=True)
assert len(done) == n
os.replace(a.out + ".part", a.out); os.remove(done_path)
print(f"done {a.out} in {time.time()-t0:.0f}s", flush=True)
