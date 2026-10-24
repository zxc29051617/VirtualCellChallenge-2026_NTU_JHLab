"""Read-only file object over HTTP range requests with a block cache, timeouts and retries.
Re-resolves redirects (signed S3 URLs from figshare expire) every few minutes or after an error."""
import time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor


class RemoteFile:
    def __init__(self, url, block=1 << 20, max_blocks=512, verbose=True):
        self.url0, self.block, self.max_blocks = url, block, max_blocks
        self.cache, self.pos, self.lock = {}, 0, threading.Lock()
        self.nreq = 0; self.nbytes = 0; self.verbose = verbose; self.t0 = time.time()
        self._resolve()

    def _resolve(self):
        req = urllib.request.Request(self.url0, headers={"Range": "bytes=0-0", "User-Agent": "rf"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1]); self.url = r.geturl()
        self.t_resolved = time.time()

    def _get(self, lo, hi):
        for attempt in range(10):
            try:
                if time.time() - self.t_resolved > 240: self._resolve()
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi - 1}", "User-Agent": "rf"})
                with urllib.request.urlopen(req, timeout=60) as r: data = r.read()
                if len(data) != hi - lo: raise IOError("short read")
                self.nreq += 1; self.nbytes += len(data)
                if self.verbose and self.nreq % 25 == 0:
                    print(f'    [remote] {self.nreq} requests, {self.nbytes / 2**20:.0f} MiB, {time.time() - self.t0:.0f}s', flush=True)
                return data
            except Exception:
                time.sleep(1 + attempt)
                try: self._resolve()
                except Exception: pass
        raise IOError(f"range {lo}-{hi} failed")

    def _block(self, b):
        d = self.cache.get(b)
        if d is None:
            lo = b * self.block; d = self._get(lo, min(self.size, lo + self.block))
            if len(self.cache) >= self.max_blocks: self.cache.pop(next(iter(self.cache)))
            self.cache[b] = d
        return d

    def prefetch(self, lo, hi, conns=16):
        """Fetch all blocks of [lo, hi) in parallel into the cache."""
        bs = [b for b in range(lo // self.block, (hi - 1) // self.block + 1) if b not in self.cache]
        with ThreadPoolExecutor(conns) as ex:
            for b, d in zip(bs, ex.map(lambda b: self._get(b * self.block, min(self.size, (b + 1) * self.block)), bs)):
                self.cache[b] = d

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos)); out = []
        lo, hi = self.pos, self.pos + n
        if n > 4 * self.block: self.prefetch(lo, hi)
        b = lo // self.block
        while lo < hi:
            d = self._block(b); s = lo - b * self.block; e = min(len(d), s + hi - lo)
            out.append(d[s:e]); lo += e - s; b += 1
        self.pos = hi; return b"".join(out)

    def readinto(self, buf):
        d = self.read(len(buf)); buf[:len(d)] = d; return len(d)

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos

    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def close(self): pass
