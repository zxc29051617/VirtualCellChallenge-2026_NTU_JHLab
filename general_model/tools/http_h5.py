"""Read-only file object over HTTP range requests with a block cache, for h5py on a remote .h5ad (2026-10-22)."""
import io, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


class RangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20, size=None):
        self.url, self.block, self.pos, self.cache, self.nreq, self.nbytes = url, block, 0, {}, 0, 0
        if size is None:
            req = urllib.request.Request(url, method="HEAD")
            size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
        self.size = size

    def _get(self, lo, hi):                                     # bytes [lo, hi)
        for a in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi - 1}"})
                d = urllib.request.urlopen(req, timeout=300).read(); self.nreq += 1; self.nbytes += len(d); return d
            except Exception:
                time.sleep(2 * (a + 1))
        raise IOError(f"range {lo}-{hi} failed")

    def prefetch(self, ranges, workers=48):
        """Fetch the blocks covering the given byte ranges in parallel into the cache."""
        need = sorted({b for lo, hi in ranges for b in range(lo // self.block, (hi - 1) // self.block + 1)} - set(self.cache))
        def one(b):
            lo = b * self.block; return b, self._get(lo, min(self.size, lo + self.block))
        with ThreadPoolExecutor(workers) as ex:
            for b, d in ex.map(one, need): self.cache[b] = d
        return len(need)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0: return 0
        out = bytearray(); lo = self.pos
        while len(out) < n:
            b = (lo + len(out)) // self.block
            if b not in self.cache: self.cache[b] = self._get(b * self.block, min(self.size, (b + 1) * self.block))
            d = self.cache[b]; s = (lo + len(out)) - b * self.block
            out += d[s:s + n - len(out)]
        buf[:n] = out; self.pos += n; return n
