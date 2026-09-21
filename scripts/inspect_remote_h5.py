"""Bounded, read-only HTTP Range metadata inspection. Never falls back to full download."""
import argparse
import io
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, cache):
        self.url, self.size, self.position = url, size, 0
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.block = 65536
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size > 32 * 1024**2:
            raise ValueError("Metadata read exceeds 32 MiB safety budget")
        chunks = []
        while size > 0:
            block = self.position // self.block
            path = self.cache / f"{block}.bin"
            if not path.exists():
                if self.transferred > 128 * 1024**2:
                    raise ValueError("Remote transfer exceeds metadata budget")
                lo = block * self.block
                hi = min(self.size - 1, lo + self.block - 1)
                url = self.url + ("&" if "?" in self.url else "?") + f"audit_range={lo}"
                request = urllib.request.Request(url,
                    headers={"Range": f"bytes={lo}-{hi}", "User-Agent": "VCC-metadata-audit/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise ValueError("Server ignored HTTP Range; refusing full download")
                    if response.headers.get("Content-Range") != f"bytes {lo}-{hi}/{self.size}":
                        raise ValueError("Server returned the wrong HTTP byte range")
                    content = response.read(hi - lo + 1)
                if len(content) != hi - lo + 1:
                    raise ValueError(f"Unexpected range {lo}-{hi}: got {len(content)} bytes")
                path.write_bytes(content)
                self.transferred += len(content)
            content = path.read_bytes()
            offset = self.position % self.block
            take = min(size, len(content) - offset)
            if take <= 0:
                raise ValueError("Unexpected EOF")
            chunks.append(content[offset:offset + take])
            size -= take
            self.position += take
        return b"".join(chunks)


def strings(node):
    if isinstance(node, h5py.Group):
        if "categories" in node:
            node = node["categories"]
        elif "values" in node:
            node = node["values"]
        else:
            return {"keys": list(node)}
    if node.size > 200000:
        return {"shape": node.shape, "dtype": str(node.dtype)}
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(node[:]).ravel()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("size", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = RangeReader(args.url, args.size, args.output.with_suffix(".range64_cache"))
    result = {"url": args.url, "bytes": args.size}
    with h5py.File(reader, "r") as f:
        result["keys"] = list(f)
        for frame in ("obs", "var"):
            result[frame] = {name: strings(node) for name, node in f[frame].items()
                             if frame == "obs" or name in ("_index", "index", "gene",
                                                           "gene_name", "gene_names")}
            result[frame + "_columns"] = list(f[frame])
        x = f["X"]
        shape = x.shape if isinstance(x, h5py.Dataset) else x.attrs["shape"]
        result["X"] = {"type": str(type(x)),
            "shape": list(map(int, shape)) if shape is not None else None}
    result["downloaded_bytes"] = reader.transferred
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("obs", "var")}, indent=2))
