"""Read a remote HDF5 over HTTP range requests, without downloading the whole file.

The CD4 CRISPRi differential-expression statistics are a 16.8 GB HDF5 on S3, and about 850
of its 33,983 rows matter here -- the Challenge's targets across three culture conditions.
HDF5 is random access, so h5py can work through a file object that fetches byte ranges on
demand. `layers/log_fc` is contiguous rather than chunked, which means row-major, so one
row is 82 kB of adjacent bytes and the extraction costs about 0.25 GB instead of 16.8.

h5py's fileobj driver needs read/seek/tell. The block cache matters more than it looks:
opening the file alone takes dozens of small header reads that would otherwise be one
request each.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request

DEFAULT_BLOCK = 1 << 18
DEFAULT_CACHED_BLOCKS = 2048
TRANSIENT = (TimeoutError, urllib.error.URLError, ConnectionError, OSError)


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests, with a block cache."""

    def __init__(
        self,
        url: str,
        *,
        block: int = DEFAULT_BLOCK,
        cached_blocks: int = DEFAULT_CACHED_BLOCKS,
        timeout: float = 120.0,
        attempts: int = 6,
    ) -> None:
        if block < 1 or cached_blocks < 1 or attempts < 1:
            raise ValueError("block size, cache depth and attempts must be positive")
        self.url = url
        self.block = block
        self.timeout = timeout
        self.attempts = attempts
        self.requests = 0
        self.retries = 0
        self.bytes_fetched = 0
        self._pos = 0
        self._cache: dict[int, bytes] = {}
        self._order: list[int] = []
        self._cached_blocks = cached_blocks
        self.size = self._probe_size()

    def _range(self, start: int, stop: int, read):
        """One range request, retried -- thousands of these will not all succeed.

        Extracting this file costs a few thousand requests over half an hour, and a single
        timed-out read used to lose the whole run. `read` takes the open response so the
        size probe can look at headers and a data fetch can take the body.
        """
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return read(response)
            except TRANSIENT:
                if attempt == self.attempts - 1:
                    raise
                self.retries += 1
                time.sleep(min(30.0, 2.0**attempt))
        raise AssertionError("unreachable")

    def _probe_size(self) -> int:
        # Some buckets refuse HEAD but answer a one-byte range with Content-Range.
        def size_of(response) -> int:
            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
            length = response.headers.get("Content-Length")
            if length is None:
                raise OSError(f"{self.url} does not report its size")
            return int(length)

        return self._range(0, 0, size_of)

    def _fetch(self, index: int) -> bytes:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        start = index * self.block
        data = self._range(start, min(self.size, start + self.block) - 1, lambda r: r.read())
        self.requests += 1
        self.bytes_fetched += len(data)
        self._cache[index] = data
        self._order.append(index)
        if len(self._order) > self._cached_blocks:
            self._cache.pop(self._order.pop(0), None)
        return data

    def readinto(self, buffer) -> int:
        want = min(len(buffer), max(0, self.size - self._pos))
        written = 0
        while written < want:
            index, offset = divmod(self._pos, self.block)
            chunk = self._fetch(index)
            take = min(want - written, len(chunk) - offset)
            buffer[written: written + take] = chunk[offset: offset + take]
            written += take
            self._pos += take
        return written

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True
