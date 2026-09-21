import io
from typing import Self

import pytest

from vcc2026 import remote
from vcc2026.remote import HttpRangeFile

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes, every one distinguishable by position


class _Response(io.BytesIO):
    def __init__(self, data: bytes, headers: dict[str, str]) -> None:
        super().__init__(data)
        self.headers = headers

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _server(payload: bytes = PAYLOAD, *, failures: int = 0, log: list | None = None):
    """A urlopen that serves byte ranges, after refusing the first `failures` calls."""
    state = {"left": failures}

    def urlopen(request, timeout=None):  # matches urllib's signature
        if state["left"] > 0:
            state["left"] -= 1
            raise TimeoutError("the read operation timed out")
        start, stop = request.headers["Range"].removeprefix("bytes=").split("-")
        start, stop = int(start), int(stop)
        if log is not None:
            log.append((start, stop))
        chunk = payload[start: stop + 1]
        return _Response(
            chunk,
            {
                "Content-Range": f"bytes {start}-{stop}/{len(payload)}",
                "Content-Length": str(len(chunk)),
            },
        )

    return urlopen


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)


def test_reads_the_whole_payload_through_range_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote.urllib.request, "urlopen", _server())
    handle = HttpRangeFile("https://example.invalid/x.h5", block=1024)
    assert handle.size == len(PAYLOAD)
    assert handle.read() == PAYLOAD


def test_seeking_returns_the_bytes_at_that_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.urllib.request, "urlopen", _server())
    handle = HttpRangeFile("https://example.invalid/x.h5", block=1024)
    handle.seek(5000)
    assert handle.read(300) == PAYLOAD[5000:5300]
    handle.seek(-10, io.SEEK_END)
    assert handle.read() == PAYLOAD[-10:]


def test_a_cached_block_is_not_fetched_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list = []
    monkeypatch.setattr(remote.urllib.request, "urlopen", _server(log=log))
    handle = HttpRangeFile("https://example.invalid/x.h5", block=1024)
    for _ in range(5):
        handle.seek(2048)
        handle.read(16)
    assert handle.requests == 1, log


def test_a_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real fetch died on one timed-out read after thousands of good ones."""
    monkeypatch.setattr(remote.urllib.request, "urlopen", _server(failures=3))
    handle = HttpRangeFile("https://example.invalid/x.h5", block=1024, attempts=5)
    assert handle.read(2048) == PAYLOAD[:2048]
    assert handle.retries == 3


def test_failing_past_the_attempt_limit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.urllib.request, "urlopen", _server(failures=99))
    with pytest.raises(TimeoutError):
        HttpRangeFile("https://example.invalid/x.h5", block=1024, attempts=2)
