"""Streaming reader for R serialization (.rds, XDR, version 2/3), for objects too large to load.

Pass 1 (`skeleton`): parse the whole object but replace numeric vectors longer than BIG elements by a `Big` record
(type, length, order of appearance); everything small (meta.data columns, dimnames, column pointers) is kept.
Pass 2 (`stream`): parse again and hand each Big vector to a callback chunk by chunk.

Only what Seurat/Matrix objects need is supported (pairlists, S4, vectors, strings, symbols, refs, environments,
compact ALTREP). Run with the rdsenv or pertrelay venv; reads untrusted data only as bytes, never evaluates anything.
"""
import gzip, bz2, lzma, struct
import numpy as np

BIG = 2_000_000


class Big:
    def __init__(self, typ, n, k): self.typ, self.n, self.k = typ, n, k
    def __repr__(self): return f"Big({self.typ},{self.n},#{self.k})"


class RObj:
    def __init__(self, typ, value=None, attr=None): self.typ, self.value, self.attr = typ, value, attr or {}
    def __repr__(self): return f"RObj({self.typ}, attrs={list(self.attr)})"


def _open(path):
    with open(path, "rb") as f: magic = f.read(6)
    if magic[:2] == b"\x1f\x8b": return gzip.open(path, "rb")
    if magic[:3] == b"BZh": return bz2.open(path, "rb")
    if magic[:6] == b"\xfd7zXZ\x00": return lzma.open(path, "rb")
    return open(path, "rb")


class Reader:
    def __init__(self, path, callback=None, chunk=8_000_000):
        self.f = _open(path); self.refs = []; self.nbig = 0; self.cb = callback; self.chunk = chunk
        assert self.f.read(2) == b"X\n", "only XDR-format rds is supported"
        ver = self.int(); self.int(); self.int()
        if ver == 3: n = self.int(); self.f.read(n)

    def int(self): return struct.unpack(">i", self.f.read(4))[0]

    def length(self):
        n = self.int()
        if n == -1: hi, lo = self.int(), self.int(); n = (hi << 32) + lo
        return n

    def vec(self, typ, n, dtype, size):
        if n > BIG:
            k = self.nbig; self.nbig += 1
            left = n
            while left:
                m = min(left, self.chunk); buf = self.f.read(m * size)
                if self.cb: self.cb(k, typ, np.frombuffer(buf, dtype=dtype))
                left -= m
            return Big(typ, n, k)
        return np.frombuffer(self.f.read(n * size), dtype=dtype).astype(dtype.newbyteorder("="))

    def pairlist_attr(self):
        out = {}; obj = self.item()
        node = obj
        while isinstance(node, RObj) and node.typ == "pairlist":
            for tag, val in node.value: out[tag] = val
            break
        return out

    def item(self):
        flags = self.int(); t = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == 254: return None                                        # NILVALUE
        if t in (253, 252, 251, 247, 242, 241): return RObj("special%d" % t)
        if t == 255:                                                    # REFSXP
            idx = flags >> 8
            if idx == 0: idx = self.int()
            return self.refs[idx - 1]
        if t in (248, 249, 250):                                        # PERSIST / NAMESPACE / PACKAGE
            self.int(); n = self.int(); v = [self.item() for _ in range(n)]
            o = RObj("ns", v); self.refs.append(o); return o
        if t == 1:                                                      # SYMSXP
            s = self.item(); o = RObj("sym", s); self.refs.append(o); return o
        if t == 9:                                                      # CHARSXP
            n = self.int(); return None if n == -1 else self.f.read(n).decode("utf-8", "replace")
        if t in (2, 6, 3, 5, 17, 240, 239):                             # pairlist-like (iterative over CDR)
            items = []
            attr = None
            while True:
                if has_attr: attr = self.item()
                tag = self.item() if has_tag else None
                car = self.item()
                items.append((tag.value if isinstance(tag, RObj) and tag.typ == "sym" else tag, car))
                flags = self.int(); t2 = flags & 0xFF
                if t2 not in (2, 6, 3, 5, 17, 240, 239):
                    # CDR is not another cons cell: put the int back by handling it here
                    rest = self._item_from_flags(flags)
                    break
                has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
            return RObj("pairlist", items)
        return self._item_from_flags(flags, consumed=True)

    def _item_from_flags(self, flags, consumed=True):
        t = flags & 0xFF; has_attr = bool(flags & (1 << 9))
        if t == 254: return None
        if t in (253, 252, 251, 247, 242, 241): return RObj("special%d" % t)
        if t == 255:
            idx = flags >> 8
            if idx == 0: idx = self.int()
            return self.refs[idx - 1]
        if t == 4:                                                      # ENVSXP
            self.int(); o = RObj("env"); self.refs.append(o)
            o.value = (self.item(), self.item(), self.item()); o.attr = self.attrs(); return o
        if t == 9:
            n = self.int(); return None if n == -1 else self.f.read(n).decode("utf-8", "replace")
        if t in (10, 13):
            v = self.vec("int", self.length(), np.dtype(">i4"), 4)
        elif t == 14:
            v = self.vec("real", self.length(), np.dtype(">f8"), 8)
        elif t == 15:
            v = self.vec("cplx", self.length(), np.dtype(">c16"), 16)
        elif t == 24:
            n = self.length(); v = self.f.read(n)
        elif t == 16:
            n = self.length(); v = [self.item() for _ in range(n)]
        elif t in (19, 20):
            n = self.length(); v = [self.item() for _ in range(n)]
        elif t == 25:
            v = None
        elif t == 238:                                                  # ALTREP: info, state, attr
            info = self.item(); state = self.item(); attr = self.item()
            cls = info.value[0][1].value if isinstance(info, RObj) and info.typ == "pairlist" else None
            v = state
            sv = state.value if isinstance(state, RObj) else state
            if cls == "compact_intseq":
                n, start, step = np.asarray(sv).tolist(); v = (start + step * np.arange(int(n))).astype(np.int64)
            elif cls == "compact_realseq":
                n, start, step = np.asarray(sv).tolist(); v = start + step * np.arange(int(n))
            elif cls == "deferred_string" and isinstance(state, RObj) and state.typ == "pairlist":
                base = state.value[0][1]; arr = base.value if isinstance(base, RObj) else base
                v = [str(int(x)) if float(x).is_integer() else str(x) for x in np.asarray(arr)]
            elif cls and cls.startswith("wrap_") and isinstance(state, RObj):
                inner = state.value[0]; v = inner.value if isinstance(inner, RObj) else inner
            o = RObj("altrep:%s" % cls, v)
            o.attr = self._pl_to_dict(attr); return o
        elif t in (2, 6, 3, 5, 17, 240, 239):
            raise ValueError("nested pairlist in unexpected position")
        else:
            raise ValueError(f"unsupported SEXP type {t}")
        o = RObj({10: "lgl", 13: "int", 14: "real", 15: "cplx", 24: "raw", 16: "str", 19: "list", 20: "expr", 25: "S4"}[t], v)
        if has_attr: o.attr = self.attrs()
        return o

    def _pl_to_dict(self, pl):
        if isinstance(pl, RObj) and pl.typ == "pairlist": return {k: v for k, v in pl.value}
        return {}

    def attrs(self):
        return self._pl_to_dict(self.item())


def skeleton(path):
    r = Reader(path); obj = r.item(); return obj


def stream(path, callback):
    r = Reader(path, callback=callback); return r.item()


def slot(o, name):
    return o.attr.get(name) if isinstance(o, RObj) else None


def tree(o, depth=0, maxdepth=6, name="", out=None):
    """Readable outline of an RObj skeleton."""
    out = [] if out is None else out
    pad = "  " * depth
    if isinstance(o, RObj):
        cls = o.attr.get("class"); cls = cls.value if isinstance(cls, RObj) else cls
        ln = len(o.value) if isinstance(o.value, (list, np.ndarray)) else (o.value.n if isinstance(o.value, Big) else "")
        out.append(f"{pad}{name}: {o.typ}{' ' + str(cls) if cls is not None else ''} {ln}")
        if depth < maxdepth:
            for k, v in o.attr.items():
                if k in ("class",): continue
                tree(v, depth + 1, maxdepth, "@" + str(k), out)
            if o.typ == "list" and isinstance(o.value, list):
                names = o.attr.get("names"); names = names.value if isinstance(names, RObj) else [None] * len(o.value)
                for nm, v in zip(names, o.value): tree(v, depth + 1, maxdepth, str(nm), out)
    elif isinstance(o, Big):
        out.append(f"{pad}{name}: {o}")
    elif isinstance(o, np.ndarray):
        out.append(f"{pad}{name}: array {o.dtype} {o.shape}")
    else:
        out.append(f"{pad}{name}: {type(o).__name__} {str(o)[:60]}")
    return out
