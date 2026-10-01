"""Forward-only reader for a .tgz holding one appended-raw VTK-XML UnstructuredGrid: arrays are
pulled from the inflated stream as they pass (the .vtu never hits disk), in offset order.
"""
import queue
import threading
import zlib
from xml.etree import ElementTree

import numpy as np

_TAR_BLOCK = 512
_XML_MAX = 1 << 20        # sanity cap on the VTK-XML prefix
_READ_CHUNK = 1 << 24     # 16 MB of compressed bytes per prefetch read
_PREFETCH = 4             # inflight compressed chunks
_SELECT_CHUNK = 1 << 26   # 64 MB scratch when streaming a block through a row filter


class _GzStream:
    """Forward-only reader over the inflated bytes of a gzip file (reads prefetched on a
    thread, so file reads and zlib inflation overlap)."""

    def __init__(self, path, chunk=_READ_CHUNK, depth=_PREFETCH):
        self.path = path
        self.pos = 0                      # decompressed bytes consumed so far
        self._q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._do = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._buf = memoryview(b"")
        self._eof = False
        self._thread = threading.Thread(target=self._read_ahead, args=(chunk,), daemon=True)
        self._thread.start()

    def _read_ahead(self, chunk):
        """Pump compressed chunks into the queue until EOF or close()."""
        try:
            with open(self.path, "rb", buffering=0) as fh:
                while not self._stop.is_set():
                    b = fh.read(chunk)
                    while not self._stop.is_set():
                        try:
                            self._q.put(b, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if not b:
                        return
        except BaseException as e:  # noqa: BLE001 - re-raised on the consumer thread
            try:
                self._q.put(e, timeout=5)
            except queue.Full:
                pass

    def _refill(self):
        """Ensure self._buf is non-empty. False once the stream is exhausted."""
        while not len(self._buf):
            if self._eof:
                return False
            b = self._q.get()
            if isinstance(b, BaseException):
                raise b
            if b:
                out = self._do.decompress(b)
            else:
                self._eof = True
                out = self._do.flush()
            if out:
                self._buf = memoryview(out)
        return True

    def readinto(self, dst):
        """Fill `dst` (writable, C-contiguous) exactly. Raises EOFError on a short read."""
        dst = memoryview(dst).cast("B")
        off = 0
        while off < len(dst):
            if not self._refill():
                raise EOFError(f"{self.path}: EOF at {self.pos} with {len(dst) - off} to go")
            k = min(len(dst) - off, len(self._buf))
            dst[off:off + k] = self._buf[:k]
            self._buf = self._buf[k:]
            off += k
            self.pos += k

    def read(self, n):
        b = bytearray(n)
        self.readinto(b)
        return bytes(b)

    def skip(self, n):
        """Inflate and discard `n` bytes."""
        while n > 0:
            if not self._refill():
                raise EOFError(f"{self.path}: EOF at {self.pos} with {n} to skip")
            k = min(n, len(self._buf))
            self._buf = self._buf[k:]
            n -= k
            self.pos += k

    def unread(self, b):
        """Push bytes back onto the front of the stream."""
        if not b:
            return
        self._buf = memoryview(bytes(b) + bytes(self._buf))
        self.pos -= len(b)

    def read_until(self, marker, limit):
        """Consume up to and including `marker` and return it; the remainder stays buffered."""
        out = bytearray()
        while len(out) < limit:
            if not self._refill():
                raise EOFError(f"{self.path}: EOF before {marker!r}")
            out += self._buf
            self.pos += len(self._buf)
            self._buf = memoryview(b"")
            i = out.find(marker)
            if i >= 0:
                end = i + len(marker)
                self.unread(bytes(out[end:]))
                return bytes(out[:end])
        raise ValueError(f"{self.path}: no {marker!r} in the first {limit} bytes")

    def close(self):
        self._stop.set()
        while True:      # unblock a reader thread parked on a full queue
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _tar_member_size(hdr):
    """Size field of a tar header (GNU base-256 for members over 8 GB, else octal)."""
    raw = hdr[124:136]
    if raw[0] & 0x80:
        return int.from_bytes(bytes([raw[0] & 0x7F]) + raw[1:], "big")
    return int(raw.rstrip(b"\0 ").split(b"\0")[0] or b"0", 8)


class _Block:
    __slots__ = ("name", "dtype", "ncomp", "offset", "nrows")

    def __init__(self, name, dtype, ncomp, offset, nrows):
        self.name, self.dtype, self.ncomp = name, dtype, ncomp
        self.offset, self.nrows = offset, nrows


_VTK_DTYPES = {"Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2",
               "Int32": "i4", "UInt32": "u4", "Int64": "i8", "UInt64": "u8",
               "Float32": "f4", "Float64": "f8"}


class AppendedVTU:
    """Appended-raw VTK-XML UnstructuredGrid read forward-only out of a single-member .tgz."""

    def __init__(self, path):
        self._s = _GzStream(path)
        self.path = path
        try:
            hdr = self._s.read(_TAR_BLOCK)
            if hdr[257:262] not in (b"ustar", b"ustar".ljust(5, b"\0")):
                assert hdr[:1] != b"\0", f"{path}: empty/invalid tar header"
            self.member = hdr[:100].rstrip(b"\0").decode()
            self.member_size = _tar_member_size(hdr)

            prefix = self._s.read_until(b"<AppendedData", _XML_MAX)
            # The raw blocks start at the '_' that follows the opening tag.
            tag_tail = self._s.read_until(b"_", 1 << 12)
            self._data_start = self._s.pos
            xml = ElementTree.fromstring(prefix[:-len(b"<AppendedData")] + b"</VTKFile>")
            self._parse(xml, tag_tail)
        except BaseException:
            self._s.close()
            raise

    def _parse(self, root, tag_tail):
        assert root.get("byte_order") == "LittleEndian", f"{self.path}: big-endian"
        assert root.get("header_type", "UInt32") == "UInt64", \
            f"{self.path}: header_type={root.get('header_type')} (expected UInt64)"
        assert root.get("compressor") is None, \
            f"{self.path}: compressed appended data ({root.get('compressor')}) is unsupported"
        assert b'encoding="raw"' in tag_tail, f"{self.path}: appended data is not raw"

        piece = root.find(".//Piece")
        self.n_points = int(piece.get("NumberOfPoints"))
        self.n_cells = int(piece.get("NumberOfCells"))

        self.blocks = {}
        for section in ("Points", "Cells", "PointData", "CellData"):
            parent = piece.find(section)
            if parent is None:
                continue
            for da in parent.findall("DataArray"):
                assert da.get("format") == "appended", \
                    f"{self.path}: {da.get('Name')} is format={da.get('format')}"
                name = da.get("Name") or section        # <Points> has no Name
                ncomp = int(da.get("NumberOfComponents", 1))
                nrows = self.n_cells if section == "CellData" else self.n_points
                if section == "Cells":
                    nrows = None                        # sized from the block header
                self.blocks[name] = _Block(name, np.dtype("<" + _VTK_DTYPES[da.get("type")]),
                                           ncomp, int(da.get("offset")), nrows)

    def _seek_block(self, name):
        """Advance to `name`'s payload and return (block, n_rows_in_block)."""
        b = self.blocks[name]
        target = self._data_start + b.offset
        assert target >= self._s.pos, (
            f"{self.path}: {name} at {b.offset} is behind the stream cursor "
            f"({self._s.pos - self._data_start}) -- blocks must be read in offset order")
        self._s.skip(target - self._s.pos)
        nbytes = int.from_bytes(self._s.read(8), "little")   # UInt64 block header
        row = b.dtype.itemsize * b.ncomp
        assert nbytes % row == 0, \
            f"{self.path}: {name} nbytes={nbytes} is not a multiple of {row}"
        n = nbytes // row
        if b.nrows is not None:
            assert n == b.nrows, f"{self.path}: {name} has {n} rows, expected {b.nrows}"
        return b, n

    def read_array(self, name, take=None, dst=None):
        """Read one block; with `take` (sorted unique rows) only those rows are kept, streamed
        through a 64 MB scratch buffer. `dst` (requires take) receives them by assignment."""
        b, n = self._seek_block(name)
        if take is None:
            assert dst is None, f"{self.path}: dst requires take (readinto needs contiguity)"
            out = np.empty((n, b.ncomp), b.dtype)
            self._s.readinto(out.reshape(-1))
            return out
        assert take.ndim == 1 and (take.size == 0 or (take[0] >= 0 and take[-1] < n)), \
            f"{self.path}: take out of range for {name} ({n} rows)"
        out = np.empty((take.size, b.ncomp), b.dtype) if dst is None else dst
        rows = max(1, _SELECT_CHUNK // (b.dtype.itemsize * b.ncomp))
        scratch = np.empty((rows, b.ncomp), b.dtype)
        r0 = a = 0
        while r0 < n:
            k = min(rows, n - r0)
            view = scratch[:k]
            self._s.readinto(view)
            j = int(np.searchsorted(take, r0 + k))
            if j > a:
                out[a:j] = view[take[a:j] - r0]
                a = j
            r0 += k
        assert a == take.size, f"{self.path}: {name} kept {a} of {take.size} rows"
        return out

    def close(self):
        self._s.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
