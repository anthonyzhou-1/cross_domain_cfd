"""Paths, manifests, splits and atomic .npy / .json writes."""
import glob
import io
import json
import os

import numpy as np
from numpy.lib import format as npf


def dataset_root(arg, name):
    """`--root`, else $CFD_DATA_ROOT/<name>; one of the two is required."""
    if arg:
        return arg
    base = os.environ.get("CFD_DATA_ROOT")
    if not base:
        raise SystemExit(f"pass --root (the {name} dataset directory) or set $CFD_DATA_ROOT")
    return os.path.join(base, name)


def save_npy(path, arr):
    """np.save through a temp file + os.replace (temp keeps .npy so np.save does not rename)."""
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj, atomic=False, **kw):
    tmp = path + ".tmp" if atomic else path
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    if atomic:
        os.replace(tmp, path)


def n_rows(path):
    return int(np.load(path, mmap_mode="r").shape[0])


def read_split(splits_dir, name):
    return load_json(os.path.join(splits_dir, name + ".json"))


def write_splits(splits_dir, indent=None, **splits):
    os.makedirs(splits_dir, exist_ok=True)
    for name, stems in splits.items():
        write_json(os.path.join(splits_dir, name + ".json"), stems, indent=indent)
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(splits[a]).isdisjoint(splits[b]), f"{a}/{b} overlap"
    print("splits: " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))


def all_on_disk(samples_dir, stems):
    return all(os.path.exists(os.path.join(samples_dir, s + ".npy")) for s in stems)


def rows_from_disk(samples_dir, stem_re, row_fn):
    """Manifest rows rebuilt from samples/*.npy (so partial/resumed builds never drop rows)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(samples_dir, "*.npy"))):
        stem = os.path.basename(f)[:-len(".npy")]
        if stem_re.match(stem):
            rows.append(row_fn(stem, n_rows(f)))
    return rows


def write_manifest(out_dir, rows, errors=None, key="run_id"):
    rows.sort(key=lambda r: r[key])
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "manifest.json"), rows)
    if errors:
        write_json(os.path.join(out_dir, "errors.json"),
                   [[list(a) if isinstance(a, tuple) else a, e] for a, e in errors], indent=2)
    n_pts = sum(r.get("n_points", r.get("n_cells", 0)) for r in rows)
    print(f"wrote manifest.json ({len(rows)} samples, {n_pts:,} points, "
          f"{len(errors) if errors else 0} errors)")
    return rows


# --- npy headers ----------------------------------------------------------------------

def read_header(path):
    """(shape, fortran_order, dtype, data_offset, filesize); reads the header only."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        version = npf.read_magic(f)
        if version != (1, 0):
            raise ValueError(f"{path}: expected npy v1.0, got v{version[0]}.{version[1]}")
        shape, fortran_order, dtype = npf.read_array_header_1_0(f)
        off = f.tell()
    return shape, fortran_order, dtype, off, size


def expected_size(n_rows_, n_cols, dtype=np.float32):
    """Exact byte size np.save gives an [n_rows, n_cols] C-order array."""
    buf = io.BytesIO()
    npf.write_array_header_1_0(buf, dict(descr=np.dtype(dtype).str, fortran_order=False,
                                         shape=(int(n_rows_), int(n_cols))))
    return buf.tell() + int(n_rows_) * int(n_cols) * np.dtype(dtype).itemsize


def n_rows_from_size(size, n_cols, dtype=np.float32):
    """Row count of an np.save'd [n, n_cols] file from its byte size alone, or None."""
    row = n_cols * np.dtype(dtype).itemsize
    for header in (128, 64, 192, 256):
        n, rem = divmod(size - header, row)
        if rem == 0 and n > 0 and expected_size(n, n_cols, dtype) == size:
            return int(n)
    return None


def list_run_files(root, run_re, pattern):
    """Sorted (stem, run_id, first match of `pattern`) over the run dirs of `root`."""
    runs = []
    for d in os.listdir(root):
        m = run_re.match(d)
        if not m:
            continue
        matches = sorted(glob.glob(os.path.join(root, d, pattern)))
        if matches:
            runs.append((d, int(m.group(1)), matches[0]))
    return sorted(runs, key=lambda r: r[1])


def merge_manifest(out_dir, stem_re, row_fn, fresh=(), errors=None, key="run_id"):
    """Manifest over every samples/*.npy on disk: row_fn(stem, n), then the old manifest's
    fields, then this run's non-None fields. A partial build therefore never drops rows."""
    path = os.path.join(out_dir, "manifest.json")
    old = {r["stem"]: r for r in load_json(path)} if os.path.exists(path) else {}
    new = {r["stem"]: r for r in fresh}
    rows = []
    for row in rows_from_disk(os.path.join(out_dir, "samples"), stem_re, row_fn):
        s = row["stem"]
        derived = {k for k, v in row.items() if v is not None}
        row.update({k: v for k, v in old.get(s, {}).items() if k not in derived})
        row.update({k: v for k, v in new.get(s, {}).items() if v is not None})
        rows.append(row)
    return write_manifest(out_dir, rows, errors, key)
