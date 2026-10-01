"""pyvista readers."""
import numpy as np
import pyvista as pv


def read_mesh(path, point_fields):
    """Read a mesh enabling only `point_fields`; falls back to a full read on any reader quirk."""
    reader = pv.get_reader(path)
    if hasattr(reader, "disable_all_point_arrays") and hasattr(reader, "enable_point_array"):
        try:
            reader.disable_all_point_arrays()
            for nm in point_fields:
                reader.enable_point_array(nm)
            mesh = reader.read()
            if all(nm in mesh.point_data for nm in point_fields):
                return mesh
        except Exception:  # noqa: BLE001
            pass
    return pv.read(path)


def find_field(mesh, prefix):
    """cell_data array whose name starts with `prefix` (names carry units, incl. non-ASCII)."""
    for name in mesh.cell_data.keys():
        if name.startswith(prefix):
            return np.asarray(mesh.cell_data[name])
    raise KeyError(f"no cell_data array starting with {prefix!r}; have {list(mesh.cell_data)}")


def f32(a, ncomp=None):
    """Contiguous float32 copy, reshaped to [-1, ncomp] when given."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a if ncomp is None else a.reshape(-1, ncomp)
