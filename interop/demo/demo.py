"""Drives libmesh.so from Python. Self-checking: exits nonzero on failure."""
import ctypes as C
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))
import mdx

HERE = pathlib.Path(__file__).resolve().parent
lib = mdx.Lib(HERE / "libmesh.so")
Mesh = lib.objtype("mesh")                      # the only line naming a C++ type

build = lib.checked(lib.fn("mesh_build", None,
                           [C.c_void_p, C.c_int32, C.c_int32, C.c_int32]))
scale = lib.checked(lib.fn("mesh_scale_coords", None,
                           [C.c_void_p, C.POINTER(mdx.MDesc)]))
total = lib.fn("mesh_total", C.c_double, [C.c_void_p])

nfail = 0


def check(label, cond):
    global nfail
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        nfail += 1


def raises(fn, needle):
    try:
        fn()
    except Exception as e:                       # noqa: BLE001 - this is the point
        return needle in str(e)
    return False


m = Mesh()
build(m._ptr, 5, 4, 2)

print(repr(m))
print("scalars")
check("dim", m.dim == 2)
check("np", m.np == 5)
check("nglobal (int64)", m.nglobal == 5_000_000)
check("h", abs(m.h - 0.25) < 1e-15)
check("curved (bool)", m.curved is True)

print("strings")
check("name", m.name == "unit-square")
check("bctags (runtime length)",
      m.bctags == ["wall", "inlet", "outlet", "symmetry"])

print("nested structs")
check("bbox.lo (C array)", np.array_equal(m.bbox.lo, [-1.0, -2.0, -3.0]))
check("bbox.centre (sarray)", np.array_equal(m.bbox.centre, [0.0, 0.5, 1.0]))
check("patches length", len(m.patches) == 2)
check("patches[1].name", m.patches[1].name == "patch1")
check("patches[1].faces", np.array_equal(m.patches[1].faces, [10, 11, 12]))

print("arrays, zero copy (numpy shape is the reverse of the mdxarray shape)")
check("p shape", m.p.shape == (2, 5))
check("t dtype", m.t.dtype == np.int32)
m.p[0, 0] = -99.0
check("write visible in C++", abs(total(m._ptr) - m.p.sum()) < 1e-12)

print("writes back")
m.name = "renamed"
m.curved = False
m.nglobal = 7
check("name write", m.name == "renamed")
check("bool write", m.curved is False)
check("int64 write", m.nglobal == 7)
before = m.p.copy()
w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
d, _keep = mdx.desc(w)
scale(m._ptr, C.byref(d))
check("host array passed in", np.allclose(m.p, before * w[np.newaxis, :]))

print("guards")
check("read-only C array",
      raises(lambda: setattr(m.bbox, "lo", np.zeros(3)), "read-only"))
check("C++ exception surfaces",
      raises(lambda: build(m._ptr, -1, 4, 2), "must be positive"))
bad, _k = mdx.desc(np.array([1.0, 2.0]))
check("shape mismatch surfaces",
      raises(lambda: scale(m._ptr, C.byref(bad)), "equal np"))

if nfail:
    print(f"\n{nfail} check(s) FAILED")
    sys.exit(1)
print("\nall checks passed")
