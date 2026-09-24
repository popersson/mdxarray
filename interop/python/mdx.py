"""mdx -- generic Python runtime for libraries exporting the md_abi field table.

Knows nothing about any particular struct: attribute access is built from the
table the library exports. Pure ctypes and numpy; no Cython, no pybind11.
The only layouts mirrored are md_desc, md_str and md_field, which md_abi.h
defines and static_asserts.
"""
import ctypes as C
import numpy as np

MAX_RANK = 8
K_I32, K_I64, K_F64, K_F32, K_BOOL, K_STR, K_ARRAY, K_STRUCT = range(8)
FLAG_READONLY = 1

class MDesc(C.Structure):
    _fields_ = [("data", C.c_void_p), ("rank", C.c_int32),
                ("elem", C.c_int32), ("n", C.c_int32 * MAX_RANK)]

class MStr(C.Structure):
    _fields_ = [("p", C.c_char_p), ("len", C.c_int32)]

class MType(C.Structure):
    pass

class MField(C.Structure):
    _fields_ = [("name", C.c_char_p), ("kind", C.c_int32), ("elem", C.c_int32),
                ("rank", C.c_int32), ("flags", C.c_int32), ("pad", C.c_int32),
                # int32(void*), or null for a field that is not a list
                ("count", C.CFUNCTYPE(C.c_int32, C.c_void_p)),
                ("get", C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int32)),
                ("set", C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int32)),
                ("subtype", C.CFUNCTYPE(C.POINTER(MType)))]

MType._fields_ = [("name", C.c_char_p), ("nfields", C.c_int32),
                  ("pad", C.c_int32), ("fields", C.POINTER(MField))]

_DTYPE = {0: np.float64, 1: np.float32, 2: np.int32, 3: np.int64,
          4: np.uint8, 5: np.complex128, 6: np.complex64}
_CODE = {np.dtype(v): k for k, v in _DTYPE.items()}

_SCALAR = {K_I32: C.c_int32, K_I64: C.c_int64, K_F64: C.c_double,
           K_F32: C.c_float, K_BOOL: C.c_uint8}


def _wrap(d):
    """md_desc -> zero-copy numpy array.

    layout_left is column-major, so the numpy shape is the REVERSE of the
    mdxarray shape and the result is an ordinary C-order array over the same
    bytes. No copy, no transpose, no order='F'.
    """
    if not d.data:
        return None
    shape = tuple(d.n[i] for i in range(d.rank))[::-1]
    dt = np.dtype(_DTYPE[d.elem])
    nbytes = int(np.prod(shape)) * dt.itemsize if shape else 0
    buf = (C.c_byte * nbytes).from_address(d.data)
    return np.frombuffer(buf, dtype=dt).reshape(shape)


def desc(a):
    a = np.ascontiguousarray(a)
    d = MDesc()
    d.data, d.rank, d.elem = a.ctypes.data, a.ndim, _CODE[a.dtype]
    for i, s in enumerate(a.shape[::-1]):        # reverse back to mdxarray order
        d.n[i] = s
    return d, a


class Obj:
    """A C++ object. Attribute access is generated from the library's table."""

    def __init__(self, ptr, ty, deleter=None, parent=None):
        object.__setattr__(self, "_ptr", ptr)
        object.__setattr__(self, "_ty", ty)
        object.__setattr__(self, "_del", deleter)
        object.__setattr__(self, "_parent", parent)   # pins an owning parent
        object.__setattr__(self, "_refs", {})         # keeps host arrays alive

    def __del__(self):
        d = getattr(self, "_del", None)
        if d and getattr(self, "_ptr", None):
            d(self._ptr)
            object.__setattr__(self, "_ptr", None)

    def __dir__(self):
        return list(self._ty[1]) + list(super().__dir__())

    def __repr__(self):
        return f"{self._ty[0]}({', '.join(self._ty[1])})"

    def __getattr__(self, name):
        ty = object.__getattribute__(self, "_ty")
        if name not in ty[1]:
            raise AttributeError(name)
        f = ty[1][name]
        p = object.__getattribute__(self, "_ptr")
        k = f.kind
        if k == K_ARRAY:
            d = MDesc(); f.get(p, C.byref(d), 0); return _wrap(d)
        if k == K_STRUCT:
            ty_sub = _read_type(f.subtype())
            n = f.count(p) if f.count else -1
            if n >= 0:                                  # a list of sub-objects
                return [self._sub(f, ty_sub, i) for i in range(n)]
            return self._sub(f, ty_sub, 0)
        if k == K_STR:
            n = f.count(p) if f.count else -1
            if n >= 0:
                return [_getstr(p, f, i) for i in range(n)]
            return _getstr(p, f, 0)
        v = _SCALAR[k]()
        f.get(p, C.byref(v), 0)
        return bool(v.value) if k == K_BOOL else v.value

    def _sub(self, f, ty_sub, i):
        """Borrow a sub-object; it pins self so the owner outlives the child."""
        c = C.c_void_p()
        f.get(object.__getattribute__(self, "_ptr"), C.byref(c), i)
        return Obj(c.value, ty_sub, None, self) if c.value else None

    def __setattr__(self, name, val):
        ty = object.__getattribute__(self, "_ty")
        if name not in ty[1]:
            return object.__setattr__(self, name, val)
        f = ty[1][name]
        if not f.set:
            raise AttributeError(f"field {name!r} is read-only")
        p = object.__getattribute__(self, "_ptr")
        k = f.kind
        if k == K_ARRAY:
            d, keep = desc(val)
            self._refs[name] = keep
            f.set(p, C.byref(d), 0)
        elif k == K_STR:
            n = f.count(p) if f.count else -1
            if n >= 0:
                if len(val) != n:
                    raise ValueError(f"field {name!r} expects {n} strings, got {len(val)}")
                for i, v in enumerate(val):
                    b = v.encode()
                    f.set(p, C.byref(MStr(b, len(b))), i)
            else:
                b = val.encode()
                f.set(p, C.byref(MStr(b, len(b))), 0)
        else:
            f.set(p, C.byref(_SCALAR[k](int(val) if k != K_F64 else float(val))), 0)


def _getstr(p, f, i):
    s = MStr(); f.get(p, C.byref(s), i)
    return s.p.decode() if s.p else ""


def _read_type(tp):
    t = tp.contents if hasattr(tp, "contents") else tp
    fields = {}
    for i in range(t.nfields):
        f = t.fields[i]
        fields[f.name.decode()] = f
    return (t.name.decode(), fields)


class Lib:
    """One loaded library. Verifies the ABI once, then builds object types."""

    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        for n in ("md_abi_version", "md_desc_sizeof", "md_field_sizeof",
                  "md_abi_max_rank"):
            getattr(self.lib, n).restype = C.c_int32
            getattr(self.lib, n).argtypes = []
        v = self.lib.md_abi_version()
        if v != 1:
            raise RuntimeError(f"md_abi version {v}, this runtime supports 1")
        if C.sizeof(MDesc) != self.lib.md_desc_sizeof():
            raise RuntimeError("MDesc layout disagrees with the library")
        if C.sizeof(MField) != self.lib.md_field_sizeof():
            raise RuntimeError("MField layout disagrees with the library")
        if self.lib.md_abi_max_rank() != MAX_RANK:
            raise RuntimeError("MAX_RANK disagrees with the library")
        self.lib.md_last_error.restype = C.c_char_p
        self.lib.md_last_error.argtypes = []

    def check(self):
        """Raise if the last entry point recorded an error.

        A C++ exception cannot cross an extern "C" boundary, so exporting
        libraries run their bodies through md::abi::guard() and store the
        message instead of letting it propagate.
        """
        msg = self.lib.md_last_error()
        if msg:
            raise RuntimeError("C++ error: " + msg.decode())

    def checked(self, fn):
        """Wrap a bound entry point so it raises on a recorded C++ error."""
        def call(*a, **kw):
            r = fn(*a, **kw)
            self.check()
            return r
        return call

    def objtype(self, prefix):
        """Build a constructor for the type exported as <prefix>_type/_new/_delete."""
        tfn = getattr(self.lib, f"{prefix}_type"); tfn.restype = C.POINTER(MType); tfn.argtypes = []
        nfn = getattr(self.lib, f"{prefix}_new"); nfn.restype = C.c_void_p; nfn.argtypes = []
        dfn = getattr(self.lib, f"{prefix}_delete"); dfn.restype = None; dfn.argtypes = [C.c_void_p]
        ty = _read_type(tfn())
        return lambda: Obj(nfn(), ty, dfn)

    def fn(self, name, restype=None, argtypes=()):
        f = getattr(self.lib, name); f.restype = restype; f.argtypes = list(argtypes)
        return f
