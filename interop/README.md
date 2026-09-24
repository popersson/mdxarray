# mdxarray interop

Driving a C++ program that uses mdxarray from Python or Julia, with no binding
generator: no pybind11, no nanobind, no Cython. Just a C ABI, `ctypes` on one
side and `ccall` on the other.

```
make -C demo check      # builds libmesh.so, runs both demos
```

## Passing a single array needs nothing at all

`md::view`'s `(pointer, extents...)` constructor already *is* the interop layer.
Expose a plain C entry point and reconstruct the view inside it:

```cpp
extern "C" void mdx_scale(double* u, const double* w, int n, int nc) {
  scale(md::view<double,2>(u, n, nc), md::view<const double,1>(w, n));
}
```

Both languages reach that zero-copy. The layout correspondence is the one thing
worth memorising:

| | native shape | mdxarray view | copy? |
|---|---|---|---|
| **Julia** `Matrix{Float64}` | `(n, nc)` column-major | `view<double,2>(p, n, nc)` | none |
| **NumPy** C-order `ndarray` | `(nc, n)` row-major | `view<double,2>(p, n, nc)` | none |

Julia matches shape for shape, because it is column-major like `layout_left`.
NumPy matches with the **shape reversed** — a C-order `(nc, n)` array has
byte-for-byte the layout of an mdxarray `(n, nc)`. No `order='F'`, no transpose.

Guard the boundary against non-contiguous inputs: `ndpointer(flags="C_CONTIGUOUS")`
in Python, and annotate `::Matrix{Float64}` rather than `::AbstractMatrix` in
Julia so a `SubArray` view cannot bind.

## Passing a struct of arrays: `md_abi.h`

The interesting objects in a solver are usually structs holding a dozen arrays
and some scalars. Mirroring such a struct field-by-field in the host language
works until someone adds a field, at which point it silently reads the wrong
memory — with no diagnostic.

`md_abi.h` avoids that by never exposing C++ object layout. A library exports a
table of accessors; the host reads it at load time and builds attribute access
from it. Adding, removing or reordering a C++ field needs no host-side change.

Two further reasons layout mirroring is a bad idea here specifically:
`std::mdspan`'s memory layout is not specified by the standard, and a struct
holding an `md::array` is not standard-layout, so `offsetof` on it is not even
valid.

### Declaring a struct

One line per member, written next to the struct:

```cpp
#include "md_abi.h"

struct Mesh {
  int32_t     dim, np, nt;
  std::string name;
  int32_t     npatch;
  Patch       patches[8];
  Bounds      bbox;
  md::array<double,2> p;
};

static const md_field mesh_fields_[] = {
  MD_SCALAR     (Mesh, dim,  md_kind_i32),
  MD_STRING     (Mesh, name),
  MD_STRUCT_LIST(Mesh, patches, patch_type, self->npatch),   // runtime length
  MD_STRUCT     (Mesh, bbox,    bounds_type),
  MD_ARRAY      (Mesh, p),
};
static const md_type mesh_type_ = MD_TYPE(Mesh, mesh_fields_);

MD_ABI_EXPORT()                       // once per library

extern "C" {
const md_type* mesh_type() { return &mesh_type_; }
Mesh*          mesh_new()  { return new Mesh(); }
void           mesh_delete(Mesh* m) { delete m; }
}
```

### Using it

```julia
lib  = MDX.MDXLib("libmesh.so")
Mesh = MDX.objtype(lib, "mesh")     # the only line naming a C++ type
m = Mesh()
m.np                                 # 5
m.name                               # "unit-square"
m.bbox.centre                        # [0.0, 0.5, 1.0]
m.patches[2].faces                   # Int32[10, 11, 12]
m.p                                  # 5x2 Matrix{Float64}, zero copy
```

```python
lib  = mdx.Lib("libmesh.so")
Mesh = lib.objtype("mesh")
m = Mesh()
m.np                                 # 5
m.bbox.centre                        # array([0. , 0.5, 1. ])
m.patches[1].faces                   # array([10, 11, 12], dtype=int32)
m.p                                  # (2, 5) float64, zero copy, shape reversed
```

Neither runtime contains the name of any C++ type. Everything above is
discovered from the table.

### Field kinds

| macro | C++ member | host type |
|---|---|---|
| `MD_SCALAR` | `int32_t`, `int64_t`, `float`, `double`, `bool` | number / bool |
| `MD_ARRAY` | `md::array`, `md::view`, `md::sarray` | zero-copy array |
| `MD_CARRAY` | `double x[3]` | zero-copy array, read-only |
| `MD_STRING` | `std::string` | string |
| `MD_STRING_LIST` | array of `std::string` | list of strings |
| `MD_STRUCT` | nested struct by value | sub-object |
| `MD_STRUCT_PTR` | nested struct by pointer | sub-object, or nothing/None |
| `MD_STRUCT_LIST` | array of nested structs | list of sub-objects |

Indexed kinds take a `COUNT` expression evaluated against `self`, so a list may
be a fixed size or a runtime value: `MD_STRING_LIST(Mesh, bctags, self->nbc)`.

Array fields declared as `md::view` **rebind** when assigned from the host, so
the host keeps ownership and nothing is copied. Fields declared as `md::array`
own their storage, so assignment copies in.

### Errors

A C++ exception must not cross an `extern "C"` boundary. Run every entry point
body through `md::abi::guard()`, which records the message instead:

```cpp
void mesh_build(Mesh* m, int np, int nt, int dim) {
  md::abi::guard([&] {
    if (np <= 0) throw std::invalid_argument("mesh_build: np must be positive");
    ...
  });
}
```

The host collects it with `MDX.@checked lib <call>` or `lib.checked(fn)`, which
raise a Julia error or a Python `RuntimeError`.

### Safety

Both runtimes verify at load time that `md_abi_version`, `sizeof(md_desc)`,
`sizeof(md_field)` and `MD_ABI_MAX_RANK` agree with the library, so a mismatched
build fails loudly rather than reading the wrong bytes. Sub-objects hold a
reference to their owning parent, so a parent cannot be finalized while a child
is alive, and host arrays assigned into a struct are kept alive by the wrapper.

## Layout

```
md_abi.h            (in the repo root, beside md.h)
interop/
├── julia/mdx.jl    generic runtime, no C++ type names
├── python/mdx.py   generic runtime, no C++ type names
└── demo/           libmesh.cpp + self-checking demo.jl and demo.py
```

The two runtimes are reference implementations rather than packaged libraries:
copy them into your project and adapt as needed.

## Not covered

Raw pointers, `std::vector`, callbacks, virtual dispatch across the border, and
ownership transfer (a C++-allocated array handed to the host to own). For the
last of these you need a finalizer on the Julia side and a capsule on the Python
side; today the host owns or the C++ object owns, never a handoff.
