# mdxarray

Multidimensional arrays and views for C++, built on `std::mdspan`.

Header-only, two files, no dependencies beyond a C++23 standard library (plus
BLAS/LAPACK if you use the optional linear-algebra shorthand).

```cpp
#include "md.h"
using md::darray;

darray A(m, n, k);            // owning, rank deduced from the argument count
md::fill(A, 0.0);
A[i, j, k] += 1.0;            // C++23 multidimensional subscript
A *= 2.0;                     // elementwise, never allocates

md::gemm(C, A.page(k), B);    // dimensions derived from the views
```

`mdxarray` is a thin convenience layer, not a framework. `std::mdspan` already
provides a well-designed view; what is missing is an owning counterpart, short
syntax for the operations a numerical code performs constantly, and a static
array with real arithmetic. That is all this is. There are no expression
templates, no hidden allocations, and no linear algebra beyond calling LAPACK.

## Requirements

**C++23 with `<mdspan>`.** On Linux this means clang with libc++ — libstdc++ does
not ship `<mdspan>` yet, even with `-std=c++26`:

```
clang++ -std=c++23 -stdlib=libc++ -O2 myprogram.cpp
```

The header emits a clear `#error` if `<mdspan>` is unavailable, rather than
several hundred lines of template diagnostics. When libstdc++ gains `<mdspan>`,
nothing here needs to change.

`md_lapack.h` additionally needs `-lblas -llapack`. `md.h` on its own has no
link dependency.

```
make check     # build and run the tests
make run       # tests and examples
```

## The four types

|                     | non-owning              | owning                   |
|---------------------|-------------------------|--------------------------|
| **dynamic extents** | `md::view<T,R>`         | `md::array<T,R>`         |
| **static extents**  | `md::sview<T,E...>`     | `md::sarray<T,E...>`     |

All four use `layout_left` (Fortran order) and a signed 32-bit index. `R` is the
rank; `E...` are compile-time extents.

Indexing is `A[i, j, k]`, the C++23 multidimensional subscript. `A(i, j, k)` is
accepted everywhere and means exactly the same thing, but brackets are the
spelling used throughout this README and the examples.

Short aliases live in `namespace md`, so bring in what you use:

```cpp
using md::darray;   // md::array<double,R>      also farray, iarray
using md::dview;    // md::view<double,R>       also fview, iview
using md::cdview;   // md::view<const double,R> also cfview, ciview
using md::dsarray;  // md::sarray<double,E...>  also fsarray, isarray
using md::dsview;   // md::sview<double,E...>   also cdsview, isview
```

Sizes, on x86-64: `view<double,2>` is 16 bytes and passes in registers,
`array<double,2>` is 24, `sview<double,4,5>` is 8 (the shape is in the type and
stored nowhere), and `sarray<double,4,5>` is exactly its 20 doubles.

## Ownership and signatures

A view is a handle. Copying one copies a pointer and a shape, never data, so
views are passed **by value**:

```cpp
void f(md::cdview<3> u);          // read-only input          <- the common case
void f(md::dview<3> u);           // in/out, writes through, shape fixed
void f(md::darray<3>& A);         // may resize or reallocate <- the only reason to take an array
md::darray<2> f(int m, int n);    // allocates and returns
```

Passing a view by value is not the same as passing it read-only: it writes
through to the caller's data exactly as a `double*` would. **Read-only lives in
the element type** (`view<const double,R>`, i.e. `cdview<R>`), not in `const` on
the view. An owning `array` does propagate `const` to its elements, as
`std::mdarray` does, but a free function taking a view parameter binds to the
mutable view inside it — so say what you mean in the signature.

`array` is move-only. A copy must be spelled `A.copy()` and a transfer
`std::move(A)`; a moved-from array is null, not dangling. A struct of arrays
inherits this and becomes non-copyable for free:

```cpp
struct Mesh {
  md::darray<2> x;       // (nnodes, dim)
  md::iarray<2> el;      // (nodes_per_elem, nelem)
};
Mesh m = read_mesh();    // returned by value: moves
m.x = std::move(newx);   // set a field from existing data
void solve(const Mesh&); // pass by reference
```

Move-only types work in `std::vector` (`emplace_back`, `push_back(rvalue)`,
`reserve`) and in `std::map::emplace`. They do not work with
`initializer_list` construction, which always copies.

## Element access and shape

```cpp
A[i, j, k]                          // index; A(i, j, k) is an exact synonym
A.n(r)                              // extent of dimension r
A.nelem()                           // number of elements
A.data()                            // raw pointer, for LAPACK and friends
if (A) ...                          // false when null
A.to_mdspan()                       // the underlying std::mdspan
```

`layout_left` means `stride(0) == 1` and `stride(1) == extent(0)`, so
`&A[0,0,k]` is the start of a contiguous page — which is what makes the LAPACK
shorthand below possible.

### Brackets inside macros

The preprocessor balances parentheses but not brackets, so a subscript with a
comma has to be parenthesised when it is a function-like macro argument:

```cpp
CHECK_EQ(A[0,0], 1.0);      // error: too many arguments to macro
CHECK_EQ((A[0,0]), 1.0);    // fine
```

Ordinary function calls are unaffected — `printf("%g", A[0,0])` is fine, because
the compiler, unlike the preprocessor, parses the brackets. `A(0,0)` also
sidesteps the problem if you prefer it inside assertion macros.

### Checking

Index and shape checking is **off by default** and enabled with
`-DMD_BOUNDS_CHECK`:

```
clang++ -std=c++23 -stdlib=libc++ -O2 -DMD_BOUNDS_CHECK ...
```

It guards element access, `assign`, `dot`, and the compound-assignment
operators. These are opt-in rather than tied to `NDEBUG` because they sit in
inner loops: `md::assign` called once per node in a nodal kernel measured a 2x
slowdown with an unconditional check. Operations called once per matrix
(`reshape`, the LAPACK wrappers) assert unconditionally, where the cost is
irrelevant — so `-DNDEBUG` is worth setting for production builds but is not
required to get full speed.

## Resizing

| | data preserved | reallocates |
|---|---|---|
| `A.reshape(...)` | yes, count must match | never |
| `A.resize(...)` | no, contents undefined | always |
| `A.ensure(...)` | when the count matches | only when it changes |

`reshape` and `resize` follow numpy and Eigen. `ensure` is the one to call in a
time-stepping loop where the size usually does not change. (`std::mdarray` has
no resizing at all — it is fixed-size once constructed.)

## Slicing

Two mirror-image slices, each returning a rank-(R-1) view:

```cpp
auto p = A.page(k);     // fixes the LAST index  -> contiguous view, free
auto r = A.row(i);      // fixes the FIRST index -> strided_view
```

The asymmetry is forced by `layout_left`: fixing the last index leaves a
gap-free block, fixing the first one cannot. `page()` is the `&A[0,0,k]` idiom;
`row()` is what a nodal loop wants.

A `strided_view` has no `begin()`/`end()` and no compound-assignment operators —
they are constrained on contiguity, so treating a strided slice as flat is a
compile error rather than silent corruption. What does accept it is every
layout-agnostic algorithm: `fill`, `apply`, `assign`, `compact`, and all the
reductions.

For a sub-block that is neither (a `(3,5)` window of a `(4,6)` array), build a
`md::strided_view` with explicit strides. This is the gap `submdspan` will close;
libc++ has not shipped it yet.

## Arithmetic

Dynamic arrays and views have compound assignment only, so nothing allocates:

```cpp
A += B;   A -= B;   A *= B;   A /= B;      // elementwise, arrays or views
A += 2.0; A *= 0.5;                        // scalar
md::fill(A, 0.0);
md::assign(dst, src);                      // either side may be strided
```

Static arrays additionally have value-returning operators, because the result
lives on the stack:

```cpp
md::dsarray<4,5> A(1.0), B(2.0);
auto C = A + B * 3.0;                      // no heap, fully unrolled
```

That asymmetry is deliberate: a hidden heap allocation is impossible to write by
accident, and expression templates become unnecessary — for static shapes the
compiler fuses the loops anyway, and for dynamic ones no temporary is built.

Note that `operator=` on a view **rebinds** the handle, matching `std::mdspan`;
it does not copy elements. Use `md::fill` or `md::assign` for that.

## Reductions and broadcasting

```cpp
md::sum(A)   md::dot(A,B)   md::norm(A)   md::infnorm(A)
md::maxval(A)   md::minval(A)   md::anynan(A)
```

These use four independent accumulators. A single-accumulator loop carries a
floating-point dependency, and since FP addition is not associative the compiler
may not vectorize it without `-ffast-math`; measured on an AVX2 machine, a naive
`dot` runs at 17 GB/s against 60 GB/s unrolled and 88 GB/s for BLAS `ddot`. The
summation order is fixed, so results stay reproducible.

Scalar functions broadcast three ways:

```cpp
auto Y = md::map([](double x){ return std::sin(x); }, X);
auto Y = md::map(md::sin, X);              // named function object
auto Y = md::sin(X);                       // shortest
md::apply(md::sqrt, X);                    // in place, static or dynamic
```

`md::map(std::sin, X)` cannot work — `std::sin` is an overload set, so the
template parameter has nothing to deduce from. `md::sin` and friends wrap the
overload set; the generated code is identical to a lambda.

## LAPACK shorthand (optional)

`md_lapack.h` exists to stop you writing out dimensions. Under `layout_left` the
leading dimension of any view is just `extent(0)`, and `m`/`n`/`k` follow from
the operands:

```cpp
// before
for (int k = 0; k < D; k++)
  cdgemm('N','N', ns, N, ng, 1, &wgfsJx[0,0,k], ns, &Fg[0,0,k], ng, 1, R, ns);

// after
for (int k = 0; k < D; k++)
  md::gemm(R, wgfsJx.page(k), Fg.page(k), 1.0, 1.0);
```

Available: `gemm`, `gemv`, `gesv`, `posv`, `syev`. Shapes are checked with
asserts.

Names keep the LAPACK root with the type prefix dropped, since the element type
already comes from the view: `dgemm` → `gemm`, `dposv` → `posv`. The root name is
canonical, maps one-to-one onto the reference manual, and never requires
inventing one — the SPD solve is `posv`, not `solve_spd`. Adding a wrapper is a
mechanical transcription of the reference page.

## A worked example: nodal flux evaluation

For a FEM solution `u` of shape `(nnodes, ncomp)` and a flux `F` of
`(nnodes, ncomp, dim)`, each node's components are strided. That is not a
problem to route around — it is what makes the loop fast, because the node index
is leftmost and every component stream is unit-stride. Measured on 10^6 nodes,
this layout beats a components-contiguous one by 1.8x.

Write the kernel once over compact static arrays and let it ignore the layout:

```cpp
md::sarray<double, NC, DIM> flux(const md::sarray<double, NC>& u) { ... }

void eval_flux(md::cdview<2> u, md::dview<3> F) {
  for (md::index_t i = 0; i < u.n(0); ++i) {
    auto ui = md::compact<NC>(u.row(i));   // u[i, :]    -> sarray<double,NC>
    md::assign(F.row(i), flux(ui));        // F[i, :, :] <- sarray<double,NC,DIM>
  }
}
```

The slicing costs nothing: this measures the same as the equivalent hand-written
index loops.

See `examples.cpp` for this and the other use cases, compiled and runnable.

## Design notes

**Fortran order throughout.** `layout_left` matches LAPACK and Julia natively,
and NumPy with reversed indices — a Fortran-ordered `(M,N)` array *is* a
C-ordered `(N,M)` array, so a binding can relabel rather than copy.

**32-bit index.** `md::index_t` is `int`: it matches the classic F77 LAPACK ABI
and keeps a rank-2 view at 16 bytes, the x86-64 threshold for register passing.
Change that one typedef to `std::int64_t` for arrays past 2^31 elements or an
ILP64 build; nothing else depends on the width.

**Rank is a template parameter** and is deduced from the constructor, so
`darray A(m,n)` gives rank 2 and no array carries a maximum-rank shape it does
not use. One gotcha: CTAD requires every declarator in a statement to deduce the
same type, so `darray A(m,n), B(m,n,k);` will not compile — split it.

**`array` derives from `view`.** Template argument deduction runs before
user-defined conversions, so a `view<T,R>` parameter would never accept an
`array<T,R>` by conversion; deduction does consider derived-to-base. That
inheritance is what lets a function take a view and a caller pass an array.

**Uninitialized storage.** `array` does not zero its memory. Accumulation loops
need an explicit `md::fill(A, 0.0)` first.

## Limitations

- No general slicing. `page()` and `row()` cover the common cases; arbitrary
  sub-blocks need a `strided_view` built by hand. `submdspan` will fix this.
- No expression templates, so `D = A + B` does not exist for dynamic arrays by
  design. Use compound assignment.
- Linear algebra is a handful of LAPACK wrappers, not a library. Add more as you
  need them; the pattern is three lines each.
- Single-threaded. No GPU.
- `double` only in `md_lapack.h`; the array types are generic.

## Calling from Python and Julia

`md_abi.h` is an optional header giving a self-describing C ABI for structs
built out of mdxarray arrays, so a solver can be driven from Python or Julia
with no binding generator — no pybind11, no nanobind, no Cython. Arrays are
shared zero-copy in both directions.

```julia
Mesh = MDX.objtype(lib, "mesh")   # the only line that names a C++ type
m.bbox.centre                     # [0.0, 0.5, 1.0]
m.p                               # 5x2 Matrix{Float64}, aliasing the C++ array
```

See [interop/README.md](interop/README.md). `md.h` itself is unaffected and
still has no dependencies.

## Layout

```
md.h           the four types, slicing, algorithms   (no dependencies)
md_lapack.h    optional BLAS/LAPACK shorthand        (-lblas -llapack)
md_abi.h       optional C ABI for host-language bindings (no dependencies)
test.cpp       self-checks, nonzero exit on failure
examples.cpp   the use cases above, compiled
interop/       generic Julia and Python runtimes, plus a demo
```

Licensed under the MIT license.
