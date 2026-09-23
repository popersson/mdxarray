// test.cpp -- mdxarray self-checks. Exits nonzero on failure.
// Build:  make test && ./test
#include "md.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <sstream>

static int failures = 0;
static int checks   = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    ++checks;                                                                   \
    if (!(cond)) {                                                              \
      std::printf("FAIL %s:%d   %s\n", __FILE__, __LINE__, #cond);              \
      ++failures;                                                               \
    }                                                                           \
  } while (0)

#define CHECK_EQ(a, b)                                                         \
  do {                                                                         \
    ++checks;                                                                   \
    if (!((a) == (b))) {                                                        \
      std::printf("FAIL %s:%d   %s == %s  (%g vs %g)\n", __FILE__, __LINE__,    \
                  #a, #b, double(a), double(b));                                \
      ++failures;                                                               \
    }                                                                           \
  } while (0)

// A requires-expression is only SFINAE-friendly inside a template, so the
// compile-time checks below go through these concepts rather than being
// written inline (where the compiler would hard-error instead of yielding false).
template <class A> concept elem_writable  = requires(A a) { a[0,0] = 1.0; };
template <class A> concept paren_writable = requires(A a) { a(0,0) = 1.0; };
template <class A> concept has_begin     = requires(A a) { a.begin(); };
template <class A> concept plus_eq       = requires(A a) { a += 1.0; };
template <class A> concept const_data    =
    std::is_same_v<decltype(std::declval<A>().data()), const double*>;

using md::darray; using md::farray; using md::iarray;
using md::dview;  using md::cdview;
using md::dsarray; using md::dsview; using md::cdsview;

// --------------------------------------------------------------------------
static void test_construction() {
  darray A(3, 4);                       // CTAD: rank from the argument count
  static_assert(std::is_base_of_v<md::array<double,2>, decltype(A)>);
  CHECK_EQ(A.n(0), 3);
  CHECK_EQ(A.n(1), 4);
  CHECK_EQ(A.nelem(), 12);
  CHECK(bool(A));

  farray B(2, 3);
  static_assert(std::is_base_of_v<md::array<float,2>, decltype(B)>);
  iarray C(2, 3, 4);
  static_assert(std::is_base_of_v<md::array<int,3>, decltype(C)>);
  CHECK_EQ(C.nelem(), 24);

  auto D = md::make<double>(5, 6);      // element type explicit, rank deduced
  CHECK_EQ(D.nelem(), 30);

  md::array<double,2> E;                // default: null
  CHECK(!bool(E));
  CHECK_EQ(E.nelem(), 0);
}

static void test_layout_and_access() {
  darray A(3, 4, 2);
  md::fill(A, 0.0);
  A[1, 2, 1] = 7.0;                     // brackets are the primary spelling
  CHECK_EQ((A[1, 2, 1]), 7.0);
  CHECK_EQ(A(1, 2, 1), 7.0);            // parentheses mean exactly the same

  // the two spellings are interchangeable on every type
  md::dview<3> v = A;
  CHECK_EQ((v[1, 2, 1]), v(1, 2, 1));
  md::dsarray<2,3> S(1.0);
  S[1, 2] = 4.0;
  CHECK_EQ((S[1, 2]), S(1, 2));
  const md::dsarray<2,3>& cS = S;
  CHECK_EQ((cS[1, 2]), cS(1, 2));

  // layout_left: stride(0) == 1, stride(1) == extent(0)
  CHECK_EQ(&(A[1,0,0]) - &(A[0,0,0]), 1);
  CHECK_EQ(&(A[0,1,0]) - &(A[0,0,0]), 3);
  CHECK_EQ(&(A[0,0,1]) - &(A[0,0,0]), 12);
}

static void test_move_and_copy() {
  darray A(3, 4);
  md::fill(A, 5.0);

  auto B = A.copy();                    // deep copy, independent
  B[0,0] = 99.0;
  CHECK_EQ((A[0,0]), 5.0);
  CHECK_EQ((B[0,0]), 99.0);

  const double* before = A.data();
  auto C = std::move(A);                // move: source becomes null, not dangling
  CHECK(C.data() == before);
  CHECK(!bool(A));
  CHECK(A.data() == nullptr);

  darray D(2, 2);
  darray E(3, 3);
  md::fill(D, 1.0); md::fill(E, 2.0);
  md::swap(D, E);
  CHECK_EQ(D.n(0), 3);
  CHECK_EQ((D[0,0]), 2.0);
  CHECK_EQ(E.n(0), 2);
  CHECK_EQ((E[0,0]), 1.0);

  static_assert(!std::is_copy_constructible_v<md::array<double,2>>);
  static_assert(std::is_nothrow_move_constructible_v<md::array<double,2>>);
}

static void test_resizing() {
  darray A(3, 4);
  md::fill(A, 1.0);
  const double* buf = A.data();

  A.reshape(4, 3);                      // same count, data kept, no realloc
  CHECK_EQ(A.n(0), 4);
  CHECK(A.data() == buf);
  CHECK_EQ((A[0,0]), 1.0);

  A.ensure(2, 6);                       // same count -> reshape
  CHECK_EQ(A.n(1), 6);
  CHECK(A.data() == buf);

  A.ensure(10, 10);                     // different count -> realloc
  CHECK_EQ(A.nelem(), 100);

  darray<2> B;                          // ensure on a null array allocates
  B.ensure(3, 3);
  CHECK_EQ(B.nelem(), 9);
  CHECK(bool(B));
}

// --------------------------------------------------------------------------
// The two defects found in review. These are the regression tests for them.
// --------------------------------------------------------------------------
static void test_strided_correctness() {
  darray F(5, 4, 3);
  md::fill(F, 1.0);

  auto r = F.row(2);                    // 4x3 = 12 elements, strided
  CHECK_EQ(r.n(0), 4);
  CHECK_EQ(r.n(1), 3);

  // nelem() must be the ELEMENT COUNT, not the memory span it reaches across.
  CHECK_EQ(r.nelem(), 12);
  CHECK(r.span_size() > r.nelem());     // strided: span is larger
  CHECK(!decltype(r)::contiguous);

  // Reductions must respect the stride, not run off across the parent.
  CHECK_EQ(md::sum(r), 12.0);
  CHECK_EQ(md::dot(r, r), 12.0);
  CHECK_EQ(md::maxval(r), 1.0);
  CHECK_EQ(md::infnorm(r), 1.0);
  CHECK(!md::anynan(r));

  // fill and apply must work on a strided slice, and must touch only it.
  md::fill(F.row(2), 7.0);
  CHECK_EQ((F[2,0,0]), 7.0);
  CHECK_EQ((F[1,0,0]), 1.0);              // neighbours untouched
  CHECK_EQ((F[3,0,0]), 1.0);
  CHECK_EQ(md::sum(F.row(2)), 7.0 * 12);

  md::apply([](double x){ return 2.0 * x; }, F.row(2));
  CHECK_EQ((F[2,0,0]), 14.0);
  CHECK_EQ((F[1,0,0]), 1.0);

  // A contiguous view still reports the same two numbers.
  darray A(3, 4);
  CHECK_EQ(A.nelem(), 12);
  CHECK_EQ(A.span_size(), 12);
}

static void test_const_correctness() {
  // A const owning array hands out const elements, as std::mdarray does.
  static_assert(!elem_writable <const md::array<double,2>&>);
  static_assert(!paren_writable<const md::array<double,2>&>);
  static_assert( elem_writable <md::array<double,2>&>);
  static_assert( paren_writable<md::array<double,2>&>);
  static_assert( const_data<const md::array<double,2>&>);

  // Read-only INTERFACES are expressed in the element type, not by const.
  static_assert(!elem_writable<md::view<const double,2>>);
  static_assert( elem_writable<md::view<double,2>>);

  darray A(2,2);
  md::fill(A, 3.0);
  const darray<2>& cA = A;
  CHECK_EQ((cA[0,0]), 3.0);               // reading is fine
  CHECK_EQ(md::sum(cA.cview()), 12.0);  // cview() for read-only handoff
}

static void test_slicing() {
  darray A(4, 5, 3);
  md::fill(A, 2.0);
  A[1, 1, 2] = 8.0;

  auto p = A.page(2);                   // trailing index: contiguous
  CHECK_EQ(p.n(0), 4);
  CHECK_EQ(p.n(1), 5);
  CHECK(decltype(p)::contiguous);
  CHECK(p.data() == &(A[0,0,2]));
  CHECK_EQ((p[1,1]), 8.0);
  CHECK_EQ(md::sum(p), 2.0 * 19 + 8.0);

  p *= 2.0;                             // contiguous, so fast arithmetic works
  CHECK_EQ((A[0,0,2]), 4.0);
  CHECK_EQ((A[0,0,1]), 2.0);              // other pages untouched

  auto r = A.row(1);                    // leading index: strided
  CHECK_EQ(r.n(0), 5);
  CHECK_EQ(r.n(1), 3);
  CHECK_EQ((r[1,2]), 16.0);
}

static void test_arithmetic() {
  darray A(3, 4);
  darray B(3, 4);
  md::fill(A, 1.0); md::fill(B, 2.0);
  dview<2> va = A, vb = B;

  A  += B;   CHECK_EQ((A[0,0]), 3.0);     // array += array
  A  += vb;  CHECK_EQ((A[0,0]), 5.0);     // array += view
  va += B;   CHECK_EQ((A[0,0]), 7.0);     // view  += array
  va += vb;  CHECK_EQ((A[0,0]), 9.0);     // view  += view
  A  -= B;   CHECK_EQ((A[0,0]), 7.0);
  A  *= 2.0; CHECK_EQ((A[0,0]), 14.0);
  A  /= 2.0; CHECK_EQ((A[0,0]), 7.0);

  CHECK_EQ(md::sum(A), 7.0 * 12);
  CHECK_EQ(md::dot(B, B), 4.0 * 12);
  CHECK_EQ(md::norm(B), std::sqrt(4.0 * 12));
  CHECK_EQ(md::maxval(A), 7.0);
  CHECK_EQ(md::minval(A), 7.0);
}

static void test_static() {
  dsarray<4,5> S(2.0);
  CHECK_EQ(S.nelem(), 20);
  CHECK_EQ(sizeof(S), 20 * sizeof(double));
  S[3,4] = 5.0;
  CHECK_EQ((S[3,4]), 5.0);

  dsarray<4,5> T(1.0);
  auto U = S + T * 3.0;                 // value arithmetic, no heap
  CHECK_EQ((U[0,0]), 2.0 + 3.0);
  CHECK_EQ((U[3,4]), 5.0 + 3.0);
  auto V = -T;
  CHECK_EQ((V[0,0]), -1.0);

  // A static-shaped function, called three ways
  auto trace45 = [](cdsview<4,5> A) { double t = 0; for (int i=0;i<4;++i) t += A[i,i]; return t; };
  CHECK_EQ(trace45(S), 2.0 * 4);                        // owning sarray

  double raw[20];
  for (double& x : raw) x = 1.0;
  dsview<4,5> W(raw);
  CHECK_EQ(trace45(W), 4.0);                            // sview over raw memory

  darray big(4, 5, 3);
  md::fill(big, 3.0);
  CHECK_EQ(trace45(&(big[0,0,2])), 12.0);                 // one-liner from a pointer
  dsview<4,5> page(&big[0,0,1]);
  page *= 2.0;
  CHECK_EQ((big[1,1,1]), 6.0);
  CHECK_EQ((big[1,1,0]), 3.0);
}

static void test_broadcast() {
  dsarray<2,3> X(0.5);
  auto Y1 = md::map([](double x){ return std::sin(x); }, X);
  auto Y2 = md::map(md::sin, X);
  auto Y3 = md::sin(X);
  CHECK_EQ((Y1[0,0]), (Y2[0,0]));
  CHECK_EQ((Y2[0,0]), (Y3[0,0]));
  CHECK_EQ((Y1[0,0]), std::sin(0.5));

  auto Z = md::map([](double a, double b){ return a * b; }, X, X);
  CHECK_EQ((Z[0,0]), 0.25);

  dsarray<2,3> W(4.0);
  md::apply(md::sqrt, W);
  CHECK_EQ((W[0,0]), 2.0);

  darray A(2,3);
  md::fill(A, 9.0);
  md::apply(md::sqrt, A);               // in place on a dynamic array
  CHECK_EQ((A[0,0]), 3.0);
}

static void test_compact_assign() {
  const int n = 6, NC = 4;
  darray u(n, NC);
  for (int c = 0; c < NC; ++c)
    for (int i = 0; i < n; ++i) u[i, c] = 10 * c + i;

  auto ui = md::compact<4>(u.row(2));   // u[2, :] into a compact static array
  CHECK_EQ(ui[0], 2.0);
  CHECK_EQ(ui[1], 12.0);
  CHECK_EQ(ui[2], 22.0);
  CHECK_EQ(ui[3], 32.0);

  darray F(n, NC);
  md::fill(F, 0.0);
  md::assign(F.row(3), ui);             // compact -> strided
  CHECK_EQ((F[3,0]), 2.0);
  CHECK_EQ((F[3,3]), 32.0);
  CHECK_EQ((F[2,0]), 0.0);                // neighbours untouched

  darray G(2,3);
  darray H(2,3);
  md::fill(G, 4.0); md::fill(H, 0.0);
  md::assign(H, G);                     // contiguous fast path
  CHECK_EQ((H[1,2]), 4.0);
}

static void test_containers() {
  std::vector<md::array<double,2>> v;
  v.emplace_back(3, 4);
  v.push_back(md::make<double>(2, 2));
  v.reserve(16);
  CHECK_EQ(v.size(), 2u);
  md::fill(v[0], 1.0);
  CHECK_EQ(md::sum(v[0]), 12.0);

  std::map<int, md::array<double,2>> m;
  m.emplace(1, md::make<double>(2, 2));
  CHECK_EQ(m.size(), 1u);

  struct Mesh { md::array<double,2> x; md::array<int,2> el; };
  static_assert(!std::is_copy_constructible_v<Mesh>);
  static_assert(std::is_move_constructible_v<Mesh>);
  Mesh mesh{ md::make<double>(10, 3), md::make<int>(4, 5) };
  md::fill(mesh.x, 0.5);
  CHECK_EQ(md::sum(mesh.x), 15.0);
  auto nx = md::make<double>(10, 3);
  md::fill(nx, 2.0);
  mesh.x = std::move(nx);
  CHECK_EQ((mesh.x[0,0]), 2.0);
  CHECK(!bool(nx));
}

static void test_misc() {
  dview<3> nothing;
  darray big(2,2,2);
  md::fill(big, 1.0);
  cdview<3> u = big;
  CHECK(!(u && nothing));
  CHECK(bool(u));
  CHECK(!bool(nothing));

  // strided views are deliberately not flat-iterable or fast-arithmetic-able
  static_assert(!has_begin<md::strided_view<double,2>>);
  static_assert(!plus_eq  <md::strided_view<double,2>>);
  static_assert( has_begin<md::view<double,2>>);
  static_assert( plus_eq  <md::view<double,2>>);

  // printing
  darray A(2,2);
  md::fill(A, 1.5);
  std::ostringstream os;
  os << A.cview();
  CHECK(os.str().find("1.5") != std::string::npos);
  CHECK(os.str().find("[2x2]") != std::string::npos);

  // zero-size array is well defined and empty
  md::array<double,2> Z(0, 5);
  CHECK_EQ(Z.nelem(), 0);
  CHECK_EQ(md::sum(Z), 0.0);
}

int main() {
  test_construction();
  test_layout_and_access();
  test_move_and_copy();
  test_resizing();
  test_strided_correctness();
  test_const_correctness();
  test_slicing();
  test_arithmetic();
  test_static();
  test_broadcast();
  test_compact_assign();
  test_containers();
  test_misc();

  if (failures) {
    std::printf("\n%d of %d checks FAILED\n", failures, checks);
    return EXIT_FAILURE;
  }
  std::printf("all %d checks passed\n", checks);
  return EXIT_SUCCESS;
}
