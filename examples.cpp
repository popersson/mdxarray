// examples.cpp -- the use cases from the README, as code that actually compiles.
// Build:  make examples && ./examples        (needs -lblas -llapack)
#include "md.h"
#include "md_lapack.h"

#include <cstdio>

using md::darray; using md::dview; using md::cdview;
using md::dsarray; using md::dsview; using md::cdsview;
using md::index_t;

// ===========================================================================
// 1. Signatures say what a function does with its arguments
// ===========================================================================
double energy(cdview<3> u)                 { return md::dot(u, u); }      // read-only
void   scale (dview<3> u, double s)        { u *= s; }                    // writes through
void   grow  (darray<3>& A, int m, int n, int k) { A.ensure(m, n, k); }   // may reallocate
darray<2> identity(int n) {                                               // allocates
  darray<2> A(n, n);
  md::fill(A, 0.0);
  for (int i = 0; i < n; ++i) A[i, i] = 1.0;
  return A;
}

// ===========================================================================
// 2. Nodal FEM flux: u is (nnodes, ncomp), F is (nnodes, ncomp, dim)
//
// Per node, u[i,:] is strided. That is not a problem to route around -- it is
// what makes the loop vectorize, because the node index is leftmost and every
// component stream is unit-stride. The kernel itself is written once over
// compact static arrays and never sees the storage layout.
// ===========================================================================
constexpr index_t NC = 4, DIM = 3;

md::sarray<double, NC, DIM> flux(const md::sarray<double, NC>& u) {
  md::sarray<double, NC, DIM> F;
  const double p = 0.4 * (u[3] - 0.5 * (u[1]*u[1] + u[2]*u[2]) / u[0]);
  for (int d = 0; d < DIM; ++d) {
    const double vd = (d < 2 ? u[1 + d] : 0.0) / u[0];
    F[0,d] = u[0]*vd;  F[1,d] = u[1]*vd;  F[2,d] = u[2]*vd;  F[3,d] = (u[3] + p)*vd;
  }
  return F;
}

void eval_flux(cdview<2> u, dview<3> F) {
  for (index_t i = 0; i < u.n(0); ++i) {
    auto ui = md::compact<NC>(u.row(i));   // u[i, :]    -> sarray<double,NC>
    md::assign(F.row(i), flux(ui));        // F[i, :, :] <- sarray<double,NC,DIM>
  }
}

// ===========================================================================
// 3. A struct of owning arrays is move-only and non-copyable for free
// ===========================================================================
struct Mesh {
  darray<2> x;      // (nnodes, dim)
  md::iarray<2> el; // (nodes_per_elem, nelem)

  index_t nnodes() const { return x.n(0); }
  index_t nelem()  const { return el.n(1); }
};

Mesh make_mesh(int nn, int ne) {
  Mesh m{ md::make<double>(nn, 3), md::make<int>(4, ne) };
  md::fill(m.x, 0.5);
  md::fill(m.el, 7);
  return m;                                // moves; a copy would not compile
}

int main() {
  std::printf("=== 1. basics ===\n");
  darray A(3, 4);                          // rank deduced from the argument count
  md::fill(A, 1.0);
  A[1, 2] = 7.0;
  std::printf("  A is %dx%d, nelem %d, A[1,2] = %g, sum = %g\n",
              A.n(0), A.n(1), A.nelem(), A[1, 2], md::sum(A));
  std::printf("  layout_left: &A[1,0]-&A[0,0] = %d, &A[0,1]-&A[0,0] = %d\n",
              int(&A[1,0] - &A[0,0]), int(&A[0,1] - &A[0,0]));

  darray B(3, 4);
  md::fill(B, 2.0);
  A += B;  A *= 0.5;                       // non-allocating, arrays or views
  std::printf("  after A += B; A *= 0.5 : A[0,0] = %g\n", A[0, 0]);

  auto I = identity(4);
  std::printf("  identity(4): I[2,2] = %g, I[2,3] = %g\n", I[2,2], I[2,3]);

  std::printf("\n=== 2. ownership ===\n");
  auto C = A.copy();                       // explicit deep copy
  C[0,0] = 99.0;
  std::printf("  A.copy() is independent: A[0,0] = %g, C[0,0] = %g\n", A[0,0], C[0,0]);
  auto D = std::move(C);                   // explicit move
  std::printf("  after std::move: D[0,0] = %g, moved-from C is null = %d\n",
              D[0,0], int(!bool(C)));

  std::printf("\n=== 3. slicing ===\n");
  darray big(4, 5, 3);
  md::fill(big, 2.0);
  auto pg = big.page(2);                   // last index: contiguous, free
  pg *= 3.0;
  std::printf("  big.page(2) is %dx%d contiguous; big[0,0,2] = %g, big[0,0,1] = %g\n",
              pg.n(0), pg.n(1), big[0,0,2], big[0,0,1]);
  auto rw = big.row(1);                    // first index: strided
  std::printf("  big.row(1) is %dx%d strided, nelem = %d, sum = %g\n",
              rw.n(0), rw.n(1), rw.nelem(), md::sum(rw));

  std::printf("\n=== 4. views in signatures ===\n");
  darray u3(2, 2, 2);
  md::fill(u3, 1.5);
  std::printf("  energy(u3) = %g\n", energy(u3));
  scale(u3, 2.0);
  std::printf("  after scale(u3, 2): u3[0,0,0] = %g\n", u3[0,0,0]);
  grow(u3, 3, 3, 3);
  std::printf("  after grow(3,3,3): nelem = %d\n", u3.nelem());

  std::printf("\n=== 5. nodal FEM flux ===\n");
  const int n = 6;
  darray u(n, NC);
  darray F(n, NC, DIM);   // one CTAD declarator per statement: the ranks differ
  for (int i = 0; i < n; ++i) {
    u[i,0] = 1.0 + 0.01*i; u[i,1] = 0.3; u[i,2] = 0.2; u[i,3] = 2.5;
  }
  md::fill(F, 0.0);
  eval_flux(u, F);
  std::printf("  u[2,:] = (%g %g %g %g), component stride = %d\n",
              u[2,0], u[2,1], u[2,2], u[2,3], int(&u[0,1] - &u[0,0]));
  std::printf("  F[2,:,0] = (%.4f %.4f %.4f %.4f)\n", F[2,0,0], F[2,1,0], F[2,2,0], F[2,3,0]);

  std::printf("\n=== 6. static arrays ===\n");
  dsarray<4,5> S(2.0);
  S[3,4] = 5.0;
  auto T = S * 2.0 + S;                    // value arithmetic, all on the stack
  std::printf("  S*2+S: T[3,4] = %g, sizeof(T) = %zu bytes\n", T[3,4], sizeof(T));

  auto trace45 = [](cdsview<4,5> M) { double t = 0; for (int i=0;i<4;++i) t += M[i,i]; return t; };
  std::printf("  trace45(S) = %g                  (owning static array)\n", trace45(S));
  std::printf("  trace45(&big[0,0,2]) = %g        (bare pointer, shape is in the type)\n",
              trace45(&big[0,0,2]));

  dsarray<2,3> X(0.5);
  auto Y = md::sin(X);                     // broadcast a scalar function
  md::apply(md::sqrt, X);                  // ... or in place
  std::printf("  md::sin(X)[0,0] = %.4f, after apply(md::sqrt, X): %.4f\n", Y[0,0], X[0,0]);

  std::printf("\n=== 7. struct of arrays ===\n");
  Mesh mesh = make_mesh(100, 50);
  std::printf("  mesh: %d nodes, %d elements; copy-constructible = %d\n",
              mesh.nnodes(), mesh.nelem(), int(std::is_copy_constructible_v<Mesh>));
  auto nx = md::make<double>(100, 3);
  md::fill(nx, 9.0);
  mesh.x = std::move(nx);                  // set a field from existing data
  std::printf("  after mesh.x = std::move(nx): mesh.x[0,0] = %g\n", mesh.x[0,0]);

  std::printf("\n=== 8. LAPACK shorthand ===\n");
  const int ns = 3, ng = 4, ncomp = 2, ndim = 2;
  darray wgfsJx(ns, ng, ndim);
  darray Fg(ng, ncomp, ndim);
  darray R(ns, ncomp);
  for (int k = 0; k < ndim; ++k) {
    for (int j = 0; j < ng; ++j) for (int i = 0; i < ns; ++i)
      wgfsJx[i,j,k] = 1.0 + 0.1*(i+j+k);
    for (int j = 0; j < ncomp; ++j) for (int i = 0; i < ng; ++i)
      Fg[i,j,k] = 0.5 - 0.05*(i+j-k);
  }
  md::fill(R, 0.0);
  for (int k = 0; k < ndim; ++k)
    md::gemm(R, wgfsJx.page(k), Fg.page(k), 1.0, 1.0);   // all dimensions derived
  std::printf("  gemm over pages: R[0,0] = %.4f\n", R[0,0]);

  darray M(3,3);
  darray rhs(3,1);
  darray y(3);
  darray xv(3);
  const double av[9] = {4,1,0, 1,3,1, 0,1,2};            // symmetric positive definite
  for (int j=0;j<3;++j) for (int i=0;i<3;++i) M[i,j] = av[i+3*j];
  xv[0]=1; xv[1]=2; xv[2]=3;
  md::gemv(y, M, xv);
  std::printf("  gemv: y = (%g, %g, %g)\n", y[0], y[1], y[2]);
  for (int i=0;i<3;++i) rhs[i,0] = y[i];
  int info = md::posv(M, rhs);                            // SPD solve, no invented name
  std::printf("  posv: info = %d, x = (%g, %g, %g)\n", info, rhs[0,0], rhs[1,0], rhs[2,0]);
  return 0;
}
