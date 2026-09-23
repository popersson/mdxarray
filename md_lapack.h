// md_lapack.h -- optional BLAS/LAPACK shorthand for mdxarray views.
//
// Kept in its own header because it is the only part with a link dependency:
// using the array types should never force you to link BLAS.
// Build with -lblas -llapack.
//
// The goal is not to wrap linear algebra, it is to stop writing out dimensions.
// Under layout_left the leading dimension of any view is just extent(0), and
// m/n/k all follow from the operands, so a gemm call goes from thirteen
// arguments to three.
//
// Naming keeps the LAPACK root with the type prefix dropped, since the element
// type already comes from the view: dgemm -> gemm, dposv -> posv. The root name
// is canonical and searchable, maps one-to-one onto the reference manual, and
// never requires inventing a name -- the SPD solve is posv, not solve_spd.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "md.h"
#include <cassert>

extern "C" {
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
void dgemv_(const char*, const int*, const int*, const double*, const double*,
            const int*, const double*, const int*, const double*, double*, const int*);
void dgesv_(const int*, const int*, double*, const int*, int*, double*, const int*, int*);
void dposv_(const char*, const int*, const int*, double*, const int*,
            double*, const int*, int*);
void dsyev_(const char*, const char*, const int*, double*, const int*, double*,
            double*, const int*, int*);
}

namespace md {

// C := alpha * op(A) * op(B) + beta * C
inline void gemm(view<double,2> C, view<const double,2> A, view<const double,2> B,
                 double alpha = 1.0, double beta = 0.0, char ta = 'N', char tb = 'N') {
  const int m = C.n(0), n = C.n(1);
  const int k  = (ta == 'N') ? A.n(1) : A.n(0);
  const int ka = (tb == 'N') ? B.n(0) : B.n(1);
  const int ma = (ta == 'N') ? A.n(0) : A.n(1);
  const int nb = (tb == 'N') ? B.n(1) : B.n(0);
  assert(k == ka && m == ma && n == nb && "md::gemm: shapes do not conform");
  const int lda = A.n(0), ldb = B.n(0), ldc = C.n(0);
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, A.data(), &lda, B.data(), &ldb,
         &beta, C.data(), &ldc);
}

// y := alpha * op(A) * x + beta * y
inline void gemv(view<double,1> y, view<const double,2> A, view<const double,1> x,
                 double alpha = 1.0, double beta = 0.0, char ta = 'N') {
  const int m = A.n(0), n = A.n(1);
  assert((ta == 'N' ? (y.nelem() == m && x.nelem() == n)
                    : (y.nelem() == n && x.nelem() == m)) &&
         "md::gemv: shapes do not conform");
  const int lda = A.n(0), one = 1;
  dgemv_(&ta, &m, &n, &alpha, A.data(), &lda, x.data(), &one, &beta, y.data(), &one);
}

// General solve A X = B. A is overwritten by its LU factors, B by the solution.
inline int gesv(view<double,2> A, view<double,2> B) {
  const int n = A.n(0), nrhs = B.n(1), lda = A.n(0), ldb = B.n(0);
  assert(A.n(0) == A.n(1) && B.n(0) == n && "md::gesv: shapes do not conform");
  auto ipiv = std::make_unique<int[]>(static_cast<std::size_t>(n));
  int info = 0;
  dgesv_(&n, &nrhs, A.data(), &lda, ipiv.get(), B.data(), &ldb, &info);
  return info;
}

// Symmetric positive definite solve, via Cholesky.
inline int posv(view<double,2> A, view<double,2> B, char uplo = 'L') {
  const int n = A.n(0), nrhs = B.n(1), lda = A.n(0), ldb = B.n(0);
  assert(A.n(0) == A.n(1) && B.n(0) == n && "md::posv: shapes do not conform");
  int info = 0;
  dposv_(&uplo, &n, &nrhs, A.data(), &lda, B.data(), &ldb, &info);
  return info;
}

// Symmetric eigenproblem. w receives the eigenvalues; with jobz == 'V', A is
// overwritten by the eigenvectors. Workspace is queried and allocated here.
inline int syev(view<double,2> A, view<double,1> w, char jobz = 'V', char uplo = 'L') {
  const int n = A.n(0), lda = A.n(0);
  assert(A.n(0) == A.n(1) && w.nelem() == n && "md::syev: shapes do not conform");
  int info = 0, lwork = -1;
  double wq = 0;
  dsyev_(&jobz, &uplo, &n, A.data(), &lda, w.data(), &wq, &lwork, &info);
  if (info != 0) return info;
  lwork = static_cast<int>(wq);
  auto work = std::make_unique<double[]>(static_cast<std::size_t>(lwork));
  dsyev_(&jobz, &uplo, &n, A.data(), &lda, w.data(), work.get(), &lwork, &info);
  return info;
}

}  // namespace md
