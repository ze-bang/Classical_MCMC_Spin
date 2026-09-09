// Reference solution -- exercise 03.
#pragma once
#include <cmath>

#include "np/harness.hpp"
#include "np/linalg.hpp"

namespace np::ex03 {

using np::Mat;
using np::Vec;

namespace detail {
// Newton on  y - rhs_const - c*h*f(tf, y) = 0  with Newton matrix I - c*h*J.
template <class F, class J>
void newton(F&& f, J&& jac, double tf, const Vec& rhs_const, double ch, Vec& y,
            double tol, int maxit) {
    const int n = int(y.size());
    Vec fv(n), r(n);
    Mat M(n, n), Jm(n, n);
    for (int it = 0; it < maxit; ++it) {
        f(tf, y, fv);
        for (int i = 0; i < n; ++i) r[i] = -(y[i] - rhs_const[i] - ch * fv[i]);
        if (norm_inf(r) <= tol) return;
        std::fill(Jm.a.begin(), Jm.a.end(), 0.0);
        jac(tf, y, Jm);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                M(i, j) = (i == j ? 1.0 : 0.0) - ch * Jm(i, j);
        lu_solve(M, r);
        for (int i = 0; i < n; ++i) y[i] += r[i];
        if (norm_inf(r) <= tol) return;
    }
}
}  // namespace detail

template <class F, class J>
void backward_euler_step(F&& f, J&& jac, double t, Vec& y, double h,
                         double tol = 1e-12, int maxit = 50) {
    const Vec y0 = y;
    detail::newton(f, jac, t + h, y0, h, y, tol, maxit);
}

template <class F, class J>
void trapezoid_step(F&& f, J&& jac, double t, Vec& y, double h,
                    double tol = 1e-12, int maxit = 50) {
    const int n = int(y.size());
    Vec f0(n);
    f(t, y, f0);
    Vec base(n);
    for (int i = 0; i < n; ++i) base[i] = y[i] + 0.5 * h * f0[i];
    detail::newton(f, jac, t + h, base, 0.5 * h, y, tol, maxit);
}

template <class F, class J>
void bdf2_step(F&& f, J&& jac, double t, const Vec& y_prev, Vec& y, double h,
               double tol = 1e-12, int maxit = 50) {
    const int n = int(y.size());
    Vec base(n);
    for (int i = 0; i < n; ++i)
        base[i] = (4.0 / 3.0) * y[i] - (1.0 / 3.0) * y_prev[i];
    detail::newton(f, jac, t + h, base, (2.0 / 3.0) * h, y, tol, maxit);
}

template <class F, class J>
void rosenbrock2_step(F&& f, J&& jac, double t, Vec& y, double h) {
    const int n = int(y.size());
    const double gam = 1.0 + 1.0 / std::sqrt(2.0);
    Mat Jm(n, n), W(n, n);
    jac(t, y, Jm);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            W(i, j) = (i == j ? 1.0 : 0.0) - gam * h * Jm(i, j);

    Vec k1(n), k2(n), ytmp(n), fv(n);
    f(t, y, fv);
    for (int i = 0; i < n; ++i) k1[i] = h * fv[i];
    lu_solve(W, k1);

    for (int i = 0; i < n; ++i) ytmp[i] = y[i] + k1[i];
    f(t + h, ytmp, fv);
    for (int i = 0; i < n; ++i) k2[i] = h * fv[i] - 2.0 * k1[i];
    lu_solve(W, k2);

    for (int i = 0; i < n; ++i) y[i] += 1.5 * k1[i] + 0.5 * k2[i];
}

template <class G>
void exp_euler_step(const Mat& A, G&& g, double t, Vec& y, double h) {
    const int n = A.n;
    Mat E = matrix_exp(mscale(A, h));
    Mat Em1 = E;
    for (int i = 0; i < n; ++i) Em1(i, i) -= 1.0;
    Mat phi = matmul(inverse(A), Em1);   // = h * phi1(hA)
    Vec gv(n, 0.0);
    g(t, y, gv);
    Vec out = matvec(E, y);
    Vec corr = matvec(phi, gv);
    for (int i = 0; i < n; ++i) y[i] = out[i] + corr[i];
}

}  // namespace np::ex03
