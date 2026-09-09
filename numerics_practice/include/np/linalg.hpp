// np/linalg.hpp -- small dense/tridiagonal linear algebra + matrix functions.
// Provided infrastructure: you never need to edit this file.
#pragma once
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace np {

using Vec = std::vector<double>;
using Vec3 = std::array<double, 3>;
using Cplx = std::complex<double>;

inline constexpr double pi = 3.14159265358979323846;

// ------------------------------- vector ops --------------------------------
inline double dot(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
inline double norm2(const Vec& a) { return std::sqrt(dot(a, a)); }
inline double norm_inf(const Vec& a) {
    double m = 0;
    for (double x : a) m = std::max(m, std::fabs(x));
    return m;
}
inline double rms(const Vec& a) {
    return a.empty() ? 0.0 : std::sqrt(dot(a, a) / double(a.size()));
}
inline double err_inf(const Vec& a, const Vec& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
inline double err_rms(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return a.empty() ? 0.0 : std::sqrt(s / double(a.size()));
}

// ------------------------------ 3-vector ops -------------------------------
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
inline double dot3(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline double norm3(const Vec3& a) { return std::sqrt(dot3(a, a)); }
inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline Vec3 operator*(double s, const Vec3& a) {
    return {s * a[0], s * a[1], s * a[2]};
}
inline Vec3 normalized(const Vec3& a) { return (1.0 / norm3(a)) * a; }

// --------------------------------- matrix ----------------------------------
struct Mat {
    int n = 0, m = 0;
    std::vector<double> a;
    Mat() = default;
    Mat(int n_, int m_) : n(n_), m(m_), a(size_t(n_) * size_t(m_), 0.0) {}
    double& operator()(int i, int j) { return a[size_t(i) * m + j]; }
    double operator()(int i, int j) const { return a[size_t(i) * m + j]; }
    static Mat identity(int n) {
        Mat I(n, n);
        for (int i = 0; i < n; ++i) I(i, i) = 1.0;
        return I;
    }
};

inline Mat matmul(const Mat& A, const Mat& B) {
    Mat C(A.n, B.m);
    for (int i = 0; i < A.n; ++i)
        for (int k = 0; k < A.m; ++k) {
            double aik = A(i, k);
            if (aik == 0.0) continue;
            for (int j = 0; j < B.m; ++j) C(i, j) += aik * B(k, j);
        }
    return C;
}
inline Mat transpose(const Mat& A) {
    Mat T(A.m, A.n);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j) T(j, i) = A(i, j);
    return T;
}
inline Mat madd(const Mat& A, const Mat& B) {
    Mat C = A;
    for (size_t i = 0; i < C.a.size(); ++i) C.a[i] += B.a[i];
    return C;
}
inline Mat mscale(const Mat& A, double s) {
    Mat C = A;
    for (double& x : C.a) x *= s;
    return C;
}
inline Vec matvec(const Mat& A, const Vec& x) {
    Vec y(A.n, 0.0);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j) y[i] += A(i, j) * x[j];
    return y;
}
inline double mat_norm_inf(const Mat& A) {
    double m = 0;
    for (int i = 0; i < A.n; ++i) {
        double s = 0;
        for (int j = 0; j < A.m; ++j) s += std::fabs(A(i, j));
        m = std::max(m, s);
    }
    return m;
}

// Solve A x = b in place (b -> x). Gaussian elimination, partial pivoting.
inline void lu_solve(Mat A, Vec& b) {
    const int n = A.n;
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(A(i, k)) > std::fabs(A(piv, k))) piv = i;
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(A(k, j), A(piv, j));
            std::swap(b[k], b[piv]);
        }
        double d = A(k, k);
        if (d == 0.0) throw std::runtime_error("lu_solve: singular matrix");
        for (int i = k + 1; i < n; ++i) {
            double f = A(i, k) / d;
            if (f == 0.0) continue;
            for (int j = k; j < n; ++j) A(i, j) -= f * A(k, j);
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A(i, j) * b[j];
        b[i] = s / A(i, i);
    }
}

inline Mat inverse(const Mat& A) {
    Mat X(A.n, A.n);
    for (int j = 0; j < A.n; ++j) {
        Vec e(A.n, 0.0);
        e[j] = 1.0;
        lu_solve(A, e);
        for (int i = 0; i < A.n; ++i) X(i, j) = e[i];
    }
    return X;
}

// Matrix exponential, scaling-and-squaring with a Pade(6,6) core.
inline Mat matrix_exp(const Mat& A) {
    const int n = A.n;
    double nrm = mat_norm_inf(A);
    int s = 0;
    while (nrm > 0.5) { nrm *= 0.5; ++s; }
    Mat As = mscale(A, std::ldexp(1.0, -s));
    static const double c[7] = {1.0, 1.0 / 2, 5.0 / 44, 1.0 / 66,
                                1.0 / 792, 1.0 / 15840, 1.0 / 665280};
    Mat A2 = matmul(As, As), A4 = matmul(A2, A2), A6 = matmul(A4, A2);
    Mat even = madd(madd(mscale(Mat::identity(n), c[0]), mscale(A2, c[2])),
                    madd(mscale(A4, c[4]), mscale(A6, c[6])));
    Mat oddc = madd(mscale(Mat::identity(n), c[1]),
                    madd(mscale(A2, c[3]), mscale(A4, c[5])));
    Mat odd = matmul(As, oddc);
    Mat num = madd(even, odd), den = madd(even, mscale(odd, -1.0));
    Mat X = matmul(inverse(den), num);
    for (int i = 0; i < s; ++i) X = matmul(X, X);
    return X;
}

// Lower-triangular L with A = L L^T. Semidefinite-tolerant (clamps tiny <0).
inline Mat cholesky(const Mat& A) {
    const int n = A.n;
    Mat L(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = A(i, j);
            for (int k = 0; k < j; ++k) s -= L(i, k) * L(j, k);
            if (i == j) {
                if (s < 0) {
                    if (s < -1e-10 * std::fabs(A(i, i)) - 1e-300)
                        throw std::runtime_error("cholesky: not PSD");
                    s = 0;
                }
                L(i, j) = std::sqrt(s);
            } else {
                L(i, j) = (L(j, j) == 0.0) ? 0.0 : s / L(j, j);
            }
        }
    return L;
}

// Discrete Lyapunov: solve  S = M S M^T + Q  by squaring (Smith's method).
inline Mat dlyap(const Mat& M, const Mat& Q) {
    Mat S = Q, A = M;
    for (int it = 0; it < 64; ++it) {
        S = madd(S, matmul(matmul(A, S), transpose(A)));
        A = matmul(A, A);
        if (mat_norm_inf(A) < 1e-300) break;
    }
    return S;
}

// Thomas algorithm: solve the tridiagonal system with sub-diagonal a[1..n-1],
// diagonal b[0..n-1], super-diagonal c[0..n-2], rhs d. Writes x (may alias d).
inline void thomas(const Vec& a, const Vec& b, const Vec& c, const Vec& d,
                   Vec& x) {
    const size_t n = b.size();
    static thread_local Vec cp, dp;
    cp.assign(n, 0.0);
    dp.assign(n, 0.0);
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (size_t i = 1; i < n; ++i) {
        double m = b[i] - a[i] * cp[i - 1];
        cp[i] = (i + 1 < n) ? c[i] / m : 0.0;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    x.resize(n);
    x[n - 1] = dp[n - 1];
    for (size_t i = n - 1; i-- > 0;) x[i] = dp[i] - cp[i] * x[i + 1];
}

}  // namespace np
