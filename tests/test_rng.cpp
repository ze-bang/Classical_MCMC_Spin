// test_rng.cpp — random-number layer: reproducibility, stream independence,
// uniform range and uniformity of the sphere sampler.
#include "physics_test_util.h"

#include <set>

using namespace phys_test;

namespace {

void test_reproducibility() {
    std::printf("\n== Seeding ==\n");
    seed_lehman(42);
    std::vector<uint64_t> a(16), b(16);
    for (auto& x : a) x = lehman_next();
    seed_lehman(42);
    for (auto& x : b) x = lehman_next();
    check(a == b, "same seed reproduces the stream");
    check(a[0] != 0, "first draw after seeding is not zero");

    seed_lehman(43);
    int equal = 0;
    for (int i = 0; i < 16; ++i) equal += (lehman_next() == a[i]);
    check(equal == 0, "adjacent seeds give unrelated streams");

    // Worker threads reseed from (master, tid) after every seed_lehman call.
#ifdef _OPENMP
    auto thread_draws = [] {
        std::vector<uint64_t> v(4);
        #pragma omp parallel num_threads(4)
        { v[omp_get_thread_num()] = lehman_next(); }
        return v;
    };
    seed_lehman(7);
    auto t1 = thread_draws();
    (void)thread_draws();  // advance every stream
    seed_lehman(7);
    auto t2 = thread_draws();
    check(t1 == t2, "thread streams are reproducible after reseeding");
    check(std::set<uint64_t>(t1.begin(), t1.end()).size() == 4, "thread streams are distinct");
#endif
    seed_lehman(9);
    seed_lehman_from_rank(0);
    const uint64_t r0 = lehman_next();
    seed_lehman(9);
    seed_lehman_from_rank(1);
    check(lehman_next() != r0, "rank-derived streams differ");
}

void test_uniforms() {
    std::printf("\n== Uniform deviates ==\n");
    seed_lehman(1);
    const int n = 2000000;
    double s = 0, s2 = 0, mn = 1, mx = 0;
    for (int i = 0; i < n; ++i) {
        const double u = random_double_lehman(0.0, 1.0);
        s += u; s2 += u * u;
        mn = std::min(mn, u); mx = std::max(mx, u);
    }
    check(mn >= 0.0 && mx < 1.0, "random_double_lehman in [0, 1)");
    check_close(s / n, 0.5, 5 * std::sqrt(1.0 / 12.0 / n), "mean of U[0,1)");
    check_close(s2 / n - (s / n) * (s / n), 1.0 / 12.0, 2e-3, "variance of U[0,1)");

    std::vector<int> hist(7, 0);
    for (int i = 0; i < 700000; ++i) ++hist.at(random_int_lehman(7));
    double chi2 = 0;
    for (int h : hist) chi2 += (h - 1e5) * (h - 1e5) / 1e5;
    check(chi2 < 30.0, "random_int_lehman(7) uniform (chi2 = " + std::to_string(chi2) + ", 6 dof)");

    double m = 0, v = 0;
    for (int i = 0; i < n; ++i) { const double z = random_normal_lehman(); m += z; v += z * z; }
    check_close(m / n, 0.0, 5.0 / std::sqrt(n), "normal mean");
    check_close(v / n, 1.0, 5e-3, "normal variance");
}

void test_sphere(size_t dim) {
    seed_lehman(3);
    const int n = 400000;
    const double r = 1.7;
    std::vector<double> x(dim);
    double m2 = 0, m4 = 0, max_norm_err = 0, mean0 = 0;
    for (int i = 0; i < n; ++i) {
        random_point_on_sphere(x.data(), dim, r);
        double nn = 0;
        for (double c : x) nn += c * c;
        max_norm_err = std::max(max_norm_err, std::abs(std::sqrt(nn) - r));
        m2 += x[0] * x[0];
        m4 += x[dim - 1] * x[dim - 1] * x[dim - 1] * x[dim - 1];
        mean0 += x[1];
    }
    const std::string tag = " (n=" + std::to_string(dim) + ")";
    check_close(max_norm_err, 0.0, 1e-12, "radius" + tag);
    // Uniform on S^{n-1}: E[x^2] = r^2/n, E[x^4] = 3 r^4 / (n (n + 2)).
    const double e2 = r * r / dim, e4 = 3.0 * std::pow(r, 4) / (dim * (dim + 2.0));
    check_close(mean0 / n, 0.0, 5 * r / std::sqrt(dim * double(n)), "E[x]" + tag);
    check_close(m2 / n, e2, 0.01 * e2, "E[x^2]" + tag);
    check_close(m4 / n, e4, 0.02 * e4, "E[x^4]" + tag);
}

}  // namespace

int main() {
    test_reproducibility();
    test_uniforms();
    std::printf("\n== Uniform points on spheres ==\n");
    for (size_t d : {2, 3, 8}) test_sphere(d);
    return finish("test_rng");
}
