// test_mixed_mc_exact.cpp — MixedLattice Monte Carlo vs. exact equilibrium
// statistics on CP^2 (SU(3) qutrit pure states, Fubini-Study measure) and
// S^2 (SU(2) spins).
//
//  1. One SU(3) site in a field, E = -B.n = <psi|-B.lambda|psi>: the moment
//     map pushes the Fubini-Study measure to the uniform measure on the
//     simplex of populations p_k = |<v_k|psi>|^2 of the eigenbasis of
//     -B.lambda, E = sum_k p_k e_k, so <E> and <n_3> are 2D simplex
//     integrals (Gauss-Legendre).
//  2. Two SU(3) sites, E = J n_1.n_2 = J (2x - 2/3), x = |<psi_1|psi_2>|^2
//     ~ Beta(1, 2) under Haar: <E> is a 1D integral.
//  3. Two SU(2) sites with exchange, fields and anisotropic on-site terms:
//     <E> from 4D quadrature (Metropolis-corrected overrelaxation and heat
//     bath must sample the Boltzmann weight exactly).
//  4. One Fe-Tm pair with every production coupling class: Fe field and
//     single-ion anisotropy, Tm field, Fe-Tm exchange K^- and the
//     self-coupled vertex W(S, S, n): <E> from quadrature over S^2 x CP^2.
//  5. The legacy S^7 manifold (su3_mc_manifold = sphere) still samples the
//     uniform measure on S^7: <n.h> = L I_4(K)/I_3(K).
//
// Every local-update kernel (uniform / Gaussian Metropolis, heat bath,
// overrelaxation at T) runs against each case. Tolerance 5 sigma of a
// batch-means error.
#include "physics_test_util.h"

#include "classical_spin/core/su3_coherent_state.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <functional>
#include <sstream>

using namespace phys_test;
namespace su3 = classical_spin::su3;

namespace {

constexpr double kSigma = 5.0;

const std::vector<Eigen::Vector3d> kAxes = {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
                                            Eigen::Vector3d(0, 0, 1)};

// The MixedLattice constructor is chatty; build test lattices quietly.
// Silences std::cout for its lifetime (exception-safe).
struct CoutSilencer {
    std::stringstream sink;
    std::streambuf* old = std::cout.rdbuf(sink.rdbuf());
    ~CoutSilencer() { std::cout.rdbuf(old); }
};

template <class F>
auto quiet(F&& f) {
    CoutSilencer silence;
    return f();
}

MixedLattice build(const MixedUnitCell& cell) {
    return quiet([&] { return MixedLattice(cell, 1, 1, 1, 1.0f, 1.0f); });
}

void gl01(int n, std::vector<double>& x, std::vector<double>& w) {
    gauss_legendre(n, x, w);
    for (int i = 0; i < n; ++i) { x[i] = 0.5 * (x[i] + 1.0); w[i] *= 0.5; }
}

// <f(p)> over the simplex with weight exp(-beta sum_k p_k e_k); Duffy map
// p0 = u, p1 = (1-u) v, p2 = (1-u)(1-v), Jacobian (1-u).
double simplex_average(const Eigen::Vector3d& e, double beta, const std::function<double(const double*)>& f) {
    std::vector<double> x, w;
    gl01(64, x, w);
    double num = 0.0, den = 0.0;
    const double shift = e.minCoeff();
    for (size_t a = 0; a < x.size(); ++a)
        for (size_t b = 0; b < x.size(); ++b) {
            const double p[3] = {x[a], (1.0 - x[a]) * x[b], (1.0 - x[a]) * (1.0 - x[b])};
            const double E = p[0] * e(0) + p[1] * e(1) + p[2] * e(2);
            const double wt = w[a] * w[b] * (1.0 - x[a]) * std::exp(-beta * (E - shift));
            num += wt * f(p);
            den += wt;
        }
    return num / den;
}

using Sampler = std::pair<std::string, std::function<void(MixedLattice&, double)>>;

std::vector<Sampler> samplers(bool with_or) {
    std::vector<Sampler> s = {
        {"metropolis(uniform)", [](MixedLattice& l, double T) { l.metropolis(T); }},
        {"metropolis(gaussian 0.5)", [](MixedLattice& l, double T) { l.metropolis(T, true, 0.5); }},
        {"metropolis_interleaved(gaussian 0.3)", [](MixedLattice& l, double T) { l.metropolis_interleaved(T, true, 0.3); }},
        {"heat_bath", [](MixedLattice& l, double T) { l.heat_bath(T); }},
    };
    if (with_or) {
        s.push_back({"overrelaxation(T)+metropolis(gaussian 0.5)",
                     [](MixedLattice& l, double T) { l.overrelaxation(T); l.metropolis(T, true, 0.5); }});
        s.push_back({"overrelaxation(T)+heat_bath",
                     [](MixedLattice& l, double T) { l.overrelaxation(T); l.heat_bath(T); }});
    }
    return s;
}

MeanErr sample(MixedLattice& lat, const std::function<void()>& sweep, const std::function<double()>& obs,
               size_t n_therm, size_t n_meas) {
    for (size_t i = 0; i < n_therm; ++i) sweep();
    std::vector<double> x;
    x.reserve(n_meas);
    for (size_t i = 0; i < n_meas; ++i) {
        sweep();
        x.push_back(obs());
    }
    (void)lat;
    return batch_means(x, 50);
}

// ----------------------------------------------------- 1. free SU(3) site
void test_free_su3_site() {
    std::printf("\n== One SU(3) site in a field on CP^2 (simplex integrals) ==\n");
    // (a) B along lambda_3: E = -h n_3, <n_3> = <p_0 - p_1>.
    {
        const double h = 1.0;
        UnitCell su2(3, 1, {Eigen::Vector3d::Zero()}, kAxes);
        UnitCell su3c(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)}, kAxes);
        SpinVector B = SpinVector::Zero(8);
        B(2) = h;
        su3c.set_field(B, 0);
        MixedLattice lat = build(MixedUnitCell(su2, su3c));
        for (double T : {0.25, 0.7, 2.0}) {
            // eigenvalues of -h lambda_3 in the basis (|0>, |1>, |2>): (-h, h, 0)
            const double exact = simplex_average(Eigen::Vector3d(-h, h, 0.0), 1.0 / T,
                                                 [](const double* p) { return p[0] - p[1]; });
            for (const auto& [name, step] : samplers(true)) {
                seed_lehman(1234);
                lat.init_random();
                const auto r = sample(lat, [&] { step(lat, T); }, [&] { return lat.spins_SU3[0](2); },
                                      2000, 200000);
                check_stat(r.mean, r.err, exact, "<n_3> " + name + " T=" + std::to_string(T), kSigma);
            }
        }
    }
    // (b) generic field: <E> with E = sum_k p_k e_k in the eigenbasis of -B.lambda.
    {
        UnitCell su2(3, 1, {Eigen::Vector3d::Zero()}, kAxes);
        UnitCell su3c(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)}, kAxes);
        SpinVector B(8);
        B << 0.3, -0.5, 0.2, 0.4, 0.1, -0.6, 0.25, 0.7;
        su3c.set_field(B, 0);
        MixedLattice lat = build(MixedUnitCell(su2, su3c));
        const SpinVector mB = -B;
        Eigen::SelfAdjointEigenSolver<su3::Matrix3c> es(su3::gell_mann_sum(mB.data()));
        const Eigen::Vector3d e = es.eigenvalues();
        for (double T : {0.3, 1.5}) {
            const double exact = simplex_average(e, 1.0 / T, [&](const double* p) {
                return p[0] * e(0) + p[1] * e(1) + p[2] * e(2);
            });
            for (const auto& [name, step] : samplers(true)) {
                seed_lehman(99);
                lat.init_random();
                const auto r = sample(lat, [&] { step(lat, T); }, [&] { return lat.total_energy(); }, 2000, 150000);
                check_stat(r.mean, r.err, exact, "<E> generic field " + name + " T=" + std::to_string(T), kSigma);
            }
        }
    }
}

// ----------------------------------------------- 2. two coupled SU(3) sites
void test_su3_dimer() {
    std::printf("\n== Two SU(3) sites, E = J n_1.n_2 (x = |<psi_1|psi_2>|^2 ~ Beta(1,2)) ==\n");
    std::vector<double> x, w;
    gl01(96, x, w);
    for (double J : {1.0, -1.0}) {
        UnitCell su2(3, 1, {Eigen::Vector3d::Zero()}, kAxes);
        UnitCell su3c(8, 2, {Eigen::Vector3d(0.25, 0.25, 0.25), Eigen::Vector3d(0.75, 0.75, 0.75)}, kAxes);
        su3c.set_bilinear_interaction(J * SpinMatrix::Identity(8, 8), 0, 1, Eigen::Vector3i(0, 0, 0));
        MixedLattice lat = build(MixedUnitCell(su2, su3c));
        for (double T : {0.3, 1.0}) {
            double num = 0.0, den = 0.0;
            for (size_t k = 0; k < x.size(); ++k) {
                const double E = J * (2.0 * x[k] - 2.0 / 3.0);
                const double wt = w[k] * 2.0 * (1.0 - x[k]) * std::exp(-(E + std::abs(J)) / T);
                num += wt * E;
                den += wt;
            }
            const double exact = num / den;
            for (const auto& [name, step] : samplers(true)) {
                seed_lehman(4242);
                lat.init_random();
                const auto r = sample(lat, [&] { step(lat, T); }, [&] { return lat.total_energy(); }, 2000, 150000);
                check_stat(r.mean, r.err, exact,
                           "<E> J=" + std::to_string(J) + " " + name + " T=" + std::to_string(T), kSigma);
            }
        }
    }
}

// ------------------------------------- 3. SU(2) dimer with anisotropy (S^2 x S^2)
void test_su2_dimer_anisotropic() {
    std::printf("\n== Two SU(2) sites with anisotropic exchange, fields and on-site anisotropy ==\n");
    Eigen::Matrix3d J;
    J << 0.9, 0.3, -0.2, -0.1, 0.7, 0.25, 0.15, -0.3, 1.1;
    Eigen::Matrix3d A0, A1;
    A0 << 0.8, 0.2, 0.0, 0.2, -0.4, 0.1, 0.0, 0.1, 0.3;
    A1 << -0.6, 0.0, 0.3, 0.0, 0.2, 0.0, 0.3, 0.0, 0.5;
    const Eigen::Vector3d B0(0.3, -0.2, 0.5), B1(-0.4, 0.1, 0.2);
    UnitCell su2(3, 2, {Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.0, 0.0)}, kAxes);
    su2.set_bilinear_interaction(J, 0, 1, Eigen::Vector3i(0, 0, 0));
    su2.set_onsite_interaction(A0, 0);
    su2.set_onsite_interaction(A1, 1);
    su2.set_field(B0, 0);
    su2.set_field(B1, 1);
    UnitCell su3c(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)}, kAxes);
    MixedLattice lat = build(MixedUnitCell(su2, su3c));

    const SphereRule q = sphere_rule(28, 56);
    const Eigen::Matrix3d As0 = 0.5 * (A0 + A0.transpose()), As1 = 0.5 * (A1 + A1.transpose());
    for (double T : {0.4, 1.2}) {
        double num = 0.0, den = 0.0;
        for (size_t a = 0; a < q.pts.size(); ++a) {
            const Eigen::Vector3d& S0 = q.pts[a];
            const double e0 = -B0.dot(S0) + S0.dot(As0 * S0);
            const Eigen::Vector3d g = J.transpose() * S0 - B1;
            for (size_t b = 0; b < q.pts.size(); ++b) {
                const Eigen::Vector3d& S1 = q.pts[b];
                const double E = e0 + g.dot(S1) + S1.dot(As1 * S1);
                const double wt = q.wts[a] * q.wts[b] * std::exp(-(E + 4.0) / T);
                num += wt * E;
                den += wt;
            }
        }
        const double exact = num / den;
        for (const auto& [name, step] : samplers(true)) {
            seed_lehman(2024);
            lat.init_random();
            const auto r = sample(lat, [&] { step(lat, T); }, [&] { return lat.total_energy(); }, 2000, 150000);
            check_stat(r.mean, r.err, exact, "<E> " + name + " T=" + std::to_string(T), kSigma);
        }
    }
}

// ------------------------------------------- 4. Fe-Tm pair: S^2 x CP^2
struct FeTmModel {
    Eigen::Vector3d B2;
    Eigen::Matrix3d A;
    SpinVector B3;
    Eigen::MatrixXd Jm;   // 3 x 8
    SpinTensor3 K;        // K[a](b, c), a, b Fe; c Tm

    double energy(const Eigen::Vector3d& S, const double* n) const {
        Eigen::Map<const Eigen::Matrix<double, 8, 1>> nv(n);
        double E = -B2.dot(S) + S.dot(A * S) - B3.dot(nv) + S.dot(Jm * nv);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) E += S(a) * S(b) * K[a].row(b).dot(nv);
        return E;
    }
};

FeTmModel fe_tm_model() {
    FeTmModel m;
    m.B2 = Eigen::Vector3d(0.2, -0.3, 0.4);
    m.A << -0.5, 0.15, 0.0, 0.15, 0.3, -0.1, 0.0, -0.1, 0.6;
    m.B3 = SpinVector(8);
    m.B3 << 0.1, 0.0, 0.485, -0.2, 0.0, 0.15, 0.0, 0.7;
    m.Jm = Eigen::MatrixXd(3, 8);
    m.Jm << 0.3, -0.2, 0.1, 0.0, 0.25, -0.1, 0.05, 0.2,
            -0.15, 0.1, 0.3, 0.2, 0.0, 0.1, -0.25, 0.0,
            0.05, 0.2, -0.1, 0.15, -0.2, 0.0, 0.1, 0.3;
    m.K.assign(3, Eigen::MatrixXd::Zero(3, 8));
    const double vals[3][3][8] = {
        {{0.2, 0, 0.1, 0, 0, -0.1, 0, 0.15}, {0.05, 0, 0, 0.1, 0, 0, 0, 0}, {0, 0, -0.1, 0, 0.05, 0, 0, 0}},
        {{0.05, 0, 0, 0.1, 0, 0, 0, 0}, {-0.15, 0, 0.2, 0, 0, 0.1, 0, -0.1}, {0, 0.1, 0, 0, 0, 0, 0.05, 0}},
        {{0, 0, -0.1, 0, 0.05, 0, 0, 0}, {0, 0.1, 0, 0, 0, 0, 0.05, 0}, {0.1, 0, -0.05, 0, 0, 0.2, 0, 0.1}}};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 8; ++c) m.K[a](b, c) = vals[a][b][c];
    return m;
}

// <E> over S^2 x CP^2. CP^2 points psi = (r0, r1 e^{i phi1}, r2 e^{i phi2}) with
// r on the positive octant of S^2 (r0 = sin t cos a, r1 = sin t sin a,
// r2 = cos t): the Fubini-Study measure is dp0 dp1 dphi1 dphi2 with p = r^2,
// i.e. r0 r1 r2 sin t dt da dphi1 dphi2 (smooth integrand: spectral GL).
double fe_tm_exact(const FeTmModel& m, double T, int n_oct, int n_phi, int n_th, int n_ph) {
    std::vector<double> x, w;
    gauss_legendre(n_oct, x, w);
    const SphereRule q = sphere_rule(n_th, n_ph);
    double num = 0.0, den = 0.0;
    for (int it = 0; it < n_oct; ++it) {
        const double t = 0.25 * M_PI * (x[it] + 1.0), wt_t = 0.25 * M_PI * w[it];
        for (int ia = 0; ia < n_oct; ++ia) {
            const double al = 0.25 * M_PI * (x[ia] + 1.0), wt_a = 0.25 * M_PI * w[ia];
            const double r0 = std::sin(t) * std::cos(al), r1 = std::sin(t) * std::sin(al), r2 = std::cos(t);
            const double jac = r0 * r1 * r2 * std::sin(t) * wt_t * wt_a;
            for (int i1 = 0; i1 < n_phi; ++i1)
                for (int i2 = 0; i2 < n_phi; ++i2) {
                    const double p1 = 2.0 * M_PI * i1 / n_phi, p2 = 2.0 * M_PI * i2 / n_phi;
                    const su3::Complex psi[3] = {r0, std::polar(r1, p1), std::polar(r2, p2)};
                    double n[8];
                    su3::pure_expectations(psi, n);
                    Eigen::Map<const Eigen::Matrix<double, 8, 1>> nv(n);
                    // E(S, n) = S^T M S + g.S + c
                    Eigen::Matrix3d M = m.A;
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) M(a, b) += m.K[a].row(b).dot(nv);
                    const Eigen::Vector3d g = m.Jm * nv - m.B2;
                    const double c = -m.B3.dot(nv);
                    for (size_t s = 0; s < q.pts.size(); ++s) {
                        const Eigen::Vector3d& S = q.pts[s];
                        const double E = S.dot(M * S) + g.dot(S) + c;
                        const double wt = jac * q.wts[s] * std::exp(-(E + 4.0) / T);
                        num += wt * E;
                        den += wt;
                    }
                }
        }
    }
    return num / den;
}

void test_fe_tm_pair() {
    std::printf("\n== Fe-Tm pair: field, anisotropy, K^- and self-coupled W (S^2 x CP^2 quadrature) ==\n");
    const FeTmModel m = fe_tm_model();
    UnitCell su2(3, 1, {Eigen::Vector3d::Zero()}, kAxes);
    su2.set_field(m.B2, 0);
    su2.set_onsite_interaction(m.A, 0);
    UnitCell su3c(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)}, kAxes);
    su3c.set_field(m.B3, 0);
    MixedUnitCell cell(su2, su3c);
    cell.set_mixed_bilinear(m.Jm, 0, 0, Eigen::Vector3i(0, 0, 0));
    cell.set_mixed_trilinear(m.K, 0, 0, 0, Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(0, 0, 0));
    MixedLattice lat = build(cell);
    check(!lat.self_isotropic_SU2[0] && lat.self_isotropic_SU3[0],
          "Fe site has a non-constant self energy (anisotropy + W), Tm site is linear");

    // The model energy must be the lattice energy (validates the reference).
    seed_lehman(5);
    double e_err = 0.0;
    for (int k = 0; k < 20; ++k) {
        lat.init_random();
        const Eigen::Vector3d S(lat.spins_SU2[0].data());
        e_err = std::max(e_err, std::abs(lat.total_energy() - m.energy(S, lat.spins_SU3[0].data())));
    }
    check_close(e_err, 0.0, 1e-12, "reference energy function == MixedLattice::total_energy");

    for (double T : {0.6, 1.5}) {
        const double exact = fe_tm_exact(m, T, 14, 14, 16, 32);
        const double coarse = fe_tm_exact(m, T, 11, 11, 13, 26);
        check_close(exact - coarse, 0.0, 1e-7, "quadrature converged at T=" + std::to_string(T));
        for (const auto& [name, step] : samplers(true)) {
            seed_lehman(31337);
            lat.init_random();
            const auto r = sample(lat, [&] { step(lat, T); }, [&] { return lat.total_energy(); }, 2000, 200000);
            check_stat(r.mean, r.err, exact, "<E> " + name + " T=" + std::to_string(T), kSigma);
        }
    }
}

// ----------------------------------------------- 5. legacy S^7 manifold
void test_legacy_sphere() {
    std::printf("\n== Legacy manifold (su3_mc_manifold = sphere): uniform measure on S^7 ==\n");
    const double h = 0.8, T = 0.5;
    UnitCell su2(3, 1, {Eigen::Vector3d::Zero()}, kAxes);
    UnitCell su3c(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)}, kAxes);
    SpinVector B = SpinVector::Zero(8);
    B(2) = h;
    su3c.set_field(B, 0);
    MixedLattice lat = build(MixedUnitCell(su2, su3c));
    lat.set_su3_mc_manifold("sphere");
    const double K = h / T;
    const double exact = std::cyl_bessel_i(4.0, K) / std::cyl_bessel_i(3.0, K);
    for (const auto& [name, step] : samplers(false)) {
        seed_lehman(77);
        lat.init_random();
        const auto r = sample(lat, [&] { step(lat, T); }, [&] { return lat.spins_SU3[0](2); }, 2000, 150000);
        check_stat(r.mean, r.err, exact, "S^7 <n_3> " + name, kSigma);
    }
    check(std::abs(lat.spins_SU3[0].norm() - 1.0) < 1e-12, "S^7 states keep |n| = spin_length_SU3");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    test_free_su3_site();
    test_su3_dimer();
    test_su2_dimer_anisotropic();
    test_fe_tm_pair();
    test_legacy_sphere();
    const int rc = finish("test_mixed_mc_exact");
    MPI_Finalize();
    return rc;
}
