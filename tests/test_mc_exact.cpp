// test_mc_exact.cpp — Monte Carlo kernels vs. exact equilibrium statistics.
//
//  1. Free spins in a field          : u = -hS L(beta h S)            (Langevin)
//  2. 1D Heisenberg AFM ring          : u = -J L(beta J), c = 1 - K^2/sinh^2 K
//                                       (Fisher 1964; ring corrections ~ L^N)
//  3. Two-site cluster, general H     : <E> from 4D Gauss-Legendre quadrature
//     (non-symmetric J, field, easy-axis on-site anisotropy)
//  4. Three-site cluster with trilinear coupling: <E> from 6D quadrature
//  5. Structural: colour partition is a proper colouring of the bond graph.
//
// Every local-update kernel exposed by Lattice is run against each case it is
// valid for. A kernel that violates detailed balance shifts <E> by many sigma.
#include "physics_test_util.h"

#include <random>

using namespace phys_test;

namespace {

constexpr double kSigma = 5.0;

void set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

// ---------------------------------------------------------------- free spins
void test_free_spins() {
    std::printf("\n== Free spins in a field ==\n");
    const double h = 1.0;
    for (double S : {1.0, 2.0}) {
        UnitCell uc = simple_cubic_cell(1);
        uc.set_field(Eigen::Vector3d(0, 0, h), 0);
        Lattice lat(uc, 8, 8, 1, float(S));
        for (double T : {0.5, 2.0}) {
            const double exact = -h * S * langevin(h * S / T);
            seed_lehman(12345);
            auto r = sample_energy_density(lat, 500, 6000, [&] { lat.metropolis(T); });
            check_stat(r.mean, r.err, exact,
                       "metropolis(uniform) S=" + std::to_string(S) + " T=" + std::to_string(T), kSigma);
            double sigma = 0.4;
            r = sample_energy_density(lat, 500, 6000, [&] { lat.metropolis(T, true, sigma); });
            check_stat(r.mean, r.err, exact, "metropolis(gaussian 0.4) S=" + std::to_string(S) +
                       " T=" + std::to_string(T), kSigma);
            r = sample_energy_density(lat, 500, 6000, [&] { lat.overrelaxation(); lat.metropolis(T); });
            check_stat(r.mean, r.err, exact, "metropolis+overrelaxation S=" + std::to_string(S) +
                       " T=" + std::to_string(T), kSigma);
        }
    }
}

// ------------------------------------------- free spins on S^7 (spin_dim 8)
void test_free_spins_s7() {
    std::printf("\n== Free 8-component spins in a field (uniform measure on S^7) ==\n");
    // E = -h.S on S^{n-1}: <cos t> = I_{n/2}(K) / I_{n/2-1}(K), K = beta h S.
    const size_t n = 8;
    const double h = 1.0, T = 0.6;
    UnitCell uc(n, 1, {Eigen::Vector3d::Zero()},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    SpinVector f = SpinVector::Zero(n);
    f(7) = h;
    uc.set_field(f, 0);
    Lattice lat(uc, 8, 8, 1, 1.0f);
    const double K = h / T;
    const double exact = -h * std::cyl_bessel_i(n / 2.0, K) / std::cyl_bessel_i(n / 2.0 - 1.0, K);
    seed_lehman(808);
    auto r = sample_energy_density(lat, 500, 8000, [&] { lat.metropolis(T); });
    check_stat(r.mean, r.err, exact, "S^7 metropolis(uniform)", kSigma);
    double sig = 0.5;
    r = sample_energy_density(lat, 500, 8000, [&] { lat.metropolis(T, true, sig); });
    check_stat(r.mean, r.err, exact, "S^7 metropolis(gaussian)", kSigma);
}

// ------------------------------------------------------------ Heisenberg ring
void test_heisenberg_ring() {
    std::printf("\n== 1D Heisenberg AFM ring (N = 64) ==\n");
    const double J = 1.0;
    UnitCell uc = chain_cell(J * Eigen::Matrix3d::Identity());
    Lattice lat(uc, 64, 1, 1, 1.0f);
    for (double T : {0.5, 2.0}) {
        const double K = J / T;
        const double u_exact = -J * langevin(K);
        const double c_exact = 1.0 - K * K / (std::sinh(K) * std::sinh(K));
        const std::string tag = " T=" + std::to_string(T);

        struct Kernel { const char* name; std::function<void()> f; };
        double sig = 0.6;
        std::vector<Kernel> kernels = {
            {"metropolis", [&] { lat.metropolis(T); }},
            {"metropolis(gaussian)", [&] { lat.metropolis(T, true, sig); }},
            {"metropolis+overrelaxation", [&] { lat.overrelaxation(); lat.metropolis(T); }},
            {"metropolis_parallel", [&] { lat.metropolis_parallel(T); }},
            {"overrelaxation_parallel+metropolis_parallel",
             [&] { lat.overrelaxation_parallel(); lat.metropolis_parallel(T); }},
            {"wolff+metropolis", [&] { lat.wolff_sweep(T, 4); lat.metropolis(T); }},
            {"swendsen_wang+metropolis", [&] { lat.swendsen_wang_sweep(T); lat.metropolis(T); }},
        };
        for (auto& k : kernels) {
            set_threads(std::string(k.name).find("parallel") != std::string::npos ? 2 : 1);
            seed_lehman(777);
            lat.init_random();
            for (int i = 0; i < 1000; ++i) k.f();
            std::vector<double> e;
            const size_t n = 20000;
            e.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                k.f();
                e.push_back(lat.total_energy());
            }
            std::vector<double> u(n);
            for (size_t i = 0; i < n; ++i) u[i] = e[i] / double(lat.lattice_size);
            auto r = batch_means(u);
            check_stat(r.mean, r.err, u_exact, std::string(k.name) + " u" + tag, kSigma);

            // Specific heat from energy fluctuations, error from batch spread.
            const size_t nb = 20, b = n / nb;
            std::vector<double> cb(nb);
            for (size_t q = 0; q < nb; ++q) {
                double s = 0, s2 = 0;
                for (size_t i = 0; i < b; ++i) { s += e[q * b + i]; s2 += e[q * b + i] * e[q * b + i]; }
                const double m = s / b;
                cb[q] = (s2 / b - m * m) / (double(lat.lattice_size) * T * T);
            }
            auto rc = batch_means(cb, nb);
            check_stat(rc.mean, rc.err, c_exact, std::string(k.name) + " c" + tag, kSigma, 0.02);
        }
        set_threads(1);
    }
}

// ------------------------------------------- cluster moves, FM ring + field
void test_cluster_ferromagnet() {
    std::printf("\n== Cluster moves on the FM ring in a field ==\n");
    // FM Heisenberg ring in a field has no closed form at finite N, so use
    // plain Metropolis (validated above) as the reference.
    const double J = -1.0, T = 0.7;
    UnitCell uc = chain_cell(J * Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0.3));
    Lattice lat(uc, 32, 1, 1, 1.0f);
    seed_lehman(55);
    auto ref = sample_energy_density(lat, 2000, 60000, [&] { lat.metropolis(T); lat.overrelaxation(); });
    struct K { const char* name; std::function<void()> f; };
    std::vector<K> ks = {
        {"wolff(ghost)", [&] { lat.wolff_sweep(T, 4, true); }},
        {"wolff(no ghost, filtered)", [&] { lat.wolff_sweep(T, 4, false); lat.metropolis(T); }},
        {"swendsen_wang(ghost)", [&] { lat.swendsen_wang_sweep(T, true); }},
        {"swendsen_wang(no ghost, filtered)", [&] { lat.swendsen_wang_sweep(T, false); }},
    };
    for (auto& k : ks) {
        lat.init_random();
        auto r = sample_energy_density(lat, 2000, 60000, k.f);
        const double err = std::sqrt(r.err * r.err + ref.err * ref.err);
        check_stat(r.mean, err, ref.mean, std::string(k.name) + " vs metropolis", kSigma);
    }
    // Clusters must actually form for a zero-field ferromagnet at low T
    // (the sign error fixed in wolff_update produced single-site clusters).
    Lattice fm(chain_cell(J * Eigen::Matrix3d::Identity()), 32, 1, 1, 1.0f);
    fm.init_ferromagnetic(Eigen::Vector3d(0, 0, 1));
    size_t total = 0;
    for (int i = 0; i < 200; ++i) total += fm.wolff_update(0.2, false);
    check(total > 200 * 4, "wolff clusters grow for a ferromagnet (mean size " +
                               std::to_string(total / 200.0) + ")");
}

// ----------------------------------------------------- two-site general model
struct TwoSiteModel {
    Eigen::Matrix3d J, A;
    Eigen::Vector3d h;
    // Lattice(2,1,1) with one chain bond creates TWO bonds 0->1 and 1->0(wrap),
    // so the pair energy is S0^T (J + J^T) S1.
    double energy(const Eigen::Vector3d& s0, const Eigen::Vector3d& s1) const {
        return s0.dot((J + J.transpose()) * s1) - h.dot(s0 + s1) + s0.dot(A * s0) + s1.dot(A * s1);
    }
};

double exact_two_site_energy(const TwoSiteModel& m, double T) {
    const auto rule = sphere_rule(28, 56);
    const size_t n = rule.pts.size();
    double Z = 0.0, EZ = 0.0, Emin = 1e300;
    std::vector<double> E(n * n);
    for (size_t a = 0; a < n; ++a)
        for (size_t b = 0; b < n; ++b) {
            E[a * n + b] = m.energy(rule.pts[a], rule.pts[b]);
            Emin = std::min(Emin, E[a * n + b]);
        }
    for (size_t a = 0; a < n; ++a)
        for (size_t b = 0; b < n; ++b) {
            const double w = rule.wts[a] * rule.wts[b] * std::exp(-(E[a * n + b] - Emin) / T);
            Z += w;
            EZ += w * E[a * n + b];
        }
    return EZ / Z;
}

void run_two_site_case(const TwoSiteModel& m, const std::string& label, bool or_valid) {
    UnitCell uc = chain_cell(m.J, m.h, m.A);
    Lattice lat(uc, 2, 1, 1, 1.0f);

    // Energy bookkeeping must match the independent formula.
    seed_lehman(99);
    double max_err = 0.0;
    for (int t = 0; t < 50; ++t) {
        lat.init_random();
        max_err = std::max(max_err, std::abs(lat.total_energy() -
                                             m.energy(lat.spins[0], lat.spins[1])));
    }
    check_close(max_err, 0.0, 1e-12, label + " total_energy matches closed form");

    // Local energy differences must match total-energy differences.
    double max_dE_err = 0.0;
    for (int t = 0; t < 50; ++t) {
        lat.init_random();
        const size_t site = t % 2;
        const double E0 = lat.total_energy();
        SpinVector old = lat.spins[site];
        SpinVector nw = lat.gen_random_spin(1.0f);
        const double dE = lat.site_energy_diff(nw, old, site);
        const double dE_flat = lat.site_energy_diff_flat(nw.data(), old.data(), site);
        lat.spins[site] = nw;
        const double E1 = lat.total_energy();
        lat.spins[site] = old;
        max_dE_err = std::max({max_dE_err, std::abs(dE - (E1 - E0)), std::abs(dE_flat - (E1 - E0))});
    }
    check_close(max_dE_err, 0.0, 1e-12, label + " site_energy_diff == total-energy difference");

    for (double T : {0.7, 2.0}) {
        const double exact = exact_two_site_energy(m, T) / 2.0;
        const std::string tag = label + " T=" + std::to_string(T);
        seed_lehman(4242);
        auto r = sample_energy_density(lat, 2000, 200000, [&] { lat.metropolis(T); });
        check_stat(r.mean, r.err, exact, "metropolis " + tag, kSigma, 1e-4);
        double sig = 0.5;
        r = sample_energy_density(lat, 2000, 200000, [&] { lat.metropolis(T, true, sig); });
        check_stat(r.mean, r.err, exact, "metropolis(gaussian) " + tag, kSigma, 1e-4);
        if (or_valid) {
            r = sample_energy_density(lat, 2000, 200000, [&] { lat.overrelaxation(); lat.metropolis(T); });
            check_stat(r.mean, r.err, exact, "metropolis+overrelaxation " + tag, kSigma, 1e-4);
            r = sample_energy_density(lat, 2000, 200000, [&] { lat.overrelaxation(T); lat.metropolis(T); });
            check_stat(r.mean, r.err, exact, "metropolis+overrelaxation(T) " + tag, kSigma, 1e-4);
        }
        // Cluster moves: the embedded-Ising part is exact only for isotropic
        // exchange; everything else must be absorbed by the residual filter.
        r = sample_energy_density(lat, 2000, 200000, [&] { lat.wolff_update(T, false); lat.metropolis(T); });
        check_stat(r.mean, r.err, exact, "wolff+metropolis " + tag, kSigma, 1e-4);
        r = sample_energy_density(lat, 2000, 200000, [&] { lat.wolff_update(T, true); lat.metropolis(T); });
        check_stat(r.mean, r.err, exact, "wolff(ghost)+metropolis " + tag, kSigma, 1e-4);
        r = sample_energy_density(lat, 2000, 200000, [&] { lat.swendsen_wang_sweep(T, true); lat.metropolis(T); });
        check_stat(r.mean, r.err, exact, "swendsen_wang(ghost)+metropolis " + tag, kSigma, 1e-4);
        r = sample_energy_density(lat, 2000, 200000, [&] { lat.swendsen_wang_sweep(T, false); });
        check_stat(r.mean, r.err, exact, "swendsen_wang only " + tag, kSigma, 1e-4);
    }
}

void test_two_site() {
    std::printf("\n== Two-site cluster vs. 4D quadrature ==\n");
    TwoSiteModel m;
    m.J << 0.8, 0.3, -0.2,
          -0.1, 0.5, 0.4,
           0.25, -0.35, -0.6;
    m.h = Eigen::Vector3d(0.3, -0.2, 0.5);
    m.A.setZero();
    run_two_site_case(m, "[bilinear+field]", /*or_valid=*/true);

    // Easy-axis on-site anisotropy: energy is quadratic in S_i, so a
    // reflection about the local field is NOT energy conserving. Sampling
    // with overrelaxation must either be skipped or reject such moves.
    m.A = Eigen::Vector3d(0.0, 0.2, -0.9).asDiagonal();
    run_two_site_case(m, "[bilinear+field+onsite]", /*or_valid=*/true);
}

// ---------------------------------------------------- three-site trilinear
void test_three_site_trilinear() {
    std::printf("\n== Three-site cluster with trilinear coupling vs. 6D quadrature ==\n");
    std::mt19937 gen(2024);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    SpinTensor3 Tt(3, Eigen::MatrixXd::Zero(3, 3));
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c) Tt[a](b, c) = U(gen);
    Eigen::Matrix3d J = 0.4 * Eigen::Matrix3d::Identity();
    J(0, 1) = 0.2;
    Eigen::Vector3d h(0.0, 0.1, 0.3);

    UnitCell uc = chain_cell(J, h);
    uc.set_trilinear_interaction(Tt, 0, 0, 0, Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(2, 0, 0));
    Lattice lat(uc, 3, 1, 1, 1.0f);

    auto energy = [&](const Eigen::Vector3d* s) {
        double E = 0.0;
        for (int i = 0; i < 3; ++i) {
            const auto& a = s[i]; const auto& b = s[(i + 1) % 3]; const auto& c = s[(i + 2) % 3];
            E += a.dot(J * b) - h.dot(a);
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q)
                    for (int r = 0; r < 3; ++r) E += Tt[p](q, r) * a[p] * b[q] * c[r];
        }
        return E;
    };

    seed_lehman(5);
    double max_err = 0.0, max_dE_err = 0.0;
    for (int t = 0; t < 50; ++t) {
        lat.init_random();
        Eigen::Vector3d s[3] = {lat.spins[0], lat.spins[1], lat.spins[2]};
        max_err = std::max(max_err, std::abs(lat.total_energy() - energy(s)));
        const size_t site = t % 3;
        SpinVector old = lat.spins[site], nw = lat.gen_random_spin(1.0f);
        const double dE = lat.site_energy_diff_flat(nw.data(), old.data(), site);
        Eigen::Vector3d s2[3] = {s[0], s[1], s[2]};
        s2[site] = nw;
        max_dE_err = std::max(max_dE_err, std::abs(dE - (energy(s2) - energy(s))));
    }
    check_close(max_err, 0.0, 1e-12, "trilinear total_energy matches closed form");
    check_close(max_dE_err, 0.0, 1e-12, "trilinear site_energy_diff_flat matches closed form");

    // Local field (gradient) must match a finite-difference gradient: the
    // field drives overrelaxation, the T=0 quench and the LLG dynamics.
    {
        lat.init_random();
        double max_g_err = 0.0;
        for (size_t site = 0; site < 3; ++site) {
            SpinVector H = lat.get_local_field(site);
            double Hf[3];
            std::vector<double> flat(9);
            for (size_t i = 0; i < 3; ++i)
                for (int d = 0; d < 3; ++d) flat[i * 3 + d] = lat.spins[i](d);
            lat.get_local_field_flat(flat.data(), site, Hf);
            for (int d = 0; d < 3; ++d) {
                const double eps = 1e-6;
                Eigen::Vector3d sp[3] = {lat.spins[0], lat.spins[1], lat.spins[2]};
                Eigen::Vector3d sm[3] = {lat.spins[0], lat.spins[1], lat.spins[2]};
                sp[site](d) += eps;
                sm[site](d) -= eps;
                const double g = (energy(sp) - energy(sm)) / (2 * eps);
                max_g_err = std::max({max_g_err, std::abs(H(d) - g), std::abs(Hf[d] - g)});
            }
        }
        check_close(max_g_err, 0.0, 1e-6, "trilinear local field == dE/dS (finite difference)");
    }

    const auto rule = sphere_rule(10, 20);
    const size_t n = rule.pts.size();
    for (double T : {1.0, 2.5}) {
        double Z = 0, EZ = 0;
        for (size_t a = 0; a < n; ++a)
            for (size_t b = 0; b < n; ++b)
                for (size_t c = 0; c < n; ++c) {
                    Eigen::Vector3d s[3] = {rule.pts[a], rule.pts[b], rule.pts[c]};
                    const double E = energy(s);
                    const double w = rule.wts[a] * rule.wts[b] * rule.wts[c] * std::exp(-E / T);
                    Z += w;
                    EZ += w * E;
                }
        const double exact = EZ / Z / 3.0;
        seed_lehman(31337);
        auto r = sample_energy_density(lat, 2000, 150000, [&] { lat.metropolis(T); });
        check_stat(r.mean, r.err, exact, "trilinear metropolis T=" + std::to_string(T), kSigma, 2e-4);
        r = sample_energy_density(lat, 2000, 150000, [&] { lat.overrelaxation(); lat.metropolis(T); });
        check_stat(r.mean, r.err, exact, "trilinear metropolis+overrelaxation T=" + std::to_string(T),
                   kSigma, 2e-4);
        r = sample_energy_density(lat, 2000, 150000, [&] { lat.wolff_update(T); lat.metropolis(T); });
        check_stat(r.mean, r.err, exact, "trilinear wolff+metropolis T=" + std::to_string(T), kSigma, 2e-4);
    }
}

// -------------------------------------------------------- colour partition
void check_colouring(const Lattice& lat, const std::string& label) {
    bool ok = lat.n_colors > 0;
    for (size_t i = 0; i < lat.lattice_size && ok; ++i) {
        for (size_t p : lat.bilinear_partners[i])
            if (p != i && lat.color_of_site[p] == lat.color_of_site[i]) ok = false;
        for (const auto& pr : lat.trilinear_partners[i])
            for (size_t p : pr)
                if (p != i && lat.color_of_site[p] == lat.color_of_site[i]) ok = false;
    }
    check(ok, label + " colour partition is a proper colouring (" +
                  std::to_string(lat.n_colors) + " colours)");
}

void test_colouring() {
    std::printf("\n== Colour partitions ==\n");
    {
        Lattice lat(triangular_heisenberg_cell(1.0), 6, 6, 1);
        check_colouring(lat, "triangular 6x6");
    }
    {
        Lattice lat(pyrochlore_heisenberg_cell(1.0), 3, 3, 3);
        check_colouring(lat, "pyrochlore 3x3x3");
    }
    {
        Lattice lat(chain_cell(Eigen::Matrix3d::Identity()), 5, 1, 1);
        check_colouring(lat, "odd ring N=5");
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    set_threads(1);
    test_colouring();
    test_free_spins();
    test_free_spins_s7();
    test_heisenberg_ring();
    test_cluster_ferromagnet();
    test_two_site();
    test_three_site_trilinear();
    const int rc = finish("test_mc_exact");
    MPI_Finalize();
    return rc;
}
