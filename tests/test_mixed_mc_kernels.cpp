// test_mixed_mc_kernels.cpp — exact (non-statistical) properties of the
// MixedLattice Monte Carlo kernels.
//
//  1. Self-bonds and bonds longer than the lattice: the energy of a lattice
//     one cell wide equals (per cell) that of a larger lattice holding the
//     tiled configuration; the local dE equals the total-energy difference
//     for random moves; the MD field is the gradient of the same energy.
//  2. Overrelaxation: conserves the energy exactly where the local energy
//     is linear in the site's own spin (SU(2) reflection, CP^2 random-phase
//     move, S^7 reflection); leaves sites with a non-constant self energy
//     untouched at T = 0.
//  3. Every SU(3) state produced by any kernel or driver lies on CP^2:
//     |n|^2 = 4/3 and d_abc n^a n^b n^c = 8/9 to 1e-12.
//  4. Sweeps visit every site exactly once (no selection with replacement).
//  5. T = 0 descent: monotone, exact single-site minimiser with anisotropy.
//  6. Simulated annealing (adaptive Gaussian and heat bath) reaches the exact
//     ground state of a decoupled model; invalid schedules throw.
//  7. Policy parsing and the legacy-convention default manifold.
#include "physics_test_util.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/su3_coherent_state.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <filesystem>
#include <random>
#include <sstream>

using namespace phys_test;
namespace su3 = classical_spin::su3;

namespace {

const std::vector<Eigen::Vector3d> kAxes = {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
                                            Eigen::Vector3d(0, 0, 1)};

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

template <class F>
void quietly(F&& f) {
    CoutSilencer silence;
    f();
}

SpinConfig tmfeo3_config(bool anisotropy_and_W) {
    SpinConfig cfg;
    cfg.field_strength = 0.0;
    cfg.set_param("J1ab", 4.74); cfg.set_param("J1c", 5.15);
    cfg.set_param("J2ab", 0.15); cfg.set_param("J2c", 0.30);
    cfg.set_param("D1", 0.12);
    cfg.set_param("e1", 0.97); cfg.set_param("e2", 3.97);
    cfg.set_param("Kminus_2x", 0.12); cfg.set_param("Kminus_5y", -0.08); cfg.set_param("Kminus_7z", 0.05);
    cfg.set_param("Jtm_3", 0.05); cfg.set_param("Jtm_8", -0.03); cfg.set_param("Jtm_1", 0.02);
    if (anisotropy_and_W) {
        cfg.set_param("Ka", -0.16221); cfg.set_param("Kc", -0.18318);
        cfg.set_param("W3_xx", 0.05); cfg.set_param("W8_zz", -0.04); cfg.set_param("W1_xy", 0.03);
        cfg.set_param("W4_xz", 0.02); cfg.set_param("W6_yz", -0.025);
    } else {
        cfg.set_param("Ka", 0.0); cfg.set_param("Kc", 0.0);
    }
    return cfg;
}

MixedLattice tmfeo3(size_t d1, size_t d2, size_t d3, bool anisotropy_and_W = true) {
    return quiet([&] { return MixedLattice(build_tmfeo3(tmfeo3_config(anisotropy_and_W)), d1, d2, d3, 1.0f, 1.0f); });
}

// A cell whose couplings are longer than the small lattices used below, so
// that bonds wrap more than once and onto their own site: SU(2) and SU(3)
// bilinears (incl. a bond onto the site's own image), mixed bilinears, a
// self-coupled and a long mixed trilinear, an SU(2) trilinear whose
// partners wrap onto the source, and on-site terms.
MixedUnitCell long_bond_cell() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-0.5, 0.5);
    auto rnd = [&](int r, int c) { Eigen::MatrixXd M(r, c); for (int i = 0; i < r * c; ++i) M.data()[i] = u(rng); return M; };
    UnitCell su2(3, 2, {Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.0, 0.0)}, kAxes);
    UnitCell su3c(8, 2, {Eigen::Vector3d(0.25, 0.5, 0.5), Eigen::Vector3d(0.75, 0.5, 0.5)}, kAxes);
    su2.set_bilinear_interaction(rnd(3, 3), 0, 1, Eigen::Vector3i(3, 0, 0));
    su2.set_bilinear_interaction(rnd(3, 3), 0, 0, Eigen::Vector3i(0, 2, 0));    // onto its own image
    su2.set_bilinear_interaction(rnd(3, 3), 1, 0, Eigen::Vector3i(-3, 1, -2));
    su2.set_onsite_interaction(rnd(3, 3), 0);
    su2.set_field(rnd(3, 1), 1);
    SpinTensor3 T3(3);
    for (auto& m : T3) m = rnd(3, 3);
    su2.set_trilinear_interaction(T3, 0, 0, 1, Eigen::Vector3i(2, 0, 0), Eigen::Vector3i(0, -1, 0));
    su3c.set_bilinear_interaction(rnd(8, 8), 0, 0, Eigen::Vector3i(0, -2, 0));  // onto its own image
    su3c.set_bilinear_interaction(rnd(8, 8), 0, 1, Eigen::Vector3i(1, 3, 0));
    su3c.set_onsite_interaction(rnd(8, 8), 1);
    su3c.set_field(rnd(8, 1), 0);
    MixedUnitCell cell(su2, su3c);
    cell.set_mixed_bilinear(rnd(3, 8), 0, 1, Eigen::Vector3i(2, -1, 3));
    cell.set_mixed_bilinear(rnd(3, 8), 1, 0, Eigen::Vector3i(-2, 0, 0));
    SpinTensor3 K(3), K2(3);
    for (auto& m : K) m = rnd(3, 8);
    for (auto& m : K2) m = rnd(3, 8);
    cell.set_mixed_trilinear(K, 1, 1, 0, Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(0, 1, 0));    // W-like
    cell.set_mixed_trilinear(K2, 0, 1, 1, Eigen::Vector3i(-2, 2, 0), Eigen::Vector3i(1, 0, -3));
    return cell;
}

MixedLattice long_bond_lattice(size_t d1, size_t d2, size_t d3) {
    return quiet([&] { return MixedLattice(long_bond_cell(), d1, d2, d3, 1.0f, 1.0f); });
}

// Copy the configuration of `small` onto `big` periodically (big dims are multiples).
void tile(const MixedLattice& small, MixedLattice& big) {
    auto copy = [&](size_t N_atoms, const auto& src, auto& dst) {
        for (size_t i = 0; i < big.dim1; ++i)
            for (size_t j = 0; j < big.dim2; ++j)
                for (size_t k = 0; k < big.dim3; ++k)
                    for (size_t a = 0; a < N_atoms; ++a)
                        dst[big.flatten_index(i, j, k, a, N_atoms)] =
                            src[small.flatten_index(i % small.dim1, j % small.dim2, k % small.dim3, a, N_atoms)];
    };
    copy(small.N_atoms_SU2, small.spins_SU2, big.spins_SU2);
    copy(small.N_atoms_SU3, small.spins_SU3, big.spins_SU3);
}

double cells(const MixedLattice& l) { return double(l.dim1 * l.dim2 * l.dim3); }

// ------------------------------------------------ 1. self-bonds, long bonds
void check_tiling(MixedLattice& small, MixedLattice& big, const std::string& what) {
    double err = 0.0, scale = 0.0;
    for (int t = 0; t < 5; ++t) {
        small.init_random();
        tile(small, big);
        const double e_small = small.total_energy() / cells(small);
        const double e_big = big.total_energy() / cells(big);
        err = std::max(err, std::abs(e_small - e_big));
        scale = std::max(scale, std::abs(e_big));
    }
    check(err <= 1e-12 * (1.0 + scale), what + ": energy per cell == tiled larger lattice (err " +
                                            std::to_string(err) + ")");
}

void check_local_dE(MixedLattice& lat, const std::string& what) {
    std::mt19937 rng(11);
    std::normal_distribution<double> g(0.0, 1.0);
    double err2 = 0.0, err3 = 0.0, scale = 1.0;
    for (int t = 0; t < 300; ++t) {
        lat.init_random();
        const double E0 = lat.total_energy();
        scale = std::max(scale, std::abs(E0));
        const size_t i = t % lat.lattice_size_SU2, j = t % lat.lattice_size_SU3;
        // SU(2): a uniform proposal; every third move an arbitrary (off-sphere) vector.
        SpinVector s(3);
        random_point_on_sphere(s.data(), 3, 1.0);
        if (t % 3 == 0) for (int a = 0; a < 3; ++a) s(a) = g(rng);
        const SpinVector s_old = lat.spins_SU2[i];
        const double dE2 = lat.site_energy_SU2_diff(s, s_old, i);
        lat.spins_SU2[i] = s;
        err2 = std::max(err2, std::abs(dE2 - (lat.total_energy() - E0)));
        lat.spins_SU2[i] = s_old;
        // SU(3): a CP^2 state; every third move an arbitrary 8-vector.
        SpinVector n(8);
        lat.random_SU3_state(n.data());
        if (t % 3 == 0) for (int a = 0; a < 8; ++a) n(a) = g(rng);
        const SpinVector n_old = lat.spins_SU3[j];
        const double dE3 = lat.site_energy_SU3_diff(n, n_old, j);
        lat.spins_SU3[j] = n;
        err3 = std::max(err3, std::abs(dE3 - (lat.total_energy() - E0)));
        lat.spins_SU3[j] = n_old;
    }
    check(err2 <= 1e-11 * scale && err3 <= 1e-11 * scale,
          what + ": local dE == E(after) - E(before) (SU(2) err " + std::to_string(err2) + ", SU(3) err " +
              std::to_string(err3) + ")");
}

// The MD field (full gradient) is the gradient of total_energy (central differences).
void check_gradient(MixedLattice& lat, const std::string& what) {
    lat.init_random();
    const double d = 1e-5;
    double err = 0.0;
    for (size_t i = 0; i < std::min<size_t>(lat.lattice_size_SU2, 4); ++i) {
        const SpinVector H = lat.get_local_field_SU2(i);
        for (int a = 0; a < 3; ++a) {
            const double s0 = lat.spins_SU2[i](a);
            lat.spins_SU2[i](a) = s0 + d; const double Ep = lat.total_energy();
            lat.spins_SU2[i](a) = s0 - d; const double Em = lat.total_energy();
            lat.spins_SU2[i](a) = s0;
            err = std::max(err, std::abs((Ep - Em) / (2 * d) - H(a)));
        }
    }
    for (size_t j = 0; j < std::min<size_t>(lat.lattice_size_SU3, 4); ++j) {
        const SpinVector H = lat.get_local_field_SU3(j);
        for (int a = 0; a < 8; ++a) {
            const double s0 = lat.spins_SU3[j](a);
            lat.spins_SU3[j](a) = s0 + d; const double Ep = lat.total_energy();
            lat.spins_SU3[j](a) = s0 - d; const double Em = lat.total_energy();
            lat.spins_SU3[j](a) = s0;
            err = std::max(err, std::abs((Ep - Em) / (2 * d) - H(a)));
        }
    }
    check(err < 1e-6, what + ": get_local_field_SU{2,3} == dE/dS (err " + std::to_string(err) + ")");
}

void test_self_and_long_bonds() {
    std::printf("\n== Self-bonds and bonds longer than the lattice ==\n");
    seed_lehman(1);
    {
        MixedLattice l1 = tmfeo3(1, 1, 1), l2 = tmfeo3(2, 2, 2);
        check(l1.self_mode_SU2[0] != 0, "1x1x1 TmFeO3: J2 self-bonds folded into the Fe on-site matrix");
        check_tiling(l1, l2, "TmFeO3 1x1x1 vs 2x2x2");
        check_local_dE(l1, "TmFeO3 1x1x1");
        check_gradient(l1, "TmFeO3 1x1x1");
        MixedLattice s1 = tmfeo3(2, 1, 2), s2 = tmfeo3(4, 2, 4);
        check_tiling(s1, s2, "TmFeO3 2x1x2 slab vs 4x2x4");
        check_local_dE(s1, "TmFeO3 2x1x2 slab");
    }
    {
        MixedLattice l1 = long_bond_lattice(1, 1, 1), l2 = long_bond_lattice(2, 2, 2);
        check_tiling(l1, l2, "long bonds 1x1x1 vs 2x2x2");
        check_local_dE(l1, "long bonds 1x1x1");
        check_gradient(l1, "long bonds 1x1x1");
        MixedLattice m1 = long_bond_lattice(2, 1, 1), m2 = long_bond_lattice(4, 3, 2);
        check_tiling(m1, m2, "long bonds 2x1x1 vs 4x3x2");
        check_local_dE(m1, "long bonds 2x1x1");
        MixedLattice b = long_bond_lattice(1, 2, 1);
        check(b.self_mode_SU2[0] == 2 || b.self_mode_SU2[1] == 2 || b.self_mode_SU2[0] == 1,
              "long bonds 1x2x1: wrapped trilinear entries classified as self-coupled");
        check_local_dE(b, "long bonds 1x2x1");
    }
}

// ------------------------------------------------------- 2. overrelaxation
void test_overrelaxation() {
    std::printf("\n== Overrelaxation: exact where the local energy is linear ==\n");
    for (const char* manifold : {"cp2", "sphere"}) {
        MixedLattice lat = tmfeo3(2, 2, 2, /*anisotropy_and_W=*/false);
        lat.set_su3_mc_manifold(manifold);
        seed_lehman(3);
        lat.init_random();
        const double E0 = lat.total_energy();
        const auto s2 = lat.spins_SU2;
        const auto s3 = lat.spins_SU3;
        for (int k = 0; k < 20; ++k) lat.overrelaxation(1.0);
        double moved = 0.0;
        for (size_t j = 0; j < s3.size(); ++j) moved = std::max(moved, (lat.spins_SU3[j] - s3[j]).norm());
        for (size_t i = 0; i < s2.size(); ++i) moved = std::max(moved, (lat.spins_SU2[i] - s2[i]).norm());
        check_close(lat.total_energy() - E0, 0.0, 1e-11 * std::abs(E0),
                    std::string("linear sites (") + manifold + "): 20 overrelaxation sweeps conserve E");
        check(moved > 0.1, std::string("linear sites (") + manifold + "): overrelaxation moves the spins");
    }
    {
        MixedLattice lat = tmfeo3(2, 2, 2, /*anisotropy_and_W=*/true);
        seed_lehman(4);
        lat.init_random();
        const double E0 = lat.total_energy();
        const auto s2 = lat.spins_SU2;
        for (int k = 0; k < 20; ++k) lat.overrelaxation(0.0);
        double fe_moved = 0.0;
        for (size_t i = 0; i < s2.size(); ++i) fe_moved = std::max(fe_moved, (lat.spins_SU2[i] - s2[i]).norm());
        check_close(lat.total_energy() - E0, 0.0, 1e-11 * std::abs(E0),
                    "anisotropy + W at T = 0: overrelaxation conserves E (Tm phases only)");
        check(fe_moved == 0.0, "anisotropy + W at T = 0: Fe sites with a non-constant self energy are skipped");
        for (int k = 0; k < 5; ++k) lat.overrelaxation(2.0);
        fe_moved = 0.0;
        for (size_t i = 0; i < s2.size(); ++i) fe_moved = std::max(fe_moved, (lat.spins_SU2[i] - s2[i]).norm());
        check(fe_moved > 0.1, "anisotropy + W at T > 0: Metropolis-corrected reflections move the Fe spins");
    }
}

// --------------------------------------------------------- 3. CP^2 invariants
double casimir_error(const MixedLattice& lat) {
    double err = 0.0;
    for (const auto& n : lat.spins_SU3) {
        err = std::max(err, std::abs(su3::casimir2(n.data()) - 4.0 / 3.0));
        err = std::max(err, std::abs(su3::casimir3(n.data()) - 8.0 / 9.0));
    }
    return err;
}

void test_cp2_invariants() {
    std::printf("\n== Every SU(3) state stays on CP^2 (|n|^2 = 4/3, cubic Casimir 8/9) ==\n");
    MixedLattice lat = tmfeo3(2, 2, 2);
    check(lat.su3_on_cp2(), "default SU(3) Monte Carlo manifold is CP^2");
    seed_lehman(5);
    auto ck = [&](const std::string& what) {
        const double e = casimir_error(lat);
        check(e <= 1e-12 && lat.min_SU3_density_eigenvalue() >= -1e-12,
              what + ": max Casimir error " + std::to_string(e));
    };
    lat.init_random();
    ck("constructor / init_random");
    for (int k = 0; k < 200; ++k) lat.metropolis(1.0);
    ck("200 x metropolis(uniform)");
    for (int k = 0; k < 200; ++k) lat.metropolis(0.3, true, 0.4);
    ck("200 x metropolis(gaussian)");
    for (int k = 0; k < 200; ++k) lat.metropolis_interleaved(0.3, true, 0.2);
    ck("200 x metropolis_interleaved(gaussian)");
    for (int k = 0; k < 200; ++k) lat.heat_bath(0.5);
    ck("200 x heat_bath");
    for (int k = 0; k < 200; ++k) lat.overrelaxation(0.5);
    ck("200 x overrelaxation(T)");
    lat.deterministic_sweep(50);
    ck("50 x deterministic_sweep");
    SpinVector d2 = SpinVector::Zero(3), d3 = SpinVector::Zero(8);
    d2(2) = 1.0; d3(2) = 1.0;
    lat.init_ferromagnetic(d2, d3);
    ck("init_ferromagnetic(lambda_3)");
    check_close(lat.spins_SU3[0](7), 1.0 / std::sqrt(3.0), 1e-12, "init_ferromagnetic(lambda_3) gives |1>");
    // A legacy (off-manifold) state is projected by the drivers.
    for (auto& n : lat.spins_SU3) { n.setZero(); n(2) = 1.0; n(5) = 0.3; }
    quietly([&] { lat.simulated_annealing(2.0, 0.2, 20, true, 0.7, "", false, true, 50); });
    ck("simulated_annealing from off-manifold states (+ T = 0 stage)");
    for (auto& n : lat.spins_SU3) { n.setZero(); n(7) = 1.0; }
    quietly([&] { lat.greedy_quench(1e-12, 50); });
    ck("greedy_quench from off-manifold states");
}

// ------------------------------------------------------------- 4. sweeps
void test_sweeps_visit_every_site() {
    std::printf("\n== Sweeps visit every site exactly once ==\n");
    MixedLattice lat = tmfeo3(3, 3, 2);
    seed_lehman(6);
    const double T_inf = 1e12;   // every proposal is accepted
    auto all_changed = [&](const std::function<void()>& sweep) {
        lat.init_random();
        const auto s2 = lat.spins_SU2;
        const auto s3 = lat.spins_SU3;
        sweep();
        size_t same = 0;
        for (size_t i = 0; i < s2.size(); ++i) same += (lat.spins_SU2[i] - s2[i]).norm() == 0.0;
        for (size_t j = 0; j < s3.size(); ++j) same += (lat.spins_SU3[j] - s3[j]).norm() == 0.0;
        return same;
    };
    check(all_changed([&] { lat.metropolis(T_inf); }) == 0, "metropolis: every site updated in one sweep");
    check(all_changed([&] { lat.metropolis_interleaved(T_inf); }) == 0,
          "metropolis_interleaved: every site updated in one sweep");
    check(all_changed([&] { lat.heat_bath(T_inf); }) == 0, "heat_bath: every site updated in one sweep");
    check(all_changed([&] { lat.metropolis_parallel(T_inf); }) == 0, "metropolis_parallel: every site updated");
    MixedLattice lin = tmfeo3(2, 2, 2, false);
    lin.init_random();
    check_close(lin.heat_bath(0.7), 1.0, 0.0, "heat bath is rejection-free where the local energy is linear");
}

// ------------------------------------------------------- 5. T = 0 descent
void test_descent() {
    std::printf("\n== T = 0 descent ==\n");
    MixedLattice lat = tmfeo3(2, 2, 2);
    seed_lehman(8);
    lat.init_random();
    double E_prev = lat.total_energy(), worst_rise = 0.0, change = 1.0;
    for (int k = 0; k < 300; ++k) {
        change = lat.deterministic_sweep();
        const double E = lat.total_energy();
        worst_rise = std::max(worst_rise, E - E_prev);
        E_prev = E;
    }
    check(worst_rise <= 1e-10 * std::abs(E_prev), "descent never raises the energy (anisotropy + W, CP^2)");
    check(change < 1e-6, "descent converges (last max |dS| = " + std::to_string(change) + ")");
    // Stationarity: every SU(2) spin minimises its local energy exactly, so
    // one more site update changes nothing.
    check(lat.deterministic_sweep() < 1e-6, "converged state is a fixed point of the exact site minimiser");

    // Single Fe site with easy-plane anisotropy in a field: global minimum by brute force.
    UnitCell su2(3, 1, {Eigen::Vector3d::Zero()}, kAxes);
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    A(2, 2) = 1.0;
    su2.set_onsite_interaction(A, 0);
    su2.set_field(Eigen::Vector3d(0.3, 0.0, 0.0), 0);
    UnitCell su3c(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)}, kAxes);
    MixedLattice one = quiet([&] { return MixedLattice(MixedUnitCell(su2, su3c), 1, 1, 1, 1.0f, 1.0f); });
    // e(S) = -0.3 S_x + S_z^2 >= -0.3, with equality at S = x (the old
    // antiparallel-to-full-field rule stalled near the energy maximum here).
    one.spins_SU2[0] = Eigen::Vector3d(0.1, 0.2, 0.97).normalized();
    one.deterministic_sweep();
    double best = 1e300;
    const SphereRule q = sphere_rule(100, 200);
    for (const auto& S : q.pts) best = std::min(best, -0.3 * S(0) + S(2) * S(2));
    check_close(one.total_energy(), -0.3, 1e-14, "exact single-site minimiser with easy-plane anisotropy");
    check(one.total_energy() <= best, "single-site minimiser beats a dense sphere scan");
}

// ------------------------------------------------- 6. simulated annealing
void test_simulated_annealing() {
    std::printf("\n== Simulated annealing to an exactly known ground state ==\n");
    // Triangular Heisenberg AFM (E0 = -3/2 J per site, 120 degree order on
    // 3x3) decoupled from Tm sites in a field (E0 = lowest eigenvalue of -B.lambda).
    Triangular su2(3);
    const Eigen::Matrix3d J = Eigen::Matrix3d::Identity();
    su2.set_bilinear_interaction(J, 0, 0, Eigen::Vector3i(1, 0, 0));
    su2.set_bilinear_interaction(J, 0, 0, Eigen::Vector3i(0, 1, 0));
    su2.set_bilinear_interaction(J, 0, 0, Eigen::Vector3i(-1, 1, 0));
    UnitCell su3c(8, 1, {Eigen::Vector3d(0.5, 0.25, 0.0)}, su2.lattice_vectors);
    SpinVector B(8);
    B << 0.2, 0.0, 0.5, 0.0, -0.3, 0.0, 0.1, 0.8;
    su3c.set_field(B, 0);
    const SpinVector mB = -B;
    const double tm_e0 = su3::ground_state_energy(su3::gell_mann_sum(mB.data()));
    for (const char* policy : {"gaussian", "heat_bath"}) {
        MixedLattice lat = quiet([&] { return MixedLattice(MixedUnitCell(su2, su3c), 3, 3, 1, 1.0f, 1.0f); });
        lat.local_update = MixedLattice::parse_local_update(policy);
        seed_lehman(9);
        lat.init_random();
        quietly([&] { lat.simulated_annealing(3.0, 0.01, 300, false, 0.85, "", false, true, 2000); });
        const double exact = -1.5 * double(lat.lattice_size_SU2) + tm_e0 * double(lat.lattice_size_SU3);
        check_close(lat.total_energy(), exact, 1e-8, std::string("SA (") + policy + ") + T = 0 descent: exact E0");
        check(casimir_error(lat) <= 1e-12, std::string("SA (") + policy + "): SU(3) ground states on CP^2");
    }
    MixedLattice lat = tmfeo3(1, 1, 1);
    bool threw = false;
    try { quietly([&] { lat.simulated_annealing(1.0, 0.1, 10, true, 1.0); }); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "simulated_annealing rejects cooling_rate >= 1 (was an infinite loop)");
    threw = false;
    try { quietly([&] { lat.simulated_annealing(1.0, 0.0, 10, true, 0.9); }); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "simulated_annealing rejects T_end <= 0");
}

// ------------------------------------------------------ 7. policy parsing
void test_policy() {
    std::printf("\n== Policy parsing and defaults ==\n");
    check(MixedLattice::parse_local_update("heat_bath") == MixedLattice::LocalUpdate::HeatBath &&
              MixedLattice::parse_local_update("gaussian") == MixedLattice::LocalUpdate::Gaussian &&
              MixedLattice::parse_local_update("metropolis") == MixedLattice::LocalUpdate::Metropolis,
          "parse_local_update");
    bool threw = false;
    try { MixedLattice::parse_su3_manifold("torus"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "parse_su3_manifold rejects unknown names");
    SpinConfig cfg = tmfeo3_config(true);
    cfg.set_param("su3_legacy_convention", 1.0);
    MixedLattice legacy = quiet([&] { return MixedLattice(build_tmfeo3(cfg), 1, 1, 1, 1.0f, 1.0f); });
    check(!legacy.su3_on_cp2(), "su3_legacy_convention keeps the legacy S^7 Monte Carlo manifold");
    legacy.set_su3_mc_manifold("cp2");
    check(legacy.su3_on_cp2(), "su3_mc_manifold = cp2 overrides the legacy default");
    legacy.set_su3_mc_manifold("auto");
    check(!legacy.su3_on_cp2(), "su3_mc_manifold = auto restores the convention default");
    MixedLattice lat = tmfeo3(1, 1, 1);
    lat.local_update = MixedLattice::LocalUpdate::HeatBath;
    check(!lat.uses_adaptive_step(true), "heat bath has no adaptive step size");
    lat.local_update = MixedLattice::LocalUpdate::Gaussian;
    check(lat.uses_adaptive_step(false), "local_update = gaussian adapts the step size");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    test_self_and_long_bonds();
    test_overrelaxation();
    test_cp2_invariants();
    test_sweeps_visit_every_site();
    test_descent();
    test_simulated_annealing();
    test_policy();
    const int rc = finish("test_mixed_mc_kernels");
    MPI_Finalize();
    return rc;
}
