// test_phonon_dynamics.cpp — thermostats and Monte Carlo of the spin–phonon model
// (PhononLattice) against results known independently of the code:
//
//   1. Free spins in a field B: the Langevin thermostat must give <S·B̂> = L(βB) =
//      coth(βB) − 1/(βB) for Gilbert damping α = 0.1 AND α = 1. A noise variance of
//      2αT/|S| (the pre-2026-10 value, noise in precession and damping) heats the spins
//      to T(1 + α²): at α = 1 it gives L(βB/2) = 0.164 instead of 0.313.
//   2. The same exact law for the Metropolis and heat-bath kernels.
//   3. Interacting Kitaev–Γ honeycomb with the lattice sector decoupled: Langevin <E>(T)
//      equals the Monte Carlo (heat bath and Metropolis) <E>(T) for α = 0.1 and 1.
//   4. Fluctuation–dissipation for the damped zone-centre modes and the SLD momenta:
//      equipartition N<V_a²> = T for every damped mode velocity, also with the
//      magnetoelastic coupling on (kinetic equipartition is exact in the coupled
//      Gibbs state), and <p_a²>/m = T for the site momenta.
//   5. Joint spin–lattice Monte Carlo (mc_sample_lattice): N ω² <Q_a²> = T for a decoupled
//      mode, and for the COUPLED model the potential energy and <|Q|²> agree with the
//      Langevin dynamics — two independent samplers of the same Gibbs state.
#include "physics_test_util.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"

#include <iostream>
#include <sstream>

using namespace phys_test;

namespace {

struct Silence {                          // the lattice is chatty on stdout
    std::streambuf* saved;
    std::ostringstream sink;
    Silence() : saved(std::cout.rdbuf()) { std::cout.rdbuf(sink.rdbuf()); }
    ~Silence() { std::cout.rdbuf(saved); }
};

PhononLattice make_model(size_t L, double J, double K, double G, double Bz,
                         double omega = 2.0, double gamma = 0.0) {
    Silence quiet;
    SpinConfig config;
    config.set_param("J", J); config.set_param("K", K); config.set_param("Gamma", G);
    config.set_param("Gammap", 0.0);
    config.set_param("J2_A", 0.0); config.set_param("J2_B", 0.0); config.set_param("J3", 0.0);
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lat(uc, L, L, 1, 1.0f);
    SpinPhononCouplingParams sp;
    sp.J = J; sp.K = K; sp.Gamma = G; sp.Gammap = 0.0;
    sp.J2_A = sp.J2_B = sp.J3 = 0.0; sp.J7 = 0.0;
    PhononParams ph;
    ph.omega_E1 = omega; ph.gamma_E1 = gamma; ph.lambda_E1_quartic = 0.0;
    DriveParams dr;                                    // no drive
    lat.set_parameters(sp, ph, dr);
    lat.set_field(Eigen::Vector3d(0.0, 0.0, Bz));
    lat.set_seed(20261005ULL, 7ULL);
    return lat;
}

/// Langevin time series of an observable, sampled every `every` steps after `t_therm`.
template <class Obs>
std::vector<double> langevin_series(PhononLattice& lat, double T, double alpha, double dt,
                                    double t_therm, double t_meas, size_t every, uint64_t seed,
                                    const Obs& obs) {
    lat.langevin_temperature = T;
    lat.alpha_gilbert = alpha;
    std::vector<double> x;
    {
        Silence quiet;
        lat.integrate_langevin(0.0, t_therm, dt, "", 1000000000, seed);
        lat.integrate_langevin(t_therm, t_therm + t_meas, dt, "", every, seed + 1,
                               [&](double) { x.push_back(obs()); });
    }
    return x;
}

void test_free_spins_langevin() {
    std::printf("\n-- Langevin free spins: <S_z> = L(B/T) for alpha = 0.1 and 1 --\n");
    const double B = 1.0, T = 1.0;
    for (const double alpha : {0.1, 1.0}) {
        PhononLattice lat = make_model(10, 0.0, 0.0, 0.0, B);   // N = 200 independent spins
        lat.init_ferromagnetic(Eigen::Vector3d(0, 0, 1));
        // relaxation time ~ (1 + α²)/(α B); measure ~ 100 relaxation times
        const double tau = (1.0 + alpha * alpha) / (alpha * B);
        const auto mz = langevin_series(lat, T, alpha, 0.01, 10.0 * tau, 100.0 * tau, 20, 11,
                                        [&] { return lat.magnetization_local()(2); });
        const MeanErr me = batch_means(mz, 20);
        check_stat(me.mean, me.err, langevin(B / T),
                   "Langevin alpha=" + std::to_string(alpha) + " <S_z>", 5.0, 0.004);
    }
}

void test_free_spins_mc() {
    std::printf("\n-- Metropolis and heat bath on free spins: <S_z> = L(B/T) --\n");
    const double B = 1.0, T = 0.7;
    for (const bool hb : {false, true}) {
        PhononLattice lat = make_model(10, 0.0, 0.0, 0.0, B);
        lat.local_update = hb ? PhononLattice::LocalUpdate::HeatBath : PhononLattice::LocalUpdate::Metropolis;
        for (int s = 0; s < 200; ++s) lat.local_sweep(T);
        std::vector<double> mz;
        for (int s = 0; s < 4000; ++s) { lat.local_sweep(T); mz.push_back(lat.magnetization_local()(2)); }
        const MeanErr me = batch_means(mz, 20);
        check_stat(me.mean, me.err, langevin(B / T), hb ? "heat bath <S_z>" : "Metropolis <S_z>");
    }
}

void test_interacting_langevin_vs_mc() {
    std::printf("\n-- Interacting Kitaev-Gamma honeycomb: Langevin <E>/N == Monte Carlo <E>/N --\n");
    const double J = -1.0, K = -1.0, G = 0.3, Bz = 0.2, T = 0.6;
    auto spin_e = [](PhononLattice& L) { return L.spin_energy() / double(L.lattice_size); };
    // Monte Carlo references (heat bath and Metropolis must agree with each other too)
    MeanErr mc[2];
    for (int k = 0; k < 2; ++k) {
        PhononLattice lat = make_model(3, J, K, G, Bz);
        lat.local_update = k ? PhononLattice::LocalUpdate::HeatBath : PhononLattice::LocalUpdate::Metropolis;
        for (int s = 0; s < 2000; ++s) lat.local_sweep(T);
        std::vector<double> e;
        for (int s = 0; s < 60000; ++s) { lat.local_sweep(T); if (s % 2 == 0) e.push_back(spin_e(lat)); }
        mc[k] = batch_means(e, 30);
    }
    const double mc_err = std::hypot(mc[0].err, mc[1].err);
    check_stat(mc[1].mean - mc[0].mean, mc_err, 0.0, "heat bath - Metropolis <E>/N");
    const MeanErr ref{0.5 * (mc[0].mean + mc[1].mean), 0.5 * mc_err};
    for (const double alpha : {0.1, 1.0}) {
        PhononLattice lat = make_model(3, J, K, G, Bz);   // lattice sector decoupled (no couplings)
        const double t_meas = (alpha < 0.5) ? 4000.0 : 800.0;   // energy relaxes in ~1/(α|H|)
        const auto e = langevin_series(lat, T, alpha, 0.01, 50.0 / alpha, t_meas, 20, 23,
                                       [&] { return spin_e(lat); });
        const MeanErr me = batch_means(e, 25);
        // 5 sigma of the combined error plus an O(dt) splitting allowance of 0.3 % of |E|
        check_stat(me.mean, std::hypot(me.err, ref.err), ref.mean,
                   "Langevin alpha=" + std::to_string(alpha) + " <E>/N vs MC", 5.0, 3e-3 * std::abs(ref.mean));
    }
}

void test_lattice_fdt() {
    std::printf("\n-- Fluctuation-dissipation of the damped lattice coordinates --\n");
    const double T = 0.8;
    {
        // E1 doublet (gamma = 1) + an extra damped A1 mode, magnetoelastic coupling ON
        PhononLattice lat = make_model(4, -1.0, -1.0, 0.3, 0.0, 2.0, 1.0);
        {
            Silence quiet;
            LatticeMode a; a.irrep = LatticeMode::Irrep::A1; a.name = "A1"; a.omega = 1.5; a.gamma = 0.8;
            a.aA1 = {0.3, -0.2, 0.1, 0.0, 0.0};
            lat.set_modes({a}, {});
            lat.modes[0].cE[1] = 0.5;      // linear K channel of the E1 doublet
            lat.update_modulation_flags();
        }
        const double N = double(lat.lattice_size);
        std::vector<double> kx, ky, ka;
        lat.langevin_temperature = T;
        lat.alpha_gilbert = 0.5;
        {
            Silence quiet;
            lat.integrate_langevin(0.0, 20.0, 0.01, "", 1000000000, 5);
            lat.integrate_langevin(20.0, 1220.0, 0.01, "", 25, 6, [&](double) {
                kx.push_back(N * lat.phonons.V_x_E1 * lat.phonons.V_x_E1);
                ky.push_back(N * lat.phonons.V_y_E1 * lat.phonons.V_y_E1);
                ka.push_back(N * lat.modes[1].V1 * lat.modes[1].V1);
            });
        }
        const MeanErr mx = batch_means(kx, 30), my = batch_means(ky, 30), ma = batch_means(ka, 30);
        check_stat(mx.mean, mx.err, T, "E1 equipartition N<V_x^2> (coupled)");
        check_stat(my.mean, my.err, T, "E1 equipartition N<V_y^2> (coupled)");
        check_stat(ma.mean, ma.err, T, "A1 equipartition N<V^2> (coupled)");
    }
    {
        // Spin-lattice dynamics: <p_a^2>/m = T for the in-plane site momenta
        PhononLattice lat = make_model(4, -1.0, 0.0, 0.0, 0.0, 2.0, 0.0);
        {
            Silence quiet;
            lat.sld_mass = 1000.0; lat.sld_k = 4.0e4; lat.sld_k2 = 1.0e4; lat.sld_g = 0.2;
            lat.sld_gamma = 0.5; lat.sld_relax = 0;
            lat.enable_sld(true);
        }
        std::vector<double> kin;
        lat.langevin_temperature = T;
        lat.alpha_gilbert = 0.3;
        {
            Silence quiet;
            lat.integrate_langevin(0.0, 20.0, 0.005, "", 1000000000, 8);
            lat.integrate_langevin(20.0, 220.0, 0.005, "", 40, 9, [&](double) {
                kin.push_back(lat.sld_kinetic_energy() / double(lat.lattice_size));   // = <p²>/(2m) per site, 2 comps → T
            });
        }
        const MeanErr mk = batch_means(kin, 25);
        check_stat(mk.mean, mk.err, T, "SLD equipartition <p_x^2 + p_y^2>/(2m) per site");
    }
}


void test_joint_mc_vs_langevin() {
    std::printf("\n-- Joint spin-lattice Monte Carlo vs Langevin (coupled model) --\n");
    const double T = 0.8;
    auto build = [&]() {
        PhononLattice lat = make_model(4, -1.0, -1.0, 0.3, 0.0, 2.0, 1.0);
        Silence quiet;
        LatticeMode a; a.irrep = LatticeMode::Irrep::A1; a.name = "A1"; a.omega = 1.5; a.gamma = 0.8;
        a.aA1 = {0.3, -0.2, 0.1, 0.0, 0.0};
        lat.set_modes({a}, {});
        lat.modes[0].cE[1] = 0.5;
        lat.update_modulation_flags();
        return lat;
    };
    auto e_pot = [](PhononLattice& L) {
        const double N = double(L.lattice_size);
        double kin = 0.5 * N * (L.phonons.V_x_E1 * L.phonons.V_x_E1 + L.phonons.V_y_E1 * L.phonons.V_y_E1);
        for (size_t m = 1; m < L.modes.size(); ++m)
            kin += 0.5 * N * (L.modes[m].V1 * L.modes[m].V1 + L.modes[m].V2 * L.modes[m].V2);
        return (L.total_energy() - kin) / N;
    };
    auto q2 = [](PhononLattice& L) {
        const double N = double(L.lattice_size);
        return N * (L.phonons.Q_x_E1 * L.phonons.Q_x_E1 + L.phonons.Q_y_E1 * L.phonons.Q_y_E1);
    };
    // Monte Carlo: heat bath for the spins + lattice coordinate moves
    std::vector<double> e_mc, q_mc, qa_mc;
    {
        PhononLattice lat = build();
        lat.local_update = PhononLattice::LocalUpdate::HeatBath;
        lat.mc_sample_lattice = true;
        for (int s = 0; s < 2000; ++s) lat.local_sweep(T);
        for (int s = 0; s < 80000; ++s) {
            lat.local_sweep(T);
            if (s % 4 == 0) { e_mc.push_back(e_pot(lat)); q_mc.push_back(q2(lat));
                              qa_mc.push_back(double(lat.lattice_size) * 2.25 * lat.modes[1].Q1 * lat.modes[1].Q1); }
        }
    }
    // Langevin dynamics of the same model
    std::vector<double> e_ld, q_ld;
    {
        PhononLattice lat = build();
        lat.langevin_temperature = T;
        lat.alpha_gilbert = 0.5;
        Silence quiet;
        lat.integrate_langevin(0.0, 50.0, 0.01, "", 1000000000, 31);
        lat.integrate_langevin(50.0, 2050.0, 0.01, "", 25, 32, [&](double) { e_ld.push_back(e_pot(lat)); q_ld.push_back(q2(lat)); });
    }
    const MeanErr em = batch_means(e_mc, 30), el = batch_means(e_ld, 30);
    const MeanErr qm = batch_means(q_mc, 30), ql = batch_means(q_ld, 30), qa = batch_means(qa_mc, 30);
    check_stat(em.mean - el.mean, std::hypot(em.err, el.err), 0.0, "MC - Langevin potential energy per site", 5.0,
               2e-3 * std::abs(em.mean));
    check_stat(qm.mean - ql.mean, std::hypot(qm.err, ql.err), 0.0, "MC - Langevin N<|Q_E1|^2>");
    std::printf("    N<|Q_E1|^2>: MC %.4f, Langevin %.4f (decoupled value 2T/omega^2 = %.4f)\n", qm.mean, ql.mean, 2 * T / 4.0);
    // The A1 mode couples only through aA1 to the bond correlations: check it is sampled at all
    check(qa.mean > 0.3 * T && qa.mean < 3.0 * T, "A1 coordinate is sampled (N w^2 <Q^2> within a factor 3 of T)");
}

void test_mc_decoupled_mode() {
    std::printf("\n-- Lattice Monte Carlo of a decoupled mode: N w^2 <Q_a^2> = T --\n");
    const double T = 0.5, w = 2.0;
    PhononLattice lat = make_model(3, -1.0, 0.0, 0.0, 0.0, w, 0.0);
    std::vector<double> qx, vx;
    for (int s = 0; s < 40000; ++s) {
        lat.lattice_mc_sweep(T);
        const double N = double(lat.lattice_size);
        qx.push_back(N * w * w * lat.phonons.Q_x_E1 * lat.phonons.Q_x_E1);
        vx.push_back(N * lat.phonons.V_y_E1 * lat.phonons.V_y_E1);
    }
    const MeanErr mq = batch_means(qx, 30), mv = batch_means(vx, 30);
    check_stat(mq.mean, mq.err, T, "N w^2 <Q_x^2>");
    check_stat(mv.mean, mv.err, T, "N <V_y^2> (Maxwell redraw)");
}
}  // namespace

int main() {
    test_free_spins_langevin();
    test_free_spins_mc();
    test_interacting_langevin_vs_mc();
    test_lattice_fdt();
    test_mc_decoupled_mode();
    test_joint_mc_vs_langevin();
    return finish("test_phonon_dynamics");
}
