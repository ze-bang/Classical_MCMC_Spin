/**
 * ncto_pulse_scan.cpp — does a single THz pulse drive 3Q into zigzag?
 *
 * Loads a relaxed seed, sets J7, fires one Gaussian THz pulse at the polar E1
 * doublet, and integrates the coupled spin + phonon equations of motion with
 * Gilbert damping.  Reports the M-point order parameters as a function of time,
 * so the 3Q -> zigzag conversion (min/max of the three S(M) peaks dropping from
 * 1 towards 0) is visible directly.
 *
 * Usage:
 *   ncto_pulse_scan config.param seed.txt J7 E0 theta_pol t_end dt_save
 *
 * Everything else (omega_E1, gamma_E1, pulse shape, couplings) comes from the
 * config.  Output: CSV on stdout, one row per save.
 */
#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "spin_solver_runners.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {

void rk4_step(PhononLattice& L, PhononLattice::ODEState& x, double t, double dt,
              PhononLattice::ODEState& k1, PhononLattice::ODEState& k2,
              PhononLattice::ODEState& k3, PhononLattice::ODEState& k4,
              PhononLattice::ODEState& tmp) {
    const size_t n = x.size();
    L.ode_system(x, k1, t);
    for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + 0.5 * dt * k1[i];
    L.ode_system(tmp, k2, t + 0.5 * dt);
    for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + 0.5 * dt * k2[i];
    L.ode_system(tmp, k3, t + 0.5 * dt);
    for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + dt * k3[i];
    L.ode_system(tmp, k4, t + dt);
    for (size_t i = 0; i < n; ++i)
        x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        cerr << "Usage: " << argv[0]
             << " config.param seed.txt J7 E0 theta_pol t_end dt_save [j7_disorder.txt]\n";
        return 1;
    }
    const string config_file = argv[1];
    const string seed_file = argv[2];
    const double j7 = atof(argv[3]);
    const double E0 = atof(argv[4]);
    const double theta = atof(argv[5]);
    const double t_end = atof(argv[6]);
    const double dt_save = atof(argv[7]);

    std::ostream csv(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());

    SpinConfig config = SpinConfig::from_file(config_file);
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, config.lattice_size[0], config.lattice_size[1],
                          config.lattice_size[2], config.spin_length);

    SpinPhononCouplingParams sp;
    PhononParams ph;
    DriveParams dr;
    TimeDependentSpinPhononParams td;
    build_phonon_params(config, sp, ph, dr, td);
    sp.J7 = j7;
    dr.E0_1 = E0;
    dr.theta_1 = theta;
    dr.E0_2 = 0.0;                       // single pulse
    lattice.set_parameters(sp, ph, dr);
    lattice.set_time_dependent_spin_phonon(td);
    lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.05);

    Eigen::Vector3d B;
    B << config.field_strength * config.field_direction[0],
         config.field_strength * config.field_direction[1],
         config.field_strength * config.field_direction[2];
    lattice.set_field(B);

    if (argc > 8 && std::string(argv[8]) != "none")
        lattice.apply_plaquette_j7_disorder_from_file(argv[8]);

    lattice.load_spin_config(seed_file);
    lattice.phonons.Q_x_E1 = 0.0;
    lattice.phonons.Q_y_E1 = 0.0;
    lattice.phonons.V_x_E1 = 0.0;
    lattice.phonons.V_y_E1 = 0.0;

    const Eigen::Vector3d M1(M_PI, M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M2(0.0, 2.0 * M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M3(-M_PI, M_PI / std::sqrt(3.0), 0.0);

    const double dt = config.md_timestep > 0.0 ? config.md_timestep : 0.005;
    const size_t n_sub = std::max<size_t>(1, static_cast<size_t>(std::lround(dt_save / dt)));
    const size_t n_out = static_cast<size_t>(std::lround(t_end / (n_sub * dt)));

    cerr << "seed=" << seed_file << " N=" << lattice.lattice_size
         << " J7=" << j7 << " E0=" << E0 << " theta=" << theta
         << " dt=" << dt << " t_end=" << t_end << " alpha=" << lattice.alpha_gilbert << "\n";

    csv << "t_ps,E_per_site,Qabs,S_M1,S_M2,S_M3,min_over_max,E_phonon_per_site\n";
    csv << scientific << setprecision(10);

    PhononLattice::ODEState x = lattice.spins_to_state();
    const size_t n = x.size();
    PhononLattice::ODEState k1(n), k2(n), k3(n), k4(n), tmp(n);
    const double HBAR_PS = 0.658212;
    double t = 0.0;

    auto emit = [&]() {
        lattice.state_to_spins(x);
        const double N = double(lattice.lattice_size);
        const double s1 = lattice.structure_factor(M1);
        const double s2 = lattice.structure_factor(M2);
        const double s3 = lattice.structure_factor(M3);
        const double smax = std::max({s1, s2, s3});
        const double smin = std::min({s1, s2, s3});
        csv << t * HBAR_PS << "," << lattice.total_energy() / N << ","
             << lattice.E1_amplitude() << ","
             << s1 << "," << s2 << "," << s3 << ","
             << (smax > 0.0 ? smin / smax : 0.0) << ","
             << lattice.phonon_energy() / N << "\n";
    };

    emit();
    for (size_t o = 0; o < n_out; ++o) {
        for (size_t s = 0; s < n_sub; ++s) {
            rk4_step(lattice, x, t, dt, k1, k2, k3, k4, tmp);
            t += dt;
        }
        // renormalise the spins: RK4 conserves |S| only to O(dt^5) per step
        for (size_t i = 0; i < lattice.lattice_size; ++i) {
            const size_t idx = i * lattice.spin_dim;
            const double nrm = std::sqrt(x[idx] * x[idx] + x[idx + 1] * x[idx + 1]
                                         + x[idx + 2] * x[idx + 2]);
            const double f = lattice.spin_length / nrm;
            x[idx] *= f; x[idx + 1] *= f; x[idx + 2] *= f;
        }
        emit();
        csv.flush();
    }
    return 0;
}
