/**
 * ncto_pulse_thermal.cpp — THz pulse at FINITE temperature.
 *
 * The T = 0 scans (ncto_pulse_scan) found no 3Q -> zigzag conversion anywhere:
 * not at 100x the measured fluence, not when zigzag is already the ground state
 * by 22 ueV/site, and not with quenched J7 disorder up to sigma = 0.20.  That is
 * expected: with deterministic Gilbert damping and no noise, a trajectory cannot
 * cross a barrier at all.  Nucleation is thermally activated by construction, so
 * the whole picture (nucleation, Avrami kinetics, an Arrhenius lifetime) requires
 * finite T.
 *
 * This driver runs the same pulse through integrate_langevin(), whose docstring
 * states it is intended for exactly this ("sufficient for observing spin-state
 * hopping 3Q <-> ZZ at finite T"), and reports the M-point order parameters.
 *
 * Usage:
 *   ncto_pulse_thermal config seed.txt J7 E0 theta t_end dt_save T [disorder|none] [rngseed]
 *
 * T is k_B T in meV (6 K = 0.517).
 */
#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/phonon_config.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 9) {
        cerr << "Usage: " << argv[0]
             << " config seed.txt J7 E0 theta t_end dt_save T [disorder|none] [rngseed]\n";
        return 1;
    }
    const string config_file = argv[1];
    const string seed_file   = argv[2];
    const double j7      = atof(argv[3]);
    const double E0      = atof(argv[4]);
    const double theta   = atof(argv[5]);
    const double t_end   = atof(argv[6]);
    const double dt_save = atof(argv[7]);
    const double Temp    = atof(argv[8]);
    const string dis     = (argc > 9) ? argv[9] : "none";
    const uint64_t rseed = (argc > 10) ? strtoull(argv[10], nullptr, 10) : 12345ULL;

    std::ostream csv(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());

    SpinConfig config = SpinConfig::from_file(config_file);
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, config.lattice_size[0], config.lattice_size[1],
                          config.lattice_size[2], config.spin_length);

    SpinPhononCouplingParams sp; PhononParams ph; DriveParams dr;
    TimeDependentSpinPhononParams td;
    build_phonon_params(config, sp, ph, dr, td);
    sp.J7 = j7; dr.E0_1 = E0; dr.theta_1 = theta; dr.E0_2 = 0.0;
    lattice.set_parameters(sp, ph, dr);
    lattice.set_time_dependent_spin_phonon(td);
    lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.05);
    lattice.langevin_temperature = Temp;

    Eigen::Vector3d B;
    B << config.field_strength * config.field_direction[0],
         config.field_strength * config.field_direction[1],
         config.field_strength * config.field_direction[2];
    lattice.set_field(B);

    if (dis != "none") lattice.apply_plaquette_j7_disorder_from_file(dis);

    lattice.load_spin_config(seed_file);
    lattice.phonons.Q_x_E1 = 0.0; lattice.phonons.Q_y_E1 = 0.0;
    lattice.phonons.V_x_E1 = 0.0; lattice.phonons.V_y_E1 = 0.0;

    const Eigen::Vector3d M1(M_PI, M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M2(0.0, 2.0 * M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M3(-M_PI, M_PI / std::sqrt(3.0), 0.0);

    const double dt = config.md_timestep > 0.0 ? config.md_timestep : 0.005;
    const int n_out = static_cast<int>(std::lround(t_end / dt_save));
    const double HBAR_PS = 0.658212;

    cerr << "N=" << lattice.lattice_size << " J7=" << j7 << " E0=" << E0
         << " T=" << Temp << " (" << Temp / 0.086173 << " K) disorder=" << dis << "\n";

    csv << "t_ps,E_per_site,Qabs,S_M1,S_M2,S_M3,min_over_max,E_phonon_per_site\n";
    csv << scientific << setprecision(10);

    auto emit = [&](double t) {
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
        csv.flush();
    };

    emit(0.0);
    for (int o = 0; o < n_out; ++o) {
        const double t0 = o * dt_save, t1 = (o + 1) * dt_save;
        // fresh, reproducible noise stream per chunk (integrate_langevin turns
        // use_langevin_noise off when it returns)
        lattice.integrate_langevin(t0, t1, dt, "", 1000000000, rseed + uint64_t(o) * 7919ULL);
        emit(t1);
    }
    return 0;
}
