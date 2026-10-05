/**
 * ncto_operating_point.cpp — locate the 3Q / zigzag degeneracy and measure the
 * static magnetoelastic bias, using the audited PhononLattice Hamiltonian.
 *
 * For every (J7, Q) grid point it reloads a seed, freezes the E1 coordinate at
 * Q = (Qx, Qy), relaxes the SPINS ONLY by T = 0 deterministic sweeps, and reports
 * energies and M-point order parameters.  Freezing Q is deliberate: it isolates
 * the static bias from any dynamical response.
 *
 * Usage:
 *   ncto_operating_point config.param seed.txt n_sweeps j7_start j7_end j7_step [Qx Qy]
 *
 * Output: one CSV line per grid point on stdout (header first).
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
    if (argc < 7) {
        cerr << "Usage: " << argv[0]
             << " config.param seed.txt n_sweeps j7_start j7_end j7_step [Qx Qy]\n";
        return 1;
    }
    const string config_file = argv[1];
    const string seed_file = argv[2];
    const size_t n_sweeps = static_cast<size_t>(atol(argv[3]));
    const double j7_start = atof(argv[4]);
    const double j7_end = atof(argv[5]);
    const double j7_step = atof(argv[6]);
    const double Qx = (argc > 7) ? atof(argv[7]) : 0.0;
    const double Qy = (argc > 8) ? atof(argv[8]) : 0.0;
    // argv[9] = 1 : after the T=0 spin quench, relax the E1 coordinate and the spins
    // JOINTLY (spontaneous exchange-striction / magnetostriction).  Zigzag is nematic
    // so it exerts a net force on the polar doublet; triple-q is C3 symmetric and does
    // not.  The Q = 0 operating point is therefore NOT the equilibrium one.
    const bool joint = (argc > 9) && (atoi(argv[9]) != 0);

    // The lattice/parameter setters chatter on stdout, which would interleave with
    // the CSV.  Divert cout to cerr for the whole run; only the explicit CSV writes
    // below go to the real stdout via `csv`.
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
    dr.E0_1 = 0.0;
    dr.E0_2 = 0.0;                      // no drive: this is a static evaluation
    lattice.set_parameters(sp, ph, dr);
    lattice.set_time_dependent_spin_phonon(td);

    Eigen::Vector3d B;
    B << config.field_strength * config.field_direction[0],
         config.field_strength * config.field_direction[1],
         config.field_strength * config.field_direction[2];
    lattice.set_field(B);

    // M points of the honeycomb (same convention as analyze_phonon_m_order)
    const Eigen::Vector3d M1(M_PI, M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M2(0.0, 2.0 * M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M3(-M_PI, M_PI / std::sqrt(3.0), 0.0);

    cerr << "seed=" << seed_file << "  N=" << lattice.lattice_size
         << "  sweeps=" << n_sweeps << "  Q=(" << Qx << "," << Qy << ")\n";

    csv << "j7,qx,qy,E_per_site,E_spin_per_site,E_ring_per_site,E_me_per_site,"
         << "S_M1,S_M2,S_M3,min_over_max,converged\n";
    csv << scientific << setprecision(12);

    const int nsteps = (j7_step == 0.0)
                     ? 1
                     : static_cast<int>(std::floor((j7_end - j7_start) / j7_step + 1e-9)) + 1;

    for (int s = 0; s < nsteps; ++s) {
        const double j7 = j7_start + s * j7_step;

        // fresh seed every point: the grid points must be independent
        lattice.load_spin_config(seed_file);
        sp.J7 = j7;
        lattice.set_parameters(sp, ph, dr);
        lattice.phonons.Q_x_E1 = Qx;
        lattice.phonons.Q_y_E1 = Qy;
        lattice.phonons.V_x_E1 = 0.0;
        lattice.phonons.V_y_E1 = 0.0;

        // T = 0 quench of the spins at frozen Q; halve-and-compare for convergence
        lattice.deterministic_sweep(n_sweeps / 2);
        const double E_half = lattice.total_energy();
        lattice.deterministic_sweep(n_sweeps - n_sweeps / 2);
        const double E_quench = lattice.total_energy();
        // convergence is judged on the QUENCH, before any joint relaxation
        const double drift = std::abs(E_quench - E_half) / double(lattice.lattice_size);
        if (joint) lattice.relax_joint(1e-12, 400, 20, false);
        const double E_full = lattice.total_energy();

        const double N = double(lattice.lattice_size);
        const double E_ring = lattice.ring_exchange_energy();
        const double E_me = lattice.spin_phonon_energy();
        const double E_spin = lattice.spin_energy();
        const double s1 = lattice.structure_factor(M1);
        const double s2 = lattice.structure_factor(M2);
        const double s3 = lattice.structure_factor(M3);
        const double smax = std::max({s1, s2, s3});
        const double smin = std::min({s1, s2, s3});

        csv << j7 << "," << lattice.phonons.Q_x_E1 << "," << lattice.phonons.Q_y_E1 << ","
             << E_full / N << "," << E_spin / N << "," << E_ring / N << "," << E_me / N << ","
             << s1 << "," << s2 << "," << s3 << ","
             << (smax > 0.0 ? smin / smax : 0.0) << ","
             << (drift < 1e-10 ? 1 : 0) << "\n";
        csv.flush();
        if (drift >= 1e-10)
            cerr << "  WARNING J7=" << j7 << " not converged: per-site drift over the last "
                 << (n_sweeps - n_sweeps / 2) << " sweeps = " << drift << "\n";
    }
    return 0;
}
