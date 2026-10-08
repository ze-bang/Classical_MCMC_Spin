/**
 * bench_phonon.cpp — micro-benchmark of the PhononLattice (NCTO spin-phonon)
 * hot paths: the coupled ODE right-hand side, the Metropolis and
 * overrelaxation sweeps, the single-site local field and the total energy.
 *
 * The model is the default NCTO operating point (Krüger fit, ring exchange
 * J7 at the 3Q/zigzag degeneracy, linear E1 channel on) on an L x L honeycomb.
 *
 * Usage:
 *   bench_phonon [--L=24] [--rhs=400] [--sweeps=40] [--repeats=3] [--threads=1]
 */

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

template <class F>
double best_of(int repeats, F&& f) {
    double best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        const double t0 = now_sec();
        f();
        best = std::min(best, now_sec() - t0);
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    int L = 24, repeats = 3, threads = 1;
    long rhs = 400, sweeps = 40;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto eat = [&](const std::string& key, auto& dst) {
            if (s.rfind(key + "=", 0) != 0) return false;
            std::stringstream(s.substr(key.size() + 1)) >> dst;
            return true;
        };
        if (!(eat("--L", L) || eat("--rhs", rhs) || eat("--sweeps", sweeps) ||
              eat("--repeats", repeats) || eat("--threads", threads))) {
            std::cerr << "usage: bench_phonon [--L=24] [--rhs=400] [--sweeps=40] "
                         "[--repeats=3] [--threads=1]\n";
            return 1;
        }
    }
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    std::streambuf* saved = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());          // the constructors are chatty

    SpinConfig config;
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lat(uc, L, L, 1, 1.0f);
    SpinPhononCouplingParams sp;            // NCTO defaults: Krüger fit + J7 ring exchange
    sp.lambda_E1_J_1 = -3.447;              // linear E1 channel at the operating-point estimate
    sp.lambda_E1_K_1 = 40.0;
    sp.lambda_E1_Gamma_1 = -15.56;
    sp.lambda_E1_Gammap_1 = 14.91;
    PhononParams ph;
    DriveParams dr;
    lat.set_parameters(sp, ph, dr);
    lat.init_random();
    lat.phonons.Q_x_E1 = 0.01;
    lat.phonons.Q_y_E1 = -0.004;
    std::cout.rdbuf(saved);

    const size_t N = lat.lattice_size;
    PhononLattice::ODEState x = lat.spins_to_state(), dxdt(x.size());

    const double t_rhs = best_of(repeats, [&] {
        for (long k = 0; k < rhs; ++k) lat.ode_system(x, dxdt, 0.0);
    });
    const double t_met = best_of(repeats, [&] {
        for (long k = 0; k < sweeps; ++k) lat.metropolis(1.0);
    });
    const double t_or = best_of(repeats, [&] {
        for (long k = 0; k < sweeps; ++k) lat.overrelaxation();
    });
    double sink_d = 0.0;
    const double t_field = best_of(repeats, [&] {
        for (long k = 0; k < sweeps; ++k)
            for (size_t i = 0; i < N; ++i) sink_d += lat.get_local_field(i)(0);
    });
    const double t_energy = best_of(repeats, [&] {
        for (long k = 0; k < sweeps; ++k) sink_d += lat.total_energy();
    });

    std::cout << std::fixed << std::setprecision(3)
              << "bench_phonon  L=" << L << "  N=" << N << "  threads=" << threads << "\n"
              << "  ode_system       : " << 1e6 * t_rhs / rhs << " us/call  ("
              << 1e9 * t_rhs / (rhs * double(N)) << " ns/site)\n"
              << "  metropolis sweep : " << 1e3 * t_met / sweeps << " ms/sweep  ("
              << 1e9 * t_met / (sweeps * double(N)) << " ns/proposal)\n"
              << "  overrelaxation   : " << 1e3 * t_or / sweeps << " ms/sweep\n"
              << "  get_local_field  : " << 1e9 * t_field / (sweeps * double(N)) << " ns/call\n"
              << "  total_energy     : " << 1e6 * t_energy / sweeps << " us/call\n"
              << "  (checksum " << std::setprecision(6) << sink_d << ")\n";
    return 0;
}
