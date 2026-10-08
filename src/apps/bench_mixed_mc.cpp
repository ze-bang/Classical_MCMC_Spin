/**
 * bench_mixed_mc.cpp — single-thread MixedLattice Monte Carlo sweep timings
 * on the production TmFeO3 model (Fe SU(2) + Tm SU(3)).
 *
 * The model has every coupling class of the production Hamiltonian: Fe-Fe
 * exchange with DM, single-ion anisotropy (Ka, Kc), the Tm crystal field,
 * the Fe-Tm exchange K^- (mixed bilinear) and the Fe-Fe-Tm vertex W (mixed
 * trilinear, on-site in the Fe index). Reported per kernel: wall time per
 * sweep and nanoseconds per site update (N = N_Fe + N_Tm sites per sweep),
 * best of `--repeats` timings after a warm-up of the same length.
 *
 *   bench_mixed_mc [--L=4] [--sweeps=400] [--repeats=5] [--T=1.0] [--csv]
 *
 * Run with OMP_NUM_THREADS=1 (the serial kernels are timed).
 */

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace {

struct Args {
    int L = 4;
    long sweeps = 400;
    int repeats = 5;
    double T = 1.0;
    bool csv = false;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto val = [&](const char* key) -> const char* {
            const std::string k = std::string(key) + "=";
            return s.rfind(k, 0) == 0 ? argv[i] + k.size() : nullptr;
        };
        if (const char* v = val("--L")) a.L = std::stoi(v);
        else if (const char* v2 = val("--sweeps")) a.sweeps = std::stol(v2);
        else if (const char* v3 = val("--repeats")) a.repeats = std::stoi(v3);
        else if (const char* v4 = val("--T")) a.T = std::stod(v4);
        else if (s == "--csv") a.csv = true;
        else {
            std::cerr << "usage: bench_mixed_mc [--L=4] [--sweeps=400] [--repeats=5] [--T=1.0] [--csv]\n";
            std::exit(2);
        }
    }
    return a;
}

SpinConfig tmfeo3_config() {
    SpinConfig cfg;
    cfg.field_strength = 0.0;
    cfg.set_param("J1ab", 4.74); cfg.set_param("J1c", 5.15);
    cfg.set_param("J2ab", 0.15); cfg.set_param("J2c", 0.30);
    cfg.set_param("Ka", -0.16221); cfg.set_param("Kc", -0.18318);
    cfg.set_param("D1", 0.12);
    cfg.set_param("e1", 0.97); cfg.set_param("e2", 3.97);
    cfg.set_param("Kminus_2x", 0.12); cfg.set_param("Kminus_5y", -0.08); cfg.set_param("Kminus_7z", 0.05);
    cfg.set_param("W3_xx", 0.05); cfg.set_param("W8_zz", -0.04); cfg.set_param("W1_xy", 0.03);
    cfg.set_param("W4_xz", 0.02); cfg.set_param("W6_yz", -0.025);
    return cfg;
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    seed_lehman(20261008);

    // The constructor is chatty; keep the benchmark output clean.
    std::stringstream sink;
    std::streambuf* old = std::cout.rdbuf(sink.rdbuf());
    MixedLattice lat(build_tmfeo3(tmfeo3_config()), a.L, a.L, a.L, 1.0f, 1.0f);
    lat.init_random();
    std::cout.rdbuf(old);

    const double N = double(lat.lattice_size_SU2 + lat.lattice_size_SU3);
    const double T = a.T;
    struct Kernel { std::string name; std::function<void()> sweep; };
    const std::vector<Kernel> kernels = {
        {"metropolis_uniform",   [&] { lat.metropolis(T, false, 0.0); }},
        {"metropolis_gaussian",  [&] { lat.metropolis(T, true, 0.5); }},
        {"interleaved_gaussian", [&] { lat.metropolis_interleaved(T, true, 0.5); }},
        {"overrelaxation",       [&] { lat.overrelaxation(); }},
        {"deterministic_sweep",  [&] { lat.deterministic_sweep(); }},
        {"total_energy",         [&] { volatile double e = lat.total_energy(); (void)e; }},
    };

    if (a.csv) std::printf("kernel,L,N,sweeps,ms_per_sweep,ns_per_site\n");
    else std::printf("TmFeO3 L=%d (N=%.0f sites), T=%g, best of %d x %ld sweeps, 1 thread\n"
                     "%-22s %14s %14s\n", a.L, N, T, a.repeats, a.sweeps, "kernel", "ms/sweep", "ns/site");
    for (const auto& k : kernels) {
        for (long s = 0; s < a.sweeps; ++s) k.sweep();   // warm-up
        double best = 1e300;
        for (int r = 0; r < a.repeats; ++r) {
            const double t0 = now();
            for (long s = 0; s < a.sweeps; ++s) k.sweep();
            best = std::min(best, now() - t0);
        }
        const double per_sweep = best / double(a.sweeps);
        if (a.csv) std::printf("%s,%d,%.0f,%ld,%.6f,%.2f\n", k.name.c_str(), a.L, N, a.sweeps,
                               1e3 * per_sweep, 1e9 * per_sweep / N);
        else std::printf("%-22s %14.4f %14.1f\n", k.name.c_str(), 1e3 * per_sweep, 1e9 * per_sweep / N);
    }
    return 0;
}
