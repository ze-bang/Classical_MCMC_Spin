// test_population_annealing.cpp — the population-annealing engine against
// exact results (runs on 1 rank; the CTest entry *_mpi3 runs it on 3 ranks).
//
//  1. 1D Heisenberg ferromagnet ring (N = 32): at every temperature of the
//     schedule ln Z(beta)/N - ln Z(0)/N = ln(sinh K / K), <E>/N = -L(K) and
//     c = 1 - K^2/sinh^2 K with K = beta J (Fisher 1964; ring corrections
//     ~ 3 L(K)^N are below 1e-8 here), within 5 sigma of the engine's family
//     jackknife errors (free energy: within 5 sigma of the spread of
//     independent runs).
//  2. The result does not depend on the MPI / thread layout: the same seed
//     gives bitwise-identical statistics with 1 or 2 worker threads and, under
//     mpiexec, on all ranks vs the whole population on rank 0 alone.
//  3. Adaptive schedule: every reweighting keeps ESS >= target, the run ends
//     at T_end, and the energies agree with the exact curve.
//  4. Grouped jackknife: equal groups reproduce the textbook error of a mean.
#include "physics_test_util.h"

#include "classical_spin/mc/population_annealing.h"

#include <mpi.h>
#include <filesystem>
#include <memory>

using namespace phys_test;

namespace {

int g_rank = 0, g_size = 1, g_fail = 0;

void ck(bool ok, const std::string& what) {
    if (g_rank == 0) std::printf("[%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

void ck_stat(double v, double err, double exact, const std::string& what, double n_sigma = 5.0) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s: %.6f +- %.6f, exact %.6f (%.2f sigma)", what.c_str(), v, err, exact,
                  err > 0 ? std::abs(v - exact) / err : 0.0);
    ck(std::abs(v - exact) <= n_sigma * err, buf);
}

double langevin(double K) { return 1.0 / std::tanh(K) - 1.0 / K; }

/// Workers (one per thread) on independent copies of a ferromagnetic ring.
struct RingWorkers {
    std::vector<std::unique_ptr<Lattice>> lattices;
    std::vector<std::unique_ptr<mc::LatticePopulationWorker<Lattice>>> workers;
    std::vector<mc::LatticePopulationWorker<Lattice>*> ptrs;
    RingWorkers(size_t N, int n_threads, bool gaussian) {
        const Lattice proto(chain_cell(-Eigen::Matrix3d::Identity()), N, 1, 1, 1.0f);
        for (int t = 0; t < n_threads; ++t) {
            lattices.push_back(std::make_unique<Lattice>(proto));
            if (gaussian) lattices.back()->local_update = Lattice::LocalUpdate::Gaussian;
            workers.push_back(std::make_unique<mc::LatticePopulationWorker<Lattice>>(*lattices.back(), 0, gaussian));
            ptrs.push_back(workers.back().get());
        }
    }
};

mc::PAResult run_ring(size_t R, int n_threads, mc::PAOptions o, MPI_Comm comm, bool gaussian = false) {
    o.population = R;
    o.verbosity = 0;
    RingWorkers w(32, n_threads, gaussian);
    return mc::run_population_annealing(w.ptrs, o, comm);
}

mc::PAOptions ring_options() {
    mc::PAOptions o;
    o.T_start = 5.0;
    o.T_end = 0.4;
    o.n_temperatures = 24;
    o.sweeps = 10;
    return o;
}

void test_ring_exact() {
    if (g_rank == 0) std::printf("\n== Population annealing on the 1D Heisenberg ring (N = 32) ==\n");
    const mc::PAOptions o = ring_options();
    // Independent runs for the free-energy spread (the estimator has no
    // per-run error bar); the first one is also checked point by point.
    const int n_runs = 4;
    std::vector<mc::PAResult> runs;
    for (int r = 0; r < n_runs; ++r) {
        seed_lehman(1000 + r);
        runs.push_back(run_ring(1500, 2, o, MPI_COMM_WORLD));
    }
    const mc::PAResult& a = runs[0];
    ck(a.steps.size() == o.n_temperatures && std::abs(a.steps.back().T - o.T_end) < 1e-12,
       "schedule visits n_temperatures points ending at T_end");
    for (size_t i : {size_t(0), size_t(8), size_t(16), a.steps.size() - 1}) {
        const mc::PAStep& s = a.steps[i];
        const double K = s.beta;
        const std::string tag = " T=" + std::to_string(s.T);
        ck_stat(s.energy, s.energy_error, -langevin(K), "u" + tag);
        ck_stat(s.specific_heat, s.specific_heat_error, 1.0 - K * K / std::pow(std::sinh(K), 2), "c" + tag);
        // m^2 per site of a ring: <m^2> = (1/N)(1 + L)/(1 - L) up to L^N corrections.
        const double L = langevin(K), N = 32.0;
        ck_stat(s.obs_mean[1], s.obs_error[1], (1.0 + L) / (1.0 - L) / N, "<m^2>" + tag);
    }
    for (size_t i : {size_t(8), a.steps.size() - 1}) {
        double m = 0.0, m2 = 0.0;
        for (const auto& r : runs) { m += r.steps[i].ln_Z; m2 += r.steps[i].ln_Z * r.steps[i].ln_Z; }
        m /= n_runs;
        const double sd = std::sqrt(std::max(0.0, m2 / n_runs - m * m) * n_runs / (n_runs - 1));
        const double K = a.steps[i].beta;
        // Error of the mean of the runs, floored at 1e-4 (four runs give a rough spread).
        ck_stat(m, std::max(sd / std::sqrt(double(n_runs)), 1e-4), std::log(std::sinh(K) / K),
                "ln Z/N T=" + std::to_string(a.steps[i].T));
    }
    const mc::PAStep& f = a.steps.back();
    ck(f.rho_t >= 1.0 && f.rho_t <= 1500.0 && f.n_families >= 1 && f.ess_fraction > 0.0 && f.ess_fraction <= 1.0,
       "diagnostics in range: rho_t = " + std::to_string(f.rho_t) + ", families = " +
           std::to_string(f.n_families) + ", ESS = " + std::to_string(f.ess_fraction));
    ck(a.best_energy / 32.0 <= f.energy_min + 1e-12 && a.best_state.size() == 3 * 32,
       "best replica reported with its state");
}

bool same_steps(const mc::PAResult& x, const mc::PAResult& y) {
    if (x.steps.size() != y.steps.size()) return false;
    for (size_t i = 0; i < x.steps.size(); ++i) {
        const auto &a = x.steps[i], &b = y.steps[i];
        if (a.energy != b.energy || a.specific_heat != b.specific_heat || a.ln_Z != b.ln_Z ||
            a.rho_t != b.rho_t || a.obs_mean != b.obs_mean || a.sigma != b.sigma)
            return false;
    }
    return x.best_state == y.best_state;
}

void test_layout_independence() {
    if (g_rank == 0) std::printf("\n== Reproducibility across MPI / thread layouts ==\n");
    mc::PAOptions o = ring_options();
    o.n_temperatures = 8;
    seed_lehman(77);
    const auto one = run_ring(300, 1, o, MPI_COMM_WORLD, true);
    seed_lehman(77);
    const auto two = run_ring(300, 2, o, MPI_COMM_WORLD, true);
    ck(same_steps(one, two), "1 vs 2 worker threads: identical results (adaptive Gaussian moves)");
    // A second run in the same process is reproducible too: the engine leaves
    // every thread-local stream in a deterministic state.
    auto second_of_two = [&] {
        seed_lehman(91);
        run_ring(200, 2, o, MPI_COMM_WORLD, true);
        return run_ring(200, 2, o, MPI_COMM_WORLD, true);
    };
    const auto r1 = second_of_two();
    const auto r2 = second_of_two();
    ck(same_steps(r1, r2), "consecutive runs in one process are reproducible");
    if (g_size > 1) {
        // Whole population on rank 0 alone, same seed.
        seed_lehman(77);
        mc::PAResult solo;
        if (g_rank == 0) solo = run_ring(300, 2, o, MPI_COMM_SELF, true);
        int ok = (g_rank == 0) ? int(same_steps(one, solo)) : 1;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ck(ok == 1, std::to_string(g_size) + " ranks vs 1 rank: identical results");
    }
}

void test_adaptive() {
    if (g_rank == 0) std::printf("\n== Adaptive (ESS) schedule ==\n");
    mc::PAOptions o = ring_options();
    o.schedule = mc::PASchedule::Adaptive;
    o.target_ess = 0.7;
    seed_lehman(5);
    const auto a = run_ring(1000, 2, o, MPI_COMM_WORLD);
    double ess_min = 1.0;
    for (const auto& s : a.steps) ess_min = std::min(ess_min, s.ess_fraction);
    ck(ess_min >= 0.7 - 1e-9, "every step keeps ESS >= 0.7 (min " + std::to_string(ess_min) + ", " +
                                  std::to_string(a.steps.size()) + " steps)");
    ck(std::abs(a.steps.back().T - o.T_end) < 1e-12, "adaptive schedule ends at T_end");
    const auto& f = a.steps.back();
    ck_stat(f.energy, f.energy_error, -langevin(f.beta), "u at T_end");
}

void test_jackknife() {
    if (g_rank == 0) std::printf("\n== Grouped jackknife ==\n");
    // Equal groups: error of the mean = sqrt(sum (ybar_g - ybar)^2 / (G (G - 1))).
    std::vector<double> n = {4, 4, 4, 4, 4};
    std::vector<double> means = {1.0, 2.5, 0.5, 3.0, 2.0};
    std::vector<std::vector<double>> sums;
    for (double m : means) sums.push_back({4.0 * m});
    const auto [est, err] = mc::detail::grouped_jackknife(n, sums, [](double c, const std::vector<double>& s) {
        return s[0] / c;
    });
    double ybar = 0.0, ss = 0.0;
    for (double m : means) ybar += m / 5.0;
    for (double m : means) ss += (m - ybar) * (m - ybar);
    ck(std::abs(est - ybar) < 1e-14 && std::abs(err - std::sqrt(ss / 20.0)) < 1e-14,
       "equal groups reproduce the textbook error of a mean");
}

void test_validation() {
    if (g_rank == 0) std::printf("\n== Input validation ==\n");
    mc::PAOptions o = ring_options();
    bool threw = false;
    try {
        o.T_end = -1.0;
        run_ring(100, 1, o, MPI_COMM_WORLD);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ck(threw, "negative T_end rejected on every rank");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    test_jackknife();
    test_ring_exact();
    test_layout_independence();
    test_adaptive();
    test_validation();
    int fails = g_fail, total = 0;
    MPI_Allreduce(&fails, &total, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (g_rank == 0)
        std::printf("\ntest_population_annealing: %s\n", total ? "check(s) FAILED" : "all checks passed");
    MPI_Finalize();
    return total ? 1 : 0;
}
