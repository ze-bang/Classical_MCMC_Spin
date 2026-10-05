// test_pt_mpi.cpp — the replica-exchange engine on 4 MPI ranks against exact
// results (run with mpiexec -n 4).
//
//  1. 1D Heisenberg AFM ring: every replica's <E>/N = -J L(beta J) and
//     c = 1 - K^2/sinh^2 K (Fisher 1964; ring corrections ~ L^N negligible)
//     within 5 sigma of the engine's own Gamma-method / jackknife errors;
//     consistent DEO bookkeeping (both partners see the same decisions, edges
//     alternate), measured round trips, f(T_min) = 1, f(T_max) = 0.
//  2. Free spins in a field with adaptive Gaussian proposals: <E>/N and <m_z>
//     against the Langevin function; sigma adapted toward the target.
//  3. MixedLattice adapter: free SU(2) spins and free 8-component SU(3)
//     vectors in fields, <E>/N and both magnetisations vs the exact
//     Langevin / Bessel-ratio results (both species travel on a swap).
//  4. Bitwise reproducibility for a fixed seed.
//  5. Ladder tuning (nrpt, katzgraber): valid ladders; nrpt equalises the
//     measured edge acceptance.
//  6. Collective failure: invalid input and rank-local output errors make every
//     rank throw (no deadlock); successful output files exist.
#include "physics_test_util.h"

#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/mc/parallel_tempering.h"

#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>

using namespace phys_test;

namespace {

int g_rank = 0, g_size = 1, g_fail = 0;

void ck(bool ok, const std::string& what) {
    if (g_rank == 0) std::printf("[%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

void ck_stat(double v, double err, double exact, const std::string& what, double n_sigma = 5.0, double floor = 0.0) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s: %.6f +- %.6f, exact %.6f (%.2f sigma)", what.c_str(), v, err, exact,
                  err > 0 ? std::abs(v - exact) / err : 0.0);
    ck(std::abs(v - exact) <= n_sigma * err + floor, buf);
}

void seed_all(unsigned long long s) {
    seed_lehman(s);
    seed_lehman_from_rank(static_cast<unsigned long long>(g_rank));
}

Lattice make_ring(size_t N) { return Lattice(chain_cell(Eigen::Matrix3d::Identity()), N, 1, 1, 1.0f); }

mc::PTResult run_ring(Lattice& lat, const std::vector<double>& T, size_t n_eq, size_t n_meas, size_t ex,
                      const std::string& dir = "") {
    return lat.parallel_tempering(T, n_eq, n_meas, /*or*/ 0, ex, /*probe*/ 2, dir, {-1}, false, MPI_COMM_WORLD);
}

void test_heisenberg_ring() {
    if (g_rank == 0) std::printf("\n== PT on the 1D Heisenberg AFM ring (N = 16, 4 replicas) ==\n");
    const std::vector<double> T = {0.5, 0.75, 1.15, 1.8};
    seed_all(2024);
    Lattice lat = make_ring(16);
    lat.init_random();
    const auto r = run_ring(lat, T, 2000, 40000, 10);
    for (size_t k = 0; k < T.size(); ++k) {
        const double K = 1.0 / T[k];
        const std::string tag = " T=" + std::to_string(T[k]);
        ck_stat(r.energy[k], r.energy_error[k], -langevin(K), "u" + tag);
        ck_stat(r.specific_heat[k], r.specific_heat_error[k], 1.0 - K * K / (std::sinh(K) * std::sinh(K)),
                "c" + tag, 5.0, 0.01);
    }
    ck(r.bookkeeping_consistent, "both partners of every edge recorded identical decisions");
    ck(r.edge_attempts.size() == 3 && r.edge_attempts[0] == r.edge_attempts[2] &&
           r.edge_attempts[0] + r.edge_attempts[1] == r.exchange_rounds,
       "DEO: even edges (0,2) and the odd edge (1) alternate every round");
    uint64_t by_label = 0;
    for (uint64_t t : r.round_trips_per_replica) by_label += t;
    ck(r.round_trips > 50 && by_label == r.round_trips,
       "round trips measured from replica labels: " + std::to_string(r.round_trips));
    ck(r.up_fraction.front() == 1.0 && r.up_fraction.back() == 0.0 && r.up_fraction[1] > r.up_fraction[2],
       "f(T_min) = 1, f(T_max) = 0, f decreasing");
    const double ratio = r.round_trip_rate / r.predicted_round_trip_rate;
    ck(ratio > 0.1 && ratio < 1.5, "measured round-trip rate " + std::to_string(r.round_trip_rate) +
                                       " vs DEO prediction " + std::to_string(r.predicted_round_trip_rate));
    for (double a : r.edge_acceptance) ck(a > 0.05 && a < 0.95, "edge acceptance " + std::to_string(a));
}

void test_free_spins_adaptive() {
    if (g_rank == 0) std::printf("\n== PT on free spins in a field, adaptive Gaussian proposals ==\n");
    const std::vector<double> T = {0.3, 0.5, 0.8, 1.3};
    const double h = 1.0;
    seed_all(77);
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(0, 0, h), 0);
    Lattice lat(uc, 6, 6, 1, 1.0f);
    lat.local_update = Lattice::LocalUpdate::Gaussian;
    lat.init_random();
    const auto r = lat.parallel_tempering(T, 3000, 30000, 0, 10, 2, "", {-1}, false, MPI_COMM_WORLD);
    for (size_t k = 0; k < T.size(); ++k) {
        ck_stat(r.energy[k], r.energy_error[k], -h * langevin(h / T[k]), "u T=" + std::to_string(T[k]));
        // sigma is steered to acceptance 0.45, unless even the widest proposal
        // (sigma at its cap, effectively uniform) is accepted more often.
        const bool capped = r.step_sizes[k] >= 10.0 - 1e-9;
        ck(r.step_sizes[k] > 0.0 && (capped ? r.acceptance[k] > 0.4 : std::abs(r.acceptance[k] - 0.45) < 0.12),
           "adapted sigma " + std::to_string(r.step_sizes[k]) + ", frozen acceptance " +
               std::to_string(r.acceptance[k]));
    }
    // Per-rank order parameter <m_z> vs L(h/T), gathered to compare all replicas.
    double mine[2] = {r.thermo.order_parameters.at(0).mean.values[2], r.thermo.order_parameters.at(0).mean.errors[2]};
    std::vector<double> all(2 * size_t(g_size));
    MPI_Allgather(mine, 2, MPI_DOUBLE, all.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);
    for (size_t k = 0; k < T.size(); ++k)
        ck_stat(all[2 * k], all[2 * k + 1], langevin(h / T[k]), "<m_z> T=" + std::to_string(T[k]));
}

void test_mixed_free_species() {
    if (g_rank == 0) std::printf("\n== PT on MixedLattice: free SU(2) and SU(3) spins in fields ==\n");
    const double h2 = 1.0, h3 = 0.8;
    const std::vector<double> T = {0.4, 0.7, 1.1, 1.8};
    const auto axes = std::vector<Eigen::Vector3d>{Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
                                                   Eigen::Vector3d(0, 0, 1)};
    UnitCell su2(3, 1, {Eigen::Vector3d::Zero()}, axes);
    UnitCell su3(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.0)}, axes);
    su2.set_field(Eigen::Vector3d(0, 0, h2), 0);
    SpinVector f3 = SpinVector::Zero(8);
    f3(2) = h3;
    su3.set_field(f3, 0);
    seed_all(606);
    MixedLattice lat(MixedUnitCell(su2, su3), 4, 4, 1, 1.0f, 1.0f);
    lat.init_random();
    const auto r = lat.parallel_tempering(T, 2000, 30000, 0, 10, 2, "", {-1}, false, true, MPI_COMM_WORLD);
    auto u3 = [&](double t) {  // <n.h_hat> on S^7: I_4(K) / I_3(K)
        const double K = h3 / t;
        return std::cyl_bessel_i(4.0, K) / std::cyl_bessel_i(3.0, K);
    };
    for (size_t k = 0; k < T.size(); ++k) {
        const double exact = 0.5 * (-h2 * langevin(h2 / T[k]) - h3 * u3(T[k]));  // equal site counts
        ck_stat(r.energy[k], r.energy_error[k], exact, "mixed <E>/N T=" + std::to_string(T[k]));
    }
    double mine[4] = {r.thermo.order_parameters.at(0).mean.values[2], r.thermo.order_parameters.at(0).mean.errors[2],
                      r.thermo.order_parameters.at(1).mean.values[2], r.thermo.order_parameters.at(1).mean.errors[2]};
    std::vector<double> all(4 * size_t(g_size));
    MPI_Allgather(mine, 4, MPI_DOUBLE, all.data(), 4, MPI_DOUBLE, MPI_COMM_WORLD);
    for (size_t k = 0; k < T.size(); ++k) {
        ck_stat(all[4 * k], all[4 * k + 1], langevin(h2 / T[k]), "<m_SU2,z> T=" + std::to_string(T[k]));
        ck_stat(all[4 * k + 2], all[4 * k + 3], u3(T[k]), "<m_SU3,3> T=" + std::to_string(T[k]));
    }
    ck(r.bookkeeping_consistent && r.round_trips > 0, "mixed: consistent exchanges and round trips");
}

void test_reproducible() {
    if (g_rank == 0) std::printf("\n== Reproducibility for a fixed seed ==\n");
    const std::vector<double> T = {0.5, 0.8, 1.2, 2.0};
    std::vector<mc::PTResult> runs;
    for (int rep = 0; rep < 2; ++rep) {
        seed_all(31337);
        Lattice lat = make_ring(12);
        lat.init_random();
        runs.push_back(run_ring(lat, T, 500, 4000, 4));
    }
    ck(runs[0].energies == runs[1].energies && runs[0].round_trips == runs[1].round_trips &&
           runs[0].edge_accepts == runs[1].edge_accepts,
       "identical seeds give bitwise identical chains, exchanges and round trips");
}

void test_ladder_tuning() {
    if (g_rank == 0) std::printf("\n== Ladder tuning on the ring (T in [0.35, 3]) ==\n");
    mc::LadderTuningOptions o;
    o.T_min = 0.35;
    o.T_max = 3.0;
    o.warmup_steps = 500;
    o.steps_per_round = 400;
    o.max_rounds = 6;
    o.exchange_every = 4;
    o.verbosity = 0;
    seed_all(4711);
    Lattice lat = make_ring(16);
    lat.init_random();
    const auto tuned = lat.tune_temperature_ladder(o, 0, false, MPI_COMM_WORLD);
    bool mono = true;
    for (size_t k = 1; k < tuned.temperatures.size(); ++k) mono = mono && tuned.temperatures[k] > tuned.temperatures[k - 1];
    ck(mono && tuned.temperatures.front() == o.T_min && tuned.temperatures.back() == o.T_max &&
           tuned.rounds_used >= 2,
       "nrpt: monotone ladder with fixed end points after " + std::to_string(tuned.rounds_used) + " rounds");
    const auto prod = run_ring(lat, tuned.temperatures, 500, 12000, 4);
    const auto geo_T = mc::generate_geometric_temperature_ladder(o.T_min, o.T_max, 4);
    const auto geo = run_ring(lat, geo_T, 500, 12000, 4);
    auto spread = [](const std::vector<double>& a) {
        const auto [lo, hi] = std::minmax_element(a.begin(), a.end());
        return *hi - *lo;
    };
    ck(spread(prod.edge_acceptance) < 0.12 && spread(prod.edge_acceptance) < spread(geo.edge_acceptance),
       "nrpt equalises edge acceptance: spread " + std::to_string(spread(prod.edge_acceptance)) +
           " (geometric ladder " + std::to_string(spread(geo.edge_acceptance)) + ")");
    o.method = mc::LadderMethod::Katzgraber;
    o.max_rounds = 3;
    const auto kt = lat.tune_temperature_ladder(o, 0, false, MPI_COMM_WORLD);
    mono = true;
    for (size_t k = 1; k < kt.temperatures.size(); ++k) mono = mono && kt.temperatures[k] > kt.temperatures[k - 1];
    ck(mono && kt.temperatures.front() == o.T_min && kt.temperatures.back() == o.T_max,
       "katzgraber: monotone ladder with fixed end points");
}

void test_no_exchange() {
    if (g_rank == 0) std::printf("\n== exchange_every = 0 (independent replicas) ==\n");
    seed_all(5);
    Lattice lat = make_ring(16);
    lat.init_random();
    const auto r = run_ring(lat, {0.6, 0.9, 1.4, 2.2}, 2000, 30000, 0);
    ck(r.exchange_rounds == 0 && r.round_trips == 0 && r.edge_attempts[1] == 0, "no exchange rounds");
    const double K = 1.0 / 0.9;
    ck_stat(r.energy[1], r.energy_error[1], -langevin(K), "u T=0.9 without exchanges");
}

template <class Exception, class F>
void expect_collective_throw(F&& f, const std::string& what) {
    int caught = 0;
    try {
        f();
    } catch (const Exception&) {
        caught = 1;
    } catch (...) {
        caught = 0;
    }
    int all = 0;
    MPI_Allreduce(&caught, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ck(all == 1, what);
}

void test_failures_and_output() {
    if (g_rank == 0) std::printf("\n== Collective validation and output errors ==\n");
    seed_all(9);
    Lattice lat = make_ring(8);
    lat.init_random();
    expect_collective_throw<std::invalid_argument>(
        [&] { run_ring(lat, {0.5, 1.0, 2.0}, 10, 10, 1); }, "wrong number of temperatures: every rank throws");
    expect_collective_throw<std::invalid_argument>(
        [&] { run_ring(lat, {0.5, 1.0, 1.0, 2.0}, 10, 10, 1); }, "non-increasing ladder: every rank throws");
    expect_collective_throw<std::invalid_argument>(
        [&] { run_ring(lat, {0.5, 1.0, -1.0, 2.0}, 10, 10, 1); }, "negative temperature: every rank throws");
    expect_collective_throw<std::invalid_argument>(
        [&] { lat.parallel_tempering({0.5, 1.0, 1.5, 2.0}, 10, 10, 0, 1, 0, "", {-1}, false); },
        "probe_rate = 0: every rank throws");
    expect_collective_throw<std::invalid_argument>(
        [&] {
            std::vector<double> T = {0.5, 1.0, 1.5, 2.0};
            if (g_rank == 3) T[1] = 1.1;  // inconsistent ladders across ranks
            run_ring(lat, T, 10, 10, 1);
        },
        "ranks given different ladders: every rank throws");
    expect_collective_throw<std::runtime_error>(
        [&] { run_ring(lat, {0.5, 1.0, 1.5, 2.0}, 10, 20, 1, "/proc/pt_test_cannot_write"); },
        "unwritable output directory: every rank throws");

    namespace fs = std::filesystem;
    const fs::path base = fs::temp_directory_path() / ("pt_mpi_test_" + std::to_string(::getpid()));
    std::string base_s = base.string();
    {   // a common directory name (rank 0's pid) for all ranks
        int len = int(base_s.size());
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        base_s.resize(size_t(len));
        MPI_Bcast(base_s.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    const std::string bad = base_s + "_rank_local_error", good = base_s + "_ok";
    if (g_rank == 0) {
        fs::remove_all(bad);
        fs::remove_all(good);
        fs::create_directories(bad);
        std::ofstream(bad + "/rank_2") << "a file where rank 2 wants its directory\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    expect_collective_throw<std::runtime_error>(
        [&] { run_ring(lat, {0.5, 1.0, 1.5, 2.0}, 10, 20, 1, bad); },
        "rank-local output failure (rank 2): every rank throws, nobody hangs");
    run_ring(lat, {0.5, 1.0, 1.5, 2.0}, 10, 20, 1, good);
    MPI_Barrier(MPI_COMM_WORLD);
    if (g_rank == 0) {
        bool ok = fs::exists(good + "/pt_summary.txt");
#ifdef HDF5_ENABLED
        ok = ok && fs::exists(good + "/parallel_tempering_aggregated.h5") &&
             fs::exists(good + "/rank_3/parallel_tempering_data.h5") &&
             !fs::exists(good + "/parallel_tempering_aggregated.h5.tmp");
#endif
        ok = ok && fs::exists(good + "/rank_1/positions.txt");
        ck(ok, "summary, aggregated and per-rank files written");
        fs::remove_all(bad);
        fs::remove_all(good);
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    if (g_size != 4) {
        if (g_rank == 0) std::printf("test_pt_mpi must run on 4 ranks (got %d)\n", g_size);
        MPI_Finalize();
        return 1;
    }
    test_heisenberg_ring();
    test_free_spins_adaptive();
    test_mixed_free_species();
    test_reproducible();
    test_ladder_tuning();
    test_no_exchange();
    test_failures_and_output();
    int total = 0;
    MPI_Allreduce(&g_fail, &total, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (g_rank == 0)
        std::printf("\ntest_pt_mpi: %s\n", total == 0 ? "all checks passed" : "check(s) FAILED");
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
