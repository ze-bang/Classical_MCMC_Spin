/**
 * pt_runner_common.h — configuration glue shared by the parallel-tempering
 * runners (Lattice, MixedLattice): ladder preparation (geometric or tuned) and
 * the run lengths, so the runners only differ in the model they drive.
 */
#pragma once

#include "classical_spin/core/spin_config.h"
#include "classical_spin/mc/parallel_tempering.h"

#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pt_runner {

/**
 * Communicator of a PT run on `size` replicas. Callers that run independent
 * single-replica jobs pass size = 1 with the default MPI_COMM_WORLD (e.g. the
 * parameter sweep); they get MPI_COMM_SELF so the collective engine never
 * waits for ranks working on other jobs. Any other mismatch is an error.
 */
inline MPI_Comm replica_comm(MPI_Comm comm, int size) {
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size == size) return comm;
    if (size == 1) return MPI_COMM_SELF;
    throw std::invalid_argument("parallel tempering: " + std::to_string(size) +
                                " replicas requested on a communicator of " + std::to_string(comm_size) +
                                " ranks");
}

/// Equilibration / measurement MC steps (0 in the config = annealing_steps).
inline size_t equilibration_steps(const SpinConfig& c) {
    return c.pt_equilibration_steps ? c.pt_equilibration_steps : c.annealing_steps;
}
inline size_t measurement_steps(const SpinConfig& c) {
    return c.pt_measurement_steps ? c.pt_measurement_steps : c.annealing_steps;
}

/// Tuned ladder, cold -> hot, with the per-edge diagnostics of the last round.
inline void write_tuned_ladder(const std::string& path, const mc::LadderTuningResult& r,
                               const std::string& method) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "Warning: cannot write " << path << std::endl;
        return;
    }
    f << "# Tuned temperature ladder (" << method << "), cold -> hot\n";
    f << "# converged: " << (r.converged ? "yes" : "no") << " after " << r.rounds_used
      << " rounds (last relative change " << r.last_change << ")\n";
    f << "# barrier Lambda = " << r.barrier << ", recommended replicas ~ " << r.recommended_replicas
      << ", predicted round trips per exchange round = " << r.predicted_round_trip_rate << "\n";
    f << "# acceptance / f_up / tau_int were measured in the last round, on the ladder before its final update\n";
    f << "# k  temperature  acceptance(k,k+1)  f_up  tau_int_E[steps]\n";
    for (size_t k = 0; k < r.temperatures.size(); ++k) {
        f << k << "  " << std::scientific << std::setprecision(12) << r.temperatures[k] << "  " << std::fixed
          << std::setprecision(4) << (k < r.acceptance_rates.size() ? r.acceptance_rates[k] : 0.0) << "  "
          << (k < r.up_fraction.size() ? r.up_fraction[k] : 0.0) << "  " << std::setprecision(2)
          << (k < r.autocorrelation_times.size() ? r.autocorrelation_times[k] : 0.0) << "\n";
    }
}

/**
 * Temperature ladder for a PT run on `size` ranks: geometric in
 * [T_end, T_start], or tuned with `tune(options)` (a model's
 * tune_temperature_ladder) when pt_optimize_temperatures is set. Every rank
 * returns the same ladder.
 */
template <class TuneFn>
std::vector<double> prepare_ladder(const SpinConfig& config, int rank, int size, TuneFn&& tune) {
    if (!config.pt_optimize_temperatures || size < 3) {
        if (rank == 0) std::cout << "Using a geometric temperature ladder" << std::endl;
        return mc::generate_geometric_temperature_ladder(config.T_end, config.T_start, size_t(size));
    }
    mc::LadderTuningOptions o;
    o.T_min = config.T_end;
    o.T_max = config.T_start;
    o.method = mc::parse_ladder_method(config.pt_temperature_optimizer, rank == 0);
    o.warmup_steps = config.pt_optimization_warmup;
    o.steps_per_round = config.pt_optimization_sweeps;
    o.max_rounds = config.pt_optimization_iterations;
    o.exchange_every = std::max<size_t>(1, config.pt_exchange_frequency);
    o.tolerance = config.pt_optimization_tolerance;
    const mc::LadderTuningResult r = tune(o);
    if (rank == 0 && !config.output_dir.empty()) {
        std::error_code ec;  // never throw on one rank only
        std::filesystem::create_directories(config.output_dir, ec);
        write_tuned_ladder(config.output_dir + "/optimized_temperatures.txt", r,
                           mc::ladder_method_name(o.method));
    }
    return r.temperatures;
}

}  // namespace pt_runner
