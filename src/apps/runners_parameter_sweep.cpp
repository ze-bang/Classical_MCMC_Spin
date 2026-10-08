/**
 * runners_parameter_sweep.cpp — multi-lattice parameter sweep driver.
 *
 * Every sweep point is the base configuration with the swept keys applied
 * through SpinConfig::set (typed fields such as pump_amplitude, probe_time,
 * T_start or md_* included; values printed losslessly), validated like a
 * direct run, and run through run_simulation(), the same construction path
 * as a direct run.
 *
 * MPI: points that need one rank run on MPI_COMM_SELF, distributed
 * round-robin over the ranks; parallel-tempering points run on equal-sized
 * sub-communicators (extra ranks idle). The only collective on the sweep's
 * communicator is the final barrier, reached once by every rank, so uneven
 * point counts cannot deadlock and no point's collectives see another
 * point's ranks.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace std;

namespace {

struct SweepPoint {
    vector<double> values;   // one per axis
    SpinConfig config;       // base config with the values applied
};

/// Cartesian product of the axes; every point validated before anything runs.
vector<SweepPoint> make_sweep_points(const SpinConfig& base, const vector<SpinConfig::SweepAxis>& axes) {
    vector<SweepPoint> points;
    vector<double> current;
    function<void(size_t)> rec = [&](size_t depth) {
        if (depth == axes.size()) {
            SweepPoint pt;
            pt.values = current;
            pt.config = base;
            pt.config.simulation = base.sweep_base_simulation;
            stringstream dir;
            dir << base.output_dir;
            for (size_t p = 0; p < axes.size(); ++p) {
                pt.config.set_value(axes[p].name, current[p]);
                dir << "/" << axes[p].name << "_" << scientific << current[p];
            }
            pt.config.output_dir = dir.str();
            const vector<string> errors = pt.config.validation_errors();
            if (!errors.empty()) {
                string msg = "sweep point " + pt.config.output_dir + " is invalid:";
                for (const string& e : errors) msg += "\n  " + e;
                throw invalid_argument(msg);
            }
            points.push_back(std::move(pt));
            return;
        }
        for (double v : axes[depth].values) {
            current.push_back(v);
            rec(depth + 1);
            current.pop_back();
        }
    };
    rec(0);
    return points;
}

string describe(const vector<SpinConfig::SweepAxis>& axes, const vector<double>& values) {
    ostringstream s;
    s << setprecision(12);
    for (size_t p = 0; p < axes.size(); ++p) s << (p ? ", " : "") << axes[p].name << "=" << values[p];
    return s.str();
}

/// One sweep point on `comm` (rank 0 of comm writes the point's run_info.txt).
void run_point(const SweepPoint& pt, MPI_Comm comm, int world_size) {
    if (comm_rank(comm) == 0) {
        filesystem::create_directories(pt.config.output_dir);
        write_run_info(pt.config, pt.config.output_dir + "/run_info.txt", world_size);
    }
    MPI_Barrier(comm);   // directory exists before any rank of the point writes
    run_simulation(pt.config, comm);
}

}  // namespace

/**
 * Run the multi-lattice parameter sweep dispatcher.
 */
void run_parameter_sweep(const SpinConfig& base_config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    const vector<SpinConfig::SweepAxis> axes = base_config.sweep_axes();
    const vector<SweepPoint> points = make_sweep_points(base_config, axes);

    // Check if GPU is needed for the base simulation
    bool needs_gpu = base_config.use_gpu && (
        base_config.sweep_base_simulation == SimulationType::MOLECULAR_DYNAMICS ||
        base_config.sweep_base_simulation == SimulationType::PUMP_PROBE ||
        base_config.sweep_base_simulation == SimulationType::TWOD_COHERENT_SPECTROSCOPY
    );

    if (rank == 0) {
        cout << "Running " << axes.size() << "D parameter sweep..." << endl;
        for (size_t p = 0; p < axes.size(); ++p) {
            cout << "  Parameter " << (p + 1) << ": " << axes[p].name << " (" << axes[p].values.size()
                 << " points, " << axes[p].values.front() << " -> " << axes[p].values.back() << ")" << endl;
        }
        cout << "Base simulation: ";
        switch (base_config.sweep_base_simulation) {
            case SimulationType::SIMULATED_ANNEALING: cout << "Simulated Annealing"; break;
            case SimulationType::POPULATION_ANNEALING: cout << "Population Annealing"; break;
            case SimulationType::PARALLEL_TEMPERING: cout << "Parallel Tempering"; break;
            case SimulationType::MOLECULAR_DYNAMICS: cout << "Molecular Dynamics"; break;
            case SimulationType::PUMP_PROBE: cout << "Pump-Probe"; break;
            case SimulationType::TWOD_COHERENT_SPECTROSCOPY: cout << "2DCS Spectroscopy"; break;
            default: cout << "Unknown"; break;
        }
        cout << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Total sweep points: " << points.size() << endl;
        if (needs_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else if (base_config.use_gpu) {
            cout << "GPU acceleration: Not used by base simulation type" << endl;
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }

#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    // Do this ONCE before the sweep loop to avoid repeated setup
    if (needs_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            cout << "[Rank " << rank << "] Assigned to GPU " << device_id
                 << " (parameter sweep, " << device_count << " GPU(s) available)" << endl;
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif

    const bool is_parallel_tempering = (base_config.sweep_base_simulation == SimulationType::PARALLEL_TEMPERING);
    if (is_parallel_tempering) {
        // ====================================================================
        // PARALLEL TEMPERING: each point runs PT on a group of ranks (one
        // replica per rank). Groups have equal size, so every point gets the
        // same ladder; ranks left over by the division idle.
        // ====================================================================
        int ranks_per_point = base_config.pt_ranks_per_point;
        if (ranks_per_point <= 0) {
            // Auto: share the ranks among the points, at least 2 replicas each
            ranks_per_point = max(2, size / static_cast<int>(points.size()));
        }
        ranks_per_point = min(ranks_per_point, size);
        const int num_groups = size / ranks_per_point;
        const int color = (rank < num_groups * ranks_per_point) ? rank / ranks_per_point : MPI_UNDEFINED;
        MPI_Comm group_comm = MPI_COMM_NULL;
        MPI_Comm_split(comm, color, rank, &group_comm);

        if (rank == 0) {
            cout << "\nParallel Tempering in Parameter Sweep Mode:" << endl;
            cout << "  Replicas (ranks) per sweep point: " << ranks_per_point << endl;
            cout << "  Concurrent sweep points: " << num_groups << endl;
            if (size % ranks_per_point != 0)
                cout << "  Warning: " << (size % ranks_per_point) << " rank(s) idle (" << size
                     << " is not a multiple of " << ranks_per_point << ")" << endl;
            if (ranks_per_point < 3)
                cout << "  Warning: " << ranks_per_point << " replicas per point is a degenerate ladder; "
                     << "set pt_ranks_per_point" << endl;
        }

        if (group_comm != MPI_COMM_NULL) {
            const int group_rank = comm_rank(group_comm);
            for (size_t i = static_cast<size_t>(color); i < points.size(); i += static_cast<size_t>(num_groups)) {
                if (group_rank == 0)
                    cout << "[Group " << color << "] Processing point " << i + 1 << "/" << points.size() << ": "
                         << describe(axes, points[i].values) << endl;
                run_point(points[i], group_comm, size);
                if (group_rank == 0)
                    cout << "[Group " << color << "] Completed point " << i + 1 << ": "
                         << describe(axes, points[i].values) << endl;
            }
            MPI_Comm_free(&group_comm);
        }
    } else {
        // ====================================================================
        // One rank per point, on MPI_COMM_SELF: the point's runner collectives
        // never involve ranks working on other points.
        // ====================================================================
        for (size_t i = static_cast<size_t>(rank); i < points.size(); i += static_cast<size_t>(size)) {
            cout << "[Rank " << rank << "] Processing point " << i + 1 << "/" << points.size() << ": "
                 << describe(axes, points[i].values) << endl;
            run_point(points[i], MPI_COMM_SELF, size);
            cout << "[Rank " << rank << "] Completed point " << i + 1 << ": " << describe(axes, points[i].values)
                 << endl;
        }
    }

    // Synchronize all ranks (the only collective over comm after the setup)
    MPI_Barrier(comm);

    if (rank == 0) {
        cout << "Parameter sweep completed (" << points.size() << " points)." << endl;
    }
}
