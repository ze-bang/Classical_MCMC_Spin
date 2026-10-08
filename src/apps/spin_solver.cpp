/**
 * spin_solver.cpp — simulation executable entry point.
 *
 * The heavy lifting lives in sibling TUs: system_factory.cpp (one
 * construction path for every system, the mode dispatch, run_info.txt),
 * `runners_lattice.cpp`, `runners_phonon.cpp`, `runners_mixed.cpp`,
 * `runners_parameter_sweep.cpp`, all declared in `spin_solver_runners.h`.
 * This file is CLI parsing, MPI lifetime and error policy:
 *
 *   - rank 0 reads the configuration file and broadcasts its text, so every
 *     rank parses identical input (a file visible on some nodes only used to
 *     make ranks disagree and hang);
 *   - configuration errors are detected identically on every rank before any
 *     collective work, reported once, and end the run with exit status 1;
 *   - an exception on any rank during the run is printed with its rank and
 *     ends the job with MPI_Abort (never MPI_Finalize while peers may sit in
 *     a collective); HDF5 C++ exceptions, which do not derive from
 *     std::exception, are reported with their detail message;
 *   - "completed successfully" is printed only when everything ran.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/simple_linear_alg.h"

#include <H5Cpp.h>
#include <mpi.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

/// Read the configuration on rank 0 and broadcast its text (or the read error).
string broadcast_config_text(const string& path, int rank, string& error) {
    string text;
    int status = 0;   // 0 = ok, 1 = cannot read
    if (rank == 0) {
        ifstream in(path);
        if (!in) {
            status = 1;
            text = "Cannot open config file: " + path;
        } else {
            stringstream ss;
            ss << in.rdbuf();
            text = ss.str();
        }
    }
    long long len = static_cast<long long>(text.size());
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&len, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    text.resize(static_cast<size_t>(len));
    // MPI counts are int: broadcast in chunks for very large files.
    for (long long off = 0; off < len; off += (1 << 30)) {
        const int n = static_cast<int>(std::min<long long>(len - off, 1 << 30));
        MPI_Bcast(text.data() + off, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    if (status != 0) {
        error = text;
        return string();
    }
    return text;
}

[[noreturn]] void abort_with(int rank, const string& what) {
    cerr << "[rank " << rank << "] error: " << what << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);   // not reached
}

}  // namespace

int main(int argc, char** argv) {
    // The library runs OpenMP regions between MPI calls made by the master thread.
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED && rank == 0)
        cerr << "Warning: the MPI library does not provide MPI_THREAD_FUNNELED" << endl;
    // HDF5 errors are reported through the exception handler below (with the
    // failing call and file); the automatic stack dump would print per rank.
    H5::Exception::dontPrint();

    const string config_file = (argc > 1) ? argv[1] : "simulation.param";

    // --- Configuration: identical text, parsing and validation on every rank.
    SpinConfig config;
    string error;
    const string text = broadcast_config_text(config_file, rank, error);
    if (error.empty()) {
        try {
            config = SpinConfig::from_string(text, config_file, /*verbose=*/rank == 0);
            const vector<string> problems = config.validation_errors();
            for (const string& p : problems) error += "\n  " + p;
            if (!problems.empty()) error = "invalid configuration " + config_file + ":" + error;
        } catch (const exception& e) {
            error = e.what();
        }
    }
    int failed = error.empty() ? 0 : 1, any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_failed) {
        // Every rank parsed the same text, so they agree; no rank is inside a
        // collective, so finalising is safe.
        if (rank == 0) {
            cerr << "[rank 0] error: " << (error.empty() ? string("configuration rejected on another rank") : error)
                 << "\nUsage: " << argv[0] << " [config_file]" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    try {
        // Seed the process RNG once: a user seed, or a random one drawn on
        // rank 0 and broadcast. Every rank then derives its own stream from
        // (seed, rank); nothing reseeds from the wall clock afterwards.
        unsigned long long seed = config.seed;
        if (seed == 0 && rank == 0) {
            std::random_device rd;
            seed = (static_cast<unsigned long long>(rd()) << 32) ^ rd();
            if (seed == 0) seed = 1;
        }
        MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        config.seed = seed;
        seed_lehman(seed);
        seed_lehman_from_rank(static_cast<unsigned long long>(rank));

        // Provenance (rank 0): the resolved configuration with seed, MPI size,
        // OpenMP threads, git describe and build flags.
        int io_ok = 1;
        string io_error;
        if (rank == 0) {
            cout << "RNG seed: " << seed << " (set `seed = " << seed << "` to reproduce this run)" << endl;
            try {
                if (!config.output_dir.empty()) {
                    std::filesystem::create_directories(config.output_dir);
                    std::ofstream(config.output_dir + "/seed.txt") << seed << "\n";
                    write_run_info(config, config.output_dir + "/run_info.txt", size);
                }
            } catch (const exception& e) {
                io_ok = 0;
                io_error = e.what();
            }
            config.print();
        }
        MPI_Bcast(&io_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!io_ok) {
            if (rank == 0) cerr << "[rank 0] error: " << io_error << endl;
            MPI_Finalize();
            return 1;
        }

        if (config.simulation == SimulationType::PARAMETER_SWEEP) {
            run_parameter_sweep(config, MPI_COMM_WORLD);
        } else {
            run_simulation(config, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } catch (const H5::Exception& e) {
        abort_with(rank, "HDF5: " + e.getFuncName() + ": " + e.getDetailMsg());
    } catch (const exception& e) {
        abort_with(rank, e.what());
    } catch (...) {
        abort_with(rank, "unknown exception");
    }

    MPI_Finalize();
    if (rank == 0) {
        cout << "\n=== Simulation completed successfully ===" << endl;
    }
    return 0;
}
