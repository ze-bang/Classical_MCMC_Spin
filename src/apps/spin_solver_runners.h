#pragma once
/**
 * spin_solver_runners.h — private declarations for spin_solver.
 *
 * spin_solver is split into one .cpp per lattice family (regular Lattice,
 * PhononLattice, MixedLattice), the parameter-sweep driver, and
 * system_factory.cpp, which owns the ONE construction path of every system
 * (unit cell, lattice, initial state) and the dispatch from SimulationType to
 * a runner. main() and every sweep point go through run_simulation(), so a
 * configuration describes the same system whichever way it is run.
 *
 * MPI: every runner is collective over the communicator it is given and
 * never touches MPI_COMM_WORLD. A sweep point that runs on one rank passes
 * MPI_COMM_SELF; a parallel-tempering sweep point passes its sub-communicator.
 * Trials are distributed round-robin over the ranks of that communicator.
 *
 * Notes:
 *   - This is an executable-private header. It lives next to spin_solver.cpp
 *     and is not installed, not in include/classical_spin/.
 *   - Only declarations live here, so including it stays cheap.
 *     Each runner's .cpp includes the corresponding lattice header.
 */

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

// --- Forward declarations -------------------------------------------------

struct SpinConfig;

class UnitCell;
class Lattice;
class PhononLattice;
class MixedLattice;

// --- System factory (system_factory.cpp) ----------------------------------

/// Unit cell of a Lattice-family system (every system except tmfeo3 / ncto).
/// Throws std::invalid_argument for the others.
UnitCell make_unit_cell(const SpinConfig& config);

/// Lattice-family system with its local update and initial state
/// (use_ferromagnetic_init, else initial_spin_config, else random).
std::unique_ptr<Lattice> make_lattice(const SpinConfig& config);

/// TmFeO3 mixed SU(2)+SU(3) system with its initial state and damping.
std::unique_ptr<MixedLattice> make_mixed_lattice(const SpinConfig& config);

// NCTO: make_ncto_lattice(config), classical_spin/lattice/phonon_config.h.

/**
 * Build the system of `config` and run config.simulation (not a parameter
 * sweep) collectively on `comm`. Throws std::invalid_argument for a
 * system/simulation combination that is not implemented.
 */
void run_simulation(const SpinConfig& config, MPI_Comm comm);

/**
 * Provenance record: comment lines with the seed, MPI size, OpenMP threads,
 * git describe, compiler and build flags, followed by the full resolved
 * configuration (SpinConfig::write), so the file itself reproduces the run:
 * `spin_solver run_info.txt`. Throws std::runtime_error if it cannot be written.
 */
void write_run_info(const SpinConfig& config, const std::string& path, int mpi_size);

/**
 * Per-trial results of a runner, gathered on rank 0 of `comm` into
 * output_dir/trial_summary.txt (trial, owning rank, energy per site,
 * result file). `energy` may be NaN when a runner has none.
 */
struct TrialResult {
    int trial = 0;
    double energy = 0.0;
    std::string file;
};
void write_trial_summary(const SpinConfig& config, const std::string& what,
                         const std::vector<TrialResult>& mine, MPI_Comm comm);

// --- Regular Lattice (SU(2), Heisenberg / Kitaev / BCAO / pyrochlore) -----

void run_simulated_annealing   (Lattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_parallel_tempering    (Lattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_molecular_dynamics    (Lattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_pump_probe            (Lattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_2dcs_spectroscopy     (Lattice& lattice, const SpinConfig& config, MPI_Comm comm);

// --- PhononLattice (spin + discrete phonon modes) -------------------------

void run_simulated_annealing_phonon (PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_molecular_dynamics_phonon  (PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_pump_probe_phonon          (PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_2dcs_phonon                (PhononLattice& lattice, const SpinConfig& config, MPI_Comm comm);

// --- MixedLattice (SU(2) + SU(3), TmFeO3) ---------------------------------

void run_simulated_annealing_mixed  (MixedLattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_parallel_tempering_mixed   (MixedLattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_molecular_dynamics_mixed   (MixedLattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_pump_probe_mixed           (MixedLattice& lattice, const SpinConfig& config, MPI_Comm comm);
void run_2dcs_spectroscopy_mixed    (MixedLattice& lattice, const SpinConfig& config, MPI_Comm comm);

// --- Parameter sweep driver (runners_parameter_sweep.cpp) -----------------

/// Collective over `comm` (MPI_COMM_WORLD from main).
void run_parameter_sweep(const SpinConfig& base_config, MPI_Comm comm);

// --- Small MPI helpers ------------------------------------------------------

inline int comm_rank(MPI_Comm comm) {
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}
inline int comm_size(MPI_Comm comm) {
    int s = 1;
    MPI_Comm_size(comm, &s);
    return s;
}
/// Rank in the whole job, for GPU binding: a sweep point runs on MPI_COMM_SELF
/// (rank 0 there), and binding by that rank put every point on device 0.
inline int job_rank() { return comm_rank(MPI_COMM_WORLD); }
