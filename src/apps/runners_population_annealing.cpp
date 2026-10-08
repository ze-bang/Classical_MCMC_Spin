/**
 * runners_population_annealing.cpp — spin_solver runner for population
 * annealing (simulation = population_annealing) on the regular Lattice.
 *
 * The population is spread over all MPI ranks and, within a rank, over the
 * OpenMP threads (one worker copy of the lattice per thread); see
 * include/classical_spin/mc/population_annealing.h for the algorithm.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/mc/population_annealing.h"

#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

void run_population_annealing(Lattice& lattice, const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm), size = comm_size(comm);
    mc::PAOptions o;
    o.population = config.pa_population;
    o.T_start = config.T_start;
    o.T_end = config.T_end;
    o.n_temperatures = config.pa_temperatures;
    o.schedule = mc::parse_pa_schedule(config.pa_schedule);
    o.target_ess = config.pa_target_ess;
    o.max_temperatures = config.pa_max_temperatures;
    o.sweeps = config.pa_sweeps;
    o.output_dir = config.output_dir;
    if (rank == 0) {
        cout << "Running population annealing: R = " << o.population << " replicas on " << size
             << " rank(s), " << o.sweeps << " sweeps per temperature, schedule "
             << mc::pa_schedule_name(o.schedule) << endl;
    }

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    vector<unique_ptr<Lattice>> copies;
    vector<unique_ptr<mc::LatticePopulationWorker<Lattice>>> owned;
    vector<mc::LatticePopulationWorker<Lattice>*> workers;
    for (int t = 0; t < n_threads; ++t) {
        copies.push_back(make_unique<Lattice>(lattice));
        owned.push_back(make_unique<mc::LatticePopulationWorker<Lattice>>(
            *copies.back(), config.overrelaxation_rate, config.gaussian_move));
        workers.push_back(owned.back().get());
    }

    const mc::PAResult res = mc::run_population_annealing(workers, o, comm);

    // The lowest-energy replica at T_end becomes the lattice state; rank 0
    // writes it (an equilibrium sample, not a quenched ground state).
    owned[0]->unpack_state(res.best_state.data());
    lattice.spins = copies[0]->spins;
    if (rank == 0) {
        lattice.save_spin_config(config.output_dir + "/pa_best_spins.txt");
        lattice.save_positions(config.output_dir + "/positions.txt");
        const mc::PAStep& f = res.steps.back();
        ofstream out(config.output_dir + "/final_energy.txt");
        out << setprecision(17) << "Energy Density: " << res.best_energy / double(lattice.lattice_size) << "\n"
            << "Population mean E/N at T_end: " << f.energy << " +- " << f.energy_error << "\n"
            << "ln Z/N - ln Z(T=inf)/N at T_end: " << f.ln_Z << "\n";
        cout << "Population annealing completed: <E>/N = " << f.energy << " +- " << f.energy_error
             << " at T = " << f.T << ", lowest E/N = " << res.best_energy / double(lattice.lattice_size)
             << " (" << res.steps.size() << " temperatures; summary in " << config.output_dir
             << "/pa_summary.txt)" << endl;
    }
}
