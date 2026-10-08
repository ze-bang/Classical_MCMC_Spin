/**
 * spin_solver.cpp — simulation executable entry point.
 *
 * The heavy lifting (every `run_<simulation>_<lattice>()` function, plus
 * the parameter-sweep driver and the phonon parameter builders)
 * lives in sibling TUs: `runners_lattice.cpp`, `runners_phonon.cpp`,
 * `runners_mixed.cpp`, `runners_parameter_sweep.cpp`.
 * All of them are declared in `spin_solver_runners.h`. This file is now
 * just CLI parsing, MPI lifetime management, and the lattice-family
 * dispatch.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/phonon_config.h"

#include <mpi.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace std;


int main(int argc, char** argv) {
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    string config_file = "simulation.param";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    // Load configuration
    SpinConfig config;
    try {
        config = SpinConfig::from_file(config_file);
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error loading configuration: " << e.what() << endl;
            cerr << "Usage: " << argv[0] << " [config_file]\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Validate configuration
    if (!config.validate()) {
        if (rank == 0) {
            cerr << "Configuration validation failed\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Seed the process RNG once: a user seed, or a random one drawn on rank 0
    // and broadcast. Every rank then derives its own stream from (seed, rank);
    // nothing reseeds from the wall clock afterwards.
    {
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
        if (rank == 0) {
            cout << "RNG seed: " << seed << " (set `seed = " << seed
                 << "` to reproduce this run)" << endl;
            if (!config.output_dir.empty()) {
                std::filesystem::create_directories(config.output_dir);
                std::ofstream(config.output_dir + "/seed.txt") << seed << "\n";
            }
        }
    }

    // Print configuration on rank 0
    if (rank == 0) {
        config.print();
    }
    
    // Build system and run simulation
    try {
        if (config.system == SystemType::NCTO) {
            // PhononLattice spin-phonon coupled system (honeycomb)
            if (rank == 0) {
                cout << "\nBuilding PhononLattice spin-phonon lattice..." << endl;
            }
            
            // One construction path for the NCTO model (also used by the parameter
            // sweep and the ncto_* tools): couplings, modes, disorder, fields, damping,
            // initial state.
            PhononLattice phonon_lattice = make_ncto_lattice(config);

            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    throw std::invalid_argument("simulation type not supported for PhononLattice "
                                                "(supported: SA, MD, pump_probe, 2dcs, parameter_sweep)");
            }
        } else if (config.system == SystemType::TMFEO3) {
            // Mixed SU(2)+SU(3) system
            if (rank == 0) {
                cout << "\nBuilding TmFeO3 system..." << endl;
            }
            
            auto mixed_uc = build_tmfeo3(config);
            MixedLattice mixed_lattice(mixed_uc, 
                                       config.lattice_size[0],
                                       config.lattice_size[1],
                                       config.lattice_size[2],
                                       config.spin_length,
                                       config.spin_length_su3);
            // Monte Carlo policy: local update kernel and SU(3) state space
            // (CP^2 by default, legacy S^7 with su3_legacy_convention or
            // su3_mc_manifold = sphere); set before the spins are initialised.
            mixed_lattice.local_update = MixedLattice::parse_local_update(config.local_update);
            mixed_lattice.set_su3_mc_manifold(config.su3_mc_manifold);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                // Create SU2 direction from config (use general dimension)
                SpinVector dir_su2(mixed_lattice.spin_dim_SU2);
                for (size_t d = 0; d < mixed_lattice.spin_dim_SU2; ++d) {
                    dir_su2(d) = (d < config.ferromagnetic_direction.size()) ? 
                                 config.ferromagnetic_direction[d] : 0.0;
                }
                // Create SU3 direction
                SpinVector dir_su3 = SpinVector::Zero(mixed_lattice.spin_dim_SU3);
                const int su3_init_component = static_cast<int>(config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < static_cast<int>(mixed_lattice.spin_dim_SU3)) {
                    dir_su3(su3_init_component) = 1.0;
                } else {
                    dir_su3(2) = 1.0;  // Default to λ3
                }
                mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
            } else if (!config.initial_spin_config.empty()) {
                mixed_lattice.load_spin_config(config.initial_spin_config);
            } else {
                mixed_lattice.init_random();
            }

            // Gilbert damping (optional; defaults to 0 = undamped)
            mixed_lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
            if (mixed_lattice.alpha_gilbert > 0.0 && rank == 0) {
                cout << "Gilbert damping: α = " << mixed_lattice.alpha_gilbert << endl;
            }
            
            // Whether to save full spin state trajectories (large, for diagnosis)
            const bool save_spin_trajectories = (config.get_param("save_spin_trajectories", 0.0) > 0.5);

            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for mixed lattice\n";
                    }
                    break;
            }
        } else {
            // Regular SU(2) system
            if (rank == 0) {
                cout << "\nBuilding unit cell..." << endl;
            }
            
            UnitCell* uc_ptr = nullptr;
            switch (config.system) {
                case SystemType::HONEYCOMB_BCAO:
                    uc_ptr = new UnitCell(build_bcao_honeycomb(config));
                    break;
                case SystemType::HONEYCOMB_KITAEV:
                    uc_ptr = new UnitCell(build_kitaev_honeycomb(config));
                    break;
                case SystemType::PYROCHLORE:
                    uc_ptr = new UnitCell(build_pyrochlore(config));
                    break;
                case SystemType::TRIANGULAR_ANISOTROPIC:
                    uc_ptr = new UnitCell(build_triangular_anisotropic(config));
                    break;
                case SystemType::PYROCHLORE_NON_KRAMER:
                    uc_ptr = new UnitCell(build_pyrochlore_non_kramer(config));
                    break;
                case SystemType::TMFEO3_FE:
                    uc_ptr = new UnitCell(build_tmfeo3_fe(config));
                    break;
                case SystemType::TMFEO3_TM:
                    uc_ptr = new UnitCell(build_tmfeo3_tm(config));
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Error: Unknown system type for parameter sweep" << endl;
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            
            Lattice lattice(*uc_ptr, 
                          config.lattice_size[0],
                          config.lattice_size[1],
                          config.lattice_size[2],
                          config.spin_length);
            lattice.lattice_type = system_type_to_string(config.system);
            lattice.local_update = Lattice::parse_local_update(config.local_update);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                // Initialize all spins in same direction (use general spin_dim)
                SpinVector dir(lattice.spin_dim);
                for (size_t d = 0; d < lattice.spin_dim; ++d) {
                    dir(d) = (d < config.ferromagnetic_direction.size()) ? 
                             config.ferromagnetic_direction[d] : 0.0;
                }
                dir.normalize();
                dir *= config.spin_length;
                for (size_t i = 0; i < lattice.lattice_size; ++i) {
                    lattice.spins[i] = dir;
                }
            } else if (!config.initial_spin_config.empty()) {
                lattice.load_spin_config(config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing(lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering(lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics(lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe(lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy(lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not implemented\n";
                    }
                    break;
            }
        }
    } catch (const exception& e) {
        cerr << "[Rank " << rank << "] Error during simulation: " << e.what() << endl;
        // Other ranks may be blocked in a collective: never finalize from one rank alone.
        if (size > 1) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
    
    // Finalize MPI
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
    
    if (rank == 0) {
        cout << "\n=== Simulation completed successfully ===" << endl;
    }
    
    return 0;
}
