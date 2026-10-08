/**
 * system_factory.cpp — the one construction path of every spin_solver system
 * and the dispatch from SimulationType to a runner.
 *
 * spin_solver's main() and the parameter sweep used to carry their own copies
 * of the unit-cell switch, the initial-state logic and the runner switch; the
 * copies drifted (the sweep built NCTO without its extra modes, disorder and
 * pinning, and TmFeO3 without damping), so one config could describe
 * different systems depending on how it was run. Both now call
 * run_simulation().
 *
 * Also: the provenance record (run_info.txt) and the per-trial summary
 * gathered on rank 0.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/phonon_config.h"

#include "classical_spin_build_info.h"

#include <mpi.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace std;

UnitCell make_unit_cell(const SpinConfig& config) {
    switch (config.system) {
        case SystemType::HONEYCOMB_BCAO: return build_bcao_honeycomb(config);
        case SystemType::HONEYCOMB_KITAEV: return build_kitaev_honeycomb(config);
        case SystemType::PYROCHLORE: return build_pyrochlore(config);
        case SystemType::PYROCHLORE_NON_KRAMER: return build_pyrochlore_non_kramer(config);
        case SystemType::TRIANGULAR_ANISOTROPIC: return build_triangular_anisotropic(config);
        case SystemType::TMFEO3_FE: return build_tmfeo3_fe(config);
        case SystemType::TMFEO3_TM: return build_tmfeo3_tm(config);
        default:
            throw invalid_argument("make_unit_cell: system '" + system_type_to_string(config.system) +
                                   "' is not a Lattice-family system");
    }
}

namespace {

/// Unit vector of length `dim` along config.ferromagnetic_direction (zero-padded).
SpinVector ferromagnetic_direction(const SpinConfig& config, size_t dim, const char* what) {
    SpinVector dir = SpinVector::Zero(dim);
    for (size_t d = 0; d < dim && d < config.ferromagnetic_direction.size(); ++d)
        dir(d) = config.ferromagnetic_direction[d];
    if (!(dir.norm() > 0.0))
        throw invalid_argument(string("ferromagnetic_direction has no component along the ") + what + " spin space");
    return dir;
}

}  // namespace

std::unique_ptr<Lattice> make_lattice(const SpinConfig& config) {
    auto lattice = std::make_unique<Lattice>(make_unit_cell(config), config.lattice_size[0], config.lattice_size[1],
                                             config.lattice_size[2], config.spin_length);
    lattice->lattice_type = system_type_to_string(config.system);
    lattice->local_update = Lattice::parse_local_update(config.local_update);
    if (config.use_ferromagnetic_init) {
        lattice->init_ferromagnetic(ferromagnetic_direction(config, lattice->spin_dim, "lattice"));
    } else if (!config.initial_spin_config.empty()) {
        lattice->load_spin_config(config.initial_spin_config);
    }
    // else: random, from the constructor
    return lattice;
}

std::unique_ptr<MixedLattice> make_mixed_lattice(const SpinConfig& config) {
    auto lattice = std::make_unique<MixedLattice>(build_tmfeo3(config), config.lattice_size[0],
                                                  config.lattice_size[1], config.lattice_size[2],
                                                  config.spin_length, config.spin_length_su3);
    if (config.use_ferromagnetic_init) {
        SpinVector dir_su3 = SpinVector::Zero(lattice->spin_dim_SU3);
        const int c = static_cast<int>(config.get_param("su3_init_component", 2.0));
        dir_su3((c >= 0 && c < static_cast<int>(lattice->spin_dim_SU3)) ? c : 2) = 1.0;   // default lambda_3
        lattice->init_ferromagnetic(ferromagnetic_direction(config, lattice->spin_dim_SU2, "SU(2)"), dir_su3);
    } else if (!config.initial_spin_config.empty()) {
        lattice->load_spin_config(config.initial_spin_config);
    } else {
        lattice->init_random();
    }
    // Gilbert damping (the dynamics runners set it again per trial, from the same key)
    lattice->alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
    return lattice;
}

void run_simulation(const SpinConfig& config, MPI_Comm comm) {
    const int rank = comm_rank(comm);
    if (!simulation_supported(config.system, config.simulation) ||
        config.simulation == SimulationType::PARAMETER_SWEEP)
        throw invalid_argument("run_simulation: simulation mode not supported for system '" +
                               system_type_to_string(config.system) + "'");

    if (config.system == SystemType::NCTO) {
        if (rank == 0) cout << "\nBuilding PhononLattice spin-phonon lattice..." << endl;
        PhononLattice lattice = make_ncto_lattice(config);
        switch (config.simulation) {
            case SimulationType::SIMULATED_ANNEALING: run_simulated_annealing_phonon(lattice, config, comm); break;
            case SimulationType::MOLECULAR_DYNAMICS: run_molecular_dynamics_phonon(lattice, config, comm); break;
            case SimulationType::PUMP_PROBE: run_pump_probe_phonon(lattice, config, comm); break;
            case SimulationType::TWOD_COHERENT_SPECTROSCOPY: run_2dcs_phonon(lattice, config, comm); break;
            default: throw invalid_argument("run_simulation: mode not supported for PhononLattice");
        }
    } else if (config.system == SystemType::TMFEO3) {
        if (rank == 0) cout << "\nBuilding TmFeO3 system..." << endl;
        std::unique_ptr<MixedLattice> lattice = make_mixed_lattice(config);
        switch (config.simulation) {
            case SimulationType::SIMULATED_ANNEALING: run_simulated_annealing_mixed(*lattice, config, comm); break;
            case SimulationType::PARALLEL_TEMPERING: run_parallel_tempering_mixed(*lattice, config, comm); break;
            case SimulationType::MOLECULAR_DYNAMICS: run_molecular_dynamics_mixed(*lattice, config, comm); break;
            case SimulationType::PUMP_PROBE: run_pump_probe_mixed(*lattice, config, comm); break;
            case SimulationType::TWOD_COHERENT_SPECTROSCOPY: run_2dcs_spectroscopy_mixed(*lattice, config, comm); break;
            default: throw invalid_argument("run_simulation: mode not supported for MixedLattice");
        }
    } else {
        if (rank == 0) cout << "\nBuilding unit cell..." << endl;
        std::unique_ptr<Lattice> lattice = make_lattice(config);
        switch (config.simulation) {
            case SimulationType::SIMULATED_ANNEALING: run_simulated_annealing(*lattice, config, comm); break;
            case SimulationType::PARALLEL_TEMPERING: run_parallel_tempering(*lattice, config, comm); break;
            case SimulationType::MOLECULAR_DYNAMICS: run_molecular_dynamics(*lattice, config, comm); break;
            case SimulationType::PUMP_PROBE: run_pump_probe(*lattice, config, comm); break;
            case SimulationType::TWOD_COHERENT_SPECTROSCOPY: run_2dcs_spectroscopy(*lattice, config, comm); break;
            default: throw invalid_argument("run_simulation: mode not supported for Lattice");
        }
    }
}

void write_run_info(const SpinConfig& config, const std::string& path, int mpi_size) {
    const filesystem::path p(path);
    if (p.has_parent_path()) filesystem::create_directories(p.parent_path());
    ofstream out(path);
    if (!out) throw runtime_error("cannot write " + path);
    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    char date[64] = "unknown";
    const std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    out << "# spin_solver run information. The keys below are the full resolved configuration:\n"
        << "# `spin_solver run_info.txt` repeats this run.\n"
        << "# date (UTC):   " << date << "\n"
        << "# host:         " << host << " (rank 0)\n"
        << "# seed:         " << config.seed << "\n"
        << "# mpi_size:     " << mpi_size << "\n"
        << "# omp_threads:  " << omp_get_max_threads() << " per rank\n"
        << "# git:          " << CLASSICAL_SPIN_GIT_DESCRIBE << "\n"
        << "# compiler:     " << CLASSICAL_SPIN_CXX_COMPILER << "\n"
        << "# build_type:   " << CLASSICAL_SPIN_BUILD_TYPE << "\n"
        << "# cxx_flags:    " << CLASSICAL_SPIN_CXX_FLAGS << "\n";
    config.write(out);
    out.close();
    if (!out) throw runtime_error("writing " + path + " failed");
}

void write_trial_summary(const SpinConfig& config, const std::string& what, const std::vector<TrialResult>& mine,
                         MPI_Comm comm) {
    const int rank = comm_rank(comm);
    const int size = comm_size(comm);
    // One text record per trial: "trial rank energy file\n" (file names have no newlines).
    std::ostringstream rec;
    rec << std::setprecision(17);
    for (const TrialResult& r : mine) rec << r.trial << ' ' << rank << ' ' << r.energy << ' ' << r.file << '\n';
    const std::string local = rec.str();
    int len = static_cast<int>(local.size());
    std::vector<int> lens(rank == 0 ? size : 0), offs(rank == 0 ? size : 0);
    MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, comm);
    std::string all;
    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < size; ++r) {
            offs[r] = total;
            total += lens[r];
        }
        all.resize(static_cast<size_t>(total));
    }
    MPI_Gatherv(local.data(), len, MPI_CHAR, rank == 0 ? all.data() : nullptr, lens.data(), offs.data(), MPI_CHAR, 0,
                comm);
    if (rank != 0) return;

    struct Row {
        int trial, rank;
        double energy;
        std::string file;
    };
    std::vector<Row> rows;
    std::istringstream in(all);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        Row r;
        std::string e;
        ls >> r.trial >> r.rank >> e;
        r.energy = (e == "nan" || e == "-nan") ? std::numeric_limits<double>::quiet_NaN() : std::stod(e);
        std::getline(ls >> std::ws, r.file);
        rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.trial < b.trial; });

    const std::string path = config.output_dir + "/trial_summary.txt";
    filesystem::create_directories(config.output_dir);
    ofstream out(path);
    if (!out) throw runtime_error("cannot write " + path);
    out << "# " << what << ": " << rows.size() << " of " << config.num_trials << " trial(s), " << size
        << " rank(s)\n";
    double sum = 0.0, sum2 = 0.0, emin = std::numeric_limits<double>::infinity();
    size_t n = 0;
    for (const Row& r : rows)
        if (std::isfinite(r.energy)) {
            sum += r.energy;
            sum2 += r.energy * r.energy;
            emin = std::min(emin, r.energy);
            ++n;
        }
    if (n > 0) {
        const double mean = sum / double(n);
        const double var = (n > 1) ? std::max(0.0, (sum2 - double(n) * mean * mean) / double(n - 1)) : 0.0;
        out << std::setprecision(17) << "# energy per site: mean " << mean << ", std " << std::sqrt(var) << ", min "
            << emin << "\n";
    }
    out << "# trial  rank  energy_per_site  result\n" << std::setprecision(17);
    for (const Row& r : rows) out << r.trial << "  " << r.rank << "  " << r.energy << "  " << r.file << "\n";
    out.close();
    if (!out) throw runtime_error("writing " + path + " failed");
    if (static_cast<int>(rows.size()) != config.num_trials)
        cerr << "Warning: " << path << " lists " << rows.size() << " trials, expected " << config.num_trials << endl;
}
