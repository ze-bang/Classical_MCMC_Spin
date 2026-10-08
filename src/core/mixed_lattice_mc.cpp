/**
 * @file mixed_lattice_mc.cpp
 * @brief MixedLattice Monte Carlo driver methods.
 *
 * Hosts the MC drivers: simulated_annealing, perform_final_measurements,
 * greedy_quench, perform_mc_sweeps, and the T = 0 descent (exact single-site
 * minimisers). The hot single-site kernels — Metropolis, heat bath,
 * overrelaxation — and the sweep drivers stay in the header so the
 * optimizer can inline them across site loops.
 */

#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/lattice.h"   // Lattice::minimize_quadratic_on_sphere_impl

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

namespace {

// Restore the caller's stream formatting on exit.
struct CoutFormatGuard {
    std::ios_base::fmtflags flags = std::cout.flags();
    std::streamsize precision = std::cout.precision();
    ~CoutFormatGuard() { std::cout.flags(flags); std::cout.precision(precision); }
};

double distance(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += (a[k] - b[k]) * (a[k] - b[k]);
    return std::sqrt(s);
}

}  // namespace

// ---- MixedLattice::deterministic_site_SU2 ----
    double MixedLattice::deterministic_site_SU2(size_t i) {
        double* S = spins_SU2[i].data();
        const double L = double(spin_length_SU2);
        double h[3], Sn[3];
        linear_field_SU2(i, h);
        if (self_isotropic_SU2[i]) {
            const double hn = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
            if (hn < 1e-300) return 0.0;   // no torque: leave the spin
            for (int d = 0; d < 3; ++d) Sn[d] = -L * h[d] / hn;
        } else {
            double A[9];
            self_form_SU2(i, A);
            const Eigen::Matrix3d A3 = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(A);
            Lattice::minimize_quadratic_on_sphere_impl<3>(A3, h, L, S, Sn, 3);
            // With cubic self terms the quadratic minimiser is only a candidate.
            if (self_mode_SU2[i] == 2 && local_energy_change_SU2(i, h, Sn, S) >= 0.0) return 0.0;
        }
        const double change = distance(Sn, S, 3);
        S[0] = Sn[0]; S[1] = Sn[1]; S[2] = Sn[2];
        return change;
    }

// ---- MixedLattice::deterministic_site_SU3 ----
    double MixedLattice::deterministic_site_SU3(size_t j, bool force_cp2) {
        double* n = spins_SU3[j].data();
        double h[8], nn[8];
        linear_field_SU3(j, h);
        if (force_cp2 || su3_on_cp2()) {
            // The ground state of H = g.lambda minimises g.n on CP^2 exactly.
            auto ground = [](const double* g, double* out) {
                const auto psi = classical_spin::su3::ground_state(classical_spin::su3::gell_mann_sum(g));
                const std::complex<double> p[3] = {psi(0), psi(1), psi(2)};
                classical_spin::su3::pure_expectations(p, out);
            };
            if (self_isotropic_SU3[j]) {
                ground(h, nn);
            } else {
                // Quadratic self terms: ground state of the linearised local
                // Hamiltonian (h + 2 A n).lambda, kept only if it lowers the
                // local energy (monotone descent).
                double A[64], g[8];
                self_form_SU3(j, A);
                for (int a = 0; a < 8; ++a) {
                    g[a] = h[a];
                    for (int b = 0; b < 8; ++b) g[a] += 2.0 * A[a * 8 + b] * n[b];
                }
                ground(g, nn);
                if (local_energy_change_SU3(j, h, nn, n) >= 0.0) return 0.0;
            }
        } else {
            const double L = double(spin_length_SU3);
            if (self_isotropic_SU3[j]) {
                double hn = 0.0;
                for (int a = 0; a < 8; ++a) hn += h[a] * h[a];
                hn = std::sqrt(hn);
                if (hn < 1e-300) return 0.0;
                for (int a = 0; a < 8; ++a) nn[a] = -L * h[a] / hn;
            } else {
                double A[64];
                self_form_SU3(j, A);
                const Eigen::Matrix<double, 8, 8> A8 =
                    Eigen::Map<const Eigen::Matrix<double, 8, 8, Eigen::RowMajor>>(A);
                Lattice::minimize_quadratic_on_sphere_impl<8>(A8, h, L, n, nn, 8);
                if (self_mode_SU3[j] == 2 && local_energy_change_SU3(j, h, nn, n) >= 0.0) return 0.0;
            }
        }
        const double change = distance(nn, n, 8);
        std::memcpy(n, nn, sizeof(nn));
        return change;
    }

// ---- MixedLattice::deterministic_sweep ----
    double MixedLattice::deterministic_sweep(size_t num_sweeps) {
        double max_change = 0.0;
        for (size_t s = 0; s < num_sweeps; ++s) {
            max_change = 0.0;
            for (size_t i = 0; i < lattice_size_SU2; ++i) max_change = std::max(max_change, deterministic_site_SU2(i));
            for (size_t j = 0; j < lattice_size_SU3; ++j) max_change = std::max(max_change, deterministic_site_SU3(j));
        }
        return max_change;
    }

// ---- MixedLattice::deterministic_sweep_interleaved ----
    double MixedLattice::deterministic_sweep_interleaved() {
        double max_change = 0.0;
        const size_t n_cells = dim1 * dim2 * dim3;
        for (size_t c = 0; c < n_cells; ++c) {
            for (size_t a = 0; a < N_atoms_SU2; ++a)
                max_change = std::max(max_change, deterministic_site_SU2(c * N_atoms_SU2 + a));
            for (size_t a = 0; a < N_atoms_SU3; ++a)
                max_change = std::max(max_change, deterministic_site_SU3(c * N_atoms_SU3 + a));
        }
        return max_change;
    }

// ---- MixedLattice::deterministic_sweep_SU3_exact_diag ----
    double MixedLattice::deterministic_sweep_SU3_exact_diag() {
        double max_change = 0.0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) max_change = std::max(max_change, deterministic_site_SU2(i));
        for (size_t j = 0; j < lattice_size_SU3; ++j)
            max_change = std::max(max_change, deterministic_site_SU3(j, /*force_cp2=*/true));
        return max_change;
    }

// ---- MixedLattice::greedy_quench ----
    void MixedLattice::greedy_quench(double rel_tol, size_t max_sweeps) {
        project_SU3_to_manifold();
        double E_prev = total_energy();
        for (size_t s = 0; s < max_sweeps; ++s) {
            const double max_change = deterministic_sweep();
            const double E = total_energy();
            if (std::abs(E - E_prev) <= rel_tol * (std::abs(E_prev) + 1e-18) &&
                max_change <= std::sqrt(rel_tol) * std::max(1.0, double(spin_length_SU2))) {
                cout << "Greedy quench converged after " << s + 1 << " sweeps" << endl;
                break;
            }
            E_prev = E;
        }
    }

// ---- MixedLattice::simulated_annealing ----
    void MixedLattice::simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            bool gaussian_move,
                            double cooling_rate,
                            string out_dir,
                            bool save_observables,
                            bool T_zero,
                            size_t n_deterministics,
                            size_t twist_sweep_count) {
        (void) twist_sweep_count;   // no twisted boundaries in MixedLattice
        const vector<double> schedule = mc::annealing_schedule(T_start, T_end, cooling_rate);
        ensure_directory_exists(out_dir);
        CoutFormatGuard format_guard;

        const size_t projected = project_SU3_to_manifold();
        if (projected > 0) {
            cout << "Projected " << projected << " SU(3) state(s) onto the "
                 << (su3_on_cp2() ? "CP^2" : "S^7") << " Monte Carlo manifold" << endl;
        }
        const bool adaptive = uses_adaptive_step(gaussian_move);
        const bool interleaved = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        mc::StepSizeController step_size(/*sigma0=*/2.0, /*target=*/0.45);
        double sigma = step_size.sigma();
        const char* policy = local_update == LocalUpdate::HeatBath ? "heat-bath"
                           : adaptive ? "adaptive Gaussian Metropolis" : "uniform Metropolis";

        cout << "Starting mixed lattice simulated annealing: T=" << T_start << " → " << T_end
             << " in " << schedule.size() << " temperature steps of " << n_anneal << " sweeps ("
             << policy << ", SU(3) on " << (su3_on_cp2() ? "CP^2" : "S^7")
             << (use_parallel_sweeps() ? ", coloured OpenMP sweeps" : "") << ")" << endl;
        if (T_zero) {
            cout << "T=0 stage: up to " << n_deterministics << " descent sweeps" << endl;
        }

        for (size_t k = 0; k < schedule.size(); ++k) {
            const double T = schedule[k];
            double acceptance = 0.0;
            if (adaptive && n_anneal >= 2) {
                // Adapt sigma during the first half, sample with it frozen.
                step_size.restart();
                const size_t n_adapt = n_anneal / 2;
                for (size_t s = 0; s < n_adapt; ++s) {
                    step_size.update(perform_mc_sweeps(1, T, gaussian_move, sigma, 0, interleaved));
                    sigma = step_size.sigma();
                }
                acceptance = perform_mc_sweeps(n_anneal - n_adapt, T, gaussian_move, sigma, 0, interleaved);
            } else {
                acceptance = perform_mc_sweeps(n_anneal, T, gaussian_move, sigma, 0, interleaved);
            }

            if (k % 10 == 0 || k + 1 == schedule.size()) {
                cout << "T=" << std::scientific << T << ", E/N=" << energy_density()
                     << ", |M_SU2|=" << magnetization_SU2().norm()
                     << ", |M_SU3|=" << magnetization_SU3().norm()
                     << ", acc=" << std::fixed << acceptance;
                if (adaptive) cout << ", σ=" << sigma;
                cout << endl;
            }
        }

        double E_total = total_energy();
        double E_SU2 = total_energy_SU2();
        double E_SU3 = total_energy_SU3();
        size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        cout << std::setprecision(12);
        cout << "Final energy density: " << E_total / total_sites << endl;
        cout << "  Total Energy:     " << E_total << endl;
        cout << "  SU2 Energy:       " << E_SU2 << " (E/N_SU2 = " << E_SU2 / lattice_size_SU2 << ")" << endl;
        cout << "  SU3 Energy:       " << E_SU3 << " (E/N_SU3 = " << E_SU3 / lattice_size_SU3 << ")" << endl;

        // Save spin config after annealing (before the T = 0 stage)
        if (!out_dir.empty()) {
            save_spin_config_to_dir(out_dir, "spins_T=" + std::to_string(T_end));
            save_energy_to_dir(out_dir, "energy_T=" + std::to_string(T_end));
            save_positions_to_dir(out_dir);
        }

        // Final measurements if requested (before the T = 0 stage)
        if (save_observables && !out_dir.empty()) {
            perform_final_measurements(T_end, sigma, gaussian_move, out_dir);
        }

        // T = 0: exact block-coordinate descent until converged.
        if (T_zero && n_deterministics > 0) {
            cout << "\nT=0 descent (at most " << n_deterministics << " sweeps)..." << endl;
            const double scale = std::max(1.0, double(spin_length_SU2));
            double E_prev = total_energy();
            size_t sweep = 0;
            bool converged = false;
            for (; sweep < n_deterministics; ++sweep) {
                const double max_change = deterministic_sweep();
                const double E = total_energy();
                if (sweep % 100 == 0) {
                    cout << "Descent sweep " << sweep << ", E/N=" << E / total_sites
                         << ", |M_SU2|=" << magnetization_SU2().norm()
                         << ", |M_SU3|=" << magnetization_SU3().norm() << endl;
                }
                if (max_change < 1e-10 * scale && std::abs(E - E_prev) <= 1e-14 * (std::abs(E) + 1.0)) {
                    converged = true;
                    ++sweep;
                    break;
                }
                E_prev = E;
            }
            double E_total_final = total_energy();
            double E_SU2_final = total_energy_SU2();
            double E_SU3_final = total_energy_SU3();
            cout << std::setprecision(15);
            cout << "T=0 descent " << (converged ? "converged" : "stopped (not converged)") << " after "
                 << sweep << " sweeps. Final energy: " << E_total_final / total_sites << endl;
            cout << "  Total Energy:     " << E_total_final << endl;
            cout << "  SU2 Energy:       " << E_SU2_final << " (E/N_SU2 = " << E_SU2_final / lattice_size_SU2 << ")" << endl;
            cout << "  SU3 Energy:       " << E_SU3_final << " (E/N_SU3 = " << E_SU3_final / lattice_size_SU3 << ")" << endl;
            // Save final configuration
            if (!out_dir.empty()) {
                save_spin_config_to_dir(out_dir, "spins_T=0");
                save_energy_to_dir(out_dir, "energy_T=0");
            }
        }
    }

// ---- MixedLattice::perform_final_measurements ----
    void MixedLattice::perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   const string& out_dir) {
        cout << "\n=== Final measurements at T=" << T_final << " ===" << endl;
        const bool interleaved = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);

        // Step 1: Estimate autocorrelation time
        cout << "Estimating autocorrelation time..." << endl;
        vector<double> prelim_energies;
        size_t prelim_samples = 10000;
        size_t prelim_interval = 10;
        prelim_energies.reserve(prelim_samples / prelim_interval);

        for (size_t i = 0; i < prelim_samples; i += prelim_interval) {
            perform_mc_sweeps(prelim_interval, T_final, gaussian_move, sigma, 0, interleaved);
            prelim_energies.push_back(total_energy());
        }

        AutocorrelationResult acf = compute_autocorrelation(prelim_energies, prelim_interval);
        cout << "  τ_int = " << acf.tau_int << endl;
        cout << "  Sampling interval = " << acf.sampling_interval << " sweeps" << endl;

        // Step 2: Equilibrate
        size_t equilibration = 10 * acf.sampling_interval;
        cout << "Equilibrating for " << equilibration << " sweeps..." << endl;
        perform_mc_sweeps(equilibration, T_final, gaussian_move, sigma, 0, interleaved);

        // Step 3: Collect samples with comprehensive observables
        size_t n_samples = 1000;
        size_t n_measure = n_samples * acf.sampling_interval;
        cout << "Collecting " << n_samples << " independent samples..." << endl;

        vector<double> energies;
        vector<pair<SpinVector, SpinVector>> magnetizations;
        vector<MixedMeasurement> measurements;
        energies.reserve(n_samples);
        magnetizations.reserve(n_samples);
        measurements.reserve(n_samples);

        for (size_t i = 0; i < n_measure; i += acf.sampling_interval) {
            perform_mc_sweeps(acf.sampling_interval, T_final, gaussian_move, sigma, 0, interleaved);
            measurements.push_back(measure_all_observables());
            energies.push_back(measurements.back().energy);
            magnetizations.push_back({magnetization_SU2(), magnetization_SU3()});
        }

        cout << "Collected " << energies.size() << " samples" << endl;

        // Step 4: Compute comprehensive observables with binning analysis
        MixedThermodynamicObservables obs = compute_thermodynamic_observables(measurements, T_final);

        // Print and save results
        print_thermodynamic_observables(obs);
        save_thermodynamic_observables(out_dir, obs);

        // Also save raw time series for further analysis
        save_observables(out_dir, energies, magnetizations);

        // Save sublattice magnetization time series
        save_sublattice_magnetization_timeseries(out_dir, measurements);

        // Save autocorrelation function
        save_autocorrelation_results(out_dir, acf);
    }

// ---- MixedLattice::perform_mc_sweeps ----
    double MixedLattice::perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move,
                            double& sigma, size_t overrelaxation_rate,
                            bool interleaved) {
        // Interleave the species only when mixed couplings connect them.
        const bool use_interleaved = interleaved && (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        double acc_sum = 0.0;
        size_t n_local = 0;
        for (size_t i = 0; i < n_sweeps; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation_sweep(T, use_interleaved);
                if (i % overrelaxation_rate != 0) continue;
            }
            acc_sum += local_sweep(T, gaussian_move, sigma, use_interleaved);
            ++n_local;
        }
        return n_local > 0 ? acc_sum / double(n_local) : 0.0;
    }
