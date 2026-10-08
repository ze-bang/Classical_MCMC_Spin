# Classical MCMC Spin

A C++20 / MPI / OpenMP (optional CUDA) engine for classical spin models:
Monte Carlo sampling (simulated annealing, population annealing, parallel
tempering), spin dynamics (Landau–Lifshitz–Gilbert, Langevin), and nonlinear
pump-probe / 2D coherent spectroscopy, for O(3) spins and SU(3) coherent
states on arbitrary lattices.

Every sampler and integrator is checked against results known independently
of the code (closed-form statistical mechanics, quadrature, analytic
precession and spin waves); see [Testing](#testing).

## Methods

**Monte Carlo**

| | |
|---|---|
| Local updates | uniform or adaptive-width Metropolis, exact heat bath (Miyatake et al. 1986), overrelaxation about the spin-independent field (Metropolis-corrected for anisotropic sites), Wolff / Swendsen–Wang embedded-Ising clusters with an exact filter; coloured OpenMP sweeps on large lattices |
| Simulated annealing | validated geometric schedule ending exactly at `T_end`, Robbins–Monro proposal-width control, converged T = 0 descent (exact single-site minimisation) |
| Population annealing | annealing of R replicas with Boltzmann resampling (Hukushima & Iba 2003; Machta 2010; Wang, Machta & Katzgraber 2015): equilibrium averages and the free energy at every temperature, family-jackknife errors, linear / geometric / ESS-adaptive schedules, MPI × OpenMP with layout-independent results |
| Parallel tempering | deterministic even/odd (non-reversible) exchanges (Okabe et al. 2001; Syed et al. 2022), measured round trips and f(T), ladder tuning by equal rejection (`nrpt`) or flow feedback (`katzgraber`) |
| Statistics | Wolff's Γ-method with automatic windowing, blocked jackknife for c, χ, Binder U4 |

**Spin dynamics**

| | |
|---|---|
| Integrators | geometric: spherical midpoint (McLachlan, Modin & Verdier 2014), Depondt–Mertens, Suzuki–Trotter colour splitting (2nd / 4th order); adaptive dopri5 with dense output, Cash–Karp, Fehlberg 7(8), Bulirsch–Stoer; fixed-step RK |
| Damping and noise | Gilbert damping in Landau–Lifshitz or Gilbert form; stochastic LLG with the fluctuation–dissipation noise strength (García-Palacios & Lázaro 1998) |
| Drives | Gaussian × carrier pulses with per-sublattice polarisation, passed explicitly to the equations of motion |
| Spectroscopy | pump-probe and 2DCS delay scans on exact time grids (OpenMP / MPI dynamic scheduling), dynamical structure factor S(q, ω) from thermal samples |
| SU(3) | Gell-Mann coherent states, E = ⟨ψ|H|ψ⟩, Lie–Poisson dynamics (Zhang & Batista 2021); Monte Carlo on CP² with the Fubini–Study measure (uniform, small-move, exact heat-bath and phase-randomising overrelaxation moves) |

**Models**: honeycomb (Kitaev–Γ–Γ′, BCAO), pyrochlore (XXZ, non-Kramers with
J2/J3), triangular (anisotropic), TmFeO3 (Fe SU(2) + Tm SU(3) with mixed
couplings), NCTO (honeycomb spins + zone-centre phonons), and any unit cell
built with `UnitCell`.

## Building

Requirements: CMake ≥ 3.18, a C++20 compiler (GCC ≥ 10, Clang ≥ 12), MPI,
Eigen ≥ 3.3, Boost (odeint), HDF5 with C++ bindings; optionally CUDA ≥ 11.

```bash
sudo apt-get install -y build-essential cmake ninja-build libopenmpi-dev openmpi-bin \
    libeigen3-dev libboost-all-dev libhdf5-dev
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -j4          # physics validation + MPI + end-to-end smoke tests
```

| CMake option | Default | Meaning |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` for production; `RelWithDebInfo` / `Debug` for development |
| `CLASSICAL_SPIN_NATIVE` | `ON` | tune Release builds for the build host (`-march=native`); `OFF` for portable binaries (clusters with mixed nodes, CI) |
| `CLASSICAL_SPIN_SANITIZE` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer (checked builds) |
| `CLASSICAL_SPIN_WARNINGS` | `ON` | `-Wall -Wextra` on the project's own code |
| `CLASSICAL_SPIN_WARNINGS_AS_ERRORS` | `OFF` | treat warnings as errors |
| `ENABLE_CUDA` | `OFF` | CUDA spin-dynamics kernels (models the GPU path does not implement fall back to the CPU with a warning) |
| `CMAKE_CUDA_ARCHITECTURES` | detected | GPU architectures to compile for |
| `BUILD_TESTING` | `ON` | build the test suite (`ctest`) |

Continuous integration (`.github/workflows/ci.yml`) builds a portable
Release with the full test suite and a Debug ASan/UBSan build running the
physics tests.

## Running

```bash
./build/spin_solver example_configs/Kitaev/sa_kitaev.param            # simulated annealing
mpirun -np 8 ./build/spin_solver example_configs/Kitaev/pt_kitaev.param  # one PT replica per rank
mpirun -np 4 ./build/spin_solver my_population_annealing.param        # population over all ranks
```

A configuration is a `key = value` file. A minimal population-annealing run:

```ini
system = honeycomb_kitaev
lattice_size = 12, 12, 1
K = -1.0
Gamma = 0.25
simulation_mode = population_annealing
T_start = 2.0
T_end = 0.02
pa_population = 4096
pa_sweeps = 10
pa_schedule = adaptive      # or linear_beta (pa_temperatures points), geometric
pa_target_ess = 0.9
seed = 12345                # 0: random, written to output_dir/seed.txt
output_dir = out_pa
```

| Mode (`simulation_mode`) | Main keys | Main outputs |
|---|---|---|
| `simulated_annealing` | `T_start`, `T_end`, `cooling_rate`, `annealing_steps`, `local_update`, `overrelaxation_rate`, `num_trials`, `T_zero` | `sample_k/` spins, energies |
| `population_annealing` | `T_start`, `T_end`, `pa_population`, `pa_sweeps`, `pa_schedule`, `pa_temperatures`, `pa_target_ess` | `pa_summary.txt` (E, C, ln Z, errors, diagnostics per T), `pa_best_spins.txt` |
| `parallel_tempering` | `T_start`, `T_end`, `pt_optimize_temperatures`, `pt_temperature_optimizer`, `pt_exchange_frequency`, `pt_equilibration_steps`, `pt_measurement_steps` | `pt_summary.txt`, `parallel_tempering_aggregated.h5`, `rank_k/` |
| `molecular_dynamics` | `md_integrator`, `md_timestep`, `md_time_end`, `alpha_gilbert`, `damping_form`, `langevin_temperature`, `dssf_*` | trajectory / `dssf.h5` per sample |
| `pump_probe`, `2dcs` | `pump_*`, `probe_*`, `tau_start`, `tau_end`, `tau_step` | `pump_probe_spectroscopy.h5` |
| `parameter_sweep` | `sweep_parameters`, `sweep_starts`, `sweep_ends`, `sweep_steps`, `sweep_base_simulation` | one directory per point |

More examples: [`example_configs/`](example_configs/README.md).

**Reproducibility.** One `seed` determines every random number of a run (it
is written to `output_dir/seed.txt` when drawn at random); there is no
wall-clock seeding anywhere. Population annealing gives bitwise-identical
results for any number of ranks and threads; the other modes are
reproducible for a fixed seed, MPI layout and thread count.

## Testing

`ctest` runs physics tests that compare kernels with exact results:
free spins (Langevin function), the 1D Heisenberg ring (Fisher's u(T), c(T)
and free energy), 2- and 3-site clusters against quadrature, free SU(3)
vectors (Bessel ratios), triangular and pyrochlore ground states, spin
precession, conservation laws, magnon dispersion, integrator orders,
fluctuation–dissipation, parallel tempering and population annealing on
several MPI ranks, neighbour shells of every unit-cell builder, and
end-to-end runs of every simulation mode.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layers, physical conventions, engines, extension points.
- [`docs/MIGRATION.md`](docs/MIGRATION.md) — every behaviour change, with the switch that restores the old behaviour where one exists.

## License

MIT — see [LICENSE](LICENSE).
