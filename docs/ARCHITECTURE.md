# Architecture

This document describes how the engine is organised, the physical conventions
every part of it follows, and where to add new models, algorithms and
observables. Behaviour changes relative to earlier versions are listed one by
one in [`MIGRATION.md`](MIGRATION.md).

## Layers

```
apps/        spin_solver (config -> system -> runner), runners_*.cpp, benchmarks, diagnostics
   |
models       Lattice            classical O(n) spins, any spin_dim (dynamics: 3 and 8)
             MixedLattice       SU(2) + SU(3) sublattices with mixed couplings (TmFeO3)
             PhononLattice      honeycomb spins + zone-centre phonons (NCTO)
   |
engines      mc/      annealing schedule, step-size control, parallel tempering, statistics
             dynamics/ time grids, drives, ODE method registry, grid integration,
                       geometric / Langevin spin integrators, dynamical structure factor
   |
core         RNG and seeding, SpinConfig, UnitCell / MixedUnitCell + builders,
             SU(3) coherent-state algebra, HDF5 writers (io/)
```

Engines know nothing about a particular model. A model plugs into them
through small adapters: `Lattice::DynamicsModel` for the geometric
integrators, the `mc::ReplicaModel` concept (`SpinLatticeReplica<L>`,
`MixedLatticeReplica`) for parallel tempering, and plain member functions
(`local_sweep`, `overrelaxation_sweep`, `total_energy`) for annealing.

## Physical conventions

* **Energy.** `E = Σ_i (−B_i·S_i + S_iᵀ A_i S_i) + Σ_⟨ij⟩ S_iᵀ J_ij S_j + Σ trilinear`.
  Every bond is declared once in the unit cell; the lattice adds the reverse
  bond with `Jᵀ`. On-site matrices are stored symmetrised; a bond that wraps
  onto its own site on a small lattice is folded into the on-site matrix.
* **Local field.** The lattice "field" functions (`get_local_field`,
  `get_local_field_flat`, `linear_field`) return the gradient `H_i = ∂E/∂S_i`,
  i.e. minus the effective field `B_eff = −∂E/∂S_i`.
* **Equation of motion.** `dS_i/dt = S_i × B_eff,i` (γ = ħ = 1). Damping is
  Gilbert damping, written either in Landau–Lifshitz form (`damping_form = landau_lifshitz`,
  prefactor 1) or in Gilbert form (`gilbert`, prefactor `1/(1+α²)`). The Langevin noise
  strength that makes `exp(−E/T)` stationary is `D = αT/(s(1+α²)g)`
  (García-Palacios & Lázaro 1998), see `dynamics/spin_integrators.h`.
* **SU(3) spins.** Stored as Gell-Mann expectations `n^a = ⟨ψ|λ^a|ψ⟩` of a
  qutrit pure state (`|n|² = 4/3`); every energy term is the coherent-state
  expectation `⟨ψ|H|ψ⟩`; the dynamics are the Lie–Poisson equation
  `dn_a/dt = c f_abc (∂E/∂n_b) n_c` with `c = UnitCell::poisson_bracket = 2`
  (`[λ_a, λ_b] = 2i f_abc λ_c`). `su3_legacy_convention = 1` restores the old
  encoding (`c = 1`). See `core/su3_coherent_state.h` (Zhang & Batista,
  PRB 104, 104409 (2021)).
* **Frames.** Spins are stored in the local (sublattice) frame of the unit
  cell; `sublattice_frames` map them to the global frame. Global-frame
  observables (magnetisation, S(q), S(q, ω)) apply the frames explicitly.
  The NCTO model stores spins in the crystal frame (`J_storage = Rᵀ J_cubic R`).

## Core

* **Random numbers** (`core/simple_linear_alg.{h,cpp}`). A 128-bit Lehmer
  generator per thread. `seed_lehman(seed)` expands the seed with splitmix64
  and bumps a generation counter, so every OpenMP worker reseeds its stream
  from `(master, thread id)` the next time it draws; `seed_lehman_from_rank`
  derives per-rank streams; `seed_lehman_thread(key)` gives per-replica streams
  inside parallel regions. `random_double_lehman` is uniform on [0, 1) with 53
  bits, `random_index_lehman` is unbiased (Lemire), `random_point_on_sphere`
  is uniform on S^{n−1}. No algorithm reseeds from the wall clock: the only
  entropy source is the `seed` config key (0 = `random_device` on rank 0,
  broadcast and written to `output_dir/seed.txt`).
* **Configuration** (`core/spin_config.{h,cpp}`). One table of typed keys
  (with aliases and deprecations) is the single entry point `SpinConfig::set`
  for config files, parameter sweeps and `to_file`, so every key round-trips.
  Hamiltonian parameters read by the builders are registered too: an unknown
  key is an error with a nearest-key suggestion, numbers and booleans are
  parsed strictly, and `validation_errors()` rejects inconsistent input (for
  every point of a sweep) before any work starts.
* **Spin files** (`io/spin_table.h`). One strict reader and full-precision
  writer for spin configurations, shared by the three lattice classes.
* **Unit cells** (`core/unitcell.h`, `src/core/unitcell_builders.cpp`).
  `UnitCell` holds positions, lattice vectors, frames, fields, on-site terms,
  bilinear and trilinear couplings and the SU(3) bracket coefficient;
  `validate()` (called by every lattice constructor) checks indices, shapes,
  finiteness and double-declared bonds. `bonds_at_distance(r)` and
  `neighbour_shells(n)` generate neighbour shells from the geometry, so
  builders need no hand-written offset tables;
  `tests/test_unitcell_geometry.cpp` checks every builder against them.

## Models

### Lattice

`lattice/lattice.h` with the heavy members in `src/core/lattice_{mc,md,pt,observables}.cpp`.

* **Kernel layer.** One templated kernel `accumulate_exchange_field(site,
  spin_of, H)` evaluates the bilinear + trilinear field of a site for any spin
  source (stored spins, a flat ODE state, a trial value); `linear_field`,
  `onsite_energy`, `site_energy_diff`, `get_local_field(_flat)` and
  `total_energy` are all built on it, so ΔE, fields and energies cannot drift
  apart. Bonds live in flat CSR arrays (`bi_flat_*`); twisted boundary
  conditions are folded into effective bond matrices (`refresh_twisted_bonds`
  after any change to `twist_matrices`, via `sync_twist_state`).
* **Monte Carlo.** `local_update ∈ {Metropolis, Gaussian, HeatBath}` selects
  the single-site kernel used by `local_sweep(T)`: uniform or adaptive
  small-angle Metropolis (one `propose_spin` kernel for the serial and
  coloured sweeps), or the exact heat bath (Miyatake et al. 1986;
  `u = 1 + log1p(ξ expm1(−2b))/b` for O(3) spins). 8-component spins of a
  Gell-Mann-bracket cell are qutrit pure states and are sampled on CP² with
  the Fubini–Study measure through the shared kernels of `core/su3_mc.h`
  (Haar draws, small moves, exact heat bath, phase-randomising
  overrelaxation, exact local ground state). Overrelaxation
  reflects about the field that does not depend on the spin itself and is
  Metropolis-corrected on sites with anisotropic on-site terms (exact at
  T > 0). Lattices with ≥ `parallel_sweep_min_sites` sites use the coloured
  OpenMP sweeps (a greedy graph colouring of the bond graph makes every colour
  class independent). Wolff and Swendsen–Wang cluster moves use the embedded
  Ising variables with an exact Metropolis filter for terms that do not embed.
  The T = 0 quench (`deterministic_sweep`) minimises each site's quadratic
  energy on the sphere exactly (trust-region subproblem), which makes it a
  monotone block-coordinate descent.
* **Dynamics.** One right-hand side, `landau_lifshitz_rhs(x, dxdt, t, drive)`,
  with the drive passed explicitly as an immutable `DriveSchedule`; one
  engine, `integrate_on_grid`, that samples every trajectory on the exact grid
  `t_k = t0 + k dt` whatever the method; one definition of the magnetisation
  channels (`measure_magnetizations`); one delay-scan implementation,
  `run_pump_probe_scan`, behind the OpenMP and MPI pump-probe / 2DCS drivers.

### MixedLattice and PhononLattice

`MixedLattice` (`lattice/mixed_lattice.h`, `src/core/mixed_lattice_*.cpp`)
holds an SU(2) and an SU(3) sublattice with mixed bilinear, trilinear and
pulse-modulated couplings; its dynamics use the generic
`dynamics/grid_integrate.h`, and its Monte Carlo follows the same policies as
Lattice (local-update kernels, exact overrelaxation, CP² sampling of the SU(3)
sites via `core/su3_mc.h`, allocation-free sweeps). `PhononLattice` (`lattice/phonon_lattice.h`,
`src/core/phonon_lattice.cpp`) couples honeycomb spins to zone-centre phonon
coordinates (magnetoelastic tensors in `ncto_me_tensors.h`, parameters in
`phonon_config.h`); with `mc_sample_lattice = 1` Monte Carlo samples the
lattice coordinates too, and they travel with the spins in replica exchange.

## Monte Carlo engines (`mc/`)

* **Simulated annealing.** `mc::annealing_schedule(T_start, T_end, rate)`
  (validated, ends exactly at `T_end`); `mc::StepSizeController`
  (Robbins–Monro on log σ toward 45 % acceptance, adapting only while
  equilibrating); a converged T = 0 descent at the end; final measurements in
  whole decorrelation intervals.
* **Population annealing** (`mc/population_annealing.h`; `simulation_mode =
  population_annealing`). A population of R replicas is annealed from
  infinite temperature with Boltzmann resampling at every step (Hukushima &
  Iba 2003; Machta 2010; Wang, Machta & Katzgraber 2015), so every
  temperature of the schedule is an equilibrium ensemble and
  ln Z(β) − ln Z(0) is estimated along the way. Systematic resampling keeps R
  fixed; errors come from a delete-m_j jackknife over families (replicas
  sharing an initial ancestor) and ρ_t, the family entropy and the ESS of
  every step are reported. Schedules: linear in β, geometric in T, or
  adaptive (the largest step keeping the ESS above a target). The population
  is split over MPI ranks and OpenMP threads; replica k at step i draws from a
  stream keyed on (seed, i, k), so results do not depend on the layout.
* **Parallel tempering** (`mc/parallel_tempering.h`). One replica per MPI
  rank, deterministic even/odd (non-reversible) exchange rounds (Okabe et al.
  2001; Syed et al. 2022) decided identically on both partners from a shared
  counter-based uniform; replica labels, directions and round-trip phases
  travel with the configurations, so round trips, f(T) and per-edge
  acceptance are measured. Ladder tuning with the chain itself: `nrpt`
  (equal rejection via a monotone interpolation of the communication barrier)
  or `katzgraber` (flow feedback). Inputs and I/O failures are reduced over all
  ranks so no rank is left waiting in a collective.
* **Statistics** (`mc/statistics.h`). Wolff's Gamma method (FFT
  autocovariance, automatic window), blocked jackknife for nonlinear
  estimators (c, χ, Binder U4), and the Flyvbjerg–Petersen blocking table.
  Specific heat is `Var(E_total)/(N T²)`.

## Dynamics engines (`dynamics/`)

| Header | Content |
|---|---|
| `time_grid.h` | integer-indexed sampling grids; `whole_steps` tolerant to round-off |
| `drive.h` | `Pulse`, `DriveSchedule` (Gaussian × carrier pulses, per-sublattice polarisation) |
| `ode_method.h` | the one registry of integrator names (`parse_ode_method` throws on typos) |
| `grid_integrate.h` | generic exact-grid integration (dense-output dopri5, integrate_times for the others) |
| `spin_integrators.h` | spherical midpoint (McLachlan–Modin–Verdier 2014), Depondt–Mertens, Suzuki–Trotter colour splitting (2nd and 4th order); stochastic LLG |
| `structure_factor.h` | windowed, zero-padded estimator of S^{ab}(q, ω) with an exact frequency sum rule |

Method families: fixed-step explicit RK, adaptive (dopri5 and Bulirsch–Stoer
with dense output, Cash–Karp and Fehlberg 7(8) stepping onto the samples), and
geometric (norm-preserving; the only family that supports Langevin noise).

## Applications and I/O

`spin_solver <config>` parses and validates the config on rank 0 and
broadcasts it, seeds the RNG once, writes `run_info.txt` (seed, MPI size,
threads, git revision, build flags and the full resolved config, which reruns
the job), and calls `run_simulation(config, comm)` (`src/apps/system_factory.cpp`):
the one construction path (`make_unit_cell`, `make_lattice`,
`make_mixed_lattice`, `make_ncto_lattice`) and mode dispatch, used by both
`main` and the parameter sweep. Runners (`runners_lattice.cpp`,
`runners_mixed.cpp`, `runners_phonon.cpp`, `runners_population_annealing.cpp`,
`runners_parameter_sweep.cpp`) take a communicator and never touch
`MPI_COMM_WORLD` (sweep points run on `MPI_COMM_SELF` or on equal rank
groups); every rank writes the trials it ran and rank 0 gathers
`trial_summary.txt`; an exception on any rank prints `[rank r] error: …` and
aborts the job. HDF5 writers (`io/hdf5_io.h`) produce one structured file per
trajectory / delay scan / PT rank; PT additionally writes `pt_summary.txt` and
`parallel_tempering_aggregated.h5`, population annealing `pa_summary.txt`.

Observables: the real-space correlation accumulator
(`lattice/correlation_accumulator.h`) computes S(q) and dimer correlations on
the commensurate grid by FFT (`core/fft.h`, any length) in O(N log N) per
sample, from global-frame spins with all components kept, with jackknife
error bars and an MPI reduction.

Parallelism: MPI distributes replicas (PT), trials (SA, MD) and delay points
(pump-probe / 2DCS, dynamic scheduling); OpenMP parallelises coloured sweeps
and the RHS site loop (thresholds avoid fork/join overhead on small systems).
The CUDA path implements the plain Landau–Lifshitz RHS with fixed-step RK4
or error-controlled embedded pairs (Dormand–Prince 5(4), Cash–Karp 5(4),
Fehlberg 7(8); tableaux in `gpu/ode/rk_tableaux.h`, shared with a CPU order
test); models it does not implement (damping, Langevin noise, trilinear
terms, twisted boundaries, the legacy SU(3) bracket) fall back to the CPU with
a warning (`Lattice::gpu_supports_model`). `gpu/device_select.h` binds each
process to a device by node-local rank, or reports once that it runs on the
CPU.

## Testing

`ctest` runs physics tests that compare kernels with results known
independently of the code (label `physics`):

| Test | References |
|---|---|
| `test_rng` | reproducibility, stream independence, moments of uniform / normal / sphere samples |
| `test_mc_statistics` | Gamma method and blocking on AR(1) (exact τ_int) |
| `test_mc_exact` | free spins (Langevin function), 1D Heisenberg ring u(T), c(T) (Fisher), 2- and 3-site clusters vs quadrature, S⁷ Bessel ratio, twisted BCs, cluster moves, heat bath, overrelaxation |
| `test_md_exact`, `test_md_drivers` | precession, conservation, magnon dispersion, integrator orders, damping, Langevin FDT, exact grids, 2DCS bookkeeping, DSSF sum rule |
| `test_sa_ground_state` | triangular (−3/2 J) and pyrochlore (−J) ground states |
| `test_population_annealing` | 1D Heisenberg ring: ln Z, u, c, ⟨m²⟩ at every temperature; bitwise layout independence (threads, 3 ranks vs 1); adaptive schedule |
| `test_pt_mpi`, `test_pt_ladder` | PT on 4 ranks vs Fisher / Langevin / Bessel results, DEO bookkeeping, round trips, reproducibility, ladder updates |
| `test_mixed_md`, `test_mixed_pump_probe` | SU(3) convention, exact grids, pump-probe on 1 and 4 ranks |
| `test_phonon_dynamics`, `test_phonon_mpi` | spin–phonon dynamics, Langevin bath, MPI 2DCS, replica exchange of the lattice sector |
| `test_mixed_mc_exact`, `test_mixed_mc_kernels` | MixedLattice samplers vs simplex / Beta(1,2) / quadrature references on CP² and S²×CP², exact local ΔE, overrelaxation energy conservation, Casimirs, SA ground states |
| `test_unitcell_geometry` | every builder's bonds vs geometric neighbour shells, validation |
| `test_observables`, `test_observables_mpi` | FFT vs direct sums, Bragg peaks, Néel and spiral states, sum rules, dimer coverings, jackknife, MPI reduction |
| `test_gpu_tableaux` | convergence orders of the GPU Runge–Kutta tables (run on the CPU) |
| `test_config`, `test_config_spin_io` | strict parsing, unknown keys, round trip of every key, every example config, sweep grids, spin-file failure modes and round trips |

`tests/smoke/run_smoke.sh <spin_solver>` (CTest `smoke_spin_solver`) runs
end-to-end configurations of every mode (SA, population annealing, PT, tuned
PT, MD with four integrators, pump-probe, 2DCS, sweeps, TmFeO3, NCTO), MPI
driver cases and expected failures. CI (`.github/workflows/ci.yml`) runs the
whole suite on a portable Release build and the physics tests under
AddressSanitizer + UndefinedBehaviorSanitizer.

## Extending

* **A new model on an existing lattice class**: add a builder
  `UnitCell build_<name>(const SpinConfig&)` to `unitcell_builders.cpp`, read
  its parameters with `config.get_param`, generate shells with
  `bonds_at_distance`, register the system name in `spin_config`, and add its
  bond geometry to `tests/test_unitcell_geometry.cpp`.
* **A new integrator**: implement it for the `SpinIntegrator<Model>` interface
  in `dynamics/spin_integrators.h` (geometric) or as an odeint stepper, add the
  name to `dynamics/ode_method.h`, and add an order / conservation test to
  `tests/test_md_exact.cpp`.
* **A new sampler for parallel tempering / population annealing**: write a
  type satisfying `mc::ReplicaModel` (`mc_step`, `energy`,
  `state_size`/`pack_state`/`unpack_state`, `measure`, `uses_step_size`) and
  call `mc::run_parallel_tempering`; add `randomize()` and scalar
  `observables()` (`mc::PopulationWorker`) for `mc::run_population_annealing`.
* **A new observable**: compute it from global-frame spins, give it an error
  bar through `mc::gamma_method` / `mc::blocked_jackknife`, and test it on a
  configuration where its value is known.
