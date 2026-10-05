# Migration notes

## Parallel tempering: one engine, measured round trips, NRPT ladder tuning

The three hand-copied PT drivers (Lattice, MixedLattice, `mc::parallel_tempering`
for PhononLattice), five ladder optimisers and the duplicated statistics were
replaced by `include/classical_spin/mc/parallel_tempering.h` and
`include/classical_spin/mc/statistics.h`.

Behaviour changes (all fixes of verified defects; there is no legacy switch
because the old outputs were wrong):

- **Specific heat** is `Var(E)/(N T^2)` with a blocked-jackknife error whose
  blocks are ~8 tau_int long (was 100 fixed blocks, error underestimated up to
  2.4x for correlated samples). The old `dC_V = sigma_E/(T^2 N^2)` pseudo-error
  is gone.
- **Exchange schedule**: exchanges happen every `pt_exchange_frequency` MC
  steps. The Bittner schedule that silently replaced it by
  `max(sweeps_per_temp)` (>= 10, often hundreds of sweeps) was removed together
  with the `sweeps_per_temp` parameter of `parallel_tempering(...)`.
- **Ladder tuning**: `pt_temperature_optimizer = nrpt` (new default, Syed et al.
  2022) or `katzgraber` (now the actual flow feedback of Katzgraber et al. 2006,
  using f(T) measured from replica labels; the old "Katzgraber" rule
  Delta beta ∝ A did not equalise anything). `gradient` was removed: its
  gradient missed the pathwise term, it used a stale energy after a
  first-parity swap, its stopping rule compared a variance with a
  mean-deviation tolerance, and it cited the paper with the wrong year. The name
  still parses and maps to `nrpt` with a warning. `pt_target_acceptance` is no
  longer used (the mean exchange acceptance is fixed by R and the temperature
  range; the tuner reports the barrier Lambda and the optimal R ~ 2 Lambda + 1).
  The tuned ladder is returned cold -> hot with all per-temperature arrays in
  the same order (`mc::LadderTuningResult`, alias `OptimizedTempGridResult`).
- **Tuned replicas are kept**: the runners no longer call `init_random()` after
  tuning, so production starts from configurations equilibrated near their
  temperatures (later trials still re-initialise).
- **One MC step** means the same thing in tuning and production (as in
  `perform_mc_sweeps`: with `overrelaxation_rate = k > 0`, one overrelaxation
  sweep per step plus a local sweep every k-th step). The optimisers used the
  opposite convention before.
- **Proposal width**: the hard-coded sigma = 1000 (effectively uniform moves)
  is replaced by `local_update` / `gaussian_move`: adaptive Gaussian proposals
  have their width tuned per temperature while equilibrating and frozen while
  measuring; `local_update = heat_bath` uses the heat bath.
- **Pilot phase removed**: up to 5000 extra sweeps whose tau estimate was only
  printed; tau_int(E) is now computed from the measurement series (Gamma
  method) and written to the outputs.
- **Validation**: wrong temperature count, non-positive or non-increasing
  temperatures (rank 0 must be the coldest), ladders that differ between ranks,
  `probe_rate = 0` or fewer measurement steps than `probe_rate` make every rank
  throw `std::invalid_argument`. MPI is never initialised by the library
  (Lattice/MixedLattice used to call `MPI_Init`). An I/O error on any rank makes
  every rank throw `std::runtime_error` after all ranks have attempted their
  output (previously one rank aborted while the others hung in a barrier).
- **PhononLattice**: the phonon coordinates are not sampled by the spin MC
  kernels and are not exchanged, so PT requires them to be identical on all
  ranks (checked; the swap acceptance would otherwise be wrong). Its MC RNG is
  seeded from the process seed and the rank instead of the wall clock.

New config keys: `pt_equilibration_steps`, `pt_measurement_steps` (0 = use
`annealing_steps`, the previous behaviour for both), `pt_optimization_tolerance`
(relative ladder movement that ends tuning; replaces the hard-coded 0.05).
`pt_sweeps_per_exchange`, which was parsed but never read, is now a deprecated
alias of `pt_exchange_frequency`.

Outputs per trial directory: `pt_summary.txt` (new, plain text) and
`parallel_tempering_aggregated.h5` (keeps `/temperature_scan/temperature`,
`specific_heat`, `specific_heat_error`; adds energy, tau_int, local acceptance,
step size, f(T), `/exchange/*` edge statistics and round trips, and
`/order_parameters/<name>/*` with |m|, m^2, susceptibility and Binder cumulant).
`rank_k/parallel_tempering_data.h5` keeps its layout; its `magnetization`
observable is now the global-frame magnetisation (the old value averaged
local-frame sublattice vectors).

API: `Lattice/MixedLattice/PhononLattice::parallel_tempering` return
`mc::PTResult` and lost the trailing `sweeps_per_temp` argument;
`generate_optimized_temperature_grid[_mpi]`, `optimize_temperature_ladder_roundtrip`,
`attempt_replica_exchange`, `estimate_sampling_interval` and
`gather_and_save_statistics[_comprehensive]` were removed in favour of
`tune_temperature_ladder(mc::LadderTuningOptions, overrelaxation_rate,
gaussian_move, comm)` and the engine. `mc::compute_autocorrelation` now uses
the Gamma method (FFT autocovariance, automatic window, no 1000-lag cap).
