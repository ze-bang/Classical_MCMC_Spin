# Migration notes

One section per behaviour change. Each lists what changed, why, and how to
recover the old behaviour where that is meaningful.

## Lattice dynamics: exact output grid (pulse-window chunking removed)

**What changed.** `Lattice::single_pulse_drive`, `double_pulse_drive`,
`pump_probe_spectroscopy(_mpi)` and `molecular_dynamics` now integrate every
trajectory once and sample it at `t_k = T_start + k * step` computed from the
integer `k`. Error-controlled methods (`dopri5`, `bulirsch_stoer`) use dense
output; `rk5`/`rk78` step exactly onto each sample. Trajectories therefore have
`floor((T_end - T_start)/step) + 1` samples (the old code usually dropped the
final sample), bitwise identical time stamps for every delay, and no seam lag.
The `pulse_window_chunking` argument / config key is ignored by `Lattice`
(MixedLattice and PhononLattice still read it).

**Why.** Segments were integrated with `odeint::integrate_const`, whose
end-of-interval test uses an absolute 2.2e-16 epsilon: whenever a segment end
`T_start + k T_step` and odeint's `t0 + j T_step` differed by ~1e-14, the last
step was skipped and the next segment restarted from a stale state, with a
delay-dependent lag of order `|dM/dt| T_step` that leaked into the 2DCS signal
(measured: up to 3.7x the true `M_NL`).

**Recover the old behaviour.** Not supported (it was wrong).

## Lattice dynamics: drive polarisation uses F^T

**What changed.** A pulse direction `B` given in the global frame now acts on
the spin variables of sublattice `a` as `F_a^T B` (it was `F_a B`).

**Why.** Every observable defines `S_global = F_a S`, so the Zeeman energy is
`-B . S_global = -(F_a^T B) . S`. For symmetric frames (identity, diagonal sign
frames) nothing changes; for rotated frames (e.g. the Kitaev frame of
`honeycomb_kitaev`) the old code drove the wrong components with the wrong
amplitude. MixedLattice already used `F^T`.

**Recover the old behaviour.** Pass `F_a F_a B` (i.e. pre-rotate the direction)
if a legacy result must be reproduced.

## Lattice 2DCS: corrected M1 reuse (W1), relative stationarity tolerance

**What changed.** With `reuse_m0_for_m1`, M1(τ) is synthesised by time
translation only when the configuration is stationary, τ is a multiple of
`md_timestep`, and the probe window starts after `md_time_start`
(`md_time_start <= τ - 9 pulse_width`); the reference is integrated from before
the pulse (and past `md_time_end` for τ < 0). All other delays are integrated.
The synthesised baseline uses the same staggered/global channels as every
integrated trajectory. `stationarity_tol` is, for Lattice, the relative torque
`max |S x H| / max(|S||H|)` (it was an absolute `max |dS/dt|`). The file gains
`/tau_scan/m1_synthesized` (1 where M1 was synthesised).

**Why.** The old reuse shifted M0 itself: with the default `md_time_start = 0`
the pump at t = 0 is cut in half, so every synthesised M1 was the response to
half a pulse; negative τ repeated the last M0 sample; τ was rounded to the grid;
and the antiferro baseline before the probe used a different definition (flat
index parity, local frame), which produced a step at t = τ.

## Lattice dynamics: integrator names validated; rosenbrock4/implicit_euler removed

**What changed.** An unknown `md_integrator` is a configuration error
(`SpinConfig::validate`) and an `std::invalid_argument` at the API, listing the
valid names; it used to run `dopri5` silently. `rosenbrock4`/`rb4` and
`implicit_euler`/`ie` were removed (dense N x N finite-difference Jacobian);
use `spherical_midpoint`. `md_save_interval = 0`, non-positive steps and
inconsistent delay grids are rejected. The delay count is
`floor((tau_end - tau_start)/tau_step + round-off) + 1` (0 -> 0.3 step 0.1
gives 4 delays, not 3).

## Lattice dynamics runners: trials, probe, damping, output on every rank

**What changed.**
- A configuration loaded with `initial_spin_config` is used in every trial (MD,
  pump-probe, both 2DCS modes); it used to be replaced by random spins from the
  second trial on. For pump-probe/2DCS with `T_zero = true` it is relaxed by the
  T = 0 quench first.
- `alpha_gilbert` and `langevin_temperature` are honoured by the MD and
  pump-probe runners (only 2DCS read `alpha_gilbert` before; finite-temperature
  Langevin MD needs `spherical_midpoint` or `depondt`).
- `pump_probe` applies the probe pulse (`probe_*` keys; it was ignored). Set
  `probe_amplitude = 0` for the old pump-only run. Every rank writes its own
  trials' `pump_probe_trajectory.txt` (only rank 0 wrote before), with 17
  significant digits and, with a probe, the pump-only and probe-only references.
- The MPI delay scan schedules delays dynamically over all ranks (rank 0 used to
  idle), streams results to rank 0, fails loudly on any rank instead of padding
  short trajectories with zeros, and takes a communicator argument.

## Lattice dynamics: SU(3) damping

**What changed.** `alpha_gilbert > 0` now damps `spin_dim = 8` lattices with
the double-bracket term `-(α/|S|) S × (H × S)` (structure-constant product); it
was silently ignored. Spin dimensions other than 3 and 8 are rejected by the
dynamics drivers (there is no Lie-algebra cross product for them).

## MD trajectory file: diagnostics

**What changed.** `trajectory.h5` gains `/trajectory/energy_density`,
`/trajectory/max_norm_error` and the attributes `/metadata/dt_save`,
`alpha_gilbert`, `langevin_temperature`, `integrator`; `final_spins.txt` holds
the final configuration.
