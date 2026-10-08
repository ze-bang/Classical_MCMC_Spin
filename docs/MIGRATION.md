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

## Lattice dynamics: damping_form (new key)

**What changed.** New config key `damping_form` = `landau_lifshitz` (default,
unchanged behaviour: `dS/dt = S x B - (α/s) S x (S x B)`) or `gilbert` (the same
divided by `1 + α²`, i.e. the Gilbert equation solved for `dS/dt`). The Langevin
noise strength follows the form, `D = αT/(s(1 + α²))` resp. `D = αT/s`, so both
sample the Gibbs distribution.

## Lattice MD: dynamical structure factor mode (new keys)

**What changed.** With `dssf_samples > 0`, `simulation_mode = molecular_dynamics`
computes the classical dynamical structure factor S^{ab}(q, ω) (thermal
sampling by stochastic LLG, energy-conserving trajectories, windowed and
zero-padded FFT, sample averages with errors) and writes `sample_<trial>/dssf.h5`
instead of a trajectory file. Without the new keys nothing changes. This replaces
the offline Python estimate, which summed exp(iωt) over non-uniform samples
without weights or window.

## SU(3) (Tm) convention: E = <psi|H|psi> and dn/dt = 2 f (dE/dn) n

**What changed.** For SU(3) spins stored as Gell-Mann expectations
n^a = <psi|lambda^a|psi> (Tr lambda^a lambda^b = 2 delta^ab):

1. Every term of the classical energy is now the coherent-state expectation of
   the quantum operator it stands for. The TmFeO3 crystal field
   diag(0, e1, e2) is

       diag(0, e1, e2) = (e1 + e2)/3 - (e1/2) lambda_3 - ((2 e2 - e1)/(2 sqrt3)) lambda_8,

   so the builder (`apply_tmfeo3_tm_sector`) now puts
   `B_3 = e1/2, B_8 = (2 e2 - e1)/(2 sqrt3)` into the Tm field (it used twice
   these values). Zeeman (`J = mu <lambda>`), Fe-Tm (`K^-`, `kappaE/B`, `W`),
   Tm-Tm (`Jtm_*`) and drive couplings are unchanged: they were already defined
   with their quantum weight.
2. The equation of motion is the Lie-Poisson equation of the Gell-Mann
   algebra. From `[lambda_a, lambda_b] = 2 i f_abc lambda_c` and the
   Heisenberg equation with the mean-field H = sum_b (dE/dn_b) lambda_b,

       dn_a/dt = <i [H, lambda_a]> = sum_b (dE/dn_b) i <[lambda_b, lambda_a]>
               = sum_b (dE/dn_b) i (2 i f_bac) n_c = 2 f_abc (dE/dn_b) n_c.

   The code used `f_abc` without the 2 (`cross_prod_SU3_flat` unscaled). The
   coefficient is now part of the unit cell (`UnitCell::poisson_bracket`: 1 for
   SU(2) spins, 2 for SU(3)) and is applied in MixedLattice, in the Lattice
   SU(3) path (TMFEO3_TM) and in the CUDA kernel.

**Why.** The old pair (doubled CEF, bracket f) put the bare CEF lines at the
right frequencies, but every other lambda-linear coupling (Zeeman, Fe-Tm
exchange, pulses) acted on Tm with HALF its torque, Monte Carlo weighted the
CEF twice relative to them, and the coupled Fe-Tm dynamics corresponded to no
quantum Hamiltonian (Fe-Tm hybridisation gaps off by sqrt 2, Tm Zeeman/drive
response halved). With the new convention the classical dynamics of a Fe-Tm
product state is exactly the time-dependent mean-field Schrodinger evolution
of the quantum model (tested in `tests/test_mixed_md.cpp`). Doubling the
couplings instead would not work: it doubles the exchange field on the
correctly normalised Fe spins and the Tm-Tm energies.

**What you see.** CEF frequencies are unchanged. Tm responses to fields,
exchange and pulses are stronger by up to 2x; equilibrium Tm polarisations
and SA ground states change because the CEF no longer dominates twice over;
`h_tm_a` (a constant lambda_a field) now has its quantum weight in both MC and
MD (it had 1x in MC and 0.5x in MD). Rates `gamma_su3*` are physical rates and
need no change.

**Legacy switch.** `su3_legacy_convention = 1` restores the old CEF encoding
and the old bracket (`poisson_bracket = 1`) bit for bit. The GPU path refuses
it (`check_gpu_supported`).

Also corrected: the documented pure-state Casimir is |n|^2 = 4/3 (not 8/3),
and `d_abc n^a n^b n^c = 8/9` (`su3::casimir2/3`). `su3::spherical_midpoint_step`
was a first-order theta = 1/4 scheme; it is now the unitary, second-order
implicit midpoint (`su3::implicit_midpoint_step`, Cayley form).

## MixedLattice dynamics: exact time grid, no chunk seams

- Pulse drives (`single_pulse_drive`, `double_pulse_drive`), pump-probe and MD
  sample on the exact grid `t_k = T_start + k dt`, `k = 0..n` (`t_n <= T_end`),
  computed from integers. Trajectories now have **n + 1 samples and include
  t = T_start** (the old observer skipped the first point and the pulse-window
  chunking dropped one step at seams, leaving the state one step behind its
  label and giving M0/M1/M01 different lengths).
- dopri5 uses dense output: steps are chosen by the error controller (capped
  at `min(T_step, width/4, period/4)` inside pulse windows only), samples are
  interpolated. Expect fewer RHS calls; results agree with the old ones to the
  integration tolerance except where the seam bug corrupted them.
  `pulse_window_chunking` is accepted and ignored.
- `molecular_dynamics` writes on the uniform grid `t_k = k save_interval dt`
  (it used to save every `save_interval`-th ACCEPTED adaptive step, at
  irregular times), and adds a `/diagnostics` group (energy per site, drifts
  of |S_i| and of both SU(3) Casimirs). A non-finite state throws.
- Unknown integrator names throw `std::invalid_argument` (was: silent dopri5).
  `rosenbrock4`/`implicit_euler` were removed (dense N x N finite-difference
  Jacobians for a non-stiff problem).
- Delay grids count `tau_start + i tau_step` up to `tau_end` with round-off
  tolerance (0 -> 0.3 step 0.1 now gives 4 delays, not 3); a zero step or a
  step pointing away from `tau_end` throws.

## MixedLattice pump-probe: drive and Hamiltonian fixes

- **No phantom second pulse.** `single_pulse_drive` used to leave pulse slot 2
  at t = 0 with zero direction but full amplitude, and its envelope gated the
  field-assisted Fe-Tm exchange (`kappaE/kappaB`): M0 felt 2 f(t), M1 felt
  f(t - tau) + f(t). Only active pulses (`n_active_pulses`) now drive or gate.
- **One Hamiltonian.** The Bloch-damping equilibrium was silently subtracted
  from the SU(3) leg of the Fe-Fe-Tm trilinear in the MD field only, so an
  annealed state was not stationary once `gamma_su3` was set. The equilibrium
  is now only a relaxation target. If the subtraction is wanted, set
  `tm_trilinear_reference = 1`: the runner folds `T(S_i, S_j, n_k - r_k)` with
  r = the annealed SU(3) state into the Hamiltonian
  (`set_mixed_trilinear_reference_SU3`; energy, MC and MD all see it) and
  re-minimises before the dynamics.
- **Thermal reservoir** E_dep is an ODE variable (it was an accumulator in the
  right-hand side, reset to zero thousands of times per trajectory by
  floating-point time jitter and step rejections).
- **W1** (M1 from time-shifted M0) only for delays on the time grid (it rounded
  tau to the nearest step) and when the pump has not started at T_start.
  `stationarity_tol` now bounds the dimensionless residual
  `max_i |dS_i/dt| / (|H_i||S_i|)` (it was an absolute rate).
- **M01 continued from M0** (`reuse_m0_for_m01`, default 1): before
  `tau - 9 width` the pump-probe drive equals the pump drive up to the
  1.6e-9 Gaussian tail, so M01 starts from the stored M0 state there; M_NL is
  exactly zero before the probe and M01 costs about half.
- **Same experiment on every path.** The trial-parallel / single-rank 2DCS
  path now honours `pump_direction_2`, `pump_direction_su3_2` and
  `linear_drive_torque`; MD, pump-probe and 2DCS share one damping/ablation
  setup; with a loaded configuration every trial starts from it (later trials
  used to run from random spins without annealing); pump-probe trajectories of
  ranks > 0 are written.
- **MPI 2DCS**: one writer (same file layout as the serial driver), dynamic
  scheduling with results streamed per delay, fixed message tags (the old
  `2 tau_steps + ...` tags exceeded MPI_TAG_UB), chunked large messages, and
  errors on any rank are propagated to all ranks (no zero padding of
  mismatched trajectories, no deadlock).
- New option `alpha_su3`: Casimir-preserving SU(3) Landau-Lifshitz damping
  `dn/dt += -(alpha/|n|) c f(n, P)`, P the precession (Bloch-vector form of the
  SU(N) LL damping of Dahlbom et al., PRB 106, 235154 (2022)). It relaxes
  towards the instantaneous local field and keeps pure states pure, unlike
  the fixed-target Bloch relaxation `gamma_su3*` (still available).
- The staggered SU(3) observable uses the unit cell's AFM signs and frames
  (TmFeO3: (+,-,+,-), identical to the old site-parity rule for that cell).
- GPU: `check_gpu_supported()` throws instead of silently dropping trilinear
  couplings, field-assisted exchange, damping, the thermal reservoir,
  two-colour / tabulated pulses, the drive-torque ablation or spin-state output.
- MixedLattice requires 3-component SU(2) and 8-component SU(3) spins (other
  shapes ran with frozen spins or out-of-bounds structure constants).
