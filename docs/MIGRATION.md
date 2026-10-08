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

## NCTO (PhononLattice): spin storage frame is the crystal frame

**What changed.** The J–K–Γ–Γ' exchange and the magnetoelastic tensors of the
NCTO model (`system = ncto`, `build_phonon_honeycomb`, `PhononLattice`) are
rotated from the cubic Kitaev frame into the storage frame with
`J_storage = Rᵀ J_cubic R`, where `R = [a | b | c*]` holds the crystal axes
`a = (1,1,-2)/√6`, `b = (-1,1,0)/√2`, `c* = (1,1,1)/√3` in cubic coordinates
(`S_cubic = R S_crystal`). Stored spins, `field_direction`, pinning fields,
`M_local` and spin files are therefore crystal components, with `c*` (the C3
axis, honeycomb normal) along `z` and the in-plane axes those of the site
positions. The old code used `R J Rᵀ`, a frame that is neither cubic nor
crystal: `field_direction = 0,0,1` pointed 98° away from `c*`, almost in-plane.
`kitaev_bonds.h` now owns the convention (`Frame`, `storage_from_cubic`,
`crystal_from_storage`); `build_phonon_honeycomb` no longer duplicates the
matrices.

The "global" outputs (`magnetization_global`, `magnetization_sublattice`,
`M_global` and `M_antiferro` in MD / pump-probe / 2DCS files) are crystal
components in every frame (`sublattice_frames = crystal_from_storage`), and
`M_antiferro` uses the Néel signs (+1, −1). Before, pump-probe outputs rotated
the stored spins by `R` a second time, MD wrote `M_global = M_local`, and
`M_antiferro` equalled `M_global` because both sublattice signs were +1.

**Why.** Zero-field energetics are frame independent, but every in-field result
(SA, MD, 2DCS, GNEB with `field_strength != 0`) applied the field along an
unintended direction, and the global outputs were in an undocumented,
doubly-rotated frame. Test [19]–[20] of `test_e1_phonon_regressions` check
that the pure-Γ bond sum is `diag(-1,-1,2)`, that every Kitaev axis is
perpendicular to its bond, and that the FM energy is uniaxial about the field
axis `c* = z`.

**Recover the old behaviour.** `legacy_kitaev_frame = 1` reproduces the old
Hamiltonian and field coupling exactly (`SpinPhononCouplingParams::frame =
Frame::Legacy`). Spin seed files written before this change are in the legacy
frame: either run with `legacy_kitaev_frame = 1` or convert each spin with
`S_crystal = crystal_from_storage(Frame::Legacy) · S_old = (Rᵀ)² S_old`.

## Kitaev honeycomb (Lattice): crystal-frame global outputs

**What changed.** `build_kitaev_honeycomb` keeps storing spins (and reading
`field_direction`) in the cubic Kitaev frame, but its sublattice frames are now
`Rᵀ` (cubic → crystal), so `M_global` is in crystal components `(a, b, c*)`,
and its Néel signs are (+1, −1). The bond matrices come from `kitaev_bonds.h`.

**Why.** The frame was `R`, which rotated cubic spins a second time instead of
expressing them in crystal axes, and both sublattice signs were +1.

**Recover the old behaviour.** Multiply the new `M_global` by `R²`
(`crystal_axes_in_cubic()` squared) to obtain the old numbers.

## NCTO Langevin thermostat: fluctuation–dissipation for the coupled system

**What changed.**
- Spin noise variance `2 D / dt` with `D = α T / (|S| (1 + α²))` (was
  `2 α T / |S|`). With the noise in the precession and the damping term this is
  the stationarity condition of `exp(−E/T)` (García-Palacios & Lázaro 1998);
  the old value heated the spins to `T (1 + α²)` — 100 % too hot at `α = 1`.
  The semi-quantum (Bose) thermostat uses the same classical level.
- Every damped zone-centre mode (primary E1 and extra modes) receives the
  matching noise force `N(0, 2 γ T_ph / (N dt))` (`phonon_langevin_T`, default:
  the bath temperature); before, the modes were damped but noiseless, i.e. a
  `T = 0` sink. SLD momenta keep their noise.
- Integration: the noise of a step is held fixed over one RK4 step of the
  whole system (frozen-noise / Wong–Zakai scheme), followed by spin
  renormalisation. The former split (RK4 step, then a noise kick) had an
  `O(α|H| dt)` bias in `<E>` (0.7 % at `α = 1`, `dt = 0.01`).
- `integrate_langevin` throws `std::invalid_argument` on invalid input instead
  of printing and returning, never changes `alpha_gilbert` (it used to set 0.01
  when `α ≤ 0`), runs the SLD static relaxation once per `enable_sld(true)`
  (it re-relaxed and zeroed the momenta on every call), computes
  `t = t_start + k dt`, and `seed = 0` derives the stream from the process seed
  instead of `std::random_device`.
- Output is streamed: `langevin_trajectory.txt` (unchanged columns),
  `langevin_spins.h5` (`/spins` float32 `[frame, N, 3]`, `/t`) replaces the
  text file `langevin_spins.txt`; `langevin_lattice.h5` keeps its layout.
- Finite-capacity bath: the drive work is summed over every polar mode (it was
  the primary mode only), and a time-dependent coupling schedule is rejected.

**Why.** Every thermostatted NCTO run was too hot by `(1 + α²)` and had no
thermal lattice fluctuations. `tests/test_phonon_dynamics.cpp` checks the free
spin Langevin function at `α = 0.1, 1`, Langevin `<E>` against Monte Carlo and
equipartition of every damped lattice coordinate.

**Recover the old behaviour.** Not supported (it was not a Gibbs sampler).

## NCTO runs: no silent THz drive, one damping default, one model builder

**What changed.**
- `molecular_dynamics` (and annealing) runs are undriven unless
  `pump_amplitude` / `probe_amplitude` is set in the config; the generic
  `SpinConfig` defaults (pump 1.0 at t = 0, probe 0.1 at t = 50) used to drive
  every NCTO MD run. `pump_probe` applies the pump (default amplitude 1.0) and
  the second pulse only if `probe_amplitude` is set. A pulse centred within 4σ
  of `md_time_start` prints a warning.
- Pulse frequencies/widths: a key present in the file is used as given (an
  explicit `pump_width = 10` used to be replaced by 0.273); absent keys take
  the measured NCTO pulse. The 2DCS runner uses the same resolution (it used
  the raw `pump_width` default 10 and `omega_E` 1.0) and now honours
  `tau_start`/`tau_end`/`tau_step` (they were read from the wrong map and
  always 0/100/5; those remain the defaults).
- `alpha_gilbert` defaults to 0 for every NCTO path (was 0.05 in spin_solver,
  0 in sweeps); Langevin runs use 0.05 when the key is absent.
- Linear E1 couplings `lambda_E1_X_1` default to 0 in `SpinPhononCouplingParams`
  too (the struct had the operating-point estimate, λ_K1 = 40, while configs
  defaulted to 0).
- `make_ncto_lattice(config)` (phonon_config.h) is the single builder used by
  spin_solver, the parameter sweep and the tools: sweeps now include the extra
  modes, disorder and pinning files, SLD and the damping default.
- `local_field_h` is added once (not once per trial); its directions come from
  `local_field_config` (strictly validated) or the loaded seed.
- Trials are independent: each starts from the seed file / a fresh random
  state with the lattice sector reset (extra modes and SLD included), and each
  Langevin trial has its own noise stream derived from `langevin_seed` (or the
  master `seed`) and the trial index.
- `set_parameters` rebuilds the NN/J2/J3 exchange from the coupling parameters
  (it ignored them) and keeps extra modes and quenched disorder (it wiped them).
- J2/J3 bonds are always registered by `build_phonon_honeycomb` (zero couplings
  are skipped), so SLD second-neighbour springs and J2/J3 modulations no longer
  vanish when J2 or J3 is zero.

**Recover the old behaviour.** Set the old values explicitly
(`alpha_gilbert = 0.05`, `pump_amplitude = 1`, `probe_amplitude = 0.1`, ...).

## NCTO dynamics and Monte Carlo consistency

**What changed.**
- The time-dependent coupling schedule (`lambda_time_mode = 1`) multiplies the
  whole magnetoelastic Hamiltonian (all bond increments, the ring modulation
  `J7_eff − J7` and the J2/J3 modulations) consistently in field, force and
  `total_energy(t)`; before it scaled only the bond increments in the dynamics
  and was ignored by every energy.
- `site_energy_diff` is `−ΔS·H` with the full local field, so Metropolis now
  includes the SLD exchange striction (it did not); `site_energy` includes the
  ring term. Heat-bath moves (`local_update = heat_bath`) are available, and
  `mc_sample_lattice = 1` samples the zone-centre coordinates in Monte Carlo.
- Replica exchange swaps the lattice sector with the spins.
- Unknown integration methods throw; implicit methods are limited to states
  of ≤ 1200 entries; MD and pulse-drive outputs lie on the exact grid
  `T_start + k dt` (the pulse-window chunking is ignored).
- The W1 stationarity guard uses the full right-hand side (lattice included).
- `relax_phonons` is a Newton iteration with the exact Hessian and a line
  search; `relax_joint` tolerances are per site; `deterministic_sweep` is a
  sequential sweep that stops when converged and returns the last max |ΔS|.
- Spin files, disorder and pinning files are parsed strictly (short, long or
  malformed files throw).

**Recover the old behaviour.** Not supported (inconsistent Hamiltonians).

## NCTO MPI 2DCS

**What changed.** `pump_probe_spectroscopy_mpi(..., MPI_Comm comm)` is
collective over `comm`; with one trial the 2DCS runner prepares the ground state
on rank 0, broadcasts it, and distributes the delays (it called the collective
from rank 0 only and deadlocked under `mpirun -np > 1`). The full trajectory
records are gathered, so `M_global` and `O_custom` are correct for every delay
(they were zeros / uninitialised for delays not computed on rank 0), and serial
and MPI runs write identical files (all of `M_antiferro`, `M_local`,
`M_global`, `O_custom`). The NCTO runners take an optional communicator
(sweeps pass `MPI_COMM_SELF`).

**Recover the old behaviour.** Not supported.

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

## Pyrochlore: (2,3) inter-tetrahedron bond

**What changed.** `build_pyrochlore` and `build_pyrochlore_non_kramer` wired the
(2,3) "down-tetrahedron" bond with the cell offset (0,+1,-1), which connects
sites 3√2/4 apart (a third-shell pair) instead of the nearest neighbour at √2/4.
The offset is now a3 − a2 = (0,−1,+1). Every site now has its six nearest
neighbours; before, one of the six was replaced by a third neighbour.

**Why.** It was a geometry bug; `tests/test_unitcell_geometry.cpp` checks every
builder's bonds against the neighbour shells generated from the cell geometry.

**Old behaviour.** Not recoverable (no physical model has that bond). Results
for pyrochlore models obtained before this change describe a different lattice.

## Pyrochlore non-Kramers: second- and third-neighbour couplings

**What changed.** `J2` used to couple each sublattice to itself at offsets a1,
a2, a3. Those are third-shell bonds (2 r_NN), and they are the chain-type third
neighbour for sublattice 0 but the hexagon-type one for the others, which breaks
the cubic symmetry. `J2` now couples the true second shell (√3 r_NN, 12
neighbours per site, generated by `UnitCell::bonds_at_distance`). The two
third-neighbour couplings are new keys: `J3a` (along a bond chain, a site at the
midpoint) and `J3b` (across a hexagon).

**Old behaviour.** `pyrochlore_legacy_J2 = 1`.

## Unit cells are validated

**What changed.** The lattice constructors call `UnitCell::validate()`: atom
indices, matrix sizes, finite entries, linearly independent lattice vectors,
and a bilinear bond declared again in the reverse direction (the lattice adds
every reverse bond with Jᵀ itself, so declaring both directions doubled the
coupling) throw `std::invalid_argument` / `std::out_of_range` naming the entry.
Several terms on one bond in the same orientation still add. Setters check their
input; `MixedUnitCell` checks indices, shapes and the drive envelope tag
(0 = E, 1 = |B|, 2..4 = B_x, B_y, B_z). On-site matrices are stored symmetrised
(only the symmetric part enters SᵀAS), and so are self-bonds folded into them.
Lattice dimensions of 0 or a non-positive spin length throw.

## Bonds longer than the lattice

**What changed.** Partner cells are reduced with a Euclidean (floor) modulo, and
twisted boundaries apply the twist once per period crossed. The old single-step
wrap indexed out of range (a heap overflow) for bonds longer than the lattice
extent, e.g. the honeycomb J3 offset (1,−2,0) on a lattice one cell wide. A
lattice narrower than 2|offset| + 1 along a bonded direction prints a warning:
periodic images of distinct bonds then land on the same pair and their
couplings add.

## Correlation accumulator: FFT cross spectra, global frame, error bars

**What changed.** `RealSpaceCorrelationAccumulator` (now in
`lattice/correlation_accumulator.h`) accumulates, per sample, the FFT of every
sublattice/component field over the cell grid and the cross spectra
X_ss'^ab(k) = F_sa(k) F_s'b(k)^*, instead of real-space sums over all cell
pairs. Consequences:

- Cost O(N log N) per sample and no N_c x N_c table (L = 12 pyrochlore with
  dimers: 5 ms per sample instead of ~2 s; 1.5 GB table at L = 24 gone).
- Spins are rotated to the global frame with `sublattice_frames`, so
  `structure_factor(q)` is the physical (neutron) S^ab(q); all sublattice pairs
  and all (a, b) components are kept complex, so the antisymmetric (chiral)
  part survives. `compute_Sq` returns its real part, as a 3x3 matrix.
- q must be commensurate (q.a_i L_i / 2π integer): anything else throws
  `std::invalid_argument` (the old code silently returned wrong values off the
  grid). `grid_wavevector(m1, m2, m3)` builds grid points in any zone.
- Error bars: samples go into `Options::n_bins` (default 16) contiguous bins
  that are merged pairwise when full; `structure_factor_estimate` and
  `compute_Sq_with_error` return delete-one-bin jackknife errors (the old error
  was always zero).
- Dimers: bond types are now geometrically distinct bond classes of the unit
  cell (`bond_classes_from_unit_cell`, nearest coupled shell by default) with
  the bond-centre phase; the old type = sublattice pair lumped up/down
  tetrahedron bonds together, used two different (i, j) -> type orderings in
  Lattice and the accumulator, and the wrong centre for inter-cell bonds.
  Dimer operators use global-frame spins and have their own sample count.
- `merge` merges every mean (it dropped two thirds of the dimer means), copies
  into an uninitialised accumulator and throws on any geometry mismatch;
  `mpi_reduce` is one `MPI_Reduce` on doubles after a collective geometry check.
- `save_hdf5` writes a new schema (cross spectra, per-bin means, S(q) on the
  first-zone grid with errors, bond-class table, convention attributes) and
  throws `std::runtime_error` on HDF5 failure. The old flat `spin_corr_sum` /
  `dimer_corr_sum` datasets are gone; `real_space_correlation(s, s', a, b)`
  returns C(d) by inverse FFT.
- The `n_bond_types` argument of `create_correlation_accumulator` /
  `parallel_tempering` (config key `pt_n_bond_types`) is ignored.

**Recover the old behaviour.** Not supported (the old output had wrong phases,
frames and normalisation). Local-frame correlations: initialise with empty
`Geometry::frames`.

## Lattice::structure_factor / structure_factor_tensor: global frame, all components

**What changed.** `Lattice::structure_factor(q)` returned |Σ_i S_i^x e^{iq·r_i}|²/N
of the first LOCAL component only; it is now Tr S(q) over all components in the
global frame. `structure_factor_tensor(q)` uses global-frame spins (was local)
and `structure_factor_matrix(q)` returns the full complex tensor. Both stay
normalised per site (1/N); the accumulator is per unit cell.

**Recover the old behaviour.** Compute |Σ_i S_i^x e^{iq·r_i}|²/N directly from
`spins` if needed.
