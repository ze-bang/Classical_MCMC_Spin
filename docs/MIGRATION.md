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

## Config files: unknown keys are errors

**What changed.** A key that is neither a typed `SpinConfig` field (or one of
its aliases, e.g. `dt`, `h`, `tbc`, `temperature_start`) nor a registered
Hamiltonian / model parameter is an error naming file, line and the nearest
known key (`anealing_steps` → "did you mean 'annealing_steps'?"). Hamiltonian
keys are registered in `src/core/spin_config.cpp` (literal names plus key
families such as `mode<i>_omega`, `Kminus<orbit>_<λ><axis>`, `h_tm_<a>`);
`tests/test_config.cpp` scans `src/` and `include/` and fails if a literal key
read through `get_param`/`has_param`/`was_set` is missing, so the registry
cannot fall behind the builders. A line without `=` or `:` is also an error
(it was skipped). A key given twice prints a warning (the last value wins, as
before).

**Why.** Unknown keys were stored as Hamiltonian parameters and never read, so a
typo (`anealing_steps`, `Gama`) silently ran the default and a whole example
(`Pyrochlore/field_scan.param`, keys `field_scan_*`) ran something else than it
said.

**Recover the old behaviour.** `allow_unknown_keys = true` keeps unknown numeric
keys as Hamiltonian parameters with a warning (anywhere in the file).

## Config files: strict numbers and booleans

**What changed.** Integer keys accept `1e5` or `100000.0` when the value is
integral and in range, and reject fractional (`2.7`), negative (for counts) and
overflowing values; doubles must be finite and consume the whole value
(`1.5abc` is an error); `seed` is parsed exactly to 64 bits. Booleans are
`true/false`, `yes/no`, `on/off`, `1/0` in any case; anything else is an error
(`ture` used to mean false, and `auto_su3_pump`/`fix_strain` only knew
`true`/`1`). `lattice_size` needs exactly three values. Errors name the key and
the line.

**Why.** `std::stoull("1e5")` is 1 and `stoull("-1")` is 2^64 − 1, so
`annealing_steps = 1e5` ran one sweep and `lattice_size = -8` asked for 10^19
sites.

**Recover the old behaviour.** Not supported (write the value in range).

## Config: one key/value entry point (`SpinConfig::set`), lossless `to_file`

**What changed.** `SpinConfig::set(key, value)` is the single entry point for a
key: `from_file`/`from_string` call it for every line and parameter sweeps
apply their values through it (`set_value`, printed with `%.17g`). It returns
false for an unknown key. `to_file` writes every typed key and Hamiltonian
parameter at full precision under the name the parser expects
(`simulation_mode`, not `simulation`; the old name is accepted as an alias), so
`from_file(to_file(c))` reproduces `c` (tested for every key). The legacy
single-axis sweep keys `sweep_parameter/start/end/step` set element 0 of the
N-dimensional lists and are written in that form. `explicit_keys`/`was_set`
store canonical names (an alias marks its canonical key). `field_direction` is
still normalised on input; a vector that is already unit length is kept
bitwise.

**Why.** Sweeps only wrote the Hamiltonian map, so sweeping a typed field
(`pump_amplitude`, `probe_time`, `T_start`, `md_*`, `tau_*`, ...) ran N identical
simulations; `to_file` output did not parse.

## Config validation

**What changed.** `SpinConfig::validation_errors()` (used by `validate()` and by
`spin_solver` before anything is built) rejects:
`cooling_rate` outside (0, 1), `T_end <= 0` or `T_start < T_end` whenever a run
anneals (SA, and the ground-state preparation of MD / pump-probe / 2DCS without
`initial_spin_config`; the TmFeO3 2DCS runner always anneals) or runs parallel
tempering, `probe_rate = 0` or `pt_exchange_frequency = 0` (PT),
`md_timestep <= 0`, `md_save_interval = 0` or `md_time_end < md_time_start`
(every dynamics mode and family), a zero or wrong-sign `tau_step` (2DCS, every
family; NCTO checks its effective default scan), lattice sizes of 0,
`num_trials < 1`, non-positive spin lengths, a `field_direction` or `g_factor`
without 3 components, system/mode combinations the driver does not implement
(NCTO + parallel tempering, `kinetic_barrier`/`gneb`, `custom`), and parameter
sweeps without parameters, with inconsistent list lengths, unknown or
non-numeric swept keys, invalid grids, or any point that is not a valid run
of the base simulation (checked point by point). `SA` with `annealing_steps = 0` (an
energy evaluation) skips the schedule checks.

**Why.** `T_end = 0` (two shipped examples) and `cooling_rate >= 1` never
terminated, `probe_rate = 0` divided by zero, unsupported modes printed an error
and exited 0.

**Example configs fixed.** `Pyrochlore/field_scan.param` is now the field scan
it describes (a parameter sweep over `field_strength`, 0 → 5 step 0.1, with
`T_end = 0.001`, `T_zero = true`); `Pyrochlore/md_pyrochlore.param` uses
`T_end = 0.001`, `T_zero = true`; `param_sweeps/2dcs_chii_sweep.param` and
`2dcs_sweep_example.param` use the Fe–Tm coupling `Kminus_2y` instead of
`chii` (a legacy key the current TmFeO3 builder never read);
`Pyrochlore/pt_non_kramer_field_sweep.param` ends at 2.9 (the same eight
points it ran before; 1.5 → 3.0 is not a whole number of 0.2 steps).

## Spin-configuration files: strict loading, full-precision saving

**What changed.** `Lattice`, `MixedLattice` and `PhononLattice::load_spin_config`
share one reader (`classical_spin/io/spin_table.h`): exactly one spin per line
with exactly `spin_dim` finite numbers, `#` comments and blank lines allowed.
A missing file, a short file, a wrong column count, a non-finite value, extra
rows or a zero vector throw `std::runtime_error` naming the file and line, and
the current state is left unchanged (the file is parsed into a buffer first;
`MixedLattice` parses both `_SU2.txt` and `_SU3.txt` before storing either).
Every loaded spin is rescaled to its length: `spin_length` (Lattice,
PhononLattice), `spin_length_su3`/`spin_length` (MixedLattice), except that an
SU(3) vector within 1e-3 of the pure-qutrit length 2/√3 is set to exactly 2/√3
(states from `physicalize_SU3_state` stay physical); a spin already of its
length to round-off is kept bitwise, and a rescale by more than 1e-3 prints a
warning. All savers (`save_spin_config`, `save_spin_config_to_dir`) write
`max_digits10` significant digits and throw if the file cannot be written
(Lattice printed an error and continued; MixedLattice wrote 6 digits and
ignored failures; PhononLattice wrote 12 digits), so save → load is bitwise.

**Why.** The Lattice and MixedLattice loaders printed to stderr and returned
on a missing or short file, leaving random or half-overwritten spins, while
the runners skipped equilibration because a configuration had been "loaded";
extra columns or rows were silently ignored; 6-digit MixedLattice files
reloaded with |S| errors of ~1e-6 (non-stationary "ground states").

**Recover the old behaviour.** Not supported (fix the file).

## HDF5 2DCS files: delay count

**What changed.** The `tau_steps` attribute of the mixed-lattice pump-probe
file uses the round-off tolerant delay count of `dynamics::delay_grid` (shared
helper `hdf5_delay_count`); it truncated `|tau_end - tau_start| / tau_step`,
so 0 → 0.3 in steps of 0.1 recorded 3 delays while 4 were written.

## Parameter sweeps: typed keys, validated points, one construction path

**What changed.** Each sweep point is the base configuration with the swept
values applied through `SpinConfig::set` (printed with `%.17g`), so typed keys
(`pump_amplitude`, `probe_time`, `T_start`, `md_*`, `tau_*`, `annealing_steps`,
...) are swept like Hamiltonian keys. Every point is validated before anything
runs (e.g. a sweep that reaches `T_end = 0` is rejected up front), writes its
own `run_info.txt`, and is built and run by `run_simulation()` — the same
factory (`make_unit_cell`, `make_lattice`, `make_mixed_lattice`,
`make_ncto_lattice`) and mode dispatch as a direct run, so a one-point sweep
reproduces the direct run bitwise (smoke-tested). Point directories keep their
names (`<key>_<value in %e>`).

**Why.** Sweeps only wrote the Hamiltonian map (5 of the 11 shipped sweep
examples ran N identical simulations) and carried their own copies of the
unit-cell switch and initial-state logic, which had drifted from `main`.

## Parameter sweeps: grids built by index

**What changed.** A sweep axis has `n = llround((end - start) / step) + 1`
points `start + k * step`, so the endpoint is kept and no round-off
accumulates. `step = 0`, a step pointing away from `end`, and a range that is
not a whole number of steps (to 1e-6 steps) are errors. `start == end` is one
point.

**Why.** Repeated addition dropped the endpoint (0 → 2 step 0.1 gave 20
points ending at 1.9000000000000006), `step = 0` looped until memory ran out,
and a wrong-sign step silently gave no points.

**Recover the old behaviour.** For a range that is not a whole number of
steps, set `sweep_end` to the last point you want.

## MPI: runners take a communicator; sweeps never deadlock

**What changed.** Every runner takes the `MPI_Comm` it runs on (instead of
rank/size integers) and uses no other communicator. Non-PT sweep points run
on `MPI_COMM_SELF`, one rank per point; the only collective on the job
communicator is the final barrier, reached once by every rank. PT sweep points
run on equal-sized groups of `pt_ranks_per_point` ranks (auto: ranks / points,
at least 2); ranks left over idle with a warning instead of joining the last
group (unequal ladders), and fewer than 3 replicas per point prints a warning.
The TmFeO3 delay-parallel 2DCS (whose library routine runs on
`MPI_COMM_WORLD`) is used only when the communicator spans the job; on a
sub-communicator the trials run serially. Mixed-runner input errors throw
(reported with the rank) instead of calling `MPI_Abort` from rank 0's checks.
`spin_solver` initialises MPI with `MPI_THREAD_FUNNELED`.

**Why.** Single-rank sweep points still ended in `MPI_Barrier(MPI_COMM_WORLD)`
(MD, pump-probe, 2DCS, NCTO and TmFeO3 runners), so whenever the point count
was not a multiple of the rank count some ranks waited forever.

## spin_solver: error reporting and exit status

**What changed.** Rank 0 reads the configuration and broadcasts its text, so
every rank parses identical input; parse and validation errors are reported
once and exit with status 1. During the run, an exception on any rank prints
`[rank r] error: ...` and calls `MPI_Abort(MPI_COMM_WORLD, 1)` (it used to call
`MPI_Finalize` on that rank while others waited in a collective, and only
rank 0's message was printed). `H5::Exception` (not derived from
`std::exception`) is caught and its function and detail message printed;
HDF5's own error-stack dump is switched off. Unsupported system/mode
combinations (NCTO + parallel tempering, `kinetic_barrier`, `custom`) are
configuration errors (they printed to stderr and exited 0).
"Simulation completed successfully" is printed only when everything ran.

## Every trial starts from the configured initial state

**What changed.** With `initial_spin_config` (or `use_ferromagnetic_init`)
every trial of simulated annealing (Lattice, MixedLattice) and every parallel-
tempering trial after the first start from that state; without one every trial
starts from fresh random spins. MixedLattice MD/pump-probe/2DCS trials restore
a ferromagnetic start too. `annealing_steps = 0` evaluates the starting state
for every family (MixedLattice and PhononLattice used to run the cooling loop
with zero sweeps).

**Why.** Trials after the first (and, under MPI, the first trial of every rank
but 0) silently started from random spins, MixedLattice SA never used the
loaded configuration after trial 0, and PT trials > 0 discarded it.

## Outputs: every rank writes its trials; trial_summary.txt; run_info.txt

**What changed.**
- Every rank writes the results of the trials it ran: `final_energy.txt`
  (17 digits) and `spins_final*.txt` for simulated annealing (Lattice,
  MixedLattice, PhononLattice); pump-probe trajectories (MixedLattice now at
  17 digits). Rank 0 gathers one line per trial into
  `output_dir/trial_summary.txt` (trial, owning rank, energy per site — final
  for SA, of the initial state for the dynamics modes — and the result file,
  plus mean/std/min).
- Rank 0 writes `output_dir/run_info.txt`: comment lines with date, host,
  seed, MPI size, OpenMP threads, `git describe` (generated at build time;
  "unknown" outside a git checkout), compiler, build type and flags, followed
  by the full resolved configuration — the file itself reruns the job
  (`spin_solver output_dir/run_info.txt` reproduces the trial energies
  bitwise; smoke-tested). Each sweep point writes its own. `seed.txt` is still
  written.
- MixedLattice MD writes `initial_spins_SU2.txt`/`_SU3.txt` (it wrote
  `initial_spins.txt_SU2.txt`).

**Why.** SA final energies and pump-probe trajectories of trials on ranks ≠ 0
were computed and discarded (rank-0-only writes at 6 digits), and no run
recorded the configuration, seed or code version that produced it.

## Config keys that have no effect warn

**What changed.** `equilibration_steps` (use `pt_equilibration_steps`),
`num_replicas` (PT uses one replica per rank), `initial_step_size`,
`deterministic`, `pt_target_acceptance` and `use_mpi` are accepted as before
but print a warning: no simulation reads them.

## Smoke tests in CTest

**What changed.** `tests/smoke/run_smoke.sh` is registered as
`smoke_spin_solver` (label `smoke`, timeout 900 s) and covers, besides every
simulation mode: uneven MPI sweeps of a typed key, PT sweep groups, a TmFeO3
MD sweep, sweep point == direct run (Lattice and NCTO), loaded configurations
in every trial (Lattice and TmFeO3), the `run_info.txt` rerun, and clean
non-zero exits for a misspelt key, an unsupported mode, a missing seed file
(under MPI) and a non-terminating schedule. Two MPI CTests run `spin_solver`
directly (`mpi_sweep_uneven`, `mpi_runtime_error_aborts`).

## MixedLattice Monte Carlo: SU(3) sites sampled on CP^2 (`su3_mc_manifold`)

**What changed.** SU(3) (Tm) sites store the Gell-Mann expectations
n^a = <psi|lambda^a|psi> of a qutrit pure state, and since the SU(3)
convention change the energy is E = <psi|H|psi> and the dynamics conserve
that state space, CP^2 (|n|^2 = 4/3, cubic Casimir d_abc n^a n^b n^c = 8/9).
Monte Carlo now samples the same manifold with its invariant (Fubini-Study,
Haar) measure:

- uniform proposals psi = normalise(z), z a complex Gaussian 3-vector;
- small symmetric moves psi' = normalise(psi + sigma z) (the kernel depends
  only on |<psi|psi'>|, so plain Metropolis is exact); psi is recovered from
  the stored n without an eigensolver (rho = 1/3 + n.lambda/2 is rank one:
  `su3::psi_from_pure_expectations`);
- heat bath: in the eigenbasis of the local H = h.lambda the populations
  p_k = |<v_k|psi>|^2 are uniform on the simplex under Fubini-Study, so they
  are drawn exactly from exp(-beta sum_k p_k e_k), with uniform phases;
- `init_random()` draws Haar-random states, `init_ferromagnetic()` maps the
  SU(3) direction to its closest pure state (lambda_3 gives |1>), the T = 0
  descent sets each site to the ground state of its local H (exact
  diagonalisation), and SA / `greedy_quench` / PT first project SU(3) states
  that are off the manifold (e.g. loaded legacy seeds) onto it
  (`project_SU3_to_manifold`).

`spin_length_su3` is not used on CP^2 (|n| = 2/sqrt 3 is fixed by purity).

**Why.** The old sampler drew 8-vectors on the 7-sphere of radius
`spin_length_su3` (default 1, which contains no pure state at all), with a
hypercube-biased proposal before the RNG fix: 99.6 % of the proposals were not
density matrices, the Tm sector had 7 instead of 4 degrees of freedom
(equipartition 7/2 instead of 2 per site), and annealed states handed to the
dynamics were on the wrong coadjoint orbit.

**What you see.** Different Tm thermodynamics and SA ground states (now the
physical ones); SU(3) Monte Carlo outputs satisfy the Casimirs to round-off.

**Recover the old behaviour.** `su3_mc_manifold = sphere` (uniform measure on
the 7-sphere of radius `spin_length_su3`, the S^7 reflection for
overrelaxation, -L h/|h| at T = 0). It is also the default when
`su3_legacy_convention = 1`; `su3_mc_manifold = cp2` overrides that, and
`auto` (or no key) restores the default of the convention. API:
`MixedLattice::set_su3_mc_manifold("cp2" | "sphere" | "auto")`.

## MixedLattice Monte Carlo: exact local updates, self-bonds, long bonds

**What changed.**

- With all other spins frozen, the energy of a site is h.S + S^T A S (plus
  cubic terms if a trilinear has the site in all three slots). The kernels
  compute h (couplings containing the site once) from the packed buffers and
  A (symmetrised on-site matrix plus the couplings containing the site twice,
  e.g. the TmFeO3 vertex W(S_i, S_i, n_k)) from tables merged per partner.
  `site_energy_SU2_diff` / `site_energy_SU3_diff` equal the total-energy
  difference for every coupling, including non-symmetric on-site matrices
  (only the symmetric part enters S^T A S; on-site matrices are symmetrised at
  build time, so the MD field 2 A S is the true gradient).
- A bond onto the site's own periodic image (lattice one cell wide along a
  bonded direction) is folded into the on-site matrix, as in Lattice, and
  cell coordinates are wrapped with a Euclidean modulo, so bonds longer than
  the lattice no longer index out of range.
- **Overrelaxation** reflects each spin about h, which does not depend on the
  spin: exact and microcanonical where the self energy S^T A S is constant on
  the sphere; elsewhere (single-ion anisotropy, self-coupled W) the reflection
  is Metropolis-corrected at temperature T > 0 and skipped at T <= 0
  (`overrelaxation(double T = 0)`, `overrelaxation_interleaved(T)`,
  `overrelaxation_parallel(T)`; the PT adapter passes the replica temperature).
  On CP^2 the S^7 reflection would leave the manifold; SU(3) sites instead get
  random relative phases in the eigenbasis of H = h.lambda,
  psi' = V diag(1, e^{i phi_1}, e^{i phi_2}) V^dagger psi: it conserves the
  populations (hence h.n), maps CP^2 onto itself, and is a random unitary from
  an inversion-symmetric distribution (unitaries preserve the Fubini-Study
  measure), so the kernel is symmetric.
- **T = 0 descent**: `deterministic_sweep(n = 1)` sets every site, in order,
  to its exact single-site minimiser (trust-region solution of
  min h.S + S^T A S on the sphere for SU(2) sites with anisotropy or W, the
  local ground state on CP^2) and returns the largest change; the energy never
  increases. `greedy_quench` stops when both the energy change and the largest
  change are below tolerance; SA's T = 0 stage runs to convergence (at most
  `n_deterministics` sweeps). `deterministic_sweep_interleaved()` and
  `deterministic_sweep_SU3_exact_diag()` return the largest change too.
- **Sweeps** visit every site exactly once (natural order, unit cell by unit
  cell for `*_interleaved`, colour by colour for the OpenMP kernels); they
  used to draw sites at random with replacement (~37 % untouched per sweep).
  Site kernels allocate nothing (stack proposals, packed couplings).

**Why.** The old overrelaxation reflected about the full gradient h + 2 A S,
which conserves neither energy nor measure (2x2x2 TmFeO3: E 33 -> 14 in 20
sweeps with the default anisotropy); the old T = 0 rule aligned spins with the
full gradient (2-cycles / energy increases with anisotropy, unphysical SU(3)
states); self-bonds gave max |dE - ΔE| = 2.4 on a 1x1x1 TmFeO3 lattice.

**Recover the old behaviour.** Not supported (it sampled the wrong
distribution).

## MixedLattice Monte Carlo: local-update policy, heat bath, adaptive SA

**What changed.**

- The `local_update` key (metropolis | gaussian | heat_bath) now applies to
  MixedLattice (`MixedLattice::local_update`, `local_sweep()`), as for
  Lattice. `heat_bath` is rejection-free for SU(2) sites and for SU(3) sites on
  CP^2 where the local energy is linear, Metropolis-corrected for the self
  energy elsewhere; SU(3) sites on the legacy sphere use uniform Metropolis.
  Lattices with at least `parallel_sweep_min_sites` (4096) sites use the
  coloured OpenMP kernels when more than one thread is available.
- `simulated_annealing` runs on `mc::annealing_schedule(T_start, T_end,
  cooling_rate)` (ends exactly at T_end; `cooling_rate` outside (0, 1) or
  `T_end <= 0` throw instead of looping forever). With Gaussian proposals the
  width is adapted by `mc::StepSizeController` (Robbins-Monro toward 45 %
  acceptance) during the first half of the sweeps at each temperature and
  frozen for the second half (was: sigma = 1000, only ever shrunk). Final
  measurements use the same local-update policy.
- `perform_mc_sweeps` returns the mean acceptance of its local sweeps (was the
  sum).
- `measure_all_observables` computes the total energy once.
- Removed (dead or unsafe): the lazy local-field cache (`enable_field_caching`,
  `init_field_cache`, `invalidate_*`, `get_cached_local_field_*`,
  `use_field_caching`, `cached_local_field_*`, `field_valid_*`, the
  `mixed_bilinear_reverse_*` tables; it was never enabled and went stale after
  any non-interleaved update or replica exchange) and the unused
  `get_local_field_SU{2,3}_state`.
- `bench_mixed_mc` (new) times the serial kernels on TmFeO3; `bench_mc`'s
  `tmfeo3-trilinear` model now actually sets the K^- and W couplings.

New config key: `su3_mc_manifold` (see above).
