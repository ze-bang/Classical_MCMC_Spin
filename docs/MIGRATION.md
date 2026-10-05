# Migration notes

One section per behaviour change. Each lists what changed, why, and how to
recover the old behaviour where that is meaningful.

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
