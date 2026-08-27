# Na2Co2TeO6 THz experiment (NCTO_phonon_v0) — simulation set-up

All numbers below are what the experiment fixes; everything else is a stated choice.
Time unit 0.658212 ps, energy meV, k_B = 0.086173 meV/K. The protocol files are
assembled as `cat common_ncto_6K.inc <protocol>.inc > run.param` by the scripts in
`ncto_phonon/runs/`.

## What the experiment fixes

| quantity | experiment | code |
|---|---|---|
| sample temperature | 6 K for the pulse/fluence/double-pulse data (T-series 6–30 K) | baseline: T = 0 relaxed 3Q; thermal variant: Langevin, `langevin_temperature = 0.517` (6 K) |
| applied field | none | `field_strength = 0` |
| ground state | triple-q (r₃ = 1), T_N ≈ 27 K | SA + T=0 sweeps at J₇ = −0.4096; verify r₃ = 1 |
| driven phonon | polar E₁ doublet; dominant line 4.2 THz, ring-down 5–10 ps | `omega_E1 = 17.37`, `gamma_E1 = 0.20` (6.6 ps); nothing else driven (`n_extra_modes = 0`) |
| pulse shape | ~1.5 cycles, envelope FWHM ≈ 0.6 ps, 1–5 THz | `pump_width = 0.273` (σ = 0.18 ps), `pump_frequency = 12.41` (3.0 THz), CEP 0 |
| peak field / fluence | 300 kV/cm / 33 µJ cm⁻² at maximum | `pump_amplitude = E0MAX` (calibrated, see below) |
| polarization | linear, in-plane, orientation to the crystal axes not reported | `pump_polarization` = 0 and π/2 (relative to the x-bond line) |
| single-pulse window | −100 … 500 ps, rise time ≈ 70 ps | `md_time_end = 770` (500 ps after the pulse at t₀ = 10) |
| phonon window | 0 … 40 ps, 0.24 ps period | separate short run, save every 0.013 ps |
| fluence series (Fig. 3f) | E²/E²_max = 0.05 … 1, saturation I₀ = 1.7 E²_max | 12-point array in E0 = E0MAX·√(E²/E²_max) |
| double pulse (Fig. 4) | second pulse ≈ 0.6 × first, τ ∈ [−5, 5] ps, t ≤ 600 ps, modulation = M₁₂ − M₁ − M₂ | `probe_amplitude = 0.6·E0`, `probe_time = 10 + τ`, three runs per τ, `md_time_end = 920` |
| repetition rate | 200/500 Hz, decay 0.86 ms | single-shot simulation (each pulse sees a fresh 3Q state) — cannot be reproduced |

## Couplings: first order is the leading term

For a polar E₁ mode of one D₃ layer the leading magnetoelastic term is LINEAR,
δX_γ = λ_{X,1} Q·d̂_γ (bond projection of the displacement). Defaults: λ_{K,1} = 40 meV/Q with
Grüneisen ratios λ_{X,1} = (X/K)λ_{K,1} for J, Γ, Γ′ — a 5 % modulation of every exchange at
the physical amplitude |Q| ≈ 0.01. The second-order bilinear terms λ_{X,0}, λ_{X,2} are OFF:
the old λ_{X,2} = (X/K)·0.02 came from applying D₆ to a single layer, and the rectified
E₂-shear physics it modelled arises automatically at O(λ₁²/ω²) once the linear modulation is
integrated in time. The ring exchange is A₁ and couples only through |Q|² (`lambda_E1_J7_0`);
it is off in the baseline and on (10⁻³) in scenario 2. The other five linear E doublets
(`mode0_cE5..9`) have no estimate yet and stay 0.

**Q normalisation (one dictionary, code and estimates):** E_ph = N·½ω²|Q|², E_drive = −N Z*E·Q,
δX = λ₁ Q·d̂_γ, Q̈ = −ω²Q − γQ̇ + Z*E(t) − (1/N)∂H_ME/∂Q; Q = √(M_cell/2)·u/ħ, so 7 µeV/Co
of absorbed energy at 4.2 THz ↔ |Q| ≈ 0.007 ↔ Z*E₀ ≈ 1 (impulsive response 0.0097 per unit
Z*E₀). See the `PhononState` header comment.

## E0: fixed by the energy budget, not by the switching response

Z*E0 is not known microscopically, but the absorbed fluence fixes the phonon energy:
33 µJ cm⁻² over an absorption depth d gives 7 µeV per Co for d = 20 µm (70 µeV for 2 µm,
1.4 µeV for 100 µm). With the code's phonon energy N·½ω²|Q|² (ω = 17.37) this is
|Q| ≈ 0.007 (0.02 for d = 2 µm), and the impulsive response of the matched pulse is
|Q|max = 0.0097 per unit Z*E0, so **E0MAX ≈ 1 (range 0.7–3)** — see
`ncto_phonon/audit/NCTO_mechanism_exercise.md`. This is the experimental operating point;
`pump_amplitude = 1.0` is the SpinConfig default. The earlier resonant runs used 20 at
0.97 THz (|Q| ≈ 14–28), i.e. 10³–10⁴ times the physical phonon energy.

Do **not** calibrate E0 to reproduce switching: at E0 ≈ 1 the rectified bias λ_K2|Q|² is
~10⁻⁶ meV against a 3Q/ZZ splitting of 5.5 µeV, so a null coherent response is the
expected (and informative) result. The fluence array spans E0 = E0MAX·√(E²/E²_max) with
E0MAX = 1; the pilot bracket {100 … 6400} is retained only to locate where the coherent
mechanism *would* switch, for the record.

## Observables to extract (post-processing of the HDF5 spin trajectories)

- r₃ = min/max S(M_i) over the three M points (3Q ↔ zigzag diagnostic);
- zigzag overlap per domain; switched fraction;
- scalar chirality / spin vorticity κ = Σ_△ S_i·(S_j×S_k) — the quantity the 800 nm
  circular-dichroism probe is sensitive to (Jin et al. 2025: Faraday rotation from the
  3Q vorticity), i.e. the model counterpart of Δη in Fig. 3b;
- Q_x(t), Q_y(t) and their FFT (Fig. 2b,c counterpart); note Δη in Fig. 2 is the
  probe birefringence ∝ lattice displacement, not the spins;
- double pulse: M_NL(t, τ) = M₁₂ − M₁ − M₂ for κ and r₃, FFT along τ (Fig. 4b,c).

## Code defaults = this operating point

Since commit "defaults(ncto)", every value in `common_ncto_6K.inc` that lives in
`hamiltonian_params` (J…J₇, λ's, ω_E1, γ_E1, Z*, phonon_per_site, polarization) is also
the *default* of `SpinPhononCouplingParams` / `PhononParams` / `DriveParams` and of
`build_phonon_params`, so an NCTO config that omits them runs at the experimental point.
The pulse carrier and width fall back to 3.0 THz / σ = 0.273 whenever `pump_frequency` /
`pump_width` are absent (SpinConfig generic values); `pump_time`, `md_timestep`,
`md_integrator`, `md_time_end` and `lattice_size` are generic SpinConfig members and must
still be set explicitly (the common block does).

## Files

- `common_ncto_6K.inc` — everything above except the protocol;
- `single_pulse.inc`, `phonon_window.inc`, `double_pulse.inc`, `langevin_6K.inc` — protocol blocks;
- `../../../ncto_phonon/runs/*.sbatch` — pilot, fluence array, double-pulse array.
