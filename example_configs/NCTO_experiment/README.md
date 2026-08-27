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

## E0 calibration (the one number the experiment cannot give directly)

Z*E0 is the mode effective charge times the field in mass-normalised units and is not
known for NCTO. Fig. 3f fixes it *relative to the switching response*: the measured
amplitude follows A = A₀/(1 + I₀/E²) with I₀ = 1.7 E²_max, i.e. the strongest pulse reaches
A/A₀ = 1/(1+1.7) = 0.37 of saturation. Procedure:

1. pilot scan E0 ∈ {50, 100, 200, 400, 800, 1600, 3200} at L = 24 to bracket the switching
   threshold (the impulsive 4.2 THz response is |Q|max = 0.0097 per unit Z*E0; the earlier
   resonant runs at 0.97 THz had 0.85, so thresholds move by ~×87);
2. refine, then define E0MAX as the amplitude where the switched fraction (or Δr₃) is 0.37
   of its saturated value;
3. run the 12-point fluence array at E0MAX·√(E²/E²_max).

Plausibility check to report alongside: the impulsive-limit physical amplitude at
300 kV/cm is 1–5 pm (percent-level exchange modulation); compare with |Q|max·λ_K2/|K|
at E0MAX.

## Observables to extract (post-processing of the HDF5 spin trajectories)

- r₃ = min/max S(M_i) over the three M points (3Q ↔ zigzag diagnostic);
- zigzag overlap per domain; switched fraction;
- scalar chirality / spin vorticity κ = Σ_△ S_i·(S_j×S_k) — the quantity the 800 nm
  circular-dichroism probe is sensitive to (Jin et al. 2025: Faraday rotation from the
  3Q vorticity), i.e. the model counterpart of Δη in Fig. 3b;
- Q_x(t), Q_y(t) and their FFT (Fig. 2b,c counterpart); note Δη in Fig. 2 is the
  probe birefringence ∝ lattice displacement, not the spins;
- double pulse: M_NL(t, τ) = M₁₂ − M₁ − M₂ for κ and r₃, FFT along τ (Fig. 4b,c).

## Files

- `common_ncto_6K.inc` — everything above except the protocol;
- `single_pulse.inc`, `phonon_window.inc`, `double_pulse.inc`, `langevin_6K.inc` — protocol blocks;
- `../../../ncto_phonon/runs/*.sbatch` — pilot, fluence array, double-pulse array.
