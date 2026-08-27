# ============================================================================
# Na2Co2TeO6 THz pump-probe: common block, set up as in NCTO_phonon_v0 (Gedik group)
# ============================================================================
# Units: energy meV, time hbar/meV = 0.658212 ps, temperature in meV (k_B = 0.086173 meV/K).
# This block is concatenated with a protocol block (single pulse / phonon window /
# fluence point / double pulse) by the run scripts in ncto_phonon/runs.
system = NCTO
lattice_size = 36,36,1              # 2592 Co; per-site phonon normalisation makes the
                                    # lattice dynamics size-independent, domains need size
num_trials = 1

# ---- spin Hamiltonian: Kruger et al. neutron fit, J7 at the corrected 3Q/ZZ near-degeneracy
J = 0.68
K = -7.89
Gamma = 3.07
Gammap = -2.94
J2_A = -0.06
J2_B = -0.70
J3 = 0.52
J7 = -0.4096
field_strength = 0.0                # experiment: zero applied field
field_direction = 0,0,1
alpha_gilbert = 0.05                # NOT measured; magnon damping knob (manuscript value)

# ---- the ONLY driven lattice coordinate: the polar E1 doublet at the dominant 4.2 THz line
omega_E1 = 17.37                    # h * 4.2 THz  (period 0.238 ps = 0.362 code units)
gamma_E1 = 0.20                     # amplitude ring-down 2/gamma = 6.6 ps (measured 5-10 ps)
lambda_E1_quartic = 0.0
Z_star = 1.0                        # absorbed into E0
phonon_per_site = 1
n_extra_modes = 0                   # no Raman / A2 / strain coordinate is driven
n_anharmonic = 0

# ---- magnetoelastic couplings (unknown constants; only the FORM is fixed by symmetry)
# second order (rectifying), Grueneisen-scaled lambda_X2 = (X/K) lambda_K2, lambda_K2 = 0.02:
lambda_E1_J_0 = 0.0
lambda_E1_J_2 = -0.0017
lambda_E1_K_0 = 0.0
lambda_E1_K_2 = 0.02
lambda_E1_Gamma_0 = 0.0
lambda_E1_Gamma_2 = -0.0078
lambda_E1_Gammap_0 = 0.0
lambda_E1_Gammap_2 = 0.0075
lambda_E1_J7_0 = 1.0e-3             # phonon effect on the ring exchange: J7 -> J7 + 1e-3 |Q|^2
# first order (odd in Q, D3-allowed): off in the baseline; scan in a separate series
lambda_E1_J_1 = 0.0
lambda_E1_K_1 = 0.0
lambda_E1_Gamma_1 = 0.0
lambda_E1_Gammap_1 = 0.0
mode0_cE5 = 0.0
mode0_cE6 = 0.0
mode0_cE7 = 0.0
mode0_cE8 = 0.0
mode0_cE9 = 0.0

# ---- THz pulse as measured (SI Fig. S1b,c): ~1.5-cycle transient, envelope FWHM ~0.6 ps,
#      spectrum 1-5 THz peaking at 3-4 THz.  Gaussian x cos with sigma_t = 0.18 ps = 0.273 units,
#      carrier 3.0 THz = 12.41 meV, CEP 0 (spectral FWHM 2.1 THz, 99.9% weight in 1-5 THz).
pump_frequency = 12.41
pump_width = 0.273
pump_phase = 0.0
pump_time = 10.0                    # pulse centre at 6.6 ps after the start (field = 0 at t = 0)
pump_polarization = 0.0             # angle from the x-bond LINE; crystal orientation vs E not
                                    # reported -> run 0 and pi/2 (see README)
# E0 is the free amplitude (Z* E0); calibrated against Fig. 3f (see README): E0MAX below.

# ---- integration: 72 RK4 steps per 4.2 THz period
md_integrator = rk4
md_timestep = 0.005

# ---- initial state: relaxed triple-q at T = 0 (baseline) via simulated annealing + T=0 sweeps,
#      lattice relaxed adiabatically.  Check r3 = S(M1)/S(M3) = 1 in the log before trusting a run.
adiabatic_phonons = false
relax_phonons = true
T_start = 20.0
T_end = 1e-3
T_zero = true
annealing_steps = 80000
cooling_rate = 0.9
overrelaxation_rate = 0
n_deterministics = 5000
# initial_spin_config = /path/to/3Q_L36_seed.txt   # uncomment to skip SA and start from a seed
