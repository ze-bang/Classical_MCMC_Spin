# numerics_practice

A self-contained C++20 practice package for ODE and PDE integrators, building up
to state-of-the-art Langevin dynamics. Ten graded exercises; you fill in the
stubs, the tests grade you.

Nothing here depends on the rest of `ClassicalSpin_Cpp` — it is header-only and
has no external dependencies beyond the standard library.

## Build and run

```bash
cd numerics_practice
cmake -S . -B build && cmake --build build -j
cd build && ctest --output-on-failure     # red until you implement things
```

Run one exercise directly, optionally filtered by a substring of the test name:

```bash
./build/test_ex09
./build/test_ex09 baoab
```

To check your work against the reference implementation:

```bash
cmake -S . -B build-sol -DNP_SOLUTIONS=ON && cmake --build build-sol -j
cd build-sol && ctest --output-on-failure   # all green
```

`exercises/` and `solutions/` expose identical header names and namespaces, so
the test files never change — only the include path does. Work in
`exercises/`; peek at `solutions/` when you are stuck, ideally after you have
seen the test fail and formed a hypothesis about why.

A test reports `TODO` while its stub still throws, `FAIL` with the numbers when
your implementation is wrong, and `OK` when it is right. `ctest` stays red until
everything is implemented.

## The curriculum

| # | File | Topic | Key result the test forces you to reproduce |
|---|------|-------|--------------------------------------|
| 01 | `ex01_ode_explicit` | Euler, RK2, RK4, Dormand–Prince 5(4), PI step control | fitted convergence orders 1/2/4; adaptive driver hits a requested tolerance |
| 02 | `ex02_symplectic` | symplectic Euler, velocity Verlet, leapfrog, Yoshida 4 | `det J = 1`; bounded energy error vs RK4's secular drift; angular momentum exact to roundoff |
| 03 | `ex03_implicit_stiff` | backward Euler, trapezoid, BDF2, Rosenbrock ROS2, exponential Euler | A- vs L-stability measured as an amplification factor; stiff Van der Pol; exponential integrator exact for constant forcing |
| 04 | `ex04_parabolic` | FTCS, Crank–Nicolson, Peaceman–Rachford ADI | the `r ≤ 1/2` cliff; CN's undershoot at a step; ADI 2nd order and unconditionally stable |
| 05 | `ex05_hyperbolic` | upwind, Lax–Wendroff, MUSCL+minmod, Rusanov for Burgers | Godunov's theorem in action: LW oscillates, MUSCL is TVD; correct shock speed from conservative form |
| 06 | `ex06_elliptic_spectral` | Jacobi/GS/SOR, geometric multigrid, radix-2 FFT, spectral derivatives, ETDRK4 | grid-independent V-cycle count; spectral accuracy beats 2nd-order FD by 10¹¹; ETDRK4 4th order with no stability limit |
| 07 | `ex07_sde_basics` | Euler–Maruyama, Milstein, stochastic Heun | strong order 0.5 vs 1.0, weak order 1.0; Itô and Stratonovich are different processes |
| 08 | `ex08_overdamped` | Brownian dynamics, exact OU, Leimkuhler–Matthews, MALA | invariant-measure bias O(dt) → O(dt²) → 0; Euler–Maruyama is *transient* on a quartic well |
| 09 | `ex09_underdamped` | BAOAB / ABOBA / OBABO / GJF splittings | the letter order decides your accuracy: exact `⟨q²⟩` for BAOAB/GJF, exact `⟨p²⟩` for OBABO, 100× separation on an anharmonic well |
| 10 | `ex10_gle_sllg` | Markovian GLE (colored-noise) thermostat; stochastic Landau–Lifshitz–Gilbert with Heun and semi-implicit SIB | FDT for the extended bath; exact configurational sampling with colored noise; `⟨S_z⟩ = L(B/kT)` independent of damping |

## Two testing techniques worth stealing

**Separate space error from time error.** Comparing a PDE solution to the
analytic PDE solution measures whichever error dominates, and you get garbage
convergence rates. Compare instead against the exact solution of the
*semi-discrete* system — for the heat equation, `sin(πx_j) exp(λ_h t)` with
`λ_h = -α(2-2cos(π dx))/dx²`. Now the fitted order is the time order alone.
(`ex04`)

**Measure a stochastic integrator's sampling bias with zero Monte Carlo.** On a
harmonic potential every Langevin scheme is an affine-Gaussian map
`z' = M z + b + Lξ`. `np::stationary_covariance()` recovers `M` and `L` by
calling *your* step function with scripted noise, then solves
`Σ = MΣMᵀ + LLᵀ`. The sampling bias becomes a deterministic number you can fit
a convergence order to — no error bars, no 10⁹-step runs. (`ex08`–`ex10`)

## Suggested path

01 → 02 → 03 is the ODE core; do it first, it is short and everything later
reuses it. 04 → 05 → 06 is the PDE core and can be skipped or deferred if you
only care about dynamics. 07 → 08 → 09 → 10 is the Langevin ladder and is the
point of the package — each rung is a small delta on the previous one:

```
Euler-Maruyama                              (07)  strong 0.5, weak 1
  + exact OU propagator for the linear part (08)  the "O" operator
  + noise averaging / Metropolis            (08)  bias O(dt²) / bias 0
  + operator splitting of the full dynamics (09)  BAOAB, GJF
  + auxiliary momenta and a memory kernel   (10)  GLE thermostat
  + a curved state space                    (10)  Cayley rotations, sLLG
```

## Where this connects to `ClassicalSpin_Cpp`

`ex10`'s `sllg_sib_step` is the Mentink semi-implicit scheme, and its
`⟨S_z⟩ = L(B/kT)` check is the sharpest single-number validation available for a
finite-temperature spin dynamics code: it simultaneously pins the noise
amplitude, the Stratonovich handling, and the damping normalisation, and it must
come out independent of α. `ex09`'s Lyapunov prober is directly reusable for
auditing any thermostat in the main repo.

## Reading

- Hairer, Nørsett & Wanner, *Solving ODEs I/II* — orders, stiffness, step control
- Hairer, Lubich & Wanner, *Geometric Numerical Integration* — shadow Hamiltonians, splitting
- LeVeque, *Finite Volume Methods for Hyperbolic Problems* — limiters, TVD, Riemann solvers
- Trefethen, *Spectral Methods in MATLAB*; Kassam & Trefethen (2005) — ETDRK4 and the contour trick
- Kloeden & Platen, *Numerical Solution of SDEs* — strong/weak orders, Itô–Taylor
- Leimkuhler & Matthews, *Molecular Dynamics* (2015) — BAOAB and the splitting analysis
- Grønbech-Jensen & Farago (2013); Ceriotti, Bussi & Parrinello (2010) — GJF and the GLE thermostat
- Mentink et al., J. Phys.: Condens. Matter 22, 176001 (2010) — the SIB scheme for sLLG
