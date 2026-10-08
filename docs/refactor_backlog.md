# Backlog

Structural work that is scoped but not done. Each item says why it matters
and what blocks it. Behaviour changes that did land are in
[`MIGRATION.md`](MIGRATION.md); the architecture is described in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Code structure

- **Split the lattice headers.** `lattice.h` (≈4.6 kLOC) and
  `mixed_lattice.h` (≈4.2 kLOC) still declare the class and inline most
  method bodies, so any change rebuilds every translation unit that includes
  them. The kernel layer, the dynamics engine and the PT / PA engines are
  already separate; what remains is moving the Monte Carlo sweeps, I/O and
  observables into their own headers or `.cpp` files behind the class
  declaration. Blocked only by effort: it touches every method and must land
  in one change.
- **Flat spin storage.** Spins are `std::vector<Eigen::VectorXd>` (one heap
  allocation per site). The hot kernels already work on raw pointers
  (`FlatView`, packed bond tables); a contiguous `n_sites × spin_dim` array
  would remove the last indirection and simplify packing for MPI and HDF5.
- **Header hygiene.** `simple_linear_alg.h`, `spin_config.h` and `unitcell.h`
  contain `using namespace std;` at namespace scope, which leaks into every
  includer. Mechanical to remove (clang-tidy
  `google-global-names-in-headers`), but it touches thousands of lines.
- **One FFT.** `core/fft.h` handles any length; `dynamics/structure_factor.h`,
  `mc/statistics.h` and `phonon_lattice.cpp` still carry their own
  power-of-two transforms.

## Algorithms

- **Geometric SU(3) integrators.** SU(3) spins (8-component `Lattice` and the
  MixedLattice Tm sector) integrate with explicit Runge–Kutta methods, so the
  norm and Casimirs drift at the integrator tolerance and Langevin noise is
  unavailable for them. `core/su3_coherent_state.h` already contains the
  coherent-state implicit-midpoint step (Dahlbom et al., PRB 106, 054423
  (2022)); it needs to be wired into the integration engines as a method.
- **Population annealing for MixedLattice and PhononLattice.** The engine
  (`mc/population_annealing.h`) is generic; these classes need a
  `PopulationWorker` adapter (random state, scalar observables) and a runner.
- **Checkpoint / restart.** Long SA / PT / PA / MD runs cannot resume after a
  walltime limit; the PT and PA engines already hold their whole state in
  packed buffers, which would make a checkpoint format straightforward.

## GPU

The CUDA path cannot be built in the CI container (no toolchain); its changes
were compile-checked with clang against the CUDA headers only.

- Gilbert damping, Langevin noise, trilinear couplings and twisted boundaries
  are not implemented on the device (those models run on the CPU, with a
  warning); neither are the geometric integrators or Monte Carlo.
- Drivers copy the full state to the host at every sample and reduce
  magnetisations with atomics (non-deterministic sums).
- The TmFeO3 delay-parallel 2DCS routine is hard-wired to `MPI_COMM_WORLD`;
  on a sub-communicator (parameter sweeps) its trials run serially.
