// su3_coherent_state.h
// -----------------------------------------------------------------------------
// SU(3) coherent-state classical spin dynamics for the TmFeO3 Tm sector.
//
// References (widely-accepted method in the SU(N) classical spin community):
//
//   [1] H. Zhang and C. D. Batista,
//       "Classical spin dynamics based on SU(N) coherent states,"
//       Phys. Rev. B 104, 104409 (2021).
//       arXiv:2106.14125
//
//   [2] D. Dahlbom, H. Zhang, C. Miles, X. Bai, C. D. Batista, K. Barros,
//       "Geometric integration of classical spin dynamics via a mean-field
//        Schrödinger equation,"
//       Phys. Rev. B 106, 054423 (2022).
//       arXiv:2204.07563
//
//   [3] D. Dahlbom et al.,
//       "Langevin dynamics of generalized spins as SU(N) coherent states,"
//       Phys. Rev. B 106, 235154 (2022).
//
//   [4] D. Dahlbom et al., Sunny.jl: A Julia package for spin dynamics,
//       Journal of Open Source Software 10(116), 8138 (2025).
//
// Zhang and Batista explicitly identify our regime — "S ≥ 1 systems with
// large single-ion anisotropy" — as the case where the SU(N) generalization
// is required rather than optional. The local Tm three-level problem
// (E1, E2, E3) is exactly an N = 3 single-ion anisotropy site.
//
// State representation
// --------------------
//   psi ∈ C^3 with ⟨psi|psi⟩ = 1   (a point on the projective manifold CP^2).
//   The Gell-Mann expectation values are
//       n^a = ⟨psi| λ^a |psi⟩,    a = 1..8,
//   which by construction always come from a physical pure-state density
//   matrix ρ = |psi⟩⟨psi|, so positivity and the Casimir constraints are
//   automatic — none of the "8-vector outside the positive cone" failure
//   modes of the bare Bloch parameterization can occur.
//
// Convention (energy and bracket)
// ------------------------------
//   The classical energy of every term is the coherent-state expectation of
//   the quantum operator it represents, E_classical(n) = ⟨psi|H|psi⟩ with
//   n^a = ⟨psi|λ^a|psi⟩. For a term linear in λ, H = Σ_a c_a λ^a has
//   E = c·n; the CEF diag(0, e1, e2) is therefore
//       diag(0, e1, e2) = (e1+e2)/3 · 1 − (e1/2) λ^3 − ((2 e2 − e1)/(2√3)) λ^8.
//   Since [λ^a, λ^b] = 2i f_{abc} λ^c, the Heisenberg equation
//   d⟨λ^a⟩/dt = ⟨i[H, λ^a]⟩ with the mean-field H = Σ_b (∂E/∂n^b) λ^b gives
//   the Lie-Poisson equation of motion
//       dn^a/dt = 2 f_{abc} (∂E/∂n^b) n^c,            (kGellMannBracket = 2)
//   i.e. the bracket {n^a, n^b} = 2 f_{abc} n^c, which is what makes the
//   coupled SU(2)+SU(3) dynamics reproduce the time-dependent mean-field
//   (product-state) Schrödinger evolution of the quantum model term by term.
//   The SU(2) analogue is dS/dt = (∂E/∂S) × S for S = ⟨S_op⟩.
//
//   Releases before 2026-10 evolved dn^a/dt = f_{abc} (∂E/∂n^b) n^c and
//   encoded the CEF at twice its Gell-Mann coefficients, so the bare CEF
//   lines had the right frequencies but every other λ-linear coupling
//   (Zeeman, Fe-Tm exchange, drives) acted on Tm with half its torque, and
//   MC energies weighted the CEF twice. `kLegacyBracket` (with the legacy
//   CEF encoding, config key `su3_legacy_convention = 1`) reproduces that
//   behaviour exactly; see docs/MIGRATION.md.
//
//   Pure states satisfy |n|^2 = 4/3 (Tr ρ^2 = 1/3 + |n|^2/2 = 1) and
//   d_{abc} n^a n^b n^c = 8/9; both Casimirs are conserved by the
//   Lie-Poisson flow for any Hamiltonian.
//
// Time integration
// ----------------
//   The mean-field Schrödinger equation i dψ/dt = H_loc[ψ] ψ, with
//   H_loc = (bracket/2) Σ_a h^a λ^a and h = ∂E/∂n, is equivalent to the
//   Bloch equation above. For frozen neighbours over a step Δt the exact
//   local propagator is the unitary U(Δt) = exp(−i H_loc Δt)
//   (`propagator_exact`); `implicit_midpoint_step` is the symplectic,
//   norm-preserving, second-order midpoint rule of Ref. [2] for
//   self-consistent H_loc[ψ].
//
// Static T = 0 update
// -------------------
//   For minimisation / annealing, the local update consistent with the
//   above is local exact diagonalization: with all neighbours frozen,
//   set ψ to the lowest eigenvector of H_loc. This is a direct upgrade
//   of the present "align 8-vector antiparallel to local field" rule in
//   `MixedLattice::deterministic_sweep_local_field`, which can place ρ
//   outside the positive cone (see seed file
//   `example_configs/TmFeO3/tmfeo3_gamma2_1x1x1_seed_SU3.txt`, whose
//   stored (λ_3, λ_8) imply a Tm-level population p_3 ≈ −0.19).
//
// Scope
// -----
//   This header is intentionally self-contained and N = 3 only — the
//   only case the Tm sector needs. Generalisation to other N would
//   replace the closed-form Gell-Mann constants below with the
//   structure-constant routines already in
//   `classical_spin/core/simple_linear_alg.h`.
// -----------------------------------------------------------------------------
#ifndef CLASSICAL_SPIN_SU3_COHERENT_STATE_H
#define CLASSICAL_SPIN_SU3_COHERENT_STATE_H

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <complex>

namespace classical_spin {
namespace su3 {

using Complex   = std::complex<double>;
using Vector3c  = Eigen::Matrix<Complex, 3, 1>;
using Matrix3c  = Eigen::Matrix<Complex, 3, 3>;
using Vector8r  = Eigen::Matrix<double,  8, 1>;

// -----------------------------------------------------------------------------
// Gell-Mann matrices (1-indexed convention, returned as 0-indexed array).
// Normalisation: Tr(λ^a λ^b) = 2 δ^{ab}.
// -----------------------------------------------------------------------------
inline const std::array<Matrix3c, 8>& gell_mann() {
    static const std::array<Matrix3c, 8> kLambda = [] {
        std::array<Matrix3c, 8> L;
        const Complex I(0.0, 1.0);
        for (auto& m : L) m.setZero();

        // λ_1
        L[0](0, 1) = 1.0;  L[0](1, 0) = 1.0;
        // λ_2
        L[1](0, 1) = -I;   L[1](1, 0) =  I;
        // λ_3
        L[2](0, 0) = 1.0;  L[2](1, 1) = -1.0;
        // λ_4
        L[3](0, 2) = 1.0;  L[3](2, 0) = 1.0;
        // λ_5
        L[4](0, 2) = -I;   L[4](2, 0) =  I;
        // λ_6
        L[5](1, 2) = 1.0;  L[5](2, 1) = 1.0;
        // λ_7
        L[6](1, 2) = -I;   L[6](2, 1) =  I;
        // λ_8
        const double s = 1.0 / std::sqrt(3.0);
        L[7](0, 0) = s;    L[7](1, 1) = s;    L[7](2, 2) = -2.0 * s;
        return L;
    }();
    return kLambda;
}

// -----------------------------------------------------------------------------
// State conversions
// -----------------------------------------------------------------------------

// n^a = ⟨ψ| λ^a |ψ⟩.
inline Vector8r expectations_from_psi(const Vector3c& psi) {
    const auto& L = gell_mann();
    Vector8r n;
    for (int a = 0; a < 8; ++a) {
        const Complex z = psi.adjoint() * (L[a] * psi);
        n(a) = z.real();
    }
    return n;
}

// Build the qutrit density matrix ρ from a Bloch vector:
//   ρ = (1/3) 1 + (1/2) Σ_a n^a λ^a.
// (Valid as a density matrix iff ρ ≥ 0 and Tr ρ = 1; the second is
//  automatic, the first is the physical positivity constraint that the
//  bare 8-vector parameterization can violate.)
inline Matrix3c density_from_expectations(const Vector8r& n) {
    const auto& L = gell_mann();
    Matrix3c rho = (1.0 / 3.0) * Matrix3c::Identity();
    for (int a = 0; a < 8; ++a) {
        rho.noalias() += 0.5 * n(a) * L[a];
    }
    return rho;
}

// Closest pure-state ψ to a given Bloch vector n^a, via the largest-eigenvalue
// eigenvector of the implied ρ. If n^a comes from a physical pure state, this
// recovers ψ up to a global phase; otherwise it returns the maximum-overlap
// projector approximation. The largest-eigenvalue itself is returned in
// `out_purity` (= 1 iff ψ is exact, < 1 if the input n^a is mixed / unphysical).
inline Vector3c psi_from_expectations(const Vector8r& n, double* out_purity = nullptr) {
    Matrix3c rho = density_from_expectations(n);
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(rho);
    const int top = 2;  // Eigen sorts ascending; largest is index 2.
    if (out_purity) *out_purity = es.eigenvalues()(top);
    Vector3c psi = es.eigenvectors().col(top);
    psi.normalize();
    return psi;
}

// -----------------------------------------------------------------------------
// Local Hamiltonian and propagators
// -----------------------------------------------------------------------------

// Lie-Poisson bracket coefficient c in dn^a/dt = c f_{abc} (∂E/∂n^b) n^c for
// n = ⟨λ⟩ (Tr λ^a λ^b = 2 δ^{ab}): [λ^a, λ^b] = 2i f_{abc} λ^c gives c = 2.
inline constexpr double kGellMannBracket = 2.0;
// Coefficient used before the convention fix (see the header comment).
inline constexpr double kLegacyBracket = 1.0;

// Local 3×3 mean-field Hamiltonian H_loc = h_0 1 + Σ_a h^a λ^a, the operator
// whose expectation is the classical energy h_0 + h·n.
inline Matrix3c local_hamiltonian(const Vector8r& h_a, double h_0 = 0.0) {
    const auto& L = gell_mann();
    Matrix3c H = h_0 * Matrix3c::Identity();
    for (int a = 0; a < 8; ++a) {
        H.noalias() += h_a(a) * L[a];
    }
    return H;
}

// Schrödinger-picture generator of the Bloch dynamics driven by the local
// field H_field = ∂E/∂n returned by `get_local_field_SU3(_flat_into)`:
//     H_loc = (bracket/2) Σ_a H_field^a λ^a,
// so that d⟨λ^a⟩/dt = i⟨[H_loc, λ^a]⟩ = bracket f_{abc} H_field^b n^c
// (use [λ^b, λ^a] = 2i f_{bac} λ^c and f_{bac} = −f_{abc}). With the default
// bracket this is `local_hamiltonian(H_field)`. A positive rescaling does not
// change eigenvectors, so ground states do not depend on the bracket.
inline Matrix3c local_hamiltonian_from_field(const Vector8r& H_field, double h_0 = 0.0,
                                             double bracket = kGellMannBracket) {
    return local_hamiltonian(0.5 * bracket * H_field, h_0);
}

// Exact one-step unitary propagator U(Δt) = exp(−i H_loc Δt) for a 3×3
// Hermitian H_loc, from the eigendecomposition H_loc = V D V^† (Eigen's
// complex SelfAdjointEigenSolver: Householder tridiagonalisation + QR; the
// closed-form computeDirect path exists only for real matrices).
inline Matrix3c propagator_exact(const Matrix3c& H_loc, double dt) {
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(H_loc);
    const auto& V = es.eigenvectors();
    Eigen::Matrix<Complex, 3, 1> phase;
    const Complex mi_dt(0.0, -dt);
    for (int k = 0; k < 3; ++k) phase(k) = std::exp(mi_dt * es.eigenvalues()(k));
    return V * phase.asDiagonal() * V.adjoint();
}

// Apply the exact local unitary step in-place, preserving ⟨ψ|ψ⟩ = 1 to
// floating-point precision (numerical re-normalisation is included as a
// safety net; should be a no-op modulo round-off).
inline void unitary_step_explicit(Vector3c& psi, const Matrix3c& H_loc, double dt) {
    psi = propagator_exact(H_loc, dt) * psi;
    const double n = psi.norm();
    if (n > 0.0) psi /= n;
}

// Implicit-midpoint step of the mean-field Schrödinger equation, Ref. [2]:
//     ψ_new − ψ_old = −i Δt H_loc(ψ_mid) ψ_mid,   ψ_mid = (ψ_old + ψ_new)/2.
// For fixed H = H_loc(ψ_mid) this is the Cayley transform
//     ψ_new = (1 + iΔt H/2)^{-1} (1 − iΔt H/2) ψ_old,
// which is unitary, so ⟨ψ|ψ⟩ is conserved to round-off without any
// renormalisation; the rule is symplectic, time-reversible and second order.
// The self-consistency in ψ_mid (H_loc depends on ψ through neighbours /
// on-site nonlinearities) is solved by fixed-point iteration, which
// converges for Δt ‖∂H/∂ψ‖ small. `H_loc_of` is any callable
// Matrix3c(const Vector3c& psi_mid). Returns the number of iterations used.
template <class H_loc_fn>
inline int implicit_midpoint_step(Vector3c& psi, H_loc_fn H_loc_of, double dt,
                                  int max_iters = 50, double tol = 1e-14) {
    const Vector3c psi_old = psi;
    Vector3c psi_new = psi;
    const Complex half_idt(0.0, 0.5 * dt);
    int it = 0;
    for (; it < max_iters; ++it) {
        const Vector3c psi_mid = 0.5 * (psi_old + psi_new);
        const Matrix3c H = H_loc_of(psi_mid);
        const Matrix3c A = Matrix3c::Identity() + half_idt * H;
        const Vector3c rhs = psi_old - half_idt * (H * psi_old);
        const Vector3c psi_next = A.partialPivLu().solve(rhs);
        const double err = (psi_next - psi_new).norm();
        psi_new = psi_next;
        if (err < tol) { ++it; break; }
    }
    psi = psi_new;
    return it;
}

// Former name (the old body was a first-order θ = 1/4 scheme, not the
// midpoint rule its documentation described).
template <class H_loc_fn>
inline void spherical_midpoint_step(Vector3c& psi, H_loc_fn H_loc_of, double dt,
                                    int max_iters = 50, double tol = 1e-14) {
    implicit_midpoint_step(psi, H_loc_of, dt, max_iters, tol);
}

// -----------------------------------------------------------------------------
// Casimir invariants of n (conserved by the Lie-Poisson flow).
// -----------------------------------------------------------------------------
// Quadratic Casimir |n|^2 (= 4/3 for a pure state).
inline double casimir2(const double* n) {
    double s = 0.0;
    for (int a = 0; a < 8; ++a) s += n[a] * n[a];
    return s;
}

// Cubic Casimir d_{abc} n^a n^b n^c (= 8/9 for a pure state), with the
// symmetric Gell-Mann constants d_{abc} = Tr({λ^a, λ^b} λ^c)/4.
inline double casimir3(const double* n) {
    static constexpr double r3 = 0.57735026918962576451;  // 1/sqrt(3)
    const double n1 = n[0], n2 = n[1], n3 = n[2], n4 = n[3];
    const double n5 = n[4], n6 = n[5], n7 = n[6], n8 = n[7];
    // Sum over all index orderings of the distinct non-zero d's:
    //   d_118 = d_228 = d_338 = 1/√3, d_888 = −1/√3,
    //   d_448 = d_558 = d_668 = d_778 = −1/(2√3),
    //   d_146 = d_157 = d_256 = d_344 = d_355 = 1/2,
    //   d_247 = d_366 = d_377 = −1/2.
    return 3.0 * r3 * (n1 * n1 + n2 * n2 + n3 * n3) * n8
         - r3 * n8 * n8 * n8
         - 1.5 * r3 * (n4 * n4 + n5 * n5 + n6 * n6 + n7 * n7) * n8
         + 3.0 * (n1 * n4 * n6 + n1 * n5 * n7 + n2 * n5 * n6 - n2 * n4 * n7)
         + 1.5 * n3 * (n4 * n4 + n5 * n5 - n6 * n6 - n7 * n7);
}

// -----------------------------------------------------------------------------
// Static T = 0 update: local ground state of H_loc.
// -----------------------------------------------------------------------------
inline Vector3c ground_state(const Matrix3c& H_loc) {
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(H_loc);
    Vector3c psi = es.eigenvectors().col(0);  // ascending sort -> smallest is col 0
    psi.normalize();
    return psi;
}

// Smallest eigenvalue (ground-state energy) of H_loc.
inline double ground_state_energy(const Matrix3c& H_loc) {
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(H_loc, Eigen::EigenvaluesOnly);
    return es.eigenvalues()(0);
}

// -----------------------------------------------------------------------------
// Diagnostics: physicality of a stored Gell-Mann 8-vector.
// -----------------------------------------------------------------------------
// Returns the three eigenvalues of the implied ρ. For a physical qutrit
// state these are non-negative and sum to 1; negative entries are a
// quantitative signature that the stored 8-vector does not correspond to
// any valid qutrit density matrix.
inline Eigen::Vector3d density_eigenvalues(const Vector8r& n) {
    Matrix3c rho = density_from_expectations(n);
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(rho, Eigen::EigenvaluesOnly);
    return es.eigenvalues();
}

}  // namespace su3
}  // namespace classical_spin

#endif  // CLASSICAL_SPIN_SU3_COHERENT_STATE_H
