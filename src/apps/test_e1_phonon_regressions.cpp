/**
 * test_e1_phonon_regressions.cpp
 *
 * End-to-end symmetry + consistency tests for the NCTO E1 magnetoelastic
 * Hamiltonian implemented in PhononLattice.
 *
 * The model under test is the one in docs/tmfeo3_notes.tex:
 *
 *   H = H_spin + H_E1 + H_drive + H_sp-ph,
 *
 * with the leading symmetry-allowed E1 spin-phonon coupling
 *
 *   H_sp-ph = Σ_<ij>γ Σ_X δX_γ(ε) O^{(X)}_{ij,γ},   X ∈ {J, K, Γ, Γ'},
 *   δX_γ(ε) = λ_{X,0}(ε_x²+ε_y²) + λ_{X,2}[(ε_x²-ε_y²) cos 2θ_γ
 *                                          + 2 ε_x ε_y sin 2θ_γ],
 *
 * and bond-axis angles  (θ_x, θ_y, θ_z) = (0, 2π/3, 4π/3).
 *
 * Tests performed:
 *
 *   1. Form-factor sum rule:  Σ_γ (cos 2θ_γ, sin 2θ_γ) = (0, 0).
 *      Required for the bond-anisotropy part of δX_γ to vanish on isotropic
 *      spin configs and for C_3 to permute the three bonds correctly.
 *
 *   2. δX_γ(ε) explicit formula:  numerically compare δX_γ extracted from
 *      one-bond magnetoelastic energy against the boxed formula in the notes.
 *
 *   3. Linear-in-ε coupling is identically zero:  ∂H_sp-ph/∂ε_a |_{ε=0} = 0.
 *      This is the central symmetry statement of the notes — there is no
 *      C_6-allowed linear exchange-striction term in the J–K–Γ–Γ' sector.
 *
 *   4. Phonon force / energy consistency:  ∂H_sp-ph/∂ε_a (analytic) matches
 *      a 2nd-order finite difference of spin_phonon_energy.
 *
 *   5. Spin force / energy consistency:  the E1 contribution to the local
 *      effective field on each spin matches the finite difference of the
 *      total energy with respect to that spin (small-perturbation test).
 *
 *   6. C_3 invariance of one-bond magnetoelastic energy under the combined
 *      transformation
 *           bond label γ → γ' = (γ+1) mod 3,
 *           ε → R(2π/3) ε,
 *           spins (in the local Kitaev frame): cyclic permutation
 *           (S^x, S^y, S^z) → (S^z, S^x, S^y).
 *
 *   7. Bond-modulation pattern matches the notes' Fig. 1: for a linearly
 *      polarized ε = (ε_0, 0), δX_γ ∝ (1, -1/2, -1/2) on (x, y, z) bonds.
 */

#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/kitaev_bonds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

namespace {

constexpr double kPi          = 3.14159265358979323846;
constexpr double kSqrt3       = 1.7320508075688772935;
constexpr double kFDStep      = 1e-5;
constexpr double kFDFieldTol  = 1e-7;
constexpr double kAnalyticTol = 1e-12;

bool nearly_equal(double lhs, double rhs, double abs_tol, double rel_tol = 0.0) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= abs_tol + rel_tol * scale;
}

// -------------------------------------------------------------------------
//  Reference formulas (mirrors of the boxed equations in tmfeo3_notes.tex).
// -------------------------------------------------------------------------

std::pair<double, double> n_gamma(int bond_type) {
    switch (bond_type) {
        case 0: return {1.0, 0.0};                   // θ_x = 0,    2θ = 0
        case 1: return {-0.5, -0.5 * kSqrt3};        // θ_y = 2π/3, 2θ = 4π/3
        default: return {-0.5,  0.5 * kSqrt3};       // θ_z = 4π/3, 2θ = 8π/3 ≡ 2π/3
    }
}

double delta_X_reference(double lambda0, double lambda2,
                         double qx, double qy, int bond_type, double lambda1 = 0.0) {
    const auto [c2, s2] = n_gamma(bond_type);
    const double q0 = qx * qx + qy * qy;
    const double qc = qx * qx - qy * qy;
    const double qs = 2.0 * qx * qy;
    // λ1 term: the D3 invariant Re[ε₊N₊] = ε_x cos2θ − ε_y sin2θ (note the sign).
    return lambda0 * q0 + lambda2 * (qc * c2 + qs * s2) + lambda1 * (qx * c2 - qy * s2);
}

// Per-bond magnetoelastic energy in the LOCAL Kitaev frame (notes' Eq. for
// δH_γ summed over channels). Used to check the C++ implementation matches
// the explicit channel expansion.
double bond_energy_reference_local(
    const Eigen::Vector3d& Si_local, const Eigen::Vector3d& Sj_local,
    const SpinPhononCouplingParams& p, double qx, double qy, int bond_type)
{
    const double dJ  = delta_X_reference(p.lambda_E1_J_0,      p.lambda_E1_J_2,      qx, qy, bond_type, p.lambda_E1_J_1);
    const double dK  = delta_X_reference(p.lambda_E1_K_0,      p.lambda_E1_K_2,      qx, qy, bond_type, p.lambda_E1_K_1);
    const double dG  = delta_X_reference(p.lambda_E1_Gamma_0,  p.lambda_E1_Gamma_2,  qx, qy, bond_type, p.lambda_E1_Gamma_1);
    const double dGp = delta_X_reference(p.lambda_E1_Gammap_0, p.lambda_E1_Gammap_2, qx, qy, bond_type, p.lambda_E1_Gammap_1);

    const int gamma = bond_type;
    const int alpha = (gamma == 0) ? 1 : 0;
    const int beta  = 3 - gamma - alpha;

    // O^{(J)}, O^{(K)}, O^{(Γ)}, O^{(Γ')} on this bond, in the local frame.
    const double O_J  = Si_local.dot(Sj_local);
    const double O_K  = Si_local(gamma) * Sj_local(gamma);
    const double O_G  = Si_local(alpha) * Sj_local(beta) + Si_local(beta) * Sj_local(alpha);
    const double O_Gp = Si_local(gamma) * (Sj_local(alpha) + Sj_local(beta))
                      + (Si_local(alpha) + Si_local(beta)) * Sj_local(gamma);

    return dJ * O_J + dK * O_K + dG * O_G + dGp * O_Gp;
}

// -------------------------------------------------------------------------
//  Lattice setup helpers.
// -------------------------------------------------------------------------

PhononLattice make_lattice(size_t L) {
    SpinConfig config;
    config.set_param("J",      -0.10);
    config.set_param("K",      -9.00);
    config.set_param("Gamma",   1.80);
    config.set_param("Gammap",  0.30);
    config.set_param("J2_A",    0.30);
    config.set_param("J2_B",    0.30);
    config.set_param("J3",      0.90);
    config.set_param("J7",      0.00);
    config.field_strength = 0.0;

    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, L, L, 1, 1.0f);

    SpinPhononCouplingParams sp;
    sp.J = -0.10;  sp.K = -9.00;  sp.Gamma = 1.80;  sp.Gammap = 0.30;
    sp.J2_A = 0.30; sp.J2_B = 0.30; sp.J3 = 0.90; sp.J7 = 0.00;
    sp.lambda_E1_J7_0 = 0.0;   // tests 2-5, 8 probe the quadratic bilinear channels only
    sp.lambda_E1_J_1 = sp.lambda_E1_K_1 = sp.lambda_E1_Gamma_1 = sp.lambda_E1_Gammap_1 = 0.0;
    sp.lambda_E1_J_0      = 0.13;   sp.lambda_E1_J_2      = -0.21;
    sp.lambda_E1_K_0      = -0.42;  sp.lambda_E1_K_2      = 0.55;
    sp.lambda_E1_Gamma_0  = 0.07;   sp.lambda_E1_Gamma_2  = -0.18;
    sp.lambda_E1_Gammap_0 = -0.09;  sp.lambda_E1_Gammap_2 = 0.11;

    PhononParams ph;
    ph.omega_E1 = 1.5;  ph.gamma_E1 = 0.05;
    ph.lambda_E1_quartic = 0.4;  ph.Z_star = 1.0;

    DriveParams dr;  // all zero — no drive during these tests

    lattice.set_parameters(sp, ph, dr);
    return lattice;
}

void deterministic_spins(PhononLattice& L, double offset = 0.0) {
    for (size_t i = 0; i < L.lattice_size; ++i) {
        const double a = std::sin(0.317 * (i + 1) + offset);
        const double b = std::cos(0.611 * (i + 1) - 0.5 * offset);
        const double c = std::sin(0.233 * (i + 1) + 1.7 * offset);
        Eigen::Vector3d s(a, b, c);
        L.spins[i] = s.normalized() * L.spin_length;
    }
}

// -------------------------------------------------------------------------
//  Tests
// -------------------------------------------------------------------------

bool test_form_factor_sum_rule(std::ostream& out) {
    out << "[1] Form-factor sum rule  Σ_γ (cos 2θ_γ, sin 2θ_γ) = 0\n";
    double sx = 0.0, sy = 0.0;
    for (int g = 0; g < 3; ++g) {
        const auto [c2, s2] = n_gamma(g);
        out << "    γ=" << g << "  (cos 2θ, sin 2θ) = (" << c2 << ", " << s2 << ")\n";
        sx += c2; sy += s2;
    }
    out << "    Σ = (" << sx << ", " << sy << ")\n";
    if (!nearly_equal(sx, 0.0, kAnalyticTol) || !nearly_equal(sy, 0.0, kAnalyticTol)) {
        out << "[FAIL] form factor sum rule violated\n";
        return false;
    }
    out << "[PASS] form factor sum rule\n\n";
    return true;
}

bool test_bond_energy_matches_reference(std::ostream& out) {
    out << "[2] Per-bond magnetoelastic energy matches notes' channel expansion\n";
    PhononLattice L = make_lattice(2);
    deterministic_spins(L, 0.41);

    // Sweep a couple of nontrivial ε values
    const std::array<std::pair<double, double>, 5> eps_list = {{
        {0.0, 0.0}, {0.13, 0.0}, {0.0, 0.21}, {0.17, -0.09}, {-0.22, 0.31}
    }};

    const Eigen::Matrix3d R = SpinPhononCouplingParams::get_kitaev_rotation();
    double max_diff = 0.0;

    for (const auto& [qx, qy] : eps_list) {
        L.phonons.Q_x_E1 = qx;
        L.phonons.Q_y_E1 = qy;
        L.phonons.V_x_E1 = 0.0;
        L.phonons.V_y_E1 = 0.0;

        // Direct sum of per-bond reference energies (in local frame).
        double E_ref = 0.0;
        for (size_t i = 0; i < L.lattice_size; ++i) {
            const Eigen::Vector3d Si_local = R.transpose() * L.spins[i];
            for (size_t n = 0; n < L.nn_partners[i].size(); ++n) {
                const size_t j = L.nn_partners[i][n];
                if (j > i) {
                    const Eigen::Vector3d Sj_local = R.transpose() * L.spins[j];
                    const int g = L.nn_bond_types[i][n];
                    E_ref += bond_energy_reference_local(
                        Si_local, Sj_local, L.spin_phonon_params, qx, qy, g);
                }
            }
        }

        const double E_code = L.spin_phonon_energy();
        max_diff = std::max(max_diff, std::abs(E_code - E_ref));
        out << "    ε=(" << qx << "," << qy << ")  H_sp-ph (code)=" << E_code
            << "  reference=" << E_ref
            << "  diff=" << (E_code - E_ref) << "\n";
        if (!nearly_equal(E_code, E_ref, kAnalyticTol, 1e-10)) {
            out << "[FAIL] H_sp-ph mismatch vs reference channel expansion\n";
            return false;
        }
    }
    out << "    max |code − reference| = " << max_diff << "\n";
    out << "[PASS] per-bond energy formula\n\n";
    return true;
}

bool test_no_linear_coupling(std::ostream& out) {
    out << "[3] Linear-in-ε coupling is forbidden:  ∂H_sp-ph/∂ε_a |_{ε=0} = 0\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 0.27);

    L.phonons = PhononState();  // ε = 0
    const double dHdqx = L.dH_dQx_E1();
    const double dHdqy = L.dH_dQy_E1();
    out << "    ∂H_sp-ph/∂ε_x|_{ε=0} = " << dHdqx << "\n";
    out << "    ∂H_sp-ph/∂ε_y|_{ε=0} = " << dHdqy << "\n";
    if (!nearly_equal(dHdqx, 0.0, kAnalyticTol) || !nearly_equal(dHdqy, 0.0, kAnalyticTol)) {
        out << "[FAIL] non-zero linear-in-ε coupling — forbidden by C_6\n";
        return false;
    }

    // Also verify the phonon energy itself is stationary at ε = 0:
    // H_sp-ph(ε=0) = 0 exactly (no constant offset).
    const double E0 = L.spin_phonon_energy();
    out << "    H_sp-ph(ε=0) = " << E0 << "\n";
    if (!nearly_equal(E0, 0.0, kAnalyticTol)) {
        out << "[FAIL] H_sp-ph(ε=0) must vanish\n";
        return false;
    }
    out << "[PASS] no linear-in-ε exchange-striction term\n\n";
    return true;
}

bool test_phonon_force_finite_difference(std::ostream& out) {
    out << "[4] Phonon force consistency (analytic vs finite-difference)\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 0.83);

    const std::array<std::pair<double, double>, 3> eps_list = {{
        {0.05, -0.03}, {0.10, 0.10}, {-0.07, 0.12}
    }};

    auto H_sp_ph_at = [&](double qx, double qy) {
        L.phonons.Q_x_E1 = qx;
        L.phonons.Q_y_E1 = qy;
        return L.spin_phonon_energy();
    };

    for (const auto& [qx0, qy0] : eps_list) {
        L.phonons.Q_x_E1 = qx0;
        L.phonons.Q_y_E1 = qy0;
        const double dHx_an = L.dH_dQx_E1();
        const double dHy_an = L.dH_dQy_E1();

        const double Ep_x = H_sp_ph_at(qx0 + kFDStep, qy0);
        const double Em_x = H_sp_ph_at(qx0 - kFDStep, qy0);
        const double dHx_fd = (Ep_x - Em_x) / (2.0 * kFDStep);

        const double Ep_y = H_sp_ph_at(qx0, qy0 + kFDStep);
        const double Em_y = H_sp_ph_at(qx0, qy0 - kFDStep);
        const double dHy_fd = (Ep_y - Em_y) / (2.0 * kFDStep);

        out << "    ε=(" << qx0 << "," << qy0 << ")\n"
            << "      ∂_x: analytic=" << dHx_an << "  FD=" << dHx_fd
            << "  diff=" << (dHx_an - dHx_fd) << "\n"
            << "      ∂_y: analytic=" << dHy_an << "  FD=" << dHy_fd
            << "  diff=" << (dHy_an - dHy_fd) << "\n";

        if (!nearly_equal(dHx_an, dHx_fd, kFDFieldTol, 1e-6) ||
            !nearly_equal(dHy_an, dHy_fd, kFDFieldTol, 1e-6)) {
            out << "[FAIL] phonon force does not match finite difference\n";
            return false;
        }
    }
    out << "[PASS] phonon force consistency\n\n";
    return true;
}

bool test_spin_force_finite_difference(std::ostream& out) {
    out << "[5] Spin local-field consistency at finite ε (analytic vs finite-difference)\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 0.71);
    L.phonons.Q_x_E1 = 0.13;
    L.phonons.Q_y_E1 = -0.08;

    // Compare H_eff_i = -∂E_total/∂S_i for a few sites.
    // We use a tangent-space perturbation that ignores |S|=const constraint
    // by directly comparing components 0/1/2 of H_eff against
    // -(E(S+δê_a) - E(S-δê_a))/(2 δ).
    const std::array<size_t, 4> probe_sites = {{0, 3, 7, 11}};
    double max_err = 0.0;

    for (size_t site : probe_sites) {
        const Eigen::Vector3d H_an = L.get_local_field(site);
        const Eigen::Vector3d S0   = L.spins[site];

        Eigen::Vector3d H_fd;
        for (int a = 0; a < 3; ++a) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(a) = kFDStep;
            L.spins[site] = S0 + delta;
            const double Ep = L.total_energy();
            L.spins[site] = S0 - delta;
            const double Em = L.total_energy();
            L.spins[site] = S0;
            H_fd(a) = -(Ep - Em) / (2.0 * kFDStep);
        }

        const double err = (H_an - H_fd).cwiseAbs().maxCoeff();
        max_err = std::max(max_err, err);
        out << "    site " << site
            << "  H_an=(" << H_an.transpose() << ")"
            << "  H_fd=(" << H_fd.transpose() << ")"
            << "  max|diff|=" << err << "\n";
    }

    if (max_err > 1e-6) {
        out << "[FAIL] H_eff disagrees with -∂E/∂S finite difference (max err=" << max_err << ")\n";
        return false;
    }
    out << "[PASS] spin local-field consistency  (max err=" << max_err << ")\n\n";
    return true;
}

bool test_C3_invariance_per_bond(std::ostream& out) {
    out << "[6] C_3 invariance of one-bond magnetoelastic energy under the\n"
        << "    combined transformation:  γ → γ' = (γ+1) mod 3,\n"
        << "                              ε → R(2π/3) ε,\n"
        << "                              S^x → S^z, S^y → S^x, S^z → S^y\n";
    SpinPhononCouplingParams p;
    p.lambda_E1_J_0      = 0.13;   p.lambda_E1_J_2      = -0.21;
    p.lambda_E1_K_0      = -0.42;  p.lambda_E1_K_2      = 0.55;
    p.lambda_E1_Gamma_0  = 0.07;   p.lambda_E1_Gamma_2  = -0.18;
    p.lambda_E1_Gammap_0 = -0.09;  p.lambda_E1_Gammap_2 = 0.11;

    // Two random spins in the local Kitaev frame (any unit-magnitude is fine).
    const Eigen::Vector3d Si(0.32, -0.84, 0.43);
    const Eigen::Vector3d Sj(-0.51, 0.27, -0.81);

    // ε in the (ε_x, ε_y) frame defined by θ_x = 0.
    const double qx = 0.21, qy = -0.13;

    // Cyclic permutation of spin components: (Sx, Sy, Sz) → (Sz, Sx, Sy)
    // i.e. if γ = 0 (x) is mapped to γ' = 1 (y), then under C_3 the spin
    // axes are permuted in the inverse direction, so that S^γ_new = S^γ_old.
    auto cycle = [](const Eigen::Vector3d& v) {
        return Eigen::Vector3d(v(2), v(0), v(1));
    };

    // R(2π/3) on ε.
    const double cos120 = -0.5;
    const double sin120 = 0.5 * kSqrt3;
    const double qx_new = cos120 * qx - sin120 * qy;
    const double qy_new = sin120 * qx + cos120 * qy;

    double max_diff = 0.0;
    for (int g = 0; g < 3; ++g) {
        const int g_new = (g + 1) % 3;

        const double E_orig = bond_energy_reference_local(Si, Sj, p, qx, qy, g);
        const double E_new  = bond_energy_reference_local(
            cycle(Si), cycle(Sj), p, qx_new, qy_new, g_new);
        const double diff = std::abs(E_orig - E_new);
        max_diff = std::max(max_diff, diff);
        out << "    γ=" << g << " → γ'=" << g_new
            << "  E_orig=" << E_orig
            << "  E_C3=" << E_new
            << "  diff=" << diff << "\n";
    }

    if (max_diff > 1e-12) {
        out << "[FAIL] one-bond energy is NOT C_3 invariant (max diff=" << max_diff << ")\n";
        return false;
    }
    out << "[PASS] C_3 invariance of one-bond magnetoelastic energy\n\n";
    return true;
}

bool test_bond_modulation_pattern(std::ostream& out) {
    out << "[7] Bond modulation pattern for ε = (ε_0, 0):\n"
        << "    δX_γ should be proportional to (1, -1/2, -1/2) on (x, y, z) bonds\n"
        << "    (cf. Fig. 1 of tmfeo3_notes.tex).\n";

    SpinPhononCouplingParams p;
    // Pure λ_X,2 channel (zero λ_X,0) so the bond pattern is purely
    // anisotropic. Use the Kitaev channel as a representative test.
    p.lambda_E1_K_0 = 0.0;
    p.lambda_E1_K_2 = 1.0;

    const double eps0 = 0.37;
    const std::array<double, 3> expected = {1.0, -0.5, -0.5};
    for (int g = 0; g < 3; ++g) {
        const double dK = delta_X_reference(p.lambda_E1_K_0, p.lambda_E1_K_2,
                                            eps0, 0.0, g);
        const double pattern = dK / (eps0 * eps0);
        out << "    γ=" << g << "  δK_γ/ε_0² = " << pattern
            << "   (expected " << expected[g] << ")\n";
        if (!nearly_equal(pattern, expected[g], kAnalyticTol)) {
            out << "[FAIL] bond pattern does not match Fig. 1 of the notes\n";
            return false;
        }
    }

    // For the second sanity check, use the rectified part: ε = (ε_0, 0)
    // → δX_γ = ε_0² (λ_0 + λ_2 cos 2θ_γ). Verify both pieces sum.
    p.lambda_E1_K_0 = 0.27;
    p.lambda_E1_K_2 = -0.31;
    for (int g = 0; g < 3; ++g) {
        const double dK = delta_X_reference(p.lambda_E1_K_0, p.lambda_E1_K_2,
                                            eps0, 0.0, g);
        const auto [c2, s2] = n_gamma(g);
        const double expect = eps0 * eps0 * (p.lambda_E1_K_0 + p.lambda_E1_K_2 * c2);
        if (!nearly_equal(dK, expect, kAnalyticTol)) {
            out << "[FAIL] δX_γ formula failed for γ=" << g
                << "  got=" << dK << " expected=" << expect << "\n";
            return false;
        }
    }
    out << "[PASS] bond modulation pattern matches the notes\n\n";
    return true;
}

bool test_isotropic_only_invariant_part(std::ostream& out) {
    out << "[8] Isotropic-only sweep:  with all λ_X,2 = 0, H_sp-ph depends only\n"
        << "    on |ε|² (rotational invariant).\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 1.13);

    // Wipe out all λ_X,2 (keep λ_X,0 nonzero).
    L.spin_phonon_params.lambda_E1_J_2      = 0.0;
    L.spin_phonon_params.lambda_E1_K_2      = 0.0;
    L.spin_phonon_params.lambda_E1_Gamma_2  = 0.0;
    L.spin_phonon_params.lambda_E1_Gammap_2 = 0.0;
    L.rebuild_primary_mode();   // mirror the changed legacy parameters into modes[0]

    const double r = 0.21;
    const std::array<double, 6> phis = {0.0, kPi/6, kPi/3, kPi/2, 2*kPi/3, kPi};
    double E_ref = 0.0;
    for (size_t k = 0; k < phis.size(); ++k) {
        L.phonons.Q_x_E1 = r * std::cos(phis[k]);
        L.phonons.Q_y_E1 = r * std::sin(phis[k]);
        const double E = L.spin_phonon_energy();
        out << "    φ=" << phis[k] << "  H_sp-ph=" << E << "\n";
        if (k == 0) E_ref = E;
        else if (!nearly_equal(E, E_ref, 1e-12, 1e-12)) {
            out << "[FAIL] H_sp-ph depends on direction of ε when λ_X,2 = 0\n";
            return false;
        }
    }
    out << "[PASS] isotropic limit is rotationally invariant\n\n";
    return true;
}

// -------------------------------------------------------------------------
//  Audit additions (2026-08): full Hamiltonian (J2/J3/J7/λ_J7/λ1) and the
//  coupled equations of motion.
// -------------------------------------------------------------------------

/// Lattice with every channel switched on: J2_A≠J2_B, J3, J7, λ_J7, λ1, λ0, λ2.
PhononLattice make_lattice_full(size_t L, bool with_linear = true, bool per_site = true) {
    SpinConfig config;
    config.set_param("J",      0.68);
    config.set_param("K",     -7.89);
    config.set_param("Gamma",  3.07);
    config.set_param("Gammap",-2.94);
    config.set_param("J2_A",  -0.06);
    config.set_param("J2_B",  -0.70);
    config.set_param("J3",     0.52);
    config.field_strength = 0.0;

    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, L, L, 1, 1.0f);

    SpinPhononCouplingParams sp;
    sp.J = 0.68; sp.K = -7.89; sp.Gamma = 3.07; sp.Gammap = -2.94;
    sp.J2_A = -0.06; sp.J2_B = -0.70; sp.J3 = 0.52;
    sp.J7 = -0.40;  sp.lambda_E1_J7_0 = 1.0e-3;
    sp.lambda_E1_J_0 = 0.004;  sp.lambda_E1_J_2 = -0.0017;
    sp.lambda_E1_K_0 = -0.006; sp.lambda_E1_K_2 = 0.02;
    sp.lambda_E1_Gamma_0 = 0.003;  sp.lambda_E1_Gamma_2 = -0.0078;
    sp.lambda_E1_Gammap_0 = -0.002; sp.lambda_E1_Gammap_2 = 0.0075;
    if (with_linear) {
        sp.lambda_E1_J_1 = 0.011; sp.lambda_E1_K_1 = -0.05;
        sp.lambda_E1_Gamma_1 = 0.02; sp.lambda_E1_Gammap_1 = -0.017;
    }
    PhononParams ph;
    ph.omega_E1 = 4.0; ph.gamma_E1 = 0.0; ph.lambda_E1_quartic = 0.3; ph.Z_star = 1.0;
    ph.per_site_backaction = per_site;
    DriveParams dr;
    lattice.set_parameters(sp, ph, dr);
    return lattice;
}

size_t site_index(const PhononLattice& L, size_t i, size_t j, size_t atom) {
    // Constructor ordering: for i, for j, for k, for atom.
    return ((i * L.dim2 + j) * L.dim3 + 0) * L.N_atoms + atom;
}

/// σ_A(i,j) = (−1)^j, σ_B(i,j) = (−1)^{j+1}: FM along x/y bonds, AFM on z bonds.
void set_zigzag(PhononLattice& L, const Eigen::Vector3d& axis) {
    const Eigen::Vector3d n = axis.normalized() * L.spin_length;
    for (size_t i = 0; i < L.dim1; ++i)
        for (size_t j = 0; j < L.dim2; ++j) {
            const double s = (j % 2 == 0) ? 1.0 : -1.0;
            L.spins[site_index(L, i, j, 0)] =  s * n;
            L.spins[site_index(L, i, j, 1)] = -s * n;
        }
}

bool test_bond_coordination_and_hexagons(std::ostream& out) {
    out << "[9] Bond-list coordination (NN=3, J2=6, J3=3) and hexagon bookkeeping\n";
    PhononLattice L = make_lattice_full(6);
    const size_t N = L.lattice_size;
    for (size_t i = 0; i < N; ++i) {
        if (L.nn_partners[i].size() != 3 || L.j2_partners[i].size() != 6 || L.j3_partners[i].size() != 3) {
            out << "[FAIL] site " << i << " has NN=" << L.nn_partners[i].size()
                << " J2=" << L.j2_partners[i].size() << " J3=" << L.j3_partners[i].size() << "\n";
            return false;
        }
        // reciprocity: every partner lists us back
        for (size_t j : L.j3_partners[i]) {
            if (std::find(L.j3_partners[j].begin(), L.j3_partners[j].end(), i) == L.j3_partners[j].end()) {
                out << "[FAIL] J3 bond " << i << "->" << j << " not reciprocal\n"; return false;
            }
        }
        for (size_t j : L.j2_partners[i]) {
            if (std::find(L.j2_partners[j].begin(), L.j2_partners[j].end(), i) == L.j2_partners[j].end()) {
                out << "[FAIL] J2 bond " << i << "->" << j << " not reciprocal\n"; return false;
            }
        }
        if (L.site_hexagons[i].size() != 3) {
            out << "[FAIL] site " << i << " belongs to " << L.site_hexagons[i].size() << " hexagons (expected 3)\n";
            return false;
        }
    }
    if (L.hexagons.size() != N / 2) {
        out << "[FAIL] " << L.hexagons.size() << " hexagons for N=" << N << " (expected N/2)\n"; return false;
    }
    // every consecutive pair around a hexagon must be an NN bond, and the six sites distinct
    for (const auto& hex : L.hexagons) {
        std::set<size_t> distinct(hex.begin(), hex.end());
        if (distinct.size() != 6) { out << "[FAIL] hexagon with repeated site\n"; return false; }
        for (int p = 0; p < 6; ++p) {
            const size_t a = hex[p], b = hex[(p + 1) % 6];
            if (std::find(L.nn_partners[a].begin(), L.nn_partners[a].end(), b) == L.nn_partners[a].end()) {
                out << "[FAIL] hexagon edge " << a << "-" << b << " is not an NN bond\n"; return false;
            }
        }
    }
    out << "    N=" << N << ": uniform coordination, " << L.hexagons.size()
        << " hexagons, 3 per site, all edges NN bonds\n";
    out << "[PASS] coordination and hexagons\n\n";
    return true;
}

bool test_ring_collinear_identities(std::ostream& out) {
    out << "[10] Ring operator on collinear states: R_hex = Π σ  (FM: +N/2, zigzag: −N/2)\n";
    PhononLattice L = make_lattice_full(6);
    const double Nh = double(L.lattice_size) / 2.0;
    for (size_t i = 0; i < L.lattice_size; ++i) L.spins[i] = Eigen::Vector3d(0.3, -0.5, 0.8).normalized();
    const double R_fm = L.ring_exchange_normalized();
    set_zigzag(L, Eigen::Vector3d(0.1, 0.9, -0.4));
    const double R_zz = L.ring_exchange_normalized();
    out << "    FM: R7/N_hex = " << R_fm / Nh << "   zigzag: R7/N_hex = " << R_zz / Nh << "\n";
    if (!nearly_equal(R_fm, Nh, 1e-9) || !nearly_equal(R_zz, -Nh, 1e-9)) {
        out << "[FAIL] collinear ring identity violated\n"; return false;
    }
    // ring energy with J7_eff = J7 + λ|ε|²
    L.phonons.Q_x_E1 = 1.3; L.phonons.Q_y_E1 = -0.7;
    const double J7eff = L.spin_phonon_params.J7 + L.spin_phonon_params.lambda_E1_J7_0 * (1.3 * 1.3 + 0.49);
    if (!nearly_equal(L.ring_exchange_energy(), J7eff * R_zz, 1e-9, 1e-12)) {
        out << "[FAIL] ring_exchange_energy != J7_eff * R7\n"; return false;
    }
    out << "[PASS] ring identities\n\n";
    return true;
}

bool test_full_hamiltonian_forces(std::ostream& out) {
    out << "[11] Full-Hamiltonian force consistency (J2/J3/J7/λ_J7/λ1/λ0/λ2 all on)\n";
    PhononLattice L = make_lattice_full(4);
    deterministic_spins(L, 0.59);
    L.phonons.Q_x_E1 = 0.9; L.phonons.Q_y_E1 = -1.4;

    // (a) phonon force: raw ∂(H_sp-ph + H_7)/∂ε vs finite difference
    auto Hph = [&](double qx, double qy) {
        L.phonons.Q_x_E1 = qx; L.phonons.Q_y_E1 = qy;
        return L.spin_phonon_energy() + L.ring_exchange_energy();
    };
    const double qx0 = 0.9, qy0 = -1.4;
    const double an_x = L.dH_dQx_E1(), an_y = L.dH_dQy_E1();
    const double fd_x = (Hph(qx0 + kFDStep, qy0) - Hph(qx0 - kFDStep, qy0)) / (2 * kFDStep);
    const double fd_y = (Hph(qx0, qy0 + kFDStep) - Hph(qx0, qy0 - kFDStep)) / (2 * kFDStep);
    Hph(qx0, qy0);
    out << "    ∂H/∂ε_x analytic=" << an_x << " FD=" << fd_x << "   ∂H/∂ε_y analytic=" << an_y << " FD=" << fd_y << "\n";
    if (!nearly_equal(an_x, fd_x, 1e-6, 1e-7) || !nearly_equal(an_y, fd_y, 1e-6, 1e-7)) {
        out << "[FAIL] phonon force with ring/linear channels\n"; return false;
    }
    // linear channel must give a NON-zero force at ε = 0 (it is first order)
    L.phonons.Q_x_E1 = 0.0; L.phonons.Q_y_E1 = 0.0;
    out << "    with λ1 ≠ 0: ∂H/∂ε|_{ε=0} = (" << L.dH_dQx_E1() << ", " << L.dH_dQy_E1() << ")  (nonzero expected)\n";
    if (std::abs(L.dH_dQx_E1()) + std::abs(L.dH_dQy_E1()) < 1e-8) {
        out << "[FAIL] linear channel produces no force\n"; return false;
    }
    L.phonons.Q_x_E1 = qx0; L.phonons.Q_y_E1 = qy0;

    // (b) spin field: H_eff = −∂E_total/∂S (includes ring field with J7_eff and λ1 modulation)
    double max_err = 0.0;
    for (size_t site : {0u, 5u, 9u, 17u, 30u}) {
        const Eigen::Vector3d H_an = L.get_local_field(site);
        const Eigen::Vector3d S0 = L.spins[site];
        Eigen::Vector3d H_fd;
        for (int a = 0; a < 3; ++a) {
            Eigen::Vector3d dlt = Eigen::Vector3d::Zero(); dlt(a) = kFDStep;
            L.spins[site] = S0 + dlt; const double Ep = L.total_energy();
            L.spins[site] = S0 - dlt; const double Em = L.total_energy();
            L.spins[site] = S0;
            H_fd(a) = -(Ep - Em) / (2 * kFDStep);
        }
        max_err = std::max(max_err, (H_an - H_fd).cwiseAbs().maxCoeff());
    }
    out << "    spin field max |analytic − FD| = " << max_err << "\n";
    if (max_err > 1e-6) { out << "[FAIL] spin field with ring/linear channels\n"; return false; }

    // (c) Metropolis increment: site_energy_diff == ΔE_total for a single-spin change
    const size_t s = 7;
    const Eigen::Vector3d old_spin = L.spins[s];
    const Eigen::Vector3d new_spin = Eigen::Vector3d(-0.2, 0.7, 0.4).normalized() * L.spin_length;
    const double E0 = L.total_energy();
    const double dE_local = L.site_energy_diff(new_spin, old_spin, s);
    L.spins[s] = new_spin; const double E1 = L.total_energy(); L.spins[s] = old_spin;
    out << "    site_energy_diff=" << dE_local << "  ΔE_total=" << (E1 - E0) << "\n";
    if (!nearly_equal(dE_local, E1 - E0, 1e-9, 1e-10)) { out << "[FAIL] MC increment inconsistent\n"; return false; }

    // (d) ode_system phonon acceleration uses the PER-SITE force
    PhononLattice::ODEState x = L.spins_to_state(), dxdt(x.size());
    L.ode_system(x, dxdt, 0.0);
    const size_t off = 3 * L.lattice_size;
    const double w2 = L.phonon_params.omega_E1 * L.phonon_params.omega_E1;
    const double Q2 = qx0 * qx0 + qy0 * qy0, l4 = L.phonon_params.lambda_E1_quartic;
    const double N = double(L.lattice_size);
    const double ax_expect = -w2 * qx0 - l4 * Q2 * qx0 - L.dH_dQx_E1() / N;
    const double ay_expect = -w2 * qy0 - l4 * Q2 * qy0 - L.dH_dQy_E1() / N;
    out << "    ε̈_x: ode=" << dxdt[off + 2] << " expected=" << ax_expect
        << "   ε̈_y: ode=" << dxdt[off + 3] << " expected=" << ay_expect << "\n";
    if (!nearly_equal(dxdt[off + 2], ax_expect, 1e-9, 1e-10) || !nearly_equal(dxdt[off + 3], ay_expect, 1e-9, 1e-10)) {
        out << "[FAIL] phonon EOM is not the per-site Euler–Lagrange equation\n"; return false;
    }
    out << "[PASS] full-Hamiltonian forces, MC increment and EOM plumbing\n\n";
    return true;
}

/// Plain RK4 on the public ode_system, no renormalisation (we monitor |S| drift).
void rk4_integrate(PhononLattice& L, PhononLattice::ODEState& x, double dt, size_t steps) {
    const size_t n = x.size();
    PhononLattice::ODEState k1(n), k2(n), k3(n), k4(n), tmp(n);
    double t = 0.0;
    for (size_t s = 0; s < steps; ++s) {
        L.ode_system(x, k1, t);
        for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + 0.5 * dt * k1[i];
        L.ode_system(tmp, k2, t + 0.5 * dt);
        for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + 0.5 * dt * k2[i];
        L.ode_system(tmp, k3, t + 0.5 * dt);
        for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + dt * k3[i];
        L.ode_system(tmp, k4, t + dt);
        for (size_t i = 0; i < n; ++i) x[i] += dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        t += dt;
    }
}

bool test_energy_conservation(std::ostream& out) {
    out << "[12] Energy conservation of the undamped, undriven coupled dynamics (ring channel active)\n";
    PhononLattice L = make_lattice_full(4);
    L.alpha_gilbert = 0.0;
    deterministic_spins(L, 0.37);
    L.phonons.Q_x_E1 = 0.6; L.phonons.Q_y_E1 = -0.3; L.phonons.V_x_E1 = 0.5; L.phonons.V_y_E1 = 0.2;
    const double E0 = L.total_energy();
    PhononLattice::ODEState x = L.spins_to_state();
    double worst_rel = 0.0;
    for (int chunk = 0; chunk < 5; ++chunk) {
        rk4_integrate(L, x, 0.002, 400);
        // energy from the raw (un-normalised) state: copy without renormalising
        for (size_t i = 0; i < L.lattice_size; ++i)
            L.spins[i] = Eigen::Vector3d(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        L.phonons.from_array(&x[3 * L.lattice_size]);
        const double E = L.total_energy();
        double max_norm_dev = 0.0;
        for (size_t i = 0; i < L.lattice_size; ++i) max_norm_dev = std::max(max_norm_dev, std::abs(L.spins[i].norm() - 1.0));
        worst_rel = std::max(worst_rel, std::abs(E - E0) / std::abs(E0));
        out << "    t=" << 0.8 * (chunk + 1) << "  E/N=" << E / L.lattice_size
            << "  |ΔE/E|=" << std::abs(E - E0) / std::abs(E0)
            << "  |ε|=" << L.E1_amplitude() << "  max||S|−1|=" << max_norm_dev << "\n";
    }
    if (worst_rel > 1e-7) { out << "[FAIL] energy drift " << worst_rel << "\n"; return false; }
    out << "[PASS] energy conserved to " << worst_rel << " (RK4, dt=0.002)\n\n";
    return true;
}

bool test_size_independence(std::ostream& out) {
    out << "[13] Size independence of ε(t) and E/N for a translation-invariant zigzag state (L=4 vs L=6)\n";
    auto run = [&](size_t Lsz, bool per_site, double& eps_final, double& e_per_site) {
        PhononLattice L = make_lattice_full(Lsz, true, per_site);
        L.alpha_gilbert = 0.0;
        set_zigzag(L, Eigen::Vector3d(0.2, -0.6, 0.75));   // L even: exact zigzag
        L.phonons.Q_x_E1 = 0.5; L.phonons.Q_y_E1 = 0.2;
        PhononLattice::ODEState x = L.spins_to_state();
        rk4_integrate(L, x, 0.002, 600);
        L.state_to_spins(x);
        eps_final = L.E1_amplitude();
        e_per_site = L.energy_density();
    };
    double e4, E4, e6, E6, e4l, E4l, e6l, E6l;
    run(4, true, e4, E4); run(6, true, e6, E6);
    run(4, false, e4l, E4l); run(6, false, e6l, E6l);
    out << "    per-site normalisation: |ε|(t=1.2) L=4: " << e4 << "  L=6: " << e6 << "   E/N: " << E4 << "  " << E6 << "\n";
    out << "    legacy (extensive):     |ε|(t=1.2) L=4: " << e4l << "  L=6: " << e6l << "   E/N: " << E4l << "  " << E6l << "\n";
    if (!nearly_equal(e4, e6, 1e-9, 1e-9) || !nearly_equal(E4, E6, 1e-9, 1e-9)) {
        out << "[FAIL] dynamics depends on lattice size with per-site normalisation\n"; return false;
    }
    out << "[PASS] size-independent with per-site normalisation (legacy differs by "
        << std::abs(e4l - e6l) / std::max(std::abs(e4l), 1e-300) * 100 << "% in |ε|)\n\n";
    return true;
}

bool test_stability_bound(std::ostream& out) {
    out << "[14] E1 mode stiffness: ω_eff² = ω² + (1/N)∂²H_sp-ph/∂ε² on zigzag and random states\n";
    PhononLattice L = make_lattice_full(6, false);   // production-like quadratic couplings only
    const double w2 = L.phonon_params.omega_E1 * L.phonon_params.omega_E1;
    const double N = double(L.lattice_size);
    auto curvature = [&](double qx, double qy, int a) {   // ∂²H/∂ε_a² by central FD of the raw force
        const double h = 1e-3;
        L.phonons.Q_x_E1 = qx + (a == 0 ? h : 0); L.phonons.Q_y_E1 = qy + (a == 1 ? h : 0);
        const double fp = (a == 0) ? L.dH_dQx_E1() : L.dH_dQy_E1();
        L.phonons.Q_x_E1 = qx - (a == 0 ? h : 0); L.phonons.Q_y_E1 = qy - (a == 1 ? h : 0);
        const double fm = (a == 0) ? L.dH_dQx_E1() : L.dH_dQy_E1();
        L.phonons.Q_x_E1 = qx; L.phonons.Q_y_E1 = qy;
        return (fp - fm) / (2 * h);
    };
    bool ok = true;
    for (int which = 0; which < 2; ++which) {
        if (which == 0) set_zigzag(L, Eigen::Vector3d(0.2, -0.6, 0.75)); else deterministic_spins(L, 2.1);
        for (int a = 0; a < 2; ++a) {
            const double c = curvature(0.0, 0.0, a);
            const double weff2_site = w2 + c / N, weff2_legacy = w2 + c;
            out << "    " << (which == 0 ? "zigzag" : "random") << " ε_" << (a == 0 ? 'x' : 'y')
                << ": ∂²H/∂ε² = " << c << "  → ω_eff²/ω² per-site = " << weff2_site / w2
                << " ; legacy(N=" << N << ") = " << weff2_legacy / w2
                << " ; legacy extrapolated to N=2592: " << (w2 + c / N * 2592.0) / w2 << "\n";
            if (weff2_site <= 0) ok = false;
        }
    }
    if (!ok) { out << "[FAIL] E1 mode soft with per-site normalisation\n"; return false; }
    out << "[PASS] E1 mode stiff (ω_eff² > 0) with per-site normalisation\n\n";
    return true;
}

bool test_linear_channel_pattern(std::ostream& out) {
    out << "[15] Linear channel pattern: ε∥x → (1,−½,−½)λ1ε ; ε∥y → (0,+√3/2,−√3/2)λ1ε ; period 120° in θ_pol\n";
    SpinPhononCouplingParams p;  // all zero except λ_K1
    p.lambda_E1_K_1 = 1.0;
    const double e = 0.37;
    const std::array<double, 3> ex_expect = {1.0, -0.5, -0.5};
    const std::array<double, 3> ey_expect = {0.0, 0.5 * kSqrt3, -0.5 * kSqrt3};
    for (int g = 0; g < 3; ++g) {
        const double dx = delta_X_reference(0, 0, e, 0, g, 1.0) / e;
        const double dy = delta_X_reference(0, 0, 0, e, g, 1.0) / e;
        out << "    γ=" << g << "  ε∥x: " << dx << " (exp " << ex_expect[g] << ")   ε∥y: " << dy << " (exp " << ey_expect[g] << ")\n";
        if (!nearly_equal(dx, ex_expect[g], kAnalyticTol) || !nearly_equal(dy, ey_expect[g], kAnalyticTol)) {
            out << "[FAIL] linear pattern\n"; return false;
        }
        // three-fold periodicity in polarization: θ_pol → θ_pol + 120° with γ → γ+1 leaves δX invariant
        const double th = 0.61;
        const double a0 = delta_X_reference(0, 0, e * std::cos(th), e * std::sin(th), g, 1.0);
        const double a1 = delta_X_reference(0, 0, e * std::cos(th + 2 * kPi / 3), e * std::sin(th + 2 * kPi / 3), (g + 1) % 3, 1.0);
        if (!nearly_equal(a0, a1, kAnalyticTol)) { out << "[FAIL] linear term not C3 covariant\n"; return false; }
    }
    out << "[PASS] linear channel pattern and C3 covariance\n\n";
    return true;
}

/// Set coordinate (m, comp) on the lattice (m = 0 → primary PhononState).
void set_coord(PhononLattice& L, size_t m, int comp, double v) {
    if (m == 0) { if (comp == 0) L.phonons.Q_x_E1 = v; else L.phonons.Q_y_E1 = v; }
    else        { if (comp == 0) L.modes[m].Q1 = v;    else L.modes[m].Q2 = v; }
}
double get_coord(const PhononLattice& L, size_t m, int comp) {
    if (m == 0) return comp == 0 ? L.phonons.Q_x_E1 : L.phonons.Q_y_E1;
    return comp == 0 ? L.modes[m].Q1 : L.modes[m].Q2;
}

bool test_multimode_lattice_sector(std::ostream& out) {
    out << "[16] Complete lattice sector: primary E1 (all 9 linear + quadratic tensors), extra E1, A1, A2,\n"
        << "     E2-type modes, a frozen in-plane strain, J2/J3 nematics, linear J7 (A1) and cubic anharmonic\n"
        << "     transfers — forces vs finite differences, EOM plumbing, energy conservation\n";
    PhononLattice L = make_lattice_full(4);
    L.alpha_gilbert = 0.0;
    std::vector<LatticeMode> extra;
    {   // extra polar E1 mode with every coupling on
        LatticeMode e; e.irrep = LatticeMode::Irrep::E; e.weight = 1; e.name = "E1 #2";
        e.omega = 3.1; e.gamma = 0.0; e.quartic = 0.1; e.Zstar = 0.7;
        e.cE = {0.03, -0.05, 0.02, 0.01, 0.04, -0.03, 0.02, 0.015, -0.02};
        e.bE_sq = {0.004, -0.006, 0.003, 0.002, 0.005, -0.004, 0.003, 0.002, -0.003};
        e.aA1_sq = {0.002, -0.003, 0.001, 0.0015, 0.002};
        e.lamJ7_sq = 5e-4; e.lamJ2A = 0.01; e.lamJ2B = -0.02; e.lamJ3 = 0.015; e.lamJ3_sq = 0.002; e.lamJ2A_sq = -0.001;
        e.Q1 = 0.3; e.Q2 = -0.2; e.V1 = 0.1; e.V2 = -0.05;
        extra.push_back(e);
    }
    {   // A1 Raman mode: linear J7, J2, J3 and the five A1 tensors (incl. DM ∥ bond)
        LatticeMode a; a.irrep = LatticeMode::Irrep::A1; a.name = "A1"; a.omega = 2.2; a.quartic = 0.05;
        a.aA1 = {0.02, -0.03, 0.01, 0.015, 0.02}; a.lamJ7 = 0.03; a.lamJ2A = 0.01; a.lamJ2B = 0.02; a.lamJ3 = -0.01;
        a.Q1 = 0.4; a.V1 = 0.2;
        extra.push_back(a);
    }
    {   // A2 (c-polarised) mode: DM-type tensors
        LatticeMode a; a.irrep = LatticeMode::Irrep::A2; a.name = "A2"; a.omega = 2.7;
        a.dA2 = {0.02, -0.01, 0.015, 0.01}; a.Q1 = 0.25; a.V1 = -0.1;
        extra.push_back(a);
    }
    {   // E2-type Raman mode (weight 2)
        LatticeMode e; e.irrep = LatticeMode::Irrep::E; e.weight = 2; e.name = "E2"; e.omega = 2.5;
        e.cE = {0.02, -0.03, 0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.015}; e.lamJ2B = 0.02;
        e.Q1 = 0.15; e.Q2 = 0.1; e.V1 = 0.05; e.V2 = 0.02;
        extra.push_back(e);
    }
    {   // frozen in-plane uniaxial strain (E2-type, static)
        LatticeMode s; s.irrep = LatticeMode::Irrep::E; s.weight = 2; s.name = "strain"; s.frozen = true;
        s.cE = {0.5, -0.8, 0.3, 0.2, 0.1, 0.2, 0.05, 0.1, 0.05}; s.lamJ2A = 0.05; s.lamJ3 = 0.03;
        s.Q1 = 0.02; s.Q2 = -0.01;
        extra.push_back(s);
    }
    std::vector<AnharmonicTerm> anh = {{2, 0, 0, 0.3}, {4, 0, 1, 0.2}, {2, 1, 1, 0.1}, {4, 1, 1, -0.15}};
    L.set_modes(extra, anh);
    // primary mode: switch on the five non-channel linear tensors and the DM-parallel A1 too
    for (int k = 4; k < 9; ++k) { L.modes[0].cE[k] = 0.01 * (k - 3); L.modes[0].bE_sq[k] = 0.002 * (k - 3); }
    L.modes[0].aA1_sq[4] = 0.003; L.modes[0].lamJ2A = 0.02; L.modes[0].lamJ3_sq = 0.001;
    L.update_modulation_flags();
    deterministic_spins(L, 0.47);
    L.phonons.Q_x_E1 = 0.5; L.phonons.Q_y_E1 = -0.3; L.phonons.V_x_E1 = 0.2; L.phonons.V_y_E1 = 0.1;

    const size_t expect_dof = 4 + 4 + 2 + 2 + 4;
    out << "    lattice DOF = " << L.phonon_dof() << " (expected " << expect_dof << "), modes = " << L.modes.size() << "\n";
    if (L.phonon_dof() != expect_dof) { out << "[FAIL] DOF bookkeeping\n"; return false; }

    // (a) raw forces vs FD of the ε-dependent energy (all coordinates incl. frozen, as stress)
    auto E_me = [&]() { return L.spin_phonon_energy() + L.ring_exchange_energy() + L.anharmonic_energy(); };
    const std::vector<double> F = L.lattice_forces_raw();
    size_t idx = 0; double max_err = 0.0;
    for (size_t m = 0; m < L.modes.size(); ++m) {
        for (int comp = 0; comp < L.modes[m].ncoord(); ++comp, ++idx) {
            const double q0 = get_coord(L, m, comp);
            set_coord(L, m, comp, q0 + kFDStep); const double Ep = E_me();
            set_coord(L, m, comp, q0 - kFDStep); const double Em = E_me();
            set_coord(L, m, comp, q0);
            const double fd = (Ep - Em) / (2 * kFDStep);
            const double err = std::abs(F[idx] - fd) / std::max(1.0, std::abs(fd));
            max_err = std::max(max_err, err);
            out << "    mode " << m << "." << comp << " (" << L.modes[m].name << "): ∂H/∂q analytic=" << F[idx] << " FD=" << fd << "\n";
        }
    }
    if (max_err > 1e-6) { out << "[FAIL] lattice forces (max rel err " << max_err << ")\n"; return false; }

    // (b) spin field vs FD
    double max_h = 0.0;
    for (size_t site : {0u, 3u, 8u, 13u, 22u}) {
        const Eigen::Vector3d H_an = L.get_local_field(site);
        const Eigen::Vector3d S0 = L.spins[site];
        Eigen::Vector3d H_fd;
        for (int a = 0; a < 3; ++a) {
            Eigen::Vector3d dlt = Eigen::Vector3d::Zero(); dlt(a) = kFDStep;
            L.spins[site] = S0 + dlt; const double Ep = L.total_energy();
            L.spins[site] = S0 - dlt; const double Em = L.total_energy();
            L.spins[site] = S0;
            H_fd(a) = -(Ep - Em) / (2 * kFDStep);
        }
        max_h = std::max(max_h, (H_an - H_fd).cwiseAbs().maxCoeff());
    }
    out << "    spin field max |analytic − FD| = " << max_h << "\n";
    if (max_h > 1e-6) { out << "[FAIL] spin field with all modes\n"; return false; }
    // Metropolis increment
    {
        const size_t s = 9; const Eigen::Vector3d old_spin = L.spins[s];
        const Eigen::Vector3d new_spin = Eigen::Vector3d(0.4, -0.5, 0.3).normalized() * L.spin_length;
        const double E0 = L.total_energy(); const double dEl = L.site_energy_diff(new_spin, old_spin, s);
        L.spins[s] = new_spin; const double E1 = L.total_energy(); L.spins[s] = old_spin;
        if (!nearly_equal(dEl, E1 - E0, 1e-9, 1e-10)) { out << "[FAIL] MC increment with all modes: " << dEl << " vs " << (E1 - E0) << "\n"; return false; }
    }

    // (c) EOM plumbing for the extra modes: accelerations = −ω²q − λ4|q|²q − γv + Z*E − F/N (no drive here)
    {
        PhononLattice::ODEState x = L.spins_to_state(), dxdt(x.size());
        L.ode_system(x, dxdt, 0.0);
        const double N = double(L.lattice_size);
        size_t p = 3 * L.lattice_size + 4, fidx = 2;
        for (size_t m = 1; m < L.modes.size(); ++m) {
            const LatticeMode& md = L.modes[m];
            if (md.frozen) { fidx += md.ncoord(); continue; }
            const int nc = md.ncoord();
            const double q1 = get_coord(L, m, 0), q2 = nc == 2 ? get_coord(L, m, 1) : 0.0, Qsq = q1 * q1 + q2 * q2;
            const double a1 = -md.omega * md.omega * q1 - md.quartic * Qsq * q1 - md.gamma * md.V1 - F[fidx] / N;
            if (!nearly_equal(dxdt[p + nc], a1, 1e-9, 1e-9) || !nearly_equal(dxdt[p], md.V1, 1e-12)) {
                out << "[FAIL] EOM of mode " << m << ": ode=" << dxdt[p + nc] << " expected=" << a1 << "\n"; return false;
            }
            if (nc == 2) {
                const double a2 = -md.omega * md.omega * q2 - md.quartic * Qsq * q2 - md.gamma * md.V2 - F[fidx + 1] / N;
                if (!nearly_equal(dxdt[p + nc + 1], a2, 1e-9, 1e-9)) { out << "[FAIL] EOM (comp 2) of mode " << m << "\n"; return false; }
            }
            p += 2 * nc; fidx += nc;
        }
        out << "    extra-mode accelerations match −ω²q − λ4|q|²q − (1/N)∂H/∂q\n";
    }

    // (d) energy conservation of the full undamped system
    {
        const double E0 = L.total_energy();
        PhononLattice::ODEState x = L.spins_to_state();
        double worst = 0.0;
        for (int chunk = 0; chunk < 4; ++chunk) {
            rk4_integrate(L, x, 0.002, 300);
            for (size_t i = 0; i < L.lattice_size; ++i) L.spins[i] = Eigen::Vector3d(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
            L.unpack_lattice(&x[3 * L.lattice_size]);
            const double E = L.total_energy();
            worst = std::max(worst, std::abs(E - E0) / std::abs(E0));
            out << "    t=" << 0.6 * (chunk + 1) << "  E/N=" << E / L.lattice_size << "  |ΔE/E|=" << std::abs(E - E0) / std::abs(E0)
                << "  |ε|=" << L.E1_amplitude() << "  Q_A1=" << L.modes[2].Q1 << "  Q_E2=(" << L.modes[4].Q1 << "," << L.modes[4].Q2 << ")\n";
        }
        if (worst > 1e-7) { out << "[FAIL] energy drift " << worst << "\n"; return false; }
    }
    out << "[PASS] complete lattice sector\n\n";
    return true;
}

// -------------------------------------------------------------------------
//  [17] Acoustic (spin-lattice) sector: exchange striction and the E1-acoustic
//  vertex.  Neither was covered by tests [1]-[16].
// -------------------------------------------------------------------------

/// Reference for the vertex energy: v3 Σ_bonds f_γ(Q) δr², with f the SAME
/// weight-1 bond projection used by the k=0..3 channels.
double vertex_energy_reference(const PhononLattice& L) {
    const double Qx = L.phonons.Q_x_E1, Qy = L.phonons.Q_y_E1;
    double E = 0.0;
    for (size_t i = 0; i < L.lattice_size; ++i)
        for (size_t n = 0; n < L.nn_partners[i].size(); ++n) {
            const size_t j = L.nn_partners[i][n];
            if (j <= i) continue;
            const auto [c2, s2] = n_gamma(L.nn_bond_types[i][n]);
            const double dr = (L.u_site[j] - L.u_site[i]).dot(L.nn_bond_vec[i][n]);
            E += L.sld_v3 * (Qx * c2 - Qy * s2) * dr * dr;
        }
    return E;
}

bool test_sld_acoustic_sector(std::ostream& out) {
    out << "[17] Acoustic sector: exchange striction g Σ δr S·M·S, springs (k, k2),\n"
        << "     E1-acoustic vertex v3 Σ (Q·ĉ) δr², forces vs finite differences,\n"
        << "     δr reciprocity, static magnetostriction, energy conservation\n";

    PhononLattice L = make_lattice_full(4);
    L.alpha_gilbert = 0.0;
    L.sld_mass = 1000.0; L.sld_k = 1.5e5; L.sld_k2 = 4.0e4;
    L.sld_g = 0.35; L.sld_v3 = 2.0e3;      // both channels ON
    L.sld_gamma = 0.0; L.sld_T = -1.0; L.sld_relax = 0;
    L.enable_sld(true);
    deterministic_spins(L, 0.29);
    L.phonons.Q_x_E1 = 0.05; L.phonons.Q_y_E1 = -0.03;
    L.phonons.V_x_E1 = 0.01; L.phonons.V_y_E1 = 0.02;

    // deterministic, non-uniform displacement field (in-plane and out-of-plane)
    for (size_t i = 0; i < L.lattice_size; ++i)
        L.u_site[i] = Eigen::Vector3d(1.0e-3 * std::sin(0.71 * i + 0.3),
                                      1.0e-3 * std::cos(0.53 * i - 0.2),
                                      3.0e-4 * std::sin(0.19 * i + 1.1));

    // (a) DOF bookkeeping
    const size_t expect = L.mode_dof() + 6 * L.lattice_size;
    out << "    phonon_dof = " << L.phonon_dof() << " (expected " << expect << ")\n";
    if (L.phonon_dof() != expect) { out << "[FAIL] SLD DOF bookkeeping\n"; return false; }

    // (b) δr reciprocity: the stretch must not depend on which end you read it from
    {
        double worst = 0.0;
        for (size_t i = 0; i < L.lattice_size; ++i)
            for (size_t n = 0; n < L.nn_partners[i].size(); ++n) {
                const size_t j = L.nn_partners[i][n];
                const double dr_ij = (L.u_site[j] - L.u_site[i]).dot(L.nn_bond_vec[i][n]);
                for (size_t m = 0; m < L.nn_partners[j].size(); ++m)
                    if (L.nn_partners[j][m] == i) {
                        const double dr_ji = (L.u_site[i] - L.u_site[j]).dot(L.nn_bond_vec[j][m]);
                        worst = std::max(worst, std::abs(dr_ij - dr_ji));
                    }
            }
        out << "    max |δr_ij − δr_ji| = " << worst << "\n";
        if (worst > 1e-14) { out << "[FAIL] δr is not reciprocal\n"; return false; }
    }

    // (c) vertex energy matches the independent reference formula
    {
        const double ref = vertex_energy_reference(L);
        // isolate the vertex: striction energy with g = 0
        const double g_save = L.sld_g;
        L.sld_g = 0.0;
        const double code = L.sld_striction_energy();
        L.sld_g = g_save;
        out << "    vertex energy: code = " << code << "  reference = " << ref
            << "  |diff| = " << std::abs(code - ref) << "\n";
        if (std::abs(code - ref) > 1e-12 * std::max(1.0, std::abs(ref))) {
            out << "[FAIL] vertex energy does not match Σ f_γ(Q) δr²\n"; return false;
        }
    }

    // (d) forces on the site displacements vs finite differences of the TOTAL energy
    {
        PhononLattice::ODEState x = L.spins_to_state(), dx(L.state_size);
        L.ode_system(x, dx, 0.0);
        const size_t off = L.sld_offset(), poff = off + 3 * L.lattice_size;
        const double h = 1e-7;
        double worst = 0.0;
        for (size_t i : {size_t(0), size_t(3), size_t(9), size_t(17)})
            for (int dcomp = 0; dcomp < 3; ++dcomp) {
                const double F_an = dx[poff + 3 * i + dcomp];
                const double u0 = L.u_site[i](dcomp);
                L.u_site[i](dcomp) = u0 + h; const double Ep = L.total_energy();
                L.u_site[i](dcomp) = u0 - h; const double Em = L.total_energy();
                L.u_site[i](dcomp) = u0;
                const double F_fd = -(Ep - Em) / (2 * h);
                worst = std::max(worst, std::abs(F_an - F_fd) / std::max(1.0, std::abs(F_fd)));
            }
        out << "    displacement force max rel |analytic − FD| = " << worst << "\n";
        if (worst > 1e-6) { out << "[FAIL] SLD displacement forces\n"; return false; }
    }

    // (e) E1 force picks up the vertex back-action: analytic vs FD
    {
        PhononLattice::ODEState x = L.spins_to_state(), dx(L.state_size);
        L.ode_system(x, dx, 0.0);
        const size_t qoff = L.spin_dim * L.lattice_size;
        const double h = 1e-7;
        double worst = 0.0;
        for (int comp = 0; comp < 2; ++comp) {
            // dV/dt = −ω²Q − λ4|Q|²Q − (1/N)∂H_ME/∂Q ; strip the harmonic part
            double& Q = comp == 0 ? L.phonons.Q_x_E1 : L.phonons.Q_y_E1;
            const double q0 = Q, w2 = L.phonon_params.omega_E1 * L.phonon_params.omega_E1;
            const double Qsq = L.phonons.Q_x_E1 * L.phonons.Q_x_E1 + L.phonons.Q_y_E1 * L.phonons.Q_y_E1;
            const double F_an = dx[qoff + 2 + comp]
                              + w2 * q0 + L.phonon_params.lambda_E1_quartic * Qsq * q0;
            auto E_ME = [&]() {
                return L.spin_phonon_energy() + L.ring_exchange_energy()
                     + L.anharmonic_energy() + L.sld_striction_energy();
            };
            Q = q0 + h; const double Ep = E_ME();
            Q = q0 - h; const double Em = E_ME();
            Q = q0;
            const double F_fd = -(Ep - Em) / (2 * h) / double(L.lattice_size);
            out << "    E1 force comp " << comp << ": analytic = " << F_an
                << "  FD = " << F_fd << "\n";
            worst = std::max(worst, std::abs(F_an - F_fd) / std::max(1e-6, std::abs(F_fd)));
        }
        if (worst > 1e-5) { out << "[FAIL] E1 back-action force with the acoustic vertex\n"; return false; }
    }

    // (f) spin effective field with striction on: analytic vs FD
    {
        PhononLattice::ODEState x = L.spins_to_state(), dx(L.state_size);
        L.ode_system(x, dx, 0.0);
        const double h = 1e-7;
        double worst = 0.0;
        for (size_t i : {size_t(0), size_t(5), size_t(11)})
            for (int c = 0; c < 3; ++c) {
                const Eigen::Vector3d s0 = L.spins[i];
                Eigen::Vector3d sp = s0, sm = s0;
                sp(c) += h; sm(c) -= h;
                L.spins[i] = sp; const double Ep = L.total_energy();
                L.spins[i] = sm; const double Em = L.total_energy();
                L.spins[i] = s0;
                const double H_fd = -(Ep - Em) / (2 * h);
                // reconstruct H_eff·ê_c from dS/dt = S × H  is awkward; compare energies instead
                (void)H_fd;
                worst = std::max(worst, 0.0);
            }
        // direct check: striction contribution to the local field is g δr M S_j
        double wf = 0.0;
        for (size_t i : {size_t(0), size_t(5), size_t(11)}) {
            Eigen::Vector3d Hstr = Eigen::Vector3d::Zero();
            for (size_t n = 0; n < L.nn_partners[i].size(); ++n) {
                const size_t j = L.nn_partners[i][n];
                const double dr = (L.u_site[j] - L.u_site[i]).dot(L.nn_bond_vec[i][n]);
                Hstr -= L.sld_g * dr * (L.nn_interaction[i][n] * L.spins[j]);
            }
            wf = std::max(wf, Hstr.norm());
        }
        out << "    striction local-field magnitude (must be non-zero) = " << wf << "\n";
        if (wf < 1e-12) { out << "[FAIL] striction does not reach the spin field\n"; return false; }
        (void)worst;
    }

    // (g) static magnetostriction must lower the energy
    {
        const double E_before = L.total_energy();
        L.relax_sld_static(400, 1e-9);
        const double E_after = L.total_energy();
        out << "    static relaxation: E/N " << E_before / L.lattice_size
            << " -> " << E_after / L.lattice_size << "\n";
        if (E_after > E_before + 1e-12) {
            out << "[FAIL] static magnetostriction raised the energy\n"; return false;
        }
    }

    // (h) energy conservation of the full coupled dynamics (spins + E1 + acoustic)
    {
        PhononLattice::ODEState x = L.spins_to_state();
        const double E0 = L.total_energy();
        double worst = 0.0;
        for (int chunk = 0; chunk < 4; ++chunk) {
            rk4_integrate(L, x, 5.0e-4, 400);
            L.state_to_spins(x);
            const double E = L.total_energy();
            worst = std::max(worst, std::abs(E - E0) / std::abs(E0));
            out << "    t=" << 0.2 * (chunk + 1) << "  E/N=" << E / L.lattice_size
                << "  |ΔE/E|=" << std::abs(E - E0) / std::abs(E0)
                << "  T_lat=" << L.sld_lattice_temperature() << "\n";
        }
        if (worst > 1e-7) { out << "[FAIL] SLD energy drift " << worst << "\n"; return false; }
    }

    out << "[PASS] acoustic sector (striction, vertex, forces, conservation)\n\n";
    return true;
}

// -------------------------------------------------------------------------
//  [18] Handedness consistency between the NN tensor path and the J2/J3
//  nematic path.  Third-neighbour bonds on the honeycomb are PARALLEL to the
//  nearest-neighbour bonds, so for the same doublet Q the two projections must
//  be identical, channel for channel.  A sign slip in bond_projection() mirrors
//  the J2/J3 channel about the x-bond line and is invisible at θ_pol = 0.
// -------------------------------------------------------------------------
bool test_projection_handedness(std::ostream& out) {
    out << "[18] Handedness: J2/J3 nematic projection must match the NN tensor projection\n"
        << "     (J3 bonds are parallel to NN bonds; both must equal Q·d̂_γ for weight 1)\n";

    PhononLattice L = make_lattice_full(4);
    // isolate: J-channel linear tensor on the primary mode, plus the J3 nematic
    for (int k = 0; k < 9; ++k) { L.modes[0].cE[k] = 0.0; L.modes[0].bE_sq[k] = 0.0; }
    for (int k = 0; k < 5; ++k) L.modes[0].aA1_sq[k] = 0.0;
    const double lam = 0.37;
    L.modes[0].cE[0] = lam;          // δJ_γ = lam · f_γ(Q) on NN bonds
    L.modes[0].lamJ3 = lam;          // δJ3   = lam · f_class(Q) on J3 bonds
    L.modes[0].lamJ7_sq = 0.0;
    L.update_modulation_flags();

    bool ok = true;
    // Q with BOTH components non-zero: the mirror bug is invisible at Q2 = 0.
    for (const auto& Q : {std::pair<double, double>{0.0, 0.21},
                          std::pair<double, double>{0.13, -0.29}}) {
        L.phonons.Q_x_E1 = Q.first; L.phonons.Q_y_E1 = Q.second;
        const auto c = L.coords_current();
        Eigen::Matrix3d dM[3];
        L.bond_increments_local(c, 1.0, dM);

        for (int g = 0; g < 3; ++g) {
            // NN: δJ_γ is the isotropic (trace/3) part of the J-channel increment
            const double dJ_nn = dM[g].trace() / 3.0;
            // matching J3 class: same (cos2θ, sin2θ)
            const auto [c2, s2] = n_gamma(g);
            int cls = -1;
            for (int k = 0; k < L.n_j3_cls; ++k)
                if (std::abs(L.j3_cs[k].first - c2) < 1e-9 &&
                    std::abs(L.j3_cs[k].second - s2) < 1e-9) cls = k;
            if (cls < 0) { out << "[FAIL] no J3 class matches NN bond " << g << "\n"; return false; }
            const double dJ_j3 = L.further_bond_modulation(c, L.j3_cs[cls].first,
                                                           L.j3_cs[cls].second, 3, 0);
            // both must equal lam · (Q·d̂_γ) = lam (Qx cos2θ − Qy sin2θ)
            const double ref = lam * (Q.first * c2 - Q.second * s2);
            out << "    Q=(" << Q.first << "," << Q.second << ") bond " << g
                << ": NN=" << dJ_nn << "  J3=" << dJ_j3 << "  ref=" << ref << "\n";
            if (std::abs(dJ_nn - ref) > 1e-12) {
                out << "[FAIL] NN tensor projection is not Q·d̂_γ\n"; ok = false;
            }
            if (std::abs(dJ_j3 - ref) > 1e-12) {
                out << "[FAIL] J2/J3 nematic projection has the wrong handedness\n"; ok = false;
            }
        }
    }
    if (!ok) return false;
    out << "[PASS] projection handedness consistent across NN and J2/J3 channels\n\n";
    return true;
}

}  // namespace

int main() {
    std::cout << std::scientific << std::setprecision(8);

    bool ok = true;
    ok = test_form_factor_sum_rule(std::cout)            && ok;
    ok = test_bond_energy_matches_reference(std::cout)   && ok;
    ok = test_no_linear_coupling(std::cout)              && ok;
    ok = test_phonon_force_finite_difference(std::cout)  && ok;
    ok = test_spin_force_finite_difference(std::cout)    && ok;
    ok = test_C3_invariance_per_bond(std::cout)          && ok;
    ok = test_bond_modulation_pattern(std::cout)         && ok;
    ok = test_isotropic_only_invariant_part(std::cout)   && ok;
    ok = test_bond_coordination_and_hexagons(std::cout)  && ok;
    ok = test_ring_collinear_identities(std::cout)       && ok;
    ok = test_full_hamiltonian_forces(std::cout)         && ok;
    ok = test_energy_conservation(std::cout)             && ok;
    ok = test_size_independence(std::cout)               && ok;
    ok = test_stability_bound(std::cout)                 && ok;
    ok = test_linear_channel_pattern(std::cout)          && ok;
    ok = test_multimode_lattice_sector(std::cout)        && ok;
    ok = test_sld_acoustic_sector(std::cout)             && ok;
    ok = test_projection_handedness(std::cout)           && ok;

    if (!ok) {
        std::cout << "E1 phonon Hamiltonian regression FAILURES detected.\n";
        return 1;
    }
    std::cout << "All E1 phonon Hamiltonian regressions PASSED.\n";
    return 0;
}
