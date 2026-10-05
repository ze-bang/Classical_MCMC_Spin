// test_mixed_md.cpp — MixedLattice (SU(2) Fe + SU(3) Tm) dynamics against
// results known independently of the code:
//
//   * SU(3) convention: E_classical(n(psi)) = <psi|H|psi> + const for the
//     TmFeO3 CEF; |n|^2 = 4/3 and d_abc n^a n^b n^c = 8/9 for pure states.
//   * A free Tm qutrit precesses exactly as the Schrodinger evolution under
//     diag(0, e1, e2) (both conventions), sampled on the exact output grid.
//   * A coupled Fe-Tm dimer (Zeeman + CEF field + mixed exchange K) follows
//     the time-dependent mean-field (product-state) Schrodinger evolution of
//     the quantum model -- this fails if the SU(3) bracket is f instead of
//     2f (the Tm then feels half the exchange/Zeeman torque).
//   * Energy and both SU(3) Casimirs are conserved by the undamped dynamics
//     of the full TmFeO3 model with trilinear couplings, also when a Bloch
//     damping equilibrium is set (it used to leak into the MD field).
//   * An explicit trilinear reference is one Hamiltonian for energy, MC
//     energy differences, MC fields and MD fields.
//   * A single pulse has no phantom second pulse gating the field-assisted
//     exchange; W1 time-shift synthesis equals the integrated M1 on the grid.
//   * The thermal reservoir E_dep (an ODE variable) reproduces the closed-form
//     lambda_3 response of a damped qutrit.
//   * The SU(3) implicit-midpoint step is unitary and second order.
#include "physics_test_util.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/su3_coherent_state.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <H5Cpp.h>
#include <unistd.h>

#include <complex>
#include <filesystem>
#include <random>

using phys_test::check;
using phys_test::check_close;
namespace su3 = classical_spin::su3;
using Complex = std::complex<double>;

namespace {

std::mt19937_64 rng(20261005);

su3::Vector3c random_psi() {
    std::normal_distribution<double> g(0.0, 1.0);
    su3::Vector3c psi;
    for (int i = 0; i < 3; ++i) psi(i) = Complex(g(rng), g(rng));
    return psi.normalized();
}

Eigen::VectorXd to_x(const su3::Vector8r& n) { return Eigen::VectorXd(n); }

// d_abc = Tr({lambda_a, lambda_b} lambda_c) / 4, by brute force.
double casimir3_reference(const su3::Vector8r& n) {
    const auto& L = su3::gell_mann();
    double s = 0.0;
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b)
            for (int c = 0; c < 8; ++c) {
                const double d = 0.25 * ((L[a] * L[b] + L[b] * L[a]) * L[c]).trace().real();
                s += d * n(a) * n(b) * n(c);
            }
    return s;
}

SpinConfig tm_config(double e1, double e2, bool legacy) {
    SpinConfig cfg;
    cfg.field_strength = 0.0;
    cfg.set_param("e1", e1);
    cfg.set_param("e2", e2);
    if (legacy) cfg.set_param("su3_legacy_convention", 1.0);
    return cfg;
}

// 1 Fe + 1 Tm unit cell (1x1x1 lattice): Fe field B, Tm field h, optional
// mixed exchange K (3x8) and field-assisted exchange Kd (3x8, E-envelope tag).
MixedLattice make_dimer(const Eigen::Vector3d& B, const Eigen::VectorXd& h, double bracket,
                        const Eigen::MatrixXd& K, const Eigen::MatrixXd* Kd = nullptr,
                        float spin_length_SU2 = 0.5f) {
    UnitCell fe(3, 1, {Eigen::Vector3d::Zero()},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    UnitCell tm(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    fe.set_field(B, 0);
    tm.set_field(h, 0);
    tm.poisson_bracket = bracket;
    MixedUnitCell cell(fe, tm);
    if (K.size() > 0) cell.set_mixed_bilinear(K, 0, 0, Eigen::Vector3i::Zero());
    if (Kd) cell.set_mixed_bilinear_drive(*Kd, 0, 0, Eigen::Vector3i::Zero(), 0);
    return MixedLattice(cell, 1, 1, 1, spin_length_SU2, 1.0f);
}

su3::Vector8r cef_field(double e1, double e2, bool legacy) {
    const UnitCell uc = build_tmfeo3_tm(tm_config(e1, e2, legacy));
    return uc.field[0];
}

// ---------------------------------------------------------------------------
void test_convention_and_casimirs() {
    const double e1 = 2.067834, e2 = 4.9628;
    const su3::Vector8r h_new = cef_field(e1, e2, false);
    const su3::Vector8r h_old = cef_field(e1, e2, true);
    const su3::Matrix3c H_cef = su3::Matrix3c(Eigen::Vector3cd(0.0, e1, e2).asDiagonal());

    double spread_new = 0.0, spread_old = 0.0, c2_err = 0.0, c3_err = 0.0, c3_ref_err = 0.0;
    double off_new = 0.0, off_old = 0.0;
    for (int s = 0; s < 200; ++s) {
        const su3::Vector3c psi = random_psi();
        const su3::Vector8r n = su3::expectations_from_psi(psi);
        const double Eq = (psi.adjoint() * H_cef * psi)(0, 0).real();
        const double dn = -h_new.dot(n) - Eq;        // classical E = -field . n
        const double dol = -h_old.dot(n) - 2.0 * Eq;  // legacy: twice the CEF
        if (s == 0) { off_new = dn; off_old = dol; }
        spread_new = std::max(spread_new, std::abs(dn - off_new));
        spread_old = std::max(spread_old, std::abs(dol - off_old));
        c2_err = std::max(c2_err, std::abs(su3::casimir2(n.data()) - 4.0 / 3.0));
        c3_err = std::max(c3_err, std::abs(su3::casimir3(n.data()) - 8.0 / 9.0));
        c3_ref_err = std::max(c3_ref_err, std::abs(su3::casimir3(n.data()) - casimir3_reference(n)));
    }
    check(spread_new < 1e-12, "CEF: E_classical(n(psi)) - <psi|diag(0,e1,e2)|psi> is constant");
    check(spread_old < 1e-12, "legacy CEF encoding is exactly twice the Gell-Mann coefficients");
    check_close(off_new, -(e1 + e2) / 3.0, 1e-12, "CEF constant offset = -(e1+e2)/3");
    check(c2_err < 1e-13, "pure qutrit states: |n|^2 = 4/3");
    check(c3_err < 1e-13, "pure qutrit states: d_abc n^a n^b n^c = 8/9");
    // A mixed (random 8-vector) checks the closed form of d_abc everywhere.
    su3::Vector8r v;
    for (int a = 0; a < 8; ++a) v(a) = std::sin(1.3 * a + 0.2);
    check(std::abs(su3::casimir3(v.data()) - casimir3_reference(v)) < 1e-13 && c3_ref_err < 1e-13,
          "casimir3 closed form = brute-force Tr({lambda_a,lambda_b} lambda_c)/4 contraction");
    check(build_tmfeo3_tm(tm_config(e1, e2, false)).poisson_bracket == 2.0 &&
          build_tmfeo3_tm(tm_config(e1, e2, true)).poisson_bracket == 1.0,
          "TmFeO3 Tm cell carries bracket 2 (legacy: 1)");
}

// ---------------------------------------------------------------------------
// Exact qutrit evolution under diag(0, e1, e2): psi_j(t) = psi_j(0) e^{-i E_j t}.
su3::Vector8r cef_exact(const su3::Vector3c& psi0, double e1, double e2, double t) {
    su3::Vector3c psi = psi0;
    psi(1) *= std::exp(Complex(0.0, -e1 * t));
    psi(2) *= std::exp(Complex(0.0, -e2 * t));
    return su3::expectations_from_psi(psi);
}

void test_free_qutrit_precession() {
    const double e1 = 0.9, e2 = 2.3;
    for (bool legacy : {false, true}) {
        const su3::Vector8r h = cef_field(e1, e2, legacy);
        MixedLattice lat = make_dimer(Eigen::Vector3d(0, 0, 0.3), h, legacy ? 1.0 : 2.0,
                                      Eigen::MatrixXd());
        const su3::Vector3c psi0 = su3::Vector3c(Complex(0.6, 0.0), Complex(0.5, 0.3), Complex(-0.2, 0.5)).normalized();
        lat.spins_SU2[0] = Eigen::Vector3d(0, 0, 0.5);
        lat.spins_SU3[0] = to_x(su3::expectations_from_psi(psi0));
        const std::vector<SpinVector> none2(1, SpinVector::Zero(3)), none3(1, SpinVector::Zero(8));
        // Deliberately awkward grid: the old segment chunking dropped steps here.
        const double T0 = -1.3, T1 = 6.1, dt = 0.02;
        auto traj = lat.single_pulse_drive(none2, none3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                           T0, T1, dt, "dopri5", false, nullptr, true, 1e-11, 1e-11);
        const size_t n_expected = static_cast<size_t>(std::llround((T1 - T0) / dt)) + 1;
        double err = 0.0, terr = 0.0;
        for (size_t k = 0; k < traj.size(); ++k) {
            terr = std::max(terr, std::abs(traj[k].first - (T0 + static_cast<double>(k) * dt)));
            const su3::Vector8r ex = cef_exact(psi0, e1, e2, traj[k].first - T0);
            err = std::max(err, (traj[k].second.second[1] - to_x(ex)).cwiseAbs().maxCoeff());
        }
        const std::string tag = legacy ? " (legacy convention)" : "";
        check(traj.size() == n_expected && terr == 0.0,
              "pulse drive samples exactly t_k = T_start + k dt, k = 0..n" + tag);
        check(err < 2e-8, "free Tm qutrit precesses at the CEF splittings e1, e2, e2-e1" + tag +
                          ": max error " + std::to_string(err));
    }
}

// ---------------------------------------------------------------------------
// Time-dependent mean-field Schrodinger evolution of the Fe (spin 1/2) - Tm
// (qutrit) product state for H = -B.S - h.lambda + S^T K lambda:
//   i dpsi_F/dt = (dE/dS) . sigma/2 psi_F,   i dpsi_T/dt = (dE/dn) . lambda psi_T,
// with E(S, n) = -B.S - h.n + S.K n, S = <sigma/2>, n = <lambda>.
struct ProductState {
    Eigen::Vector2cd f;
    su3::Vector3c t;
};

const std::array<Eigen::Matrix2cd, 3>& pauli() {
    static const std::array<Eigen::Matrix2cd, 3> P = [] {
        std::array<Eigen::Matrix2cd, 3> p;
        p[0] << 0, 1, 1, 0;
        p[1] << 0, Complex(0, -1), Complex(0, 1), 0;
        p[2] << 1, 0, 0, -1;
        return p;
    }();
    return P;
}

Eigen::Vector3d spin_of(const Eigen::Vector2cd& f) {
    Eigen::Vector3d S;
    for (int a = 0; a < 3; ++a) S(a) = 0.5 * (f.adjoint() * pauli()[a] * f)(0, 0).real();
    return S;
}

ProductState tdhf_rhs(const ProductState& x, const Eigen::Vector3d& B, const su3::Vector8r& h,
                      const Eigen::MatrixXd& K) {
    const Eigen::Vector3d S = spin_of(x.f);
    const su3::Vector8r n = su3::expectations_from_psi(x.t);
    const Eigen::Vector3d gS = -B + K * n;
    const su3::Vector8r gn = -h + K.transpose() * S;
    Eigen::Matrix2cd HF = Eigen::Matrix2cd::Zero();
    for (int a = 0; a < 3; ++a) HF += 0.5 * gS(a) * pauli()[a];
    const su3::Matrix3c HT = su3::local_hamiltonian(gn);
    return {Complex(0, -1) * (HF * x.f), Complex(0, -1) * (HT * x.t)};
}

void test_fe_tm_dimer_vs_mean_field_schrodinger() {
    const Eigen::Vector3d B(0.3, -0.2, 0.7);
    su3::Vector8r h;
    h << 0.1, -0.2, 0.45, 0.05, 0.3, -0.1, 0.2, 0.6;
    Eigen::MatrixXd K(3, 8);
    for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 8; ++c) K(a, c) = 0.4 * std::sin(1.7 * a + 0.9 * c + 0.3);

    ProductState x{Eigen::Vector2cd(Complex(0.8, 0.1), Complex(0.3, -0.5)).normalized(),
                   su3::Vector3c(Complex(0.5, 0.2), Complex(-0.4, 0.6), Complex(0.3, 0.1)).normalized()};

    MixedLattice lat = make_dimer(B, h, su3::kGellMannBracket, K);
    lat.spins_SU2[0] = spin_of(x.f);
    lat.spins_SU3[0] = to_x(su3::expectations_from_psi(x.t));
    const std::vector<SpinVector> none2(1, SpinVector::Zero(3)), none3(1, SpinVector::Zero(8));
    const double dt_out = 0.05, T = 6.0;
    auto traj = lat.single_pulse_drive(none2, none3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                       0.0, T, dt_out, "dopri5", false, nullptr, true, 1e-12, 1e-12);

    // Reference: classical RK4 on the Schrodinger pair with a tiny step.
    const int sub = 200;
    const double h_step = dt_out / sub;
    double errS = 0.0, errn = 0.0;
    for (size_t k = 0; k < traj.size(); ++k) {
        if (k > 0) {
            for (int i = 0; i < sub; ++i) {
                auto add = [](const ProductState& a, const ProductState& b, double s) {
                    return ProductState{a.f + s * b.f, a.t + s * b.t};
                };
                const ProductState k1 = tdhf_rhs(x, B, h, K);
                const ProductState k2 = tdhf_rhs(add(x, k1, 0.5 * h_step), B, h, K);
                const ProductState k3 = tdhf_rhs(add(x, k2, 0.5 * h_step), B, h, K);
                const ProductState k4 = tdhf_rhs(add(x, k3, h_step), B, h, K);
                x.f += (h_step / 6.0) * (k1.f + 2.0 * k2.f + 2.0 * k3.f + k4.f);
                x.t += (h_step / 6.0) * (k1.t + 2.0 * k2.t + 2.0 * k3.t + k4.t);
            }
        }
        errS = std::max(errS, (traj[k].second.first[1] - spin_of(x.f)).cwiseAbs().maxCoeff());
        errn = std::max(errn, (traj[k].second.second[1] - to_x(su3::expectations_from_psi(x.t))).cwiseAbs().maxCoeff());
    }
    check(errS < 1e-8 && errn < 1e-8,
          "Fe-Tm dimer follows mean-field Schrodinger dynamics (S err " + std::to_string(errS) +
          ", n err " + std::to_string(errn) + ")");
}

// ---------------------------------------------------------------------------
MixedLattice make_tmfeo3(int L) {
    SpinConfig cfg;
    cfg.field_strength = 0.0;
    cfg.set_param("J1ab", 4.74); cfg.set_param("J1c", 5.15);
    cfg.set_param("J2ab", 0.15); cfg.set_param("J2c", 0.30);
    cfg.set_param("Ka", -0.16221); cfg.set_param("Kc", -0.18318);
    cfg.set_param("D1", 0.12);
    cfg.set_param("e1", 0.97); cfg.set_param("e2", 3.97);
    // Fe-Tm exchange K^- and the Fe-Fe-Tm vertex W (T(S_i, S_i, n_k), on-site
    // in the Fe index), so every coupling class of the production model acts.
    cfg.set_param("Kminus_2x", 0.12); cfg.set_param("Kminus_5y", -0.08); cfg.set_param("Kminus_7z", 0.05);
    cfg.set_param("W3_xx", 0.05); cfg.set_param("W8_zz", -0.04); cfg.set_param("W1_xy", 0.03);
    cfg.set_param("W4_xz", 0.02); cfg.set_param("W6_yz", -0.025);
    MixedLattice lat(build_tmfeo3(cfg), L, L, L, 1.0f, 1.0f);
    // Random physical states: unit Fe spins, Tm n = <psi|lambda|psi>.
    std::normal_distribution<double> g(0.0, 1.0);
    for (auto& s : lat.spins_SU2) { Eigen::Vector3d v(g(rng), g(rng), g(rng)); s = v.normalized(); }
    for (auto& n : lat.spins_SU3) n = to_x(su3::expectations_from_psi(random_psi()));
    return lat;
}

void test_energy_and_casimir_conservation() {
    MixedLattice lat = make_tmfeo3(2);
    bool has_tri = false;
    for (const auto& t : lat.mixed_trilinear_partners_SU2) has_tri = has_tri || !t.empty();
    check(has_tri, "TmFeO3 test model has mixed Fe-Fe-Tm trilinear couplings");

    // A Bloch-damping equilibrium is only a relaxation target: with zero rates
    // it must not change the Hamiltonian (it used to be subtracted from the
    // SU(3) leg of the trilinear in the MD field only).
    MixedLattice::SpinConfigSU3 eq = lat.spins_SU3;
    for (auto& v : eq) v *= 0.7;
    lat.set_equilibrium_SU3(eq);

    // MC field == MD field (no drive), site by site.
    const auto x0 = lat.spins_to_state();
    const size_t off = lat.lattice_size_SU2 * 3;
    double field_diff = 0.0;
    for (size_t i = 0; i < lat.lattice_size_SU2; ++i) {
        double H[3];
        lat.get_local_field_SU2_flat_into(i, x0, off, 0.0, 0.0, H);
        field_diff = std::max(field_diff, (Eigen::Map<Eigen::Vector3d>(H) - lat.get_local_field_SU2(i)).cwiseAbs().maxCoeff());
    }
    for (size_t i = 0; i < lat.lattice_size_SU3; ++i) {
        double H[8];
        lat.get_local_field_SU3_flat_into(i, x0, off, 0.0, 0.0, H);
        field_diff = std::max(field_diff, (Eigen::Map<Eigen::VectorXd>(H, 8) - lat.get_local_field_SU3(i)).cwiseAbs().maxCoeff());
    }
    check(field_diff < 1e-12, "MD local fields equal the MC local fields (one Hamiltonian)");

    std::vector<std::vector<double>> states;
    const std::vector<SpinVector> none2(lat.N_atoms_SU2, SpinVector::Zero(3));
    const std::vector<SpinVector> none3(lat.N_atoms_SU3, SpinVector::Zero(8));
    lat.single_pulse_drive(none2, none3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 2.0, 0.05, "dopri5", false, &states, true, 1e-11, 1e-11);
    const double E0 = lat.total_energy_flat(states.front().data());
    double dE = 0.0, dC2 = 0.0, dC3 = 0.0, dS = 0.0;
    for (const auto& s : states) {
        dE = std::max(dE, std::abs(lat.total_energy_flat(s.data()) - E0));
        for (size_t i = 0; i < lat.lattice_size_SU2; ++i)
            dS = std::max(dS, std::abs(Eigen::Map<const Eigen::Vector3d>(&s[3 * i]).norm() - 1.0));
        for (size_t i = 0; i < lat.lattice_size_SU3; ++i) {
            dC2 = std::max(dC2, std::abs(su3::casimir2(&s[off + 8 * i]) - 4.0 / 3.0));
            dC3 = std::max(dC3, std::abs(su3::casimir3(&s[off + 8 * i]) - 8.0 / 9.0));
        }
    }
    const double scale = std::abs(E0) + 1.0;
    check(dE / scale < 1e-8, "full TmFeO3 model conserves the energy (rel. drift " + std::to_string(dE / scale) + ")");
    check(dS < 1e-8 && dC2 < 1e-8 && dC3 < 1e-8,
          "|S_i|, |n_i|^2 and d_abc n^a n^b n^c are conserved (" + std::to_string(dS) + ", " +
          std::to_string(dC2) + ", " + std::to_string(dC3) + ")");
}

void test_trilinear_reference_is_one_hamiltonian() {
    MixedLattice lat = make_tmfeo3(2);
    const double E_plain = lat.total_energy();
    const size_t nbi_before = lat.bilinear_partners_SU2[0].size();
    MixedLattice::SpinConfigSU3 ref = lat.spins_SU3;
    for (size_t k = 0; k < ref.size(); ++k) ref[k] = to_x(su3::expectations_from_psi(random_psi()));

    // Expected Fe field: T_i(S_j, n_k - r_k) instead of T_i(S_j, n_k).
    std::vector<Eigen::Vector3d> H_expected(lat.lattice_size_SU2);
    for (size_t i = 0; i < lat.lattice_size_SU2; ++i) {
        Eigen::Vector3d Hi = lat.get_local_field_SU2(i);
        for (size_t n = 0; n < lat.mixed_trilinear_partners_SU2[i].size(); ++n) {
            const size_t j = lat.mixed_trilinear_partners_SU2[i][n][0];
            const size_t k = lat.mixed_trilinear_partners_SU2[i][n][1];
            const auto& T = lat.mixed_trilinear_interaction_SU2[i][n];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    for (int c = 0; c < 8; ++c) Hi(a) -= T[a](b, c) * lat.spins_SU2[j](b) * ref[k](c);
        }
        H_expected[i] = Hi;
    }
    lat.set_mixed_trilinear_reference_SU3(ref);
    lat.set_mixed_trilinear_reference_SU3(ref);  // idempotent
    double hdiff = 0.0, hdiff_md = 0.0;
    const auto x = lat.spins_to_state();
    for (size_t i = 0; i < lat.lattice_size_SU2; ++i) {
        hdiff = std::max(hdiff, (lat.get_local_field_SU2(i) - H_expected[i]).cwiseAbs().maxCoeff());
        double H[3];
        lat.get_local_field_SU2_flat_into(i, x, lat.lattice_size_SU2 * 3, 0.0, 0.0, H);
        hdiff_md = std::max(hdiff_md, (Eigen::Map<Eigen::Vector3d>(H) - H_expected[i]).cwiseAbs().maxCoeff());
    }
    check(hdiff < 1e-12 && hdiff_md < 1e-12, "reference subtraction: MC and MD Fe fields = T(S_j, n_k - r_k)");

    // MC energy differences agree with total-energy differences.
    double worst = 0.0;
    for (size_t i = 0; i < lat.lattice_size_SU2; i += 5) {
        const SpinVector old_s = lat.spins_SU2[i];
        const SpinVector new_s = Eigen::Vector3d(std::cos(0.3 * i), std::sin(0.3 * i), 0.4).normalized();
        const double E_before = lat.total_energy();
        const double dE_site = lat.site_energy_SU2_diff(new_s, old_s, i);
        lat.spins_SU2[i] = new_s;
        worst = std::max(worst, std::abs(lat.total_energy() - E_before - dE_site));
        lat.spins_SU2[i] = old_s;
    }
    check(worst < 1e-10, "reference subtraction: site energy differences match the total energy");

    // Energy conservation of the modified Hamiltonian.
    std::vector<std::vector<double>> states;
    const std::vector<SpinVector> none2(lat.N_atoms_SU2, SpinVector::Zero(3));
    const std::vector<SpinVector> none3(lat.N_atoms_SU3, SpinVector::Zero(8));
    lat.single_pulse_drive(none2, none3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 1.0, 0.05, "dopri5", false, &states, true, 1e-11, 1e-11);
    double dE = 0.0;
    for (const auto& s : states) {
        dE = std::max(dE, std::abs(lat.total_energy_flat(s.data()) - lat.total_energy_flat(states[0].data())));
    }
    check(dE < 1e-8 * (1.0 + std::abs(lat.total_energy())), "reference subtraction: MD conserves the modified energy");

    lat.set_mixed_trilinear_reference_SU3({});
    check(lat.bilinear_partners_SU2[0].size() == nbi_before && std::abs(lat.total_energy() - E_plain) < 1e-11,
          "removing the reference restores the original Hamiltonian");
}

// ---------------------------------------------------------------------------
void test_no_phantom_pulse_and_w1() {
    su3::Vector8r h = cef_field(0.9, 2.3, false);
    Eigen::MatrixXd Kd(3, 8);
    for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 8; ++c) Kd(a, c) = 0.3 * std::cos(0.7 * a + 1.1 * c);
    MixedLattice lat = make_dimer(Eigen::Vector3d(0.0, 0.0, 0.8), h, 2.0, Eigen::MatrixXd(), &Kd);
    check(lat.has_mixed_bilinear_drive, "dimer has a field-assisted (pulse-gated) Fe-Tm bond");
    // Non-stationary initial state so the gated exchange would act.
    lat.spins_SU2[0] = Eigen::Vector3d(0.3, 0.2, 0.3).normalized() * 0.5;
    lat.spins_SU3[0] = to_x(su3::expectations_from_psi(su3::Vector3c(Complex(0.7, 0), Complex(0.4, 0.2), Complex(0.1, -0.5)).normalized()));
    const std::vector<SpinVector> d2(1, Eigen::Vector3d(1.0, 0.0, 0.0));
    std::vector<SpinVector> d3(1, SpinVector::Zero(8));
    d3[0](1) = 1.0;
    const std::vector<SpinVector> none2(1, SpinVector::Zero(3)), none3(1, SpinVector::Zero(8));
    // Pulse far in the future: the trajectory on [0, 4] must equal the undriven one.
    auto far = lat.single_pulse_drive(d2, d3, 1000.0, 0.5, 0.2, 3.0, 0.5, 0.2, 3.0,
                                      0.0, 4.0, 0.02, "dopri5", false, nullptr, true, 1e-11, 1e-11);
    auto free = lat.single_pulse_drive(none2, none3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                       0.0, 4.0, 0.02, "dopri5", false, nullptr, true, 1e-11, 1e-11);
    double diff = 0.0;
    for (size_t k = 0; k < far.size(); ++k) {
        diff = std::max(diff, (far[k].second.first[1] - free[k].second.first[1]).cwiseAbs().maxCoeff());
        diff = std::max(diff, (far[k].second.second[1] - free[k].second.second[1]).cwiseAbs().maxCoeff());
    }
    check(far.size() == free.size() && diff < 1e-9,
          "single pulse: no phantom second pulse at t = 0 gating the assisted exchange (diff " +
          std::to_string(diff) + ")");

    // W1: from the stationary ground state (Fe along +B, Tm in the lowest CEF
    // level), M1(tau; t) synthesised by shifting M0 equals the integrated M1.
    lat.spins_SU2[0] = Eigen::Vector3d(0, 0, 0.5);
    su3::Vector3c g(1.0, 0.0, 0.0);
    lat.spins_SU3[0] = to_x(su3::expectations_from_psi(g));
    check(lat.relative_stationarity_residual(lat.spins_to_state()) < 1e-14, "dimer ground state is stationary");
    const double T0 = -3.0, T1 = 9.0, dt = 0.02, tau = 2.4;
    auto M0 = lat.single_pulse_drive(d2, d3, 0.0, 0.5, 0.2, 3.0, 0.5, 0.2, 3.0, T0, T1, dt,
                                     "dopri5", false, nullptr, true, 1e-11, 1e-11);
    auto M1 = lat.single_pulse_drive(d2, d3, tau, 0.5, 0.2, 3.0, 0.5, 0.2, 3.0, T0, T1, dt,
                                     "dopri5", false, nullptr, true, 1e-11, 1e-11);
    const auto ground = lat.observe(lat.spins_to_state().data());
    auto M1s = lat.synthesize_M1_from_M0(M0, ground, tau, T0, T1, dt);
    double w1 = 0.0;
    for (size_t k = 0; k < M1.size(); ++k) {
        w1 = std::max(w1, (M1[k].second.first[1] - M1s[k].second.first[1]).cwiseAbs().maxCoeff());
        w1 = std::max(w1, (M1[k].second.second[1] - M1s[k].second.second[1]).cwiseAbs().maxCoeff());
    }
    check(M1.size() == M1s.size() && w1 < 1e-8, "W1: shifted M0 equals the integrated M1 sample by sample (" +
                                                std::to_string(w1) + ")");
    bool threw = false;
    try { lat.synthesize_M1_from_M0(M0, ground, tau + 0.5 * dt, T0, T1, dt); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "W1 refuses delays that are not on the time grid");
}

// ---------------------------------------------------------------------------
void test_thermal_reservoir_closed_form() {
    const double e1 = 0.9, e2 = 2.3;
    MixedLattice lat = make_dimer(Eigen::Vector3d(0, 0, 0.5), cef_field(e1, e2, false), 2.0, Eigen::MatrixXd());
    const double theta = 0.6, Gamma = 0.4, Gamma3 = 0.7, kappa = 0.3;
    su3::Vector3c psi(Complex(std::cos(theta), 0), Complex(std::sin(theta), 0), 0.0);
    const su3::Vector8r n0 = su3::expectations_from_psi(psi);
    lat.spins_SU2[0] = Eigen::Vector3d(0, 0, 0.5);
    lat.spins_SU3[0] = to_x(n0);
    SpinVector rates = SpinVector::Zero(8);
    rates(0) = Gamma; rates(1) = Gamma; rates(2) = Gamma3;
    lat.set_damping_SU3(rates);
    SpinVector eq = to_x(n0);
    eq(0) = 0.0; eq(1) = 0.0;
    lat.set_equilibrium_SU3({eq});
    lat.thermal_heat = kappa;
    check(lat.ode_state_size() == lat.spin_state_size() + 1, "thermal reservoir adds one ODE variable");

    const double A = n0(0) * n0(0) + n0(1) * n0(1);
    auto n3_exact = [&](double t) {
        // E(t) = (A/2)(1 - e^{-2 Gamma t});  dn3/dt = -Gamma3 (n3 - eq3 + kappa E).
        const double a = (1.0 - std::exp(-Gamma3 * t)) / Gamma3;
        const double b = (std::exp(-2.0 * Gamma * t) - std::exp(-Gamma3 * t)) / (Gamma3 - 2.0 * Gamma);
        return n0(2) - Gamma3 * kappa * 0.5 * A * (a - b);
    };
    const std::vector<SpinVector> none2(1, SpinVector::Zero(3)), none3(1, SpinVector::Zero(8));
    for (const char* method : {"dopri5", "rk4"}) {
        auto traj = lat.single_pulse_drive(none2, none3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                           0.0, 8.0, 0.01, method, false, nullptr, true, 1e-11, 1e-11);
        double err = 0.0;
        for (const auto& [t, o] : traj) err = std::max(err, std::abs(o.second[1](2) - n3_exact(t)));
        check(err < 1e-8, std::string("thermal reservoir: lambda3 follows the closed form (") + method +
                          ", err " + std::to_string(err) + ")");
    }
}

// ---------------------------------------------------------------------------
// SU(3) Landau-Lifshitz damping: energy decreases monotonically, both
// Casimirs are conserved (a pure state stays pure) and a free qutrit relaxes
// into the lowest CEF level.
void test_su3_ll_damping() {
    const double e1 = 0.9, e2 = 2.3;
    MixedLattice lat = make_dimer(Eigen::Vector3d(0, 0, 0.5), cef_field(e1, e2, false), 2.0, Eigen::MatrixXd());
    lat.spins_SU2[0] = Eigen::Vector3d(0, 0, 0.5);
    lat.spins_SU3[0] = to_x(su3::expectations_from_psi(
        su3::Vector3c(Complex(0.3, 0.1), Complex(0.6, -0.2), Complex(0.5, 0.5)).normalized()));
    lat.alpha_SU3 = 0.3;
    std::vector<std::vector<double>> states;
    const std::vector<SpinVector> none2(1, SpinVector::Zero(3)), none3(1, SpinVector::Zero(8));
    lat.single_pulse_drive(none2, none3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 60.0, 0.1, "dopri5", false, &states, true, 1e-11, 1e-11);
    bool monotone = true;
    double prev = 1e300, dC2 = 0.0, dC3 = 0.0;
    for (const auto& s : states) {
        const double E = lat.total_energy_flat(s.data());
        monotone = monotone && E <= prev + 1e-12;
        prev = E;
        dC2 = std::max(dC2, std::abs(su3::casimir2(&s[3]) - 4.0 / 3.0));
        dC3 = std::max(dC3, std::abs(su3::casimir3(&s[3]) - 8.0 / 9.0));
    }
    const su3::Vector8r ground = su3::expectations_from_psi(su3::Vector3c(1.0, 0.0, 0.0));
    double dist = 0.0;
    for (int a = 0; a < 8; ++a) dist = std::max(dist, std::abs(states.back()[3 + a] - ground(a)));
    check(monotone, "SU(3) LL damping: energy decreases monotonically");
    check(dC2 < 1e-8 && dC3 < 1e-8, "SU(3) LL damping conserves |n|^2 and d_abc n n n (pure stays pure)");
    check(dist < 1e-6, "SU(3) LL damping relaxes a free qutrit into the lowest CEF level (dist " +
                       std::to_string(dist) + ")");
}

// ---------------------------------------------------------------------------
void test_su3_implicit_midpoint() {
    // Nonlinear mean-field Hamiltonian H(psi) = H0 + g Σ_a n_a(psi) lambda_a.
    su3::Matrix3c H0;
    H0 << 0.3, Complex(0.1, 0.2), Complex(-0.2, 0.05),
          Complex(0.1, -0.2), 1.1, Complex(0.3, 0.1),
          Complex(-0.2, -0.05), Complex(0.3, -0.1), 2.0;
    const double gcoup = 0.7;
    auto H_of = [&](const su3::Vector3c& p) {
        su3::Vector3c q = p;  // n of the (nearly normalised) midpoint
        return su3::Matrix3c(H0 + su3::local_hamiltonian(gcoup * su3::expectations_from_psi(q)));
    };
    const su3::Vector3c psi0 = su3::Vector3c(Complex(0.5, 0.1), Complex(-0.3, 0.6), Complex(0.4, -0.2)).normalized();
    const double T = 2.0;
    // Reference: RK4 with a tiny step.
    su3::Vector3c ref = psi0;
    {
        const int N = 200000;
        const double h = T / N;
        auto f = [&](const su3::Vector3c& p) { return su3::Vector3c(Complex(0, -1) * (H_of(p) * p)); };
        for (int i = 0; i < N; ++i) {
            const su3::Vector3c k1 = f(ref), k2 = f(ref + 0.5 * h * k1), k3 = f(ref + 0.5 * h * k2), k4 = f(ref + h * k3);
            ref += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }
    double err[3], norm_dev = 0.0;
    const int steps[3] = {50, 100, 200};
    for (int s = 0; s < 3; ++s) {
        su3::Vector3c psi = psi0;
        const double dt = T / steps[s];
        for (int i = 0; i < steps[s]; ++i) {
            su3::implicit_midpoint_step(psi, H_of, dt);
            norm_dev = std::max(norm_dev, std::abs(psi.norm() - 1.0));
        }
        err[s] = (psi - ref).norm();
    }
    const double order1 = std::log2(err[0] / err[1]), order2 = std::log2(err[1] / err[2]);
    check(norm_dev < 1e-13, "implicit midpoint is unitary: |psi| = 1 to " + std::to_string(norm_dev));
    check(order1 > 1.9 && order2 > 1.9 && order1 < 2.2 && order2 < 2.2,
          "implicit midpoint is second order (observed " + std::to_string(order1) + ", " +
          std::to_string(order2) + ")");
}

// ---------------------------------------------------------------------------
std::vector<double> read_h5(const std::string& file, const std::string& path) {
    H5::H5File f(file, H5F_ACC_RDONLY);
    H5::DataSet ds = f.openDataSet(path);
    std::vector<hsize_t> dims(ds.getSpace().getSimpleExtentNdims());
    ds.getSpace().getSimpleExtentDims(dims.data());
    size_t n = 1;
    for (auto d : dims) n *= d;
    std::vector<double> v(n);
    ds.read(v.data(), H5::PredType::NATIVE_DOUBLE);
    return v;
}

void test_md_diagnostics_and_nan_guard() {
    MixedLattice lat = make_tmfeo3(2);
    const std::string dir = (std::filesystem::temp_directory_path() /
                             ("classical_spin_mixed_md_diag_" + std::to_string(::getpid()))).string();
    std::filesystem::remove_all(dir);
    lat.molecular_dynamics(0.0, 1.0, 0.01, dir, 5, "dopri5", false, 1e-10, 1e-10);
    const std::string file = dir + "/trajectory.h5";
    const auto t = read_h5(file, "/diagnostics/times");
    const auto e = read_h5(file, "/diagnostics/energy_per_site");
    const auto ds = read_h5(file, "/diagnostics/spin_length_drift_SU2");
    const auto c2 = read_h5(file, "/diagnostics/casimir2_drift_SU3");
    const auto c3 = read_h5(file, "/diagnostics/casimir3_drift_SU3");
    const auto ts = read_h5(file, "/trajectory_SU2/times");
    bool grid = t.size() == 21 && ts.size() == 21;
    for (size_t k = 0; grid && k < t.size(); ++k) grid = t[k] == 0.05 * static_cast<double>(k) && ts[k] == t[k];
    check(grid, "MD output on the uniform grid t_k = k * save_interval * dt (21 samples)");
    double de = 0.0, worst = 0.0;
    for (size_t k = 0; k < e.size(); ++k) {
        de = std::max(de, std::abs(e[k] - e[0]));
        worst = std::max({worst, ds[k], c2[k], c3[k]});
    }
    check(de < 1e-8 && worst < 1e-8, "MD diagnostics: energy, |S| and Casimir drifts recorded and small (" +
                                     std::to_string(de) + ", " + std::to_string(worst) + ")");
    std::filesystem::remove_all(dir);

    lat.spins_SU3[3](2) = std::numeric_limits<double>::quiet_NaN();
    bool threw = false;
    try {
        lat.molecular_dynamics(0.0, 0.1, 0.01, "", 1, "rk4");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "a non-finite state aborts the dynamics with an exception");
}

// ---------------------------------------------------------------------------
void test_validation() {
    MixedLattice lat = make_dimer(Eigen::Vector3d(0, 0, 0.5), cef_field(0.9, 2.3, false), 2.0, Eigen::MatrixXd());
    const std::vector<SpinVector> none2(1, SpinVector::Zero(3)), none3(1, SpinVector::Zero(8));
    auto throws = [&](auto&& f) {
        try { f(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(throws([&] { lat.single_pulse_drive(none2, none3, 0.0, 0, 1, 0, 0, 1, 0, 0.0, 1.0, 0.1, "dopri"); }),
          "unknown integrator name throws (no silent dopri5 fallback)");
    check(throws([&] { lat.single_pulse_drive(none2, none3, 0.0, 1.0, 0.0, 0, 0, 1, 0, 0.0, 1.0, 0.1); }),
          "zero pulse width with non-zero amplitude throws");
    check(throws([&] { lat.single_pulse_drive(none2, none3, 0.0, 0, 1, 0, 0, 1, 0, 0.0, 1.0, 0.0); }),
          "zero time step throws");
    check(throws([&] { lat.set_damping_SU3(SpinVector::Zero(3)); }), "mis-sized SU(3) damping rates throw");
    check(throws([&] { lat.set_equilibrium_SU3({}); }), "mis-sized SU(3) equilibrium throws");
    check(throws([&] {
              UnitCell a(3, 1), b(4, 1);
              MixedLattice bad(MixedUnitCell(a, b), 1, 1, 1);
          }),
          "non-(3, 8) spin dimensions are rejected at construction");
    check(lat.gpu_unsupported_reason().empty(), "plain bilinear dimer is GPU-compatible");
    lat.alpha_gilbert = 0.1;
    check(lat.gpu_unsupported_reason().find("Gilbert") != std::string::npos,
          "GPU support check reports the dropped Gilbert damping");
    lat.alpha_gilbert = 0.0;
    lat.su3_bracket = su3::kLegacyBracket;
    check(!lat.gpu_unsupported_reason().empty(), "GPU support check reports the legacy SU(3) bracket");
}

}  // namespace

int main() {
    test_convention_and_casimirs();
    test_free_qutrit_precession();
    test_fe_tm_dimer_vs_mean_field_schrodinger();
    test_energy_and_casimir_conservation();
    test_trilinear_reference_is_one_hamiltonian();
    test_no_phantom_pulse_and_w1();
    test_thermal_reservoir_closed_form();
    test_su3_ll_damping();
    test_su3_implicit_midpoint();
    test_md_diagnostics_and_nan_guard();
    test_validation();
    return phys_test::finish("test_mixed_md");
}
