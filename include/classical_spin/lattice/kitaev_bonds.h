#pragma once
/**
 * kitaev_bonds.h — Shared helpers for the honeycomb Kitaev bond Hamiltonian.
 *
 * The bond exchange matrices of the J–K–Γ–Γ' model are written in the cubic
 * Kitaev frame (x, y, z along the Co–O bonds; "local" frame below), where the
 * γ-bond carries K on the γγ diagonal. Spins can be STORED in another frame;
 * this header is the single place that fixes the frame conventions.
 *
 * FRAMES (S_cubic = Kitaev-frame components of a spin):
 *
 *   R = [a | b | c*]  with  a = (1,1,-2)/√6,  b = (-1,1,0)/√2,  c* = (1,1,1)/√3
 *   the crystal axes in cubic coordinates, so S_cubic = R · S_crystal.
 *
 *   Frame::Crystal  S_storage = S_crystal = Rᵀ S_cubic,  J_storage = Rᵀ J_cubic R.
 *                   c* (the C3 axis, honeycomb normal) is the storage z axis and the
 *                   in-plane axes coincide with the HoneyComb position frame: Rᵀ maps the
 *                   Kitaev bond vectors d_x, d_y, d_z onto the x, y, z bond LINES of
 *                   HoneyComb (the orientation is A←B; A→B would be the C2(c*) image,
 *                   which together with time reversal is a symmetry of H(S, Q; h), so
 *                   the two orientations are physically equivalent). A field_direction
 *                   is therefore a crystal direction: (0,0,1) is c*. Default.
 *   Frame::Cubic    S_storage = S_cubic (build_kitaev_honeycomb; field_direction in
 *                   cubic axes, e.g. (1,1,1)/√3 = c*).
 *   Frame::Legacy   S_storage = R S_cubic, J_storage = R J_cubic Rᵀ: the pre-2026-10
 *                   PhononLattice convention. It is neither the cubic nor the crystal
 *                   frame (c* sits at (0.16, 0.98, -0.14), so field_direction (0,0,1)
 *                   was almost in-plane); kept only to reproduce old runs exactly
 *                   (config key `legacy_kitaev_frame = 1`). See docs/MIGRATION.md.
 *
 * Zero-field energetics are identical in every frame (a global rotation); the frame
 * fixes what field_direction and the stored spin components mean.
 */

#include "classical_spin/core/simple_linear_alg.h"
#include <cmath>
#include <Eigen/Dense>

namespace classical_spin::kitaev {

enum class Frame { Crystal, Cubic, Legacy };

/// Config key that selects Frame::Legacy for the NCTO (PhononLattice) model.
inline constexpr const char* kLegacyFrameKey = "legacy_kitaev_frame";

/// NCTO storage frame from the value of `legacy_kitaev_frame` (0 = crystal, default).
inline Frame ncto_frame_from_flag(double legacy_flag) {
    return legacy_flag > 0.5 ? Frame::Legacy : Frame::Crystal;
}

inline const char* frame_name(Frame f) {
    switch (f) {
        case Frame::Crystal: return "crystal (a, b, c*)";
        case Frame::Cubic:   return "cubic Kitaev (x, y, z)";
        case Frame::Legacy:  return "legacy R·cubic";
    }
    return "?";
}

/// R: columns are the crystal axes a, b, c* in cubic coordinates (S_cubic = R S_crystal).
inline Eigen::Matrix3d crystal_axes_in_cubic() {
    Eigen::Matrix3d R;
    R <<  1.0 / std::sqrt(6.0), -1.0 / std::sqrt(2.0), 1.0 / std::sqrt(3.0),
          1.0 / std::sqrt(6.0),  1.0 / std::sqrt(2.0), 1.0 / std::sqrt(3.0),
         -2.0 / std::sqrt(6.0),  0.0,                  1.0 / std::sqrt(3.0);
    return R;
}

/// Same matrix R as a dynamic SpinMatrix (historic signature).
inline SpinMatrix kitaev_rotation() { return crystal_axes_in_cubic(); }

/// U with S_storage = U S_cubic, so J_storage = U J_cubic Uᵀ.
inline Eigen::Matrix3d storage_from_cubic(Frame f) {
    const Eigen::Matrix3d R = crystal_axes_in_cubic();
    switch (f) {
        case Frame::Crystal: return R.transpose();
        case Frame::Cubic:   return Eigen::Matrix3d::Identity();
        case Frame::Legacy:  return R;
    }
    return R.transpose();
}

/// C with S_crystal = C S_storage (= Rᵀ Uᵀ): the "global" (crystal-axis) output frame.
inline Eigen::Matrix3d crystal_from_storage(Frame f) {
    return crystal_axes_in_cubic().transpose() * storage_from_cubic(f).transpose();
}

/// Exchange (or magnetoelastic) matrix from the cubic Kitaev frame to storage frame f.
inline Eigen::Matrix3d to_storage_frame(const Eigen::Matrix3d& J_cubic, Frame f) {
    const Eigen::Matrix3d U = storage_from_cubic(f);
    return U * J_cubic * U.transpose();
}

/// Cubic Kitaev frame → crystal frame, Rᵀ J R (the default NCTO storage frame).
inline SpinMatrix to_global_frame(const SpinMatrix& J_local) {
    return to_storage_frame(Eigen::Matrix3d(J_local), Frame::Crystal);
}

/// x-bond (local frame) exchange matrix with Kitaev term on diagonal (0,0).
inline SpinMatrix make_Jx_local(double J, double K, double Gamma, double Gammap) {
    SpinMatrix Jx = SpinMatrix::Zero(3, 3);
    Jx << J + K, Gammap, Gammap,
          Gammap, J,     Gamma,
          Gammap, Gamma, J;
    return Jx;
}

/// y-bond (local frame) exchange matrix with Kitaev term on diagonal (1,1).
inline SpinMatrix make_Jy_local(double J, double K, double Gamma, double Gammap) {
    SpinMatrix Jy = SpinMatrix::Zero(3, 3);
    Jy << J,      Gammap, Gamma,
          Gammap, J + K,  Gammap,
          Gamma,  Gammap, J;
    return Jy;
}

/// z-bond (local frame) exchange matrix with Kitaev term on diagonal (2,2).
inline SpinMatrix make_Jz_local(double J, double K, double Gamma, double Gammap) {
    SpinMatrix Jz = SpinMatrix::Zero(3, 3);
    Jz << J,      Gamma,  Gammap,
          Gamma,  J,      Gammap,
          Gammap, Gammap, J + K;
    return Jz;
}

/// Bond γ ∈ {0, 1, 2} = {x, y, z} exchange matrix in the cubic Kitaev frame.
inline Eigen::Matrix3d make_J_local(int bond_type, double J, double K, double Gamma, double Gammap) {
    switch (bond_type) {
        case 0: return make_Jx_local(J, K, Gamma, Gammap);
        case 1: return make_Jy_local(J, K, Gamma, Gammap);
        default: return make_Jz_local(J, K, Gamma, Gammap);
    }
}

/// Isotropic Heisenberg exchange (J_ij = J * I_3). Used by J2_A, J2_B, J3.
inline SpinMatrix heisenberg_matrix(double J) {
    return J * SpinMatrix::Identity(3, 3);
}

} // namespace classical_spin::kitaev
