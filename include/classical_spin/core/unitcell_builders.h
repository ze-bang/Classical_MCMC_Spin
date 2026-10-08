/**
 * unitcell_builders.h - Unit cell builder function declarations
 * 
 * This header declares the builder functions for various lattice types.
 * Implementations are in unitcell_builders.cpp.
 */

#ifndef UNITCELL_BUILDERS_H
#define UNITCELL_BUILDERS_H

#include "spin_config.h"
#include "unitcell.h"

// Build BCAO honeycomb unit cell
UnitCell build_bcao_honeycomb(const SpinConfig& config);

// Build Kitaev honeycomb unit cell
UnitCell build_kitaev_honeycomb(const SpinConfig& config);

// Build pyrochlore unit cell
UnitCell build_pyrochlore(const SpinConfig& config);

// Build non-Kramers pyrochlore unit cell (Jpm, Jzz, Jpmpm exchange)
UnitCell build_pyrochlore_non_kramer(const SpinConfig& config);

// Build the NdMgAl11O19 anisotropic triangular unit cell
// (Jzz, Jpm, Jpmpm, Jzpm with bond phases 0, -2pi/3, +2pi/3)
UnitCell build_triangular_anisotropic(const SpinConfig& config);

// Build the kagome XXZ model with Ising 2NN + hexagon-diagonal couplings
// (Balents-Fisher-Girvin plane: Jxy, Delta1, Delta2, optional single-ion D)
UnitCell build_kagome_bfg(const SpinConfig& config);

// Build TmFeO3 mixed unit cell (SU2 Fe + SU3 Tm)
MixedUnitCell build_tmfeo3(const SpinConfig& config);

// Build TmFeO3 Fe-only unit cell (SU2 only, no Tm atoms)
UnitCell build_tmfeo3_fe(const SpinConfig& config);

// Build TmFeO3 Tm-only unit cell (SU3 only, no Fe atoms)
UnitCell build_tmfeo3_tm(const SpinConfig& config);

// Build honeycomb unit cell for phonon lattice (Kitaev-Heisenberg-Γ-Γ' with bond types)
// Includes NN with bond_type metadata, J2 (sublattice-dependent), J3, and sublattice frames
UnitCell build_phonon_honeycomb(const SpinConfig& config);


#endif // UNITCELL_BUILDERS_H
