/**
 * @file phonon_lattice.cpp
 * @brief Spin–phonon coupled honeycomb lattice (NCTO magnetoelastic model).
 *
 * Implements the model described in phonon_lattice.h:
 *   - J–K–Γ–Γ' nearest-neighbour exchange (cubic Kitaev frame rotated into the spin
 *     storage frame, the crystal frame (a, b, c*) by default; kitaev_bonds.h), plus
 *     sublattice-dependent J2, isotropic J3, six-spin ring exchange J7 and a Zeeman field.
 *   - The zone-centre lattice sector (primary E1 doublet, extra E/A1/A2 modes, frozen
 *     strains) with the D3-allowed magnetoelastic couplings of ncto_me_tensors.h, cubic
 *     anharmonic transfers, and optional spin–lattice dynamics (site displacements with
 *     exchange striction).
 *   - A polar THz drive H_drive = -N Σ_m Z*_m E(t)·Q_m on the polar modes.
 *
 * Structure: one coordinate-dependent coupling evaluation (compute_couplings), one
 * spin-field kernel (field_without_ring), one hexagon kernel (ring_hexagon) and one
 * correlation kernel (correlations_of) serve the energies, the Monte Carlo moves and
 * the equations of motion.
 */

#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/pulse_chunking.h"
#include "classical_spin/io/spin_table.h"
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <filesystem>
#include <random>
#include <algorithm>
#include <complex>
#include <mpi.h>

#include <boost/numeric/odeint.hpp>
// Boost uBLAS for the implicit solvers (rosenbrock4, implicit_euler)
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_dense_output.hpp>
#include <boost/numeric/odeint/stepper/implicit_euler.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

namespace odeint = boost::numeric::odeint;

namespace {

constexpr double SQRT3 = 1.7320508075688772935;

using classical_spin::ncto_me::N_E;
using classical_spin::ncto_me::N_A1;
using classical_spin::ncto_me::N_A2;
using Vec3 = Eigen::Vector3d;
using CMap3 = Eigen::Map<const Eigen::Vector3d>;

/// One 3x3 bond block of a tensor table (Kitaev/local frame, A->B orientation).
inline Eigen::Matrix3d tbl(const double (&T)[3][3]) {
    Eigen::Matrix3d M;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) M(a, b) = T[a][b];
    return M;
}
inline Eigen::Matrix3d TE1(int k, int g) { return tbl(classical_spin::ncto_me::NCTO_ME_E1[k][g]); }
inline Eigen::Matrix3d TE2(int k, int g) { return tbl(classical_spin::ncto_me::NCTO_ME_E2[k][g]); }
inline Eigen::Matrix3d TA1(int k, int g) { return tbl(classical_spin::ncto_me::NCTO_ME_A1[k][g]); }
inline Eigen::Matrix3d TA2(int k, int g) { return tbl(classical_spin::ncto_me::NCTO_ME_A2[k][g]); }

/// Bond-line projection of an E doublet, f = q1 cos2θ − t2 q2 sin2θ, with
/// t2 = q2sign() = +1 for a weight-1 (polar, E1-type) doublet and −1 for
/// weight-2 (E2-type).  For weight 1 this is exactly Q·d̂_γ, because the bond
/// angles satisfy cos2θ_γ = cos θ_γ and sin2θ_γ = −sin θ_γ.
///
/// The minus sign in front of t2 is REQUIRED for consistency with the compiled
/// tensor table, which stores T^{E2}_{k,γ} = −sin2θ_γ × (structure) and is
/// contracted as q1 T^{E1} + t2 q2 T^{E2}.  Before 2026-09-02 this function had
/// `+ t2 q2 s2`, which mirrored the J2/J3 nematic channel about the x-bond line
/// relative to every other channel (silent whenever q2 = 0 or λ_J2 = λ_J3 = 0).
inline double bond_projection(double q1, double q2, double t2, double c2, double s2) {
    return q1 * c2 - t2 * q2 * s2;
}

// ---------------------------------------------------------------------------
// Six-spin ring exchange: one hexagon kernel.
//
// R_hex = (1/6) Σ_{6 cyclic shifts of (i,j,k,l,m,n)} [ 2(ij)(kl)(mn) − 6(ik)(jl)(mn)
//         + 3(il)(jk)(mn) + 3(ik)(jm)(ln) − (il)(jm)(kn) ],   (ab) = S_a·S_b,
// is a cubic polynomial in the 15 pair products d_ab, every monomial containing each
// of the six spins exactly once (so R_hex is linear in each spin). With
// G_ab = ∂R_hex/∂d_ab the gradient is ∂R_hex/∂S_a = Σ_{b≠a} G_ab S_b, so one pass
// (15 dots, the merged monomial table, 6×5 vector FMAs) gives R_hex and all six
// gradients — the energy, every site field and the lattice force R_7 = Σ R_hex use
// this single formula.
// ---------------------------------------------------------------------------
constexpr int pair_index(int a, int b) {
    // pairs (0,1),(0,2),…,(4,5) → 0..14
    return a < b ? a * (11 - a) / 2 + (b - a - 1) : b * (11 - b) / 2 + (a - b - 1);
}

struct RingMonomial { double c; int p, q, r; };

const std::vector<RingMonomial>& ring_monomials() {
    static const std::vector<RingMonomial> table = [] {
        std::map<std::array<int, 3>, double> acc;
        auto add = [&](double c, int a1, int b1, int a2, int b2, int a3, int b3) {
            std::array<int, 3> key = {pair_index(a1, b1), pair_index(a2, b2), pair_index(a3, b3)};
            std::sort(key.begin(), key.end());
            acc[key] += c / 6.0;
        };
        for (int s = 0; s < 6; ++s) {
            const int i = s, j = (s + 1) % 6, k = (s + 2) % 6, l = (s + 3) % 6, m = (s + 4) % 6, n = (s + 5) % 6;
            add( 2.0, i, j, k, l, m, n);
            add(-6.0, i, k, j, l, m, n);
            add( 3.0, i, l, j, k, m, n);
            add( 3.0, i, k, j, m, l, n);
            add(-1.0, i, l, j, m, k, n);
        }
        std::vector<RingMonomial> out;
        for (const auto& [key, c] : acc)
            if (c != 0.0) out.push_back({c, key[0], key[1], key[2]});
        return out;
    }();
    return table;
}

/// R_hex of six spins and (optionally) G_ab = ∂R_hex/∂d_ab.
inline double ring_hexagon(const Vec3 (&S)[6], double* G /* 15 or nullptr */) {
    double d[15];
    for (int a = 0; a < 6; ++a)
        for (int b = a + 1; b < 6; ++b) d[pair_index(a, b)] = S[a].dot(S[b]);
    double R = 0.0;
    if (G) std::fill(G, G + 15, 0.0);
    for (const auto& mo : ring_monomials()) {
        const double dp = d[mo.p], dq = d[mo.q], dr = d[mo.r];
        R += mo.c * dp * dq * dr;
        if (G) {
            G[mo.p] += mo.c * dq * dr;
            G[mo.q] += mo.c * dp * dr;
            G[mo.r] += mo.c * dp * dq;
        }
    }
    return R;
}

/// ∂R_hex/∂S_a from the pair derivatives.
inline Vec3 ring_gradient(const Vec3 (&S)[6], const double* G, int a) {
    Vec3 g = Vec3::Zero();
    for (int b = 0; b < 6; ++b)
        if (b != a) g += G[pair_index(a, b)] * S[b];
    return g;
}

/// Largest singular value of a 3x3 matrix (bound of |Sᵢᵀ T Sⱼ| for unit spins).
inline double spectral_norm(const Eigen::Matrix3d& T) {
    return Eigen::JacobiSVD<Eigen::Matrix3d>(T).singularValues()(0);
}

}  // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

PhononLattice::PhononLattice(const UnitCell& uc, size_t d1, size_t d2, size_t d3, float spin_l)
    : N_atoms(uc.N_atoms), dim1(d1), dim2(d2), dim3(d3), spin_length(spin_l), unit_cell(uc)
{
    if (uc.N_atoms != 2)
        throw std::invalid_argument("PhononLattice: the unit cell must be the two-site honeycomb basis (N_atoms = 2), got " +
                                    std::to_string(uc.N_atoms));
    if (d1 < 2 || d2 < 2 || d3 < 1)
        throw std::invalid_argument("PhononLattice: lattice dimensions must be at least 2 x 2 x 1 (got " +
                                    std::to_string(d1) + " x " + std::to_string(d2) + " x " + std::to_string(d3) + ")");
    if (!(spin_l > 0.0f))
        throw std::invalid_argument("PhononLattice: spin_length must be positive");

    lattice_size = N_atoms * dim1 * dim2 * dim3;
    state_size = spin_dim * lattice_size + PhononState::N_DOF;

    // Initialize arrays
    spins.assign(lattice_size, SpinVector::Zero(3));
    site_positions.resize(lattice_size);
    field.assign(lattice_size, SpinVector::Zero(3));

    nn_interaction.resize(lattice_size);
    nn_partners.resize(lattice_size);
    nn_bond_types.resize(lattice_size);
    j2_coupling.resize(lattice_size);
    j2_partners.resize(lattice_size);
    j3_coupling.resize(lattice_size);
    j3_partners.resize(lattice_size);
    site_hexagons.resize(lattice_size);

    // Copy sublattice frames from UnitCell (set_parameters() replaces them by the
    // crystal-from-storage map of the coupling frame).
    sublattice_frames.resize(N_atoms);
    for (size_t atom = 0; atom < N_atoms; ++atom) {
        sublattice_frames[atom] = uc.sublattice_frames[atom];
    }
    afm_sublattice_signs = uc.afm_sublattice_signs;
    if (afm_sublattice_signs.size() != N_atoms ||
        afm_sublattice_signs[0] == afm_sublattice_signs[1]) {
        // Honeycomb Néel staggering; a unit cell with equal signs would make the
        // "staggered" magnetisation identical to the uniform one.
        afm_sublattice_signs = {1.0, -1.0};
    }

    // Private engine for Monte Carlo moves and the Langevin noise, derived from the
    // process seed (config key `seed`); the shared Lehmer stream is not reseeded here.
    rng.seed(static_cast<std::mt19937::result_type>(derive_seed_from_master(0x50484F4EULL)));

    cout << "Initializing PhononLattice with dimensions: "
         << dim1 << " x " << dim2 << " x " << dim3 << endl;
    cout << "Atoms per unit cell: " << N_atoms << endl;
    cout << "Total spin sites: " << lattice_size << endl;

    // Build lattice site positions from UnitCell
    size_t site_idx = 0;
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                for (size_t atom = 0; atom < N_atoms; ++atom) {
                    Eigen::Vector3d pos = uc.lattice_pos[atom];
                    pos += double(i) * uc.lattice_vectors[0];
                    pos += double(j) * uc.lattice_vectors[1];
                    pos += double(k) * uc.lattice_vectors[2];
                    site_positions[site_idx] = pos;

                    // Copy field from unit cell
                    field[site_idx] = uc.field[atom].head<3>();

                    ++site_idx;
                }
            }
        }
    }

    // In-plane doublet frame of the layer: e_x = geometric A→B direction of the
    // x bond (bond_type 0 from atom 0), e_y = n × e_x with n = a1 × a2.  Bond-line
    // angles of the further-neighbour bonds are measured in this frame and grouped
    // into classes with form factors (cos2θ, sin2θ) for the E-doublet modulations.
    {
        Eigen::Vector3d nrm = uc.lattice_vectors[0].cross(uc.lattice_vectors[1]);
        nrm.normalize();
        bool found = false;
        auto range0 = uc.bilinear_interaction.equal_range(0);
        for (auto it = range0.first; it != range0.second; ++it) {
            const auto& bi = it->second;
            if (bi.bond_type == 0) {
                Eigen::Vector3d r = uc.lattice_pos[bi.partner] - uc.lattice_pos[0];
                for (int k = 0; k < 3; ++k) r += double(bi.offset[k]) * uc.lattice_vectors[k];
                ex_code = r.normalized();
                ey_code = nrm.cross(ex_code).normalized();
                found = true;
                break;
            }
        }
        if (!found) { ex_code = Eigen::Vector3d::UnitX(); ey_code = Eigen::Vector3d::UnitY(); }
    }
    j2_cls.assign(lattice_size, {});
    j3_cls.assign(lattice_size, {});
    nn_bond_vec.assign(lattice_size, {});
    nn_bond_cq.assign(lattice_size, {});
    j2_bond_vec.assign(lattice_size, {});
    n_j2_cls = 0;
    n_j3_cls = 0;
    auto bond_class = [&](const Eigen::Vector3d& r, bool isJ3) {
        const double th = std::atan2(r.dot(ey_code), r.dot(ex_code));
        const double c2 = std::cos(2.0 * th), s2 = std::sin(2.0 * th);
        auto& cs = isJ3 ? j3_cs : j2_cs;
        int& n = isJ3 ? n_j3_cls : n_j2_cls;
        for (int k = 0; k < n; ++k)
            if (std::abs(cs[k].first - c2) < 1e-9 && std::abs(cs[k].second - s2) < 1e-9) return k;
        if (n >= 3) throw std::invalid_argument("PhononLattice: more than 3 bond-line classes for J2/J3");
        cs[n] = {c2, s2};
        return n++;
    };
    // J2/J3 are isotropic Heisenberg couplings in this model: store the scalar.
    auto isotropic_coupling = [](const SpinMatrix& M, const char* what) {
        const double J = M.trace() / 3.0;
        if ((M - J * SpinMatrix::Identity(3, 3)).cwiseAbs().maxCoeff() > 1e-12 * std::max(1.0, std::abs(J)))
            throw std::invalid_argument(std::string("PhononLattice: ") + what +
                                        " couplings must be isotropic (J * identity)");
        return J;
    };

    // Build interaction topology from UnitCell
    // Iterate over all bilinear interactions and classify by bond_type
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                for (size_t atom = 0; atom < N_atoms; ++atom) {
                    size_t site = flatten_index(i, j, k, atom);

                    auto range = uc.bilinear_interaction.equal_range(atom);
                    for (auto it = range.first; it != range.second; ++it) {
                        const auto& bi = it->second;
                        if (bi.partner >= N_atoms)
                            throw std::invalid_argument("PhononLattice: bond partner index out of range");

                        // Compute partner site with periodic boundaries
                        size_t partner = flatten_index_periodic(
                            (int)i + bi.offset[0],
                            (int)j + bi.offset[1],
                            (int)k + bi.offset[2],
                            bi.partner);
                        if (partner == site)
                            throw std::invalid_argument(
                                "PhononLattice: a bond wraps onto its own site (lattice " + std::to_string(dim1) + "x" +
                                std::to_string(dim2) + "x" + std::to_string(dim3) +
                                " is too small for the bond offsets); local energies would be wrong");

                        Eigen::Vector3d rbond = uc.lattice_pos[bi.partner] - uc.lattice_pos[atom];
                        for (int kk = 0; kk < 3; ++kk) rbond += double(bi.offset[kk]) * uc.lattice_vectors[kk];

                        if (bi.bond_type >= 0) {
                            if (bi.bond_type > 2 || atom == bi.partner)
                                throw std::invalid_argument("PhononLattice: NN bonds must join the two sublattices "
                                                            "and carry bond_type 0, 1 or 2");
                            // NN interaction with bond_type info -> nn_interaction
                            nn_interaction[site].push_back(Eigen::Matrix3d(bi.interaction));
                            nn_partners[site].push_back(partner);
                            nn_bond_types[site].push_back(bi.bond_type);

                            // Reverse bond
                            nn_interaction[partner].push_back(Eigen::Matrix3d(bi.interaction.transpose()));
                            nn_partners[partner].push_back(site);
                            nn_bond_types[partner].push_back(bi.bond_type);

                            // Bond geometry for the spin–lattice dynamics: unit vector i→j (both ends) and
                            // the A→B direction of the bond in the doublet frame (for the E1–acoustic vertex).
                            const Eigen::Vector3d d = rbond.normalized();
                            const Eigen::Vector3d dAB = (atom == 0) ? d : Eigen::Vector3d(-d);
                            const Eigen::Vector2d cq(dAB.dot(ex_code), dAB.dot(ey_code));
                            nn_bond_vec[site].push_back(d);
                            nn_bond_vec[partner].push_back(-d);
                            nn_bond_cq[site].push_back(cq);
                            nn_bond_cq[partner].push_back(cq);
                        } else if (atom == bi.partner) {
                            // Same sublattice -> J2.  Each bond appears exactly ONCE in
                            // uc.bilinear_interaction, so both directions are added here.
                            const int cls = bond_class(rbond, false);
                            const double J = isotropic_coupling(bi.interaction, "J2");
                            j2_coupling[site].push_back(J);
                            j2_partners[site].push_back(partner);
                            j2_cls[site].push_back(cls);
                            j2_coupling[partner].push_back(J);
                            j2_partners[partner].push_back(site);
                            j2_cls[partner].push_back(cls);
                            j2_bond_vec[site].push_back(rbond.normalized());
                            j2_bond_vec[partner].push_back(-rbond.normalized());
                        } else {
                            // Different sublattice -> J3
                            const int cls = bond_class(rbond, true);
                            const double J = isotropic_coupling(bi.interaction, "J3");
                            j3_coupling[site].push_back(J);
                            j3_partners[site].push_back(partner);
                            j3_cls[site].push_back(cls);
                            j3_coupling[partner].push_back(J);
                            j3_partners[partner].push_back(site);
                            j3_cls[partner].push_back(cls);
                        }
                    }
                }
            }
        }
    }
    nn_clean_ = nn_interaction;
    nn_delta_.assign(lattice_size, {});
    nn_scale_.assign(lattice_size, {});
    for (size_t s = 0; s < lattice_size; ++s) {
        nn_delta_[s].assign(nn_interaction[s].size(), Eigen::Matrix3d::Zero());
        nn_scale_[s].assign(nn_interaction[s].size(), 1.0);
    }
    update_exchange_flags();
    validate_topology();
    build_hexagons();

    cout << "Total ODE state size: " << state_size << endl;

    // Initialize random spins
    init_random();

    cout << "PhononLattice initialization complete!" << endl;
}

void PhononLattice::validate_topology() const {
    for (size_t i = 0; i < lattice_size; ++i) {
        if (nn_partners[i].size() != 3)
            throw std::invalid_argument("PhononLattice: site " + std::to_string(i) + " has " +
                                        std::to_string(nn_partners[i].size()) +
                                        " nearest neighbours (honeycomb requires 3)");
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            if ((i % N_atoms) == (j % N_atoms))
                throw std::invalid_argument("PhononLattice: NN bond within one sublattice");
        }
    }
    // Small lattices alias distinct bonds onto the same partner (e.g. J3 on L = 2):
    // the energies stay consistent but the model is not the infinite-lattice one.
    if (dim1 < 3 || dim2 < 3)
        cout << "  NOTE: " << dim1 << " x " << dim2
             << " lattice aliases distinct J3 bonds onto the same pair (use L >= 3 for physics runs)" << endl;
}

void PhononLattice::build_hexagons() {
    // Hexagonal plaquettes for the ring exchange. For each unit cell (i, j, k) one hexagon,
    // walked around the ring through NN bonds (bond offsets of build_phonon_honeycomb:
    // x: A(i,j)→B(i,j−1), y: A(i,j)→B(i+1,j−1), z: A(i,j)→B(i,j)):
    //   0: A(i,j)  −z−  1: B(i,j)  −x−  2: A(i,j+1)  −y−  3: B(i+1,j)
    //   −z−  4: A(i+1,j)  −x−  5: B(i+1,j−1)  −y−  back to 0.
    // The construction is checked: consecutive sites must be NN partners and the six
    // sites distinct (otherwise the lattice is too small or the unit cell not honeycomb).
    hexagons.clear();
    for (auto& v : site_hexagons) v.clear();
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                std::array<size_t, 6> hex;
                hex[0] = flatten_index(i, j, k, 0);
                hex[1] = flatten_index(i, j, k, 1);
                hex[2] = flatten_index_periodic(int(i), int(j) + 1, int(k), 0);
                hex[3] = flatten_index_periodic(int(i) + 1, int(j), int(k), 1);
                hex[4] = flatten_index_periodic(int(i) + 1, int(j), int(k), 0);
                hex[5] = flatten_index_periodic(int(i) + 1, int(j) - 1, int(k), 1);
                for (int p = 0; p < 6; ++p) {
                    for (int q = p + 1; q < 6; ++q)
                        if (hex[p] == hex[q])
                            throw std::invalid_argument("PhononLattice: hexagon with a repeated site (lattice too small)");
                    const size_t a = hex[p], b = hex[(p + 1) % 6];
                    if (std::find(nn_partners[a].begin(), nn_partners[a].end(), b) == nn_partners[a].end())
                        throw std::invalid_argument("PhononLattice: hexagon edge is not a NN bond — the unit cell "
                                                    "does not have the honeycomb bond offsets");
                }
                const size_t hex_idx = hexagons.size();
                hexagons.push_back(hex);
                for (size_t pos = 0; pos < 6; ++pos) site_hexagons[hex[pos]].push_back({hex_idx, pos});
            }
        }
    }
    plaquette_j7_offsets.assign(hexagons.size(), 0.0);
}

size_t PhononLattice::nn_index(size_t site, size_t partner, const string& context) const {
    for (size_t n = 0; n < nn_partners[site].size(); ++n)
        if (nn_partners[site][n] == partner) return n;
    throw std::runtime_error(context + ": " + std::to_string(site) + " " + std::to_string(partner) +
                             " is not a nearest-neighbour bond");
}

void PhononLattice::rebuild_nn_exchange() {
    for (size_t i = 0; i < lattice_size; ++i)
        for (size_t n = 0; n < nn_interaction[i].size(); ++n)
            nn_interaction[i][n] = nn_scale_[i][n] * nn_clean_[i][n] + nn_delta_[i][n];
}

void PhononLattice::update_exchange_flags() {
    has_j2_ = false;
    has_j3_ = false;
    for (size_t i = 0; i < lattice_size; ++i) {
        for (double J : j2_coupling[i]) if (J != 0.0) has_j2_ = true;
        for (double J : j3_coupling[i]) if (J != 0.0) has_j3_ = true;
    }
}

// ============================================================
// PARAMETER SETTING
// ============================================================

void PhononLattice::set_parameters(const SpinPhononCouplingParams& sp_params,
                                   const PhononParams& ph_params,
                                   const DriveParams& dr_params) {
    spin_phonon_params = sp_params;
    phonon_params = ph_params;
    drive_params = dr_params;

    // NN exchange from (J, K, Γ, Γ') in the storage frame; the UnitCell only fixed the
    // topology. Report a mismatch with the unit cell's matrices (a different frame or
    // parameter set in the builder) instead of silently using one of the two.
    double max_dev = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        const bool isA = (i % N_atoms == 0);
        for (size_t n = 0; n < nn_clean_[i].size(); ++n) {
            const Eigen::Matrix3d M = sp_params.nn_exchange(nn_bond_types[i][n]);
            const Eigen::Matrix3d Mi = isA ? M : Eigen::Matrix3d(M.transpose());
            max_dev = std::max(max_dev, (Mi - nn_clean_[i][n]).cwiseAbs().maxCoeff());
            nn_clean_[i][n] = Mi;
        }
        for (size_t n = 0; n < j2_coupling[i].size(); ++n)
            j2_coupling[i][n] = (i % N_atoms == 0) ? sp_params.J2_A : sp_params.J2_B;
        for (size_t n = 0; n < j3_coupling[i].size(); ++n)
            j3_coupling[i][n] = sp_params.J3;
    }
    if (max_dev > 1e-9)
        cout << "  NOTE: NN exchange rebuilt from the coupling parameters differs from the unit cell's by "
             << max_dev << " (the coupling parameters are used)" << endl;
    rebuild_nn_exchange();
    update_exchange_flags();

    // "Global" outputs are crystal components whatever the storage frame.
    const Eigen::Matrix3d C = classical_spin::kitaev::crystal_from_storage(sp_params.frame);
    for (size_t atom = 0; atom < N_atoms; ++atom) sublattice_frames[atom] = C;

    cout << "Set PhononLattice parameters (spin storage frame: "
         << classical_spin::kitaev::frame_name(sp_params.frame) << "):" << endl;
    cout << "  J=" << sp_params.J << ", K=" << sp_params.K
         << ", Γ=" << sp_params.Gamma << ", Γ'=" << sp_params.Gammap << endl;
    cout << "  J2_A=" << sp_params.J2_A << ", J2_B=" << sp_params.J2_B << endl;
    cout << "  J3=" << sp_params.J3 << ", J7=" << sp_params.J7 << endl;
    cout << "  Spin-phonon (primary E1 mode):" << endl;
    cout << "    E1 λ0(J,K,Γ,Γ')=(" << sp_params.lambda_E1_J_0 << ", "
         << sp_params.lambda_E1_K_0 << ", " << sp_params.lambda_E1_Gamma_0
         << ", " << sp_params.lambda_E1_Gammap_0 << ")" << endl;
    cout << "    E1 λ2(J,K,Γ,Γ')=(" << sp_params.lambda_E1_J_2 << ", "
         << sp_params.lambda_E1_K_2 << ", " << sp_params.lambda_E1_Gamma_2
         << ", " << sp_params.lambda_E1_Gammap_2 << ")" << endl;
    cout << "    E1 λ(J7,0)=" << sp_params.lambda_E1_J7_0
         << " so J7_eff=J7+λ(J7,0)|ε|²" << endl;
    cout << "    E1 λ1(J,K,Γ,Γ') [D3-allowed LINEAR striction ε_x cos2θ − ε_y sin2θ]=("
         << sp_params.lambda_E1_J_1 << ", " << sp_params.lambda_E1_K_1 << ", "
         << sp_params.lambda_E1_Gamma_1 << ", " << sp_params.lambda_E1_Gammap_1 << ")" << endl;
    cout << "    polarization/bond angles are measured from the x-bond LINE "
            "(geometric x-bond is at 30° from the a1 lattice vector)" << endl;
    cout << "  E1 mode: ω_E1=" << ph_params.omega_E1 << ", γ_E1=" << ph_params.gamma_E1
         << ", λ_E1(quartic)=" << ph_params.lambda_E1_quartic
         << ", Z*=" << ph_params.Z_star
         << ", back-action per site=" << (ph_params.per_site_backaction ? "yes" : "NO (legacy, size-dependent)")
         << endl;
    cout << "  Drive: E0_1=" << dr_params.E0_1 << ", ω_1=" << dr_params.omega_1
         << ", E0_2=" << dr_params.E0_2 << ", ω_2=" << dr_params.omega_2 << endl;

    // Mirror the legacy parameters into modes[0]; the extra modes of set_modes() are kept.
    rebuild_primary_mode();
    report_stability();
}

void PhononLattice::report_stability() const {
    // Rigorous a-priori stability bound for every dynamic mode.  For unit-length spins
    // |Sᵢᵀ T Sⱼ| ≤ ‖T‖ (largest singular value) and |R_hex| ≤ 15; per site there are 3/2
    // NN bonds, 3/2 J2 bonds per sublattice class (3 per A or B site, half the sites),
    // 3/2 J3 bonds and 1/2 hexagon.  The quadratic couplings of an E doublet have Hessians
    // of norm ≤ 2√2 |b| ‖T‖ ((Q⊗Q)_E part) and 2|a| ‖T‖ (|Q|² part), so
    //   |∂²H_ME/∂q²| / N ≤ (3/2) Σ_k [2√2|b_k| max‖T^E_k‖ + 2|a_k| max‖T^A1_k‖] S²
    //                      + 15 |λ_J7²| S⁶ + 3 (|λ_J2A²| + |λ_J2B²| + |λ_J3²|) S²
    // and ω_eff² = ω² + (1/N)∂²H/∂q² stays positive for EVERY spin configuration when ω²
    // exceeds the bound (cubic anharmonic transfers are not included).  In the legacy
    // (extensive) convention the bound is multiplied by N.
    const double S2 = double(spin_length) * double(spin_length);
    static const auto norms = [] {
        std::array<double, 9> nE{};
        std::array<double, 5> nA{};
        for (int k = 0; k < N_E; ++k)
            for (int g = 0; g < 3; ++g)
                nE[k] = std::max({nE[k], spectral_norm(TE1(k, g)), spectral_norm(TE2(k, g))});
        for (int k = 0; k < N_A1; ++k)
            for (int g = 0; g < 3; ++g) nA[k] = std::max(nA[k], spectral_norm(TA1(k, g)));
        return std::make_pair(nE, nA);
    }();
    const double ext = phonon_params.per_site_backaction ? 1.0 : double(lattice_size);
    for (size_t m = 0; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        if (md.frozen || md.irrep != LatticeMode::Irrep::E) continue;
        double bil = 0.0;
        for (int k = 0; k < N_E; ++k) bil += 2.0 * std::sqrt(2.0) * std::abs(md.bE_sq[k]) * norms.first[k];
        for (int k = 0; k < N_A1; ++k) bil += 2.0 * std::abs(md.aA1_sq[k]) * norms.second[k];
        bil *= 1.5 * S2;
        const double ring = 15.0 * std::abs(md.lamJ7_sq) * S2 * S2 * S2;
        const double further = 3.0 * (std::abs(md.lamJ2A_sq) + std::abs(md.lamJ2B_sq) + std::abs(md.lamJ3_sq)) * S2;
        const double bound = (bil + ring + further) * ext;
        if (bound == 0.0) continue;
        const double w2 = md.omega * md.omega;
        cout << "  mode " << m << " stability: ω² = " << w2 << ", worst-case |δω²| from spin back-action ≤ "
             << bound << " (bilinear " << bil << " + ring " << ring << " + J2/J3 " << further << ")";
        if (bound >= w2) {
            cout << "\n  WARNING: ω² does not exceed the back-action bound; the mode can go soft"
                    " (ω_eff² ≤ 0) for some spin configurations — the coupled dynamics is not"
                    " guaranteed stable." << endl;
        } else {
            cout << " → ω_eff/ω ∈ [" << std::sqrt(1.0 - bound / w2) << ", "
                 << std::sqrt(1.0 + bound / w2) << "], stable for all configurations." << endl;
        }
    }
}


// ============================================================
// MULTI-MODE LATTICE SECTOR
// ============================================================

void PhononLattice::rebuild_primary_mode() {
    // Mode 0 mirrors the legacy single-E1 parameters (coordinates live in `phonons`).
    // Only the legacy-mirrored fields are overwritten; the additional couplings of the
    // primary mode (cE[4..8], bE_sq[4..8], aA1_sq[4], J2/J3 nematics) set through
    // set_modes()/build_lattice_modes() are preserved.  Call this after changing
    // spin_phonon_params or phonon_params directly (set_parameters does it).
    if (modes.empty()) modes.resize(1);
    LatticeMode& p = modes[0];
    p.irrep = LatticeMode::Irrep::E; p.weight = 1; p.name = "E1 primary";
    p.omega = phonon_params.omega_E1; p.gamma = phonon_params.gamma_E1;
    p.quartic = phonon_params.lambda_E1_quartic; p.Zstar = phonon_params.Z_star; p.frozen = false;
    const SpinPhononCouplingParams& s = spin_phonon_params;
    p.cE[0] = s.lambda_E1_J_1; p.cE[1] = s.lambda_E1_K_1; p.cE[2] = s.lambda_E1_Gamma_1; p.cE[3] = s.lambda_E1_Gammap_1;
    p.bE_sq[0] = s.lambda_E1_J_2; p.bE_sq[1] = s.lambda_E1_K_2; p.bE_sq[2] = s.lambda_E1_Gamma_2; p.bE_sq[3] = s.lambda_E1_Gammap_2;
    p.aA1_sq[0] = s.lambda_E1_J_0; p.aA1_sq[1] = s.lambda_E1_K_0; p.aA1_sq[2] = s.lambda_E1_Gamma_0; p.aA1_sq[3] = s.lambda_E1_Gammap_0;
    p.lamJ7_sq = s.lambda_E1_J7_0;
    update_modulation_flags();
    recompute_state_size();
}

void PhononLattice::update_modulation_flags() {
    has_further_modulation = false;
    for (const auto& md : modes)
        if (md.lamJ2A != 0.0 || md.lamJ2B != 0.0 || md.lamJ3 != 0.0 ||
            md.lamJ2A_sq != 0.0 || md.lamJ2B_sq != 0.0 || md.lamJ3_sq != 0.0)
            has_further_modulation = true;
    invalidate_couplings();
}

void PhononLattice::set_modes(const std::vector<LatticeMode>& extra,
                              const std::vector<AnharmonicTerm>& anh) {
    for (const auto& md : extra) {
        if (!(md.omega > 0.0) || md.gamma < 0.0 || !std::isfinite(md.omega) || !std::isfinite(md.gamma))
            throw std::invalid_argument("set_modes: mode '" + md.name + "' needs omega > 0 and gamma >= 0");
        if (md.weight != 1 && md.weight != 2)
            throw std::invalid_argument("set_modes: mode '" + md.name + "' weight must be 1 or 2");
    }
    rebuild_primary_mode();
    modes.resize(1);
    for (const auto& md : extra) modes.push_back(md);
    anharmonic.clear();
    for (const auto& t : anh) {
        if (t.target < 0 || size_t(t.target) >= modes.size() || t.lam < 0 || size_t(t.lam) >= modes.size() ||
            t.lamp < 0 || size_t(t.lamp) >= modes.size())
            throw std::invalid_argument("set_modes: anharmonic term references an undefined mode");
        if (modes[t.target].irrep == LatticeMode::Irrep::A2)
            throw std::invalid_argument("set_modes: anharmonic target must be A1 or E");
        anharmonic.push_back(t);
    }
    update_modulation_flags();
    recompute_state_size();
    cout << "Lattice sector: " << modes.size() << " mode(s), " << anharmonic.size()
         << " anharmonic term(s), " << phonon_dof() << " lattice DOF" << endl;
    for (size_t m = 0; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        const char* irr = md.irrep == LatticeMode::Irrep::E ? (md.weight == 1 ? "E (polar/E1)" : "E (E2-type)")
                        : md.irrep == LatticeMode::Irrep::A1 ? "A1" : "A2";
        cout << "  mode " << m << " [" << irr << "] " << md.name << ": ω=" << md.omega << " γ=" << md.gamma
             << " Z*=" << md.Zstar << (md.frozen ? " FROZEN" : "") << endl;
        auto nz = [](const auto& arr) { int n = 0; for (double v : arr) if (v != 0.0) ++n; return n; };
        cout << "    linear: cE " << nz(md.cE) << "/9, aA1 " << nz(md.aA1) << "/5, dA2 " << nz(md.dA2)
             << "/4, λJ7=" << md.lamJ7 << ", λJ2A/B=" << md.lamJ2A << "/" << md.lamJ2B << ", λJ3=" << md.lamJ3 << endl;
        cout << "    quadratic: bE " << nz(md.bE_sq) << "/9, aA1 " << nz(md.aA1_sq) << "/5, λJ7|Q|²=" << md.lamJ7_sq
             << ", λJ2A/B|Q|²=" << md.lamJ2A_sq << "/" << md.lamJ2B_sq << ", λJ3|Q|²=" << md.lamJ3_sq << endl;
        if (j2_partners[0].empty() &&
            (md.lamJ2A != 0.0 || md.lamJ2B != 0.0 || md.lamJ2A_sq != 0.0 || md.lamJ2B_sq != 0.0))
            cout << "    WARNING: J2 modulation on a lattice without second-neighbour bonds" << endl;
    }
    for (const auto& t : anharmonic)
        cout << "  anharmonic: −N g Q_" << t.target << " (Q_" << t.lam << " ⊗ Q_" << t.lamp << "), g=" << t.g << endl;
    report_stability();
}

void PhononLattice::reset_lattice_sector() {
    phonons = PhononState();
    for (size_t m = 1; m < modes.size(); ++m) {
        LatticeMode& md = modes[m];
        md.V1 = md.V2 = 0.0;
        if (!md.frozen) md.Q1 = md.Q2 = 0.0;
    }
    if (sld_enabled) {
        for (auto& u : u_site) u.setZero();
        for (auto& p : p_site) p.setZero();
    }
}

// ============================================================
// KERNELS
// ============================================================

template <class SpinOf, class DispOf>
Eigen::Vector3d PhononLattice::field_without_ring(size_t i, const SpinOf& S, const DispOf& U,
                                                  const Couplings& cc) const {
    // −∂E/∂S_i for every term except the ring exchange: Zeeman, NN exchange with the
    // magnetoelastic increment (δM for the A end, δMᵀ for the B end of a bond), the
    // exchange striction g δr_ij M_ij, and the (modulated) J2/J3 couplings.
    Vec3 H = CMap3(field[i].data());
    const int sub = int(i % N_atoms);
    const Eigen::Matrix3d* dM = (sub == 0) ? cc.dM : cc.dMT;
    const bool striction = sld_enabled && sld_g != 0.0;
    const auto& partners = nn_partners[i];
    for (size_t n = 0; n < partners.size(); ++n) {
        const size_t j = partners[n];
        const Vec3 Sj = S(j);
        const Vec3 MSj = nn_interaction[i][n] * Sj;
        H -= MSj + dM[nn_bond_types[i][n]] * Sj;
        if (striction) {
            const double dr = (U(j) - U(i)).dot(nn_bond_vec[i][n]);
            H -= sld_g * dr * MSj;
        }
    }
    if (has_j2_ || has_further_modulation) {
        const auto& p2 = j2_partners[i];
        for (size_t n = 0; n < p2.size(); ++n)
            H -= (j2_coupling[i][n] + cc.dJ2[sub][j2_cls[i][n]]) * S(p2[n]);
    }
    if (has_j3_ || has_further_modulation) {
        const auto& p3 = j3_partners[i];
        for (size_t n = 0; n < p3.size(); ++n)
            H -= (j3_coupling[i][n] + cc.dJ3[j3_cls[i][n]]) * S(p3[n]);
    }
    return H;
}

template <class SpinOf>
Eigen::Vector3d PhononLattice::ring_field_site(size_t i, const SpinOf& S, double J7eff) const {
    // −∂H_7/∂S_i = −Σ_{hexagons ∋ i} J7_hex ∂R_hex/∂S_i
    Vec3 H = Vec3::Zero();
    for (const auto& [hex_idx, pos] : site_hexagons[i]) {
        const double J7 = effective_J7_for_hexagon(hex_idx, J7eff);
        if (J7 == 0.0) continue;
        const auto& hex = hexagons[hex_idx];
        Vec3 s[6];
        for (int p = 0; p < 6; ++p) s[p] = S(hex[p]);
        double G[15];
        ring_hexagon(s, G);
        H -= J7 * ring_gradient(s, G, int(pos));
    }
    return H;
}

template <class SpinOf>
double PhononLattice::ring_operator(const SpinOf& S, double J7eff, double* weighted_energy) const {
    // R_7 = Σ_hex R_hex and, optionally, H_7 = Σ_hex (J7_eff + δJ7_hex) R_hex.
    double R7 = 0.0, E = 0.0;
    for (size_t h = 0; h < hexagons.size(); ++h) {
        Vec3 s[6];
        for (int p = 0; p < 6; ++p) s[p] = S(hexagons[h][p]);
        const double R = ring_hexagon(s, nullptr);
        R7 += R;
        E += effective_J7_for_hexagon(h, J7eff) * R;
    }
    if (weighted_energy) *weighted_energy = E;
    return R7;
}

template <class SpinOf>
void PhononLattice::correlations_of(const SpinOf& S, Eigen::Matrix3d C[3],
                                    double Cj2[2][3], double Cj3[3]) const {
    // C_γ = Σ_{γ bonds} S_A S_Bᵀ, Cj2[sub][class] / Cj3[class] = Σ_bonds S_i·S_j.
    // Per-thread partial sums combined in thread order (deterministic for a fixed
    // thread count).
    constexpr int W = 27 + 6 + 3;
    // Light O(N) work: threads only pay off for large lattices.
    const bool par = lattice_size >= 4 * parallel_min_sites;
    int nt = 1;
#ifdef _OPENMP
    if (par) nt = omp_get_max_threads();
#endif
    corr_scratch_.assign(size_t(nt) * W, 0.0);
#ifdef _OPENMP
    #pragma omp parallel if(par)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double acc[W] = {};      // thread-local accumulator (no false sharing); stored once below
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (long il = 0; il < long(lattice_size); ++il) {
            const size_t i = size_t(il);
            const Vec3 Si = S(i);
            const int sub = int(i % N_atoms);
            if (sub == 0) {
                for (size_t n = 0; n < nn_partners[i].size(); ++n) {
                    const Vec3 Sj = S(nn_partners[i][n]);
                    double* Cg = acc + 9 * nn_bond_types[i][n];
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) Cg[3 * a + b] += Si(a) * Sj(b);
                }
            }
            for (size_t n = 0; n < j2_partners[i].size(); ++n) {
                const size_t j = j2_partners[i][n];
                if (j > i) acc[27 + 3 * sub + j2_cls[i][n]] += Si.dot(S(j));
            }
            for (size_t n = 0; n < j3_partners[i].size(); ++n) {
                const size_t j = j3_partners[i][n];
                if (j > i) acc[33 + j3_cls[i][n]] += Si.dot(S(j));
            }
        }
        std::copy(acc, acc + W, corr_scratch_.begin() + std::ptrdiff_t(tid) * W);
    }
    for (int g = 0; g < 3; ++g) C[g].setZero();
    for (int s = 0; s < 2; ++s) for (int k = 0; k < 3; ++k) Cj2[s][k] = 0.0;
    for (int k = 0; k < 3; ++k) Cj3[k] = 0.0;
    for (int t = 0; t < nt; ++t) {
        const double* acc = &corr_scratch_[size_t(t) * W];
        for (int g = 0; g < 3; ++g)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) C[g](a, b) += acc[9 * g + 3 * a + b];
        for (int s = 0; s < 2; ++s) for (int k = 0; k < 3; ++k) Cj2[s][k] += acc[27 + 3 * s + k];
        for (int k = 0; k < 3; ++k) Cj3[k] += acc[33 + k];
    }
}

// ============================================================
// ENERGY CALCULATIONS
// ============================================================

double PhononLattice::spin_energy() const {
    // Zeeman + bare exchange (NN incl. quenched disorder, J2, J3) + ring exchange
    // (with J7_eff(Q) and the plaquette offsets).
    double E = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        const CMap3 Si(spins[i].data());
        E -= Si.dot(CMap3(field[i].data()));
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            if (j > i) E += Si.dot(nn_interaction[i][n] * CMap3(spins[j].data()));
        }
        for (size_t n = 0; n < j2_partners[i].size(); ++n) {
            const size_t j = j2_partners[i][n];
            if (j > i) E += j2_coupling[i][n] * Si.dot(CMap3(spins[j].data()));
        }
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            const size_t j = j3_partners[i][n];
            if (j > i) E += j3_coupling[i][n] * Si.dot(CMap3(spins[j].data()));
        }
    }
    return E + ring_exchange_energy();
}

double PhononLattice::ring_exchange_energy() const {
    const Couplings& cc = couplings();
    if (!cc.ring_active) return 0.0;
    double E = 0.0;
    ring_operator([this](size_t j) { return CMap3(spins[j].data()); }, cc.J7eff, &E);
    return E;
}

double PhononLattice::ring_exchange_normalized() const {
    return ring_operator([this](size_t j) { return CMap3(spins[j].data()); }, 0.0, nullptr);
}

double PhononLattice::total_energy(double t) const {
    const double s = get_e1_coupling_scale(t);
    if (s == 1.0) return total_energy();
    // H(t) = H_bare + s(t) H_ME: rescale the magnetoelastic part of the s = 1 energy.
    const Couplings& c1 = couplings();
    Eigen::Matrix3d C[3];
    double Cj2[2][3], Cj3[3];
    bond_correlations(C);
    further_correlations(Cj2, Cj3);
    double E_me = 0.0;
    for (int g = 0; g < 3; ++g) E_me += (c1.dM[g].cwiseProduct(C[g])).sum();
    for (int sb = 0; sb < 2; ++sb) for (int k = 0; k < 3; ++k) E_me += c1.dJ2[sb][k] * Cj2[sb][k];
    for (int k = 0; k < 3; ++k) E_me += c1.dJ3[k] * Cj3[k];
    E_me += (c1.J7eff - spin_phonon_params.J7) * ring_exchange_normalized();
    return total_energy() + (s - 1.0) * E_me;
}

// ============================================================
// LATTICE COORDINATES, BOND INCREMENTS AND MAGNETOELASTIC ENERGIES
// ============================================================

size_t PhononLattice::mode_dof() const {
    size_t n = PhononState::N_DOF;
    for (size_t m = 1; m < modes.size(); ++m) n += modes[m].ndof();
    return n;
}

size_t PhononLattice::phonon_dof() const {
    return mode_dof() + (sld_enabled ? 6 * lattice_size : 0);
}

void PhononLattice::enable_sld(bool on) {
    if (on && sld_k2 != 0.0 && j2_partners[0].empty())
        throw std::invalid_argument("enable_sld: sld_k2 != 0 but the lattice has no second-neighbour bonds "
                                    "(only the NN springs would remain: z = 3 < 2d is mechanically floppy)");
    if (on && !(sld_mass > 0.0))
        throw std::invalid_argument("enable_sld: sld_mass must be positive");
    sld_enabled = on;
    sld_relax_pending_ = on && sld_relax > 0;
    u_site.assign(lattice_size, Eigen::Vector3d::Zero());
    p_site.assign(lattice_size, Eigen::Vector3d::Zero());
    recompute_state_size();
    if (on) {
        cout << "Spin–lattice dynamics ENABLED: " << 6 * lattice_size << " site DOF (in-plane u, p);"
             << " m = " << sld_mass << ", k = " << sld_k << ", k2 = " << sld_k2
             << ", striction g = " << sld_g << " /a, v3 = " << sld_v3
             << ", gamma_l = " << sld_gamma << (sld_quantum ? " (Bose-coloured noise)" : " (white noise)")
             << ", T_l = " << (sld_T >= 0.0 ? sld_T : langevin_temperature) << endl;
        cout << "  ODE state size now " << state_size << endl;
    }
}

void PhononLattice::recompute_state_size() {
    state_size = spin_dim * lattice_size + phonon_dof();
}

double PhononLattice::relax_sld_static(int max_iter, double tol) {
    if (!sld_enabled) return 0.0;
    for (auto& p : p_site) p.setZero();
    ODEState x = spins_to_state(), dxdt(state_size);
    const size_t off = sld_offset(), poff = off + 3 * lattice_size;
    // diagonal stiffness seen by one site: 3 NN springs (k, each contributing k d d^T ~ k/2 per in-plane
    // direction on average → 3k/2) plus 6 second-neighbour springs (3 k2); damped Jacobi step
    const double kdiag = 1.5 * sld_k + 3.0 * sld_k2;
    if (!(kdiag > 0.0)) throw std::invalid_argument("relax_sld_static: needs sld_k or sld_k2 > 0");
    double fmax = 0.0;
    int it = 0;
    for (; it < max_iter; ++it) {
        ode_system(x, dxdt, 0.0);
        fmax = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            for (int d = 0; d < 2; ++d) {
                const double F = dxdt[poff + 3 * i + d];        // p = 0 → pure force
                fmax = std::max(fmax, std::abs(F));
                x[off + 3 * i + d] += 0.7 * F / kdiag;
            }
        }
        if (fmax < tol) break;
    }
    state_to_spins(x);
    for (auto& p : p_site) p.setZero();
    sld_relax_pending_ = false;
    cout << "  SLD static relaxation: " << it << " iterations, max|F| = " << fmax
         << ", spring energy " << sld_spring_energy() / double(lattice_size) * 1e3
         << " ueV/site, striction energy " << sld_striction_energy() / double(lattice_size) * 1e3 << " ueV/site" << endl;
    return fmax;
}

void PhononLattice::pack_lattice(double* arr) const {
    phonons.to_array(arr);
    size_t p = PhononState::N_DOF;
    for (size_t m = 1; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        if (md.frozen) continue;
        arr[p++] = md.Q1;
        if (md.ncoord() == 2) arr[p++] = md.Q2;
        arr[p++] = md.V1;
        if (md.ncoord() == 2) arr[p++] = md.V2;
    }
    if (sld_enabled) {
        for (size_t i = 0; i < lattice_size; ++i) for (int d = 0; d < 3; ++d) arr[p++] = u_site[i](d);
        for (size_t i = 0; i < lattice_size; ++i) for (int d = 0; d < 3; ++d) arr[p++] = p_site[i](d);
    }
}

void PhononLattice::unpack_lattice(const double* arr) {
    phonons.from_array(arr);
    size_t p = PhononState::N_DOF;
    for (size_t m = 1; m < modes.size(); ++m) {
        LatticeMode& md = modes[m];
        if (md.frozen) continue;
        md.Q1 = arr[p++];
        if (md.ncoord() == 2) md.Q2 = arr[p++];
        md.V1 = arr[p++];
        if (md.ncoord() == 2) md.V2 = arr[p++];
    }
    if (sld_enabled) {
        for (size_t i = 0; i < lattice_size; ++i) for (int d = 0; d < 3; ++d) u_site[i](d) = arr[p++];
        for (size_t i = 0; i < lattice_size; ++i) for (int d = 0; d < 3; ++d) p_site[i](d) = arr[p++];
    }
}

// ---- spin–lattice dynamics energies (u, p from u_site/p_site) ----
double PhononLattice::sld_kinetic_energy() const {
    if (!sld_enabled) return 0.0;
    double E = 0.0;
    for (const auto& p : p_site) E += p.squaredNorm();
    return 0.5 * E / sld_mass;
}

double PhononLattice::sld_spring_energy() const {
    if (!sld_enabled) return 0.0;
    double E = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            if (j <= i) continue;
            const double dr = (u_site[j] - u_site[i]).dot(nn_bond_vec[i][n]);
            E += 0.5 * sld_k * dr * dr;
        }
        for (size_t n = 0; n < j2_partners[i].size(); ++n) {
            const size_t j = j2_partners[i][n];
            if (j <= i) continue;
            const double dr = (u_site[j] - u_site[i]).dot(j2_bond_vec[i][n]);
            E += 0.5 * sld_k2 * dr * dr;
        }
    }
    return E;
}

double PhononLattice::sld_striction_energy() const {
    // Σ_bonds g δr_ij S_i·M_ij S_j  +  v3 Σ_bonds (Q·d̂_γ) δr_ij²
    if (!sld_enabled) return 0.0;
    double E = 0.0;
    const double Qx = phonons.Q_x_E1, Qy = phonons.Q_y_E1;
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            if (j <= i) continue;
            const double dr = (u_site[j] - u_site[i]).dot(nn_bond_vec[i][n]);
            E += sld_g * dr * CMap3(spins[i].data()).dot(nn_interaction[i][n] * CMap3(spins[j].data()));
            if (sld_v3 != 0.0) E += sld_v3 * (Qx * nn_bond_cq[i][n](0) + Qy * nn_bond_cq[i][n](1)) * dr * dr;
        }
    }
    return E;
}

double PhononLattice::sld_energy() const {
    if (!sld_enabled) return 0.0;
    return sld_kinetic_energy() + sld_spring_energy() + sld_striction_energy();
}

PhononLattice::Coords PhononLattice::coords_current() const {
    Coords c;
    c.q1.assign(modes.size(), 0.0);
    c.q2.assign(modes.size(), 0.0);
    for (size_t m = 0; m < modes.size(); ++m) {
        if (m == 0) { c.q1[0] = phonons.Q_x_E1; c.q2[0] = phonons.Q_y_E1; }
        else        { c.q1[m] = modes[m].Q1;     c.q2[m] = modes[m].Q2; }
    }
    return c;
}

PhononLattice::Coords PhononLattice::coords_from_state(const double* arr) const {
    Coords c;
    c.q1.assign(modes.size(), 0.0);
    c.q2.assign(modes.size(), 0.0);
    c.q1[0] = arr[0];
    c.q2[0] = arr[1];
    size_t p = PhononState::N_DOF;
    for (size_t m = 1; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        if (md.frozen) { c.q1[m] = md.Q1; c.q2[m] = md.Q2; continue; }
        c.q1[m] = arr[p++];
        c.q2[m] = (md.ncoord() == 2) ? arr[p++] : 0.0;
        p += md.ncoord();   // skip the velocities
    }
    return c;
}

void PhononLattice::bond_increments_local(const Coords& c, double scale, Eigen::Matrix3d dM[3]) const {
    for (int g = 0; g < 3; ++g) dM[g].setZero();
    for (size_t m = 0; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        const double q1 = c.q1[m], q2 = c.q2[m];
        switch (md.irrep) {
        case LatticeMode::Irrep::E: {
            const double t2 = md.q2sign();
            const double P1 = q1 * q1 - q2 * q2, P2 = 2.0 * q1 * q2, sP = -t2;
            const double Qsq = q1 * q1 + q2 * q2;
            for (int g = 0; g < 3; ++g) {
                for (int k = 0; k < N_E; ++k) {
                    if (md.cE[k] != 0.0)    dM[g] += md.cE[k]    * (q1 * TE1(k, g) + t2 * q2 * TE2(k, g));
                    if (md.bE_sq[k] != 0.0) dM[g] += md.bE_sq[k] * (P1 * TE1(k, g) + sP * P2 * TE2(k, g));
                }
                for (int k = 0; k < N_A1; ++k)
                    if (md.aA1_sq[k] != 0.0) dM[g] += md.aA1_sq[k] * Qsq * TA1(k, g);
            }
            break;
        }
        case LatticeMode::Irrep::A1:
            for (int g = 0; g < 3; ++g)
                for (int k = 0; k < N_A1; ++k)
                    if (md.aA1[k] != 0.0) dM[g] += md.aA1[k] * q1 * TA1(k, g);
            break;
        case LatticeMode::Irrep::A2:
            for (int g = 0; g < 3; ++g)
                for (int k = 0; k < N_A2; ++k)
                    if (md.dA2[k] != 0.0) dM[g] += md.dA2[k] * q1 * TA2(k, g);
            break;
        }
    }
    if (scale != 1.0) for (int g = 0; g < 3; ++g) dM[g] *= scale;
}

void PhononLattice::bond_increment_derivs_local(const Coords& c, double scale, size_t m, int comp,
                                                Eigen::Matrix3d dD[3]) const {
    for (int g = 0; g < 3; ++g) dD[g].setZero();
    const LatticeMode& md = modes[m];
    const double q1 = c.q1[m], q2 = c.q2[m];
    switch (md.irrep) {
    case LatticeMode::Irrep::E: {
        const double t2 = md.q2sign(), sP = -t2;
        for (int g = 0; g < 3; ++g) {
            for (int k = 0; k < N_E; ++k) {
                if (md.cE[k] != 0.0) dD[g] += md.cE[k] * (comp == 0 ? TE1(k, g) : Eigen::Matrix3d(t2 * TE2(k, g)));
                if (md.bE_sq[k] != 0.0) {
                    if (comp == 0) dD[g] += md.bE_sq[k] * ( 2.0 * q1 * TE1(k, g) + sP * 2.0 * q2 * TE2(k, g));
                    else           dD[g] += md.bE_sq[k] * (-2.0 * q2 * TE1(k, g) + sP * 2.0 * q1 * TE2(k, g));
                }
            }
            for (int k = 0; k < N_A1; ++k)
                if (md.aA1_sq[k] != 0.0) dD[g] += md.aA1_sq[k] * 2.0 * (comp == 0 ? q1 : q2) * TA1(k, g);
        }
        break;
    }
    case LatticeMode::Irrep::A1:
        if (comp == 0)
            for (int g = 0; g < 3; ++g)
                for (int k = 0; k < N_A1; ++k)
                    if (md.aA1[k] != 0.0) dD[g] += md.aA1[k] * TA1(k, g);
        break;
    case LatticeMode::Irrep::A2:
        if (comp == 0)
            for (int g = 0; g < 3; ++g)
                for (int k = 0; k < N_A2; ++k)
                    if (md.dA2[k] != 0.0) dD[g] += md.dA2[k] * TA2(k, g);
        break;
    }
    if (scale != 1.0) for (int g = 0; g < 3; ++g) dD[g] *= scale;
}

void PhononLattice::bond_increments_global(const Coords& c, double scale, Eigen::Matrix3d dM[3]) const {
    // Kitaev-frame tensors → spin storage frame, the same rotation as the exchange.
    bond_increments_local(c, scale, dM);
    const Eigen::Matrix3d U = spin_phonon_params.storage_from_local();
    for (int g = 0; g < 3; ++g) dM[g] = U * dM[g] * U.transpose();
}

void PhononLattice::bond_increment_derivs_global(const Coords& c, double scale, size_t m, int comp,
                                                 Eigen::Matrix3d dD[3]) const {
    bond_increment_derivs_local(c, scale, m, comp, dD);
    const Eigen::Matrix3d U = spin_phonon_params.storage_from_local();
    for (int g = 0; g < 3; ++g) dD[g] = U * dD[g] * U.transpose();
}

void PhononLattice::bond_correlations(Eigen::Matrix3d C[3]) const {
    double Cj2[2][3], Cj3[3];
    correlations_of([this](size_t j) { return CMap3(spins[j].data()); }, C, Cj2, Cj3);
}

void PhononLattice::further_correlations(double Cj2[2][3], double Cj3[3]) const {
    Eigen::Matrix3d C[3];
    correlations_of([this](size_t j) { return CMap3(spins[j].data()); }, C, Cj2, Cj3);
}

double PhononLattice::further_bond_modulation(const Coords& c, double c2, double s2, int which, int sub) const {
    double dJ = 0.0;
    for (size_t m = 0; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        double lin, sq;
        if (which == 3)    { lin = md.lamJ3;  sq = md.lamJ3_sq; }
        else if (sub == 0) { lin = md.lamJ2A; sq = md.lamJ2A_sq; }
        else               { lin = md.lamJ2B; sq = md.lamJ2B_sq; }
        if (lin == 0.0 && sq == 0.0) continue;
        const double q1 = c.q1[m], q2 = c.q2[m];
        switch (md.irrep) {
        case LatticeMode::Irrep::E:
            dJ += lin * bond_projection(q1, q2, md.q2sign(), c2, s2) + sq * (q1 * q1 + q2 * q2);
            break;
        case LatticeMode::Irrep::A1: dJ += lin * q1; break;
        case LatticeMode::Irrep::A2: break;
        }
    }
    return dJ;
}

double PhononLattice::further_bond_modulation_deriv(const Coords& c, double c2, double s2, int which, int sub,
                                                    size_t m, int comp) const {
    const LatticeMode& md = modes[m];
    double lin, sq;
    if (which == 3)    { lin = md.lamJ3;  sq = md.lamJ3_sq; }
    else if (sub == 0) { lin = md.lamJ2A; sq = md.lamJ2A_sq; }
    else               { lin = md.lamJ2B; sq = md.lamJ2B_sq; }
    const double q1 = c.q1[m], q2 = c.q2[m];
    switch (md.irrep) {
    case LatticeMode::Irrep::E:
        // d/dq of bond_projection() = q1 c2 − t2 q2 s2  (see the note there)
        return comp == 0 ? (lin * c2 + sq * 2.0 * q1) : (-lin * md.q2sign() * s2 + sq * 2.0 * q2);
    case LatticeMode::Irrep::A1: return comp == 0 ? lin : 0.0;
    case LatticeMode::Irrep::A2: return 0.0;
    }
    return 0.0;
}

double PhononLattice::further_neighbour_modulation_energy(const Coords& c) const {
    if (!has_further_modulation) return 0.0;
    double Cj2[2][3], Cj3[3];
    further_correlations(Cj2, Cj3);
    double E = 0.0;
    for (int s = 0; s < 2; ++s)
        for (int k = 0; k < n_j2_cls; ++k)
            E += further_bond_modulation(c, j2_cs[k].first, j2_cs[k].second, 2, s) * Cj2[s][k];
    for (int k = 0; k < n_j3_cls; ++k)
        E += further_bond_modulation(c, j3_cs[k].first, j3_cs[k].second, 3, 0) * Cj3[k];
    return E;
}

double PhononLattice::effective_J7(const Coords& c) const {
    double J7 = spin_phonon_params.J7;
    for (size_t m = 0; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        if (md.irrep == LatticeMode::Irrep::E)       J7 += md.lamJ7_sq * (c.q1[m] * c.q1[m] + c.q2[m] * c.q2[m]);
        else if (md.irrep == LatticeMode::Irrep::A1) J7 += md.lamJ7 * c.q1[m];
    }
    return J7;
}

double PhononLattice::effective_J7() const { return couplings().J7eff; }

double PhononLattice::dJ7_dq(const Coords& c, size_t m, int comp) const {
    const LatticeMode& md = modes[m];
    if (md.irrep == LatticeMode::Irrep::E)  return 2.0 * md.lamJ7_sq * (comp == 0 ? c.q1[m] : c.q2[m]);
    if (md.irrep == LatticeMode::Irrep::A1) return comp == 0 ? md.lamJ7 : 0.0;
    return 0.0;
}

double PhononLattice::anharmonic_energy(const Coords& c) const {
    double E = 0.0;
    for (const auto& t : anharmonic) {
        const LatticeMode& tg = modes[t.target];
        const double a1 = c.q1[t.lam], a2 = c.q2[t.lam], b1 = c.q1[t.lamp], b2 = c.q2[t.lamp];
        if (tg.irrep == LatticeMode::Irrep::A1) {
            E -= t.g * c.q1[t.target] * (a1 * b1 + a2 * b2);
        } else if (tg.irrep == LatticeMode::Irrep::E) {
            const double P1 = a1 * b1 - a2 * b2, P2 = a1 * b2 + a2 * b1;
            const double tP = (tg.weight == 2) ? 1.0 : -1.0;
            E -= t.g * (c.q1[t.target] * P1 + tP * c.q2[t.target] * P2);
        }
    }
    return phonon_norm() * E;   // extensive: one term per unit cell
}

double PhononLattice::anharmonic_energy() const { return anharmonic_energy(coords_current()); }

double PhononLattice::anharmonic_deriv(const Coords& c, size_t m, int comp) const {
    double d = 0.0;
    for (const auto& t : anharmonic) {
        const LatticeMode& tg = modes[t.target];
        const double a1 = c.q1[t.lam], a2 = c.q2[t.lam], b1 = c.q1[t.lamp], b2 = c.q2[t.lamp];
        const double s1 = c.q1[t.target], s2 = c.q2[t.target];
        const double tP = (tg.weight == 2) ? 1.0 : -1.0;
        if (tg.irrep == LatticeMode::Irrep::A1) {
            if (m == size_t(t.target) && comp == 0) d -= t.g * (a1 * b1 + a2 * b2);
            if (m == size_t(t.lam))  d -= t.g * s1 * (comp == 0 ? b1 : b2);
            if (m == size_t(t.lamp)) d -= t.g * s1 * (comp == 0 ? a1 : a2);
        } else if (tg.irrep == LatticeMode::Irrep::E) {
            const double P1 = a1 * b1 - a2 * b2, P2 = a1 * b2 + a2 * b1;
            if (m == size_t(t.target)) d -= t.g * (comp == 0 ? P1 : tP * P2);
            if (m == size_t(t.lam))  d -= t.g * (comp == 0 ? (s1 * b1 + tP * s2 * b2) : (-s1 * b2 + tP * s2 * b1));
            if (m == size_t(t.lamp)) d -= t.g * (comp == 0 ? (s1 * a1 + tP * s2 * a2) : (-s1 * a2 + tP * s2 * a1));
        }
    }
    return phonon_norm() * d;
}

double PhononLattice::lattice_force_raw(const Coords& c, double scale, const Eigen::Matrix3d C[3],
                                        const double Cj2[2][3], const double Cj3[3], double R7,
                                        size_t m, int comp) const {
    // ∂/∂q [ s (Σ_γ ⟨δM_γ, C_γ⟩ + (J7_eff − J7) R_7 + Σ δJ C_J) + H_anh ]
    Eigen::Matrix3d dD[3];
    bond_increment_derivs_global(c, scale, m, comp, dD);
    double F = 0.0;
    for (int g = 0; g < 3; ++g) F += (dD[g].cwiseProduct(C[g])).sum();
    double F_mod = dJ7_dq(c, m, comp) * R7;
    if (has_further_modulation) {
        for (int s = 0; s < 2; ++s)
            for (int k = 0; k < n_j2_cls; ++k)
                F_mod += further_bond_modulation_deriv(c, j2_cs[k].first, j2_cs[k].second, 2, s, m, comp) * Cj2[s][k];
        for (int k = 0; k < n_j3_cls; ++k)
            F_mod += further_bond_modulation_deriv(c, j3_cs[k].first, j3_cs[k].second, 3, 0, m, comp) * Cj3[k];
    }
    return F + scale * F_mod + anharmonic_deriv(c, m, comp);
}

std::vector<double> PhononLattice::lattice_forces_raw() const {
    const Coords c = coords_current();
    Eigen::Matrix3d C[3];
    double Cj2[2][3], Cj3[3];
    correlations_of([this](size_t j) { return CMap3(spins[j].data()); }, C, Cj2, Cj3);
    const double R7 = ring_exchange_normalized();
    std::vector<double> F;
    for (size_t m = 0; m < modes.size(); ++m)
        for (int comp = 0; comp < modes[m].ncoord(); ++comp)
            F.push_back(lattice_force_raw(c, 1.0, C, Cj2, Cj3, R7, m, comp));
    return F;
}

double PhononLattice::phonon_energy() const {
    // All lattice modes, extensive: N·[½|V|² + ½ω²|Q|² + ¼λ4|Q|⁴].  The zone-centre
    // modes are one coordinate per unit cell, so their inertia and restoring force
    // scale with N while the coordinates stay intensive (see ode_system).
    double E = 0.0;
    {
        const double Q_sq = phonons.Q_x_E1 * phonons.Q_x_E1 + phonons.Q_y_E1 * phonons.Q_y_E1;
        E += phonons.kinetic_energy()
           + 0.5 * phonon_params.omega_E1 * phonon_params.omega_E1 * Q_sq
           + 0.25 * phonon_params.lambda_E1_quartic * Q_sq * Q_sq;
    }
    for (size_t m = 1; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        if (md.frozen) continue;
        const double Q_sq = md.Q1 * md.Q1 + md.Q2 * md.Q2;
        E += 0.5 * (md.V1 * md.V1 + md.V2 * md.V2) + 0.5 * md.omega * md.omega * Q_sq
           + 0.25 * md.quartic * Q_sq * Q_sq;
    }
    return phonon_norm() * E;
}

double PhononLattice::spin_phonon_energy() const {
    // H_ME = Σ_γ ⟨δM_γ, C_γ⟩ + further-neighbour modulations (ring modulation is
    // inside ring_exchange_energy via J7_eff; anharmonic energy is separate).
    const Couplings& cc = couplings();
    Eigen::Matrix3d C[3];
    double Cj2[2][3], Cj3[3];
    correlations_of([this](size_t j) { return CMap3(spins[j].data()); }, C, Cj2, Cj3);
    double E = 0.0;
    for (int g = 0; g < 3; ++g) E += (cc.dM[g].cwiseProduct(C[g])).sum();
    for (int s = 0; s < 2; ++s) for (int k = 0; k < 3; ++k) E += cc.dJ2[s][k] * Cj2[s][k];
    for (int k = 0; k < 3; ++k) E += cc.dJ3[k] * Cj3[k];
    return E;
}

// ============================================================
// COUPLINGS (coordinate-dependent magnetoelastic quantities)
// ============================================================

void PhononLattice::compute_couplings(const Coords& c, double scale, Couplings& out) const {
    bond_increments_global(c, scale, out.dM);
    for (int g = 0; g < 3; ++g) out.dMT[g] = out.dM[g].transpose();
    for (int s = 0; s < 2; ++s) for (int k = 0; k < 3; ++k) out.dJ2[s][k] = 0.0;
    for (int k = 0; k < 3; ++k) out.dJ3[k] = 0.0;
    if (has_further_modulation) {
        for (int s = 0; s < 2; ++s)
            for (int k = 0; k < n_j2_cls; ++k)
                out.dJ2[s][k] = scale * further_bond_modulation(c, j2_cs[k].first, j2_cs[k].second, 2, s);
        for (int k = 0; k < n_j3_cls; ++k)
            out.dJ3[k] = scale * further_bond_modulation(c, j3_cs[k].first, j3_cs[k].second, 3, 0);
    }
    const double J7 = spin_phonon_params.J7;
    out.J7eff = J7 + scale * (effective_J7(c) - J7);
    out.ring_active = (out.J7eff != 0.0) ||
                      std::any_of(plaquette_j7_offsets.begin(), plaquette_j7_offsets.end(),
                                  [](double v) { return v != 0.0; });
}

const PhononLattice::Couplings& PhononLattice::couplings() const {
    // Key: the parameter epoch, the bare J7 (a public member) and every lattice coordinate
    // (frozen ones included).
    const size_t M = modes.size();
    bool hit = (coupling_cache_epoch_ == coupling_epoch_) && coupling_cache_q_.size() == 2 * M + 1;
    if (hit) {
        hit = coupling_cache_q_[2 * M] == spin_phonon_params.J7 &&
              coupling_cache_q_[0] == phonons.Q_x_E1 && coupling_cache_q_[1] == phonons.Q_y_E1;
        for (size_t m = 1; hit && m < M; ++m)
            hit = coupling_cache_q_[2 * m] == modes[m].Q1 && coupling_cache_q_[2 * m + 1] == modes[m].Q2;
    }
    if (!hit) {
        const Coords c = coords_current();
        compute_couplings(c, 1.0, coupling_cache_);
        coupling_cache_q_.resize(2 * M + 1);
        for (size_t m = 0; m < M; ++m) {
            coupling_cache_q_[2 * m] = c.q1[m];
            coupling_cache_q_[2 * m + 1] = c.q2[m];
        }
        coupling_cache_q_[2 * M] = spin_phonon_params.J7;
        coupling_cache_epoch_ = coupling_epoch_;
    }
    return coupling_cache_;
}

// ============================================================
// LOCAL FIELD AND MONTE CARLO INCREMENTS
// ============================================================

Eigen::Vector3d PhononLattice::local_field(size_t site) const {
    const Couplings& cc = couplings();
    auto S = [this](size_t j) { return CMap3(spins[j].data()); };
    auto U = [this](size_t j) { return CMap3(u_site[j].data()); };
    Vec3 H = field_without_ring(site, S, U, cc);
    if (cc.ring_active) H += ring_field_site(site, S, cc.J7eff);
    return H;
}

SpinVector PhononLattice::get_local_field(size_t site) const {
    return SpinVector(local_field(site));
}

SpinVector PhononLattice::get_ring_exchange_field(size_t site) const {
    return get_ring_exchange_field(site, couplings().J7eff);
}

SpinVector PhononLattice::get_ring_exchange_field(size_t site, double J7eff) const {
    return SpinVector(ring_field_site(site, [this](size_t j) { return CMap3(spins[j].data()); }, J7eff));
}

double PhononLattice::site_energy(const Eigen::Vector3d& spin_here, size_t site) const {
    // Every term is linear in S_site and there are no self-bonds, so the energy of all
    // terms containing the site is exactly −S·H with H evaluated without S_site.
    return -spin_here.dot(local_field(site));
}

double PhononLattice::site_energy_diff(const Eigen::Vector3d& new_spin,
                                       const Eigen::Vector3d& old_spin,
                                       size_t site) const {
    return -(new_spin - old_spin).dot(local_field(site));
}

double PhononLattice::dH_dQx_E1() const {
    // Raw (extensive) ∂H_ME/∂Q_x of the primary E1 mode; the EOM divides by phonon_norm().
    return lattice_forces_raw()[0];
}

double PhononLattice::dH_dQy_E1() const {
    return lattice_forces_raw()[1];
}

// ============================================================
// EQUATIONS OF MOTION
// ============================================================

void PhononLattice::phonon_derivatives(
    const PhononState& ph, double t,
    double dHsp_dQx, double dHsp_dQy,
    PhononState& dph_dt) const
{
    // Polar THz drive (in-plane field in the doublet frame: x along the x-bond line)
    double Ex, Ey;
    drive_params.E_field(t, Ex, Ey);

    const double omega_sq = phonon_params.omega_E1 * phonon_params.omega_E1;
    const double Q_sq = ph.Q_x_E1 * ph.Q_x_E1 + ph.Q_y_E1 * ph.Q_y_E1;
    const double l4 = phonon_params.lambda_E1_quartic;
    const double gamma = phonon_params.gamma_E1;
    const double Z = phonon_params.Z_star;

    dph_dt.Q_x_E1 = ph.V_x_E1;
    dph_dt.V_x_E1 = -omega_sq * ph.Q_x_E1 - l4 * Q_sq * ph.Q_x_E1 - gamma * ph.V_x_E1 - dHsp_dQx + Z * Ex;
    dph_dt.Q_y_E1 = ph.V_y_E1;
    dph_dt.V_y_E1 = -omega_sq * ph.Q_y_E1 - l4 * Q_sq * ph.Q_y_E1 - gamma * ph.V_y_E1 - dHsp_dQy + Z * Ey;
}

void PhononLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) const {
    rhs(x, dxdt, t, nullptr, nullptr);
}

void PhononLattice::rhs(const ODEState& x, ODEState& dxdt, double t,
                        const double* spin_noise, const double* lattice_noise) const {
    // Flat layout: [S_0 .. S_{N-1}, Q_x, Q_y, V_x, V_y, (extra modes: q1,[q2],v1,[v2])..., (SLD u, p)].
    // spin_noise (3N) is added to every effective field (so it enters the precession AND
    // the damping term), lattice_noise (phonon_dof()) to the lattice-sector derivatives.
    if (x.size() != state_size)
        throw std::invalid_argument("ode_system: state has " + std::to_string(x.size()) +
                                    " entries, expected state_size = " + std::to_string(state_size));
    if (dxdt.size() != state_size) dxdt.resize(state_size);
    const size_t N = lattice_size;
    const size_t spin_offset = spin_dim * N;
    const double* X = x.data();
    double* D = dxdt.data();
    auto S = [X](size_t j) { return CMap3(X + 3 * j); };

    const Coords c = coords_from_state(X + spin_offset);
    const double scale = get_e1_coupling_scale(t);
    Couplings cc;
    compute_couplings(c, scale, cc);

    // Spin–lattice dynamics: site displacements/momenta live after the mode block.
    const double* Ub = sld_enabled ? X + sld_offset() : nullptr;
    const double* Pb = Ub ? Ub + 3 * N : nullptr;
    double* dU = sld_enabled ? D + sld_offset() : nullptr;
    double* dP = dU ? dU + 3 * N : nullptr;
    auto U = [Ub](size_t j) { return CMap3(Ub + 3 * j); };
    const double Qx0 = c.q1[0], Qy0 = c.q2[0];
    const double inv_m = sld_enabled ? 1.0 / sld_mass : 0.0;
    const bool par = N >= parallel_min_sites;

    // ---- Ring exchange: hexagon pass (R_hex and the six gradients, stored per hexagon) ----
    bool ring_modulated = false;
    for (const auto& md : modes)
        if (md.lamJ7 != 0.0 || md.lamJ7_sq != 0.0) ring_modulated = true;
    const size_t Nh = hexagons.size();
    double R7 = 0.0;
    if (cc.ring_active || ring_modulated) {
        ring_grad_scratch_.resize(18 * Nh);
        ring_val_scratch_.resize(Nh);
        double* RG = ring_grad_scratch_.data();
        double* RV = ring_val_scratch_.data();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(par)
#endif
        for (long hl = 0; hl < long(Nh); ++hl) {
            const size_t h = size_t(hl);
            Vec3 s[6];
            for (int p = 0; p < 6; ++p) s[p] = S(hexagons[h][p]);
            double G[15];
            RV[h] = ring_hexagon(s, G);
            for (int p = 0; p < 6; ++p) {
                const Vec3 g = ring_gradient(s, G, p);
                RG[18 * h + 3 * p] = g(0); RG[18 * h + 3 * p + 1] = g(1); RG[18 * h + 3 * p + 2] = g(2);
            }
        }
        for (size_t h = 0; h < Nh; ++h) R7 += RV[h];
    }
    const double* RG = ring_grad_scratch_.data();

    // ---- Site pass: spin EOM and SLD forces ----
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(par)
#endif
    for (long il = 0; il < long(N); ++il) {
        const size_t i = size_t(il);
        Vec3 H = field_without_ring(i, S, U, cc);
        if (cc.ring_active) {
            for (const auto& [h, pos] : site_hexagons[i]) {
                const double J7 = effective_J7_for_hexagon(h, cc.J7eff);
                H -= J7 * CMap3(RG + 18 * h + 3 * pos);
            }
        }
        if (spin_noise) H += CMap3(spin_noise + 3 * i);
        const Vec3 Si = S(i);
        const Vec3 dSdt = spin_derivative(Si, H);
        D[3 * i] = dSdt(0); D[3 * i + 1] = dSdt(1); D[3 * i + 2] = dSdt(2);

        if (Ub) {
            // forces on u_i from the springs (½k δr²), the striction (g δr S·M·S) and the
            // cubic E1–acoustic vertex (s3 δr²), δr_ij = (u_j − u_i)·d̂_ij
            const Vec3 ui = U(i);
            Vec3 Fi = Vec3::Zero();
            for (size_t n = 0; n < nn_partners[i].size(); ++n) {
                const size_t j = nn_partners[i][n];
                const Vec3& d = nn_bond_vec[i][n];
                const double dr = (U(j) - ui).dot(d);
                const double s3 = sld_v3 * (Qx0 * nn_bond_cq[i][n](0) + Qy0 * nn_bond_cq[i][n](1));
                const double SMS = (sld_g != 0.0) ? Si.dot(nn_interaction[i][n] * S(j)) : 0.0;
                Fi += (sld_k * dr + sld_g * SMS + 2.0 * s3 * dr) * d;
            }
            if (sld_k2 != 0.0) {
                for (size_t n = 0; n < j2_partners[i].size(); ++n) {
                    const Vec3& d2 = j2_bond_vec[i][n];
                    Fi += sld_k2 * (U(j2_partners[i][n]) - ui).dot(d2) * d2;
                }
            }
            // in-plane phonons only: z components stay frozen at zero
            const double px = Pb[3 * i], py = Pb[3 * i + 1];
            dU[3 * i] = px * inv_m; dU[3 * i + 1] = py * inv_m; dU[3 * i + 2] = 0.0;
            dP[3 * i] = Fi(0) - sld_gamma * px; dP[3 * i + 1] = Fi(1) - sld_gamma * py; dP[3 * i + 2] = 0.0;
        }
    }

    // ---- Lattice forces ----
    // ∂H_ME/∂q from the bond correlations (extensive sums over all bonds/hexagons), divided
    // by phonon_norm() = N so that the intensive zone-centre coordinates obey
    // ε̈ = −ω²ε − λ4|ε|²ε − γε̇ + Z*E(t) − (1/N)∂H/∂ε.  This makes the dynamics independent
    // of lattice size; the legacy extensive force is kept behind
    // PhononParams::per_site_backaction = false.
    Eigen::Matrix3d C[3];
    double Cj2[2][3], Cj3[3];
    correlations_of(S, C, Cj2, Cj3);
    double F3x = 0.0, F3y = 0.0;                       // ∂H3/∂Q (E1–acoustic vertex), extensive
    if (Ub && sld_v3 != 0.0) {
        for (size_t i = 0; i < N; i += N_atoms) {      // A sites: each NN bond counted once
            for (size_t n = 0; n < nn_partners[i].size(); ++n) {
                const double dr = (U(nn_partners[i][n]) - U(i)).dot(nn_bond_vec[i][n]);
                F3x += sld_v3 * nn_bond_cq[i][n](0) * dr * dr;
                F3y += sld_v3 * nn_bond_cq[i][n](1) * dr * dr;
            }
        }
    }
    const double inv_norm = 1.0 / phonon_norm();

    // Primary E1 mode
    PhononState ph;
    ph.from_array(X + spin_offset);
    const double F0x = (lattice_force_raw(c, scale, C, Cj2, Cj3, R7, 0, 0) + F3x) * inv_norm;
    const double F0y = (lattice_force_raw(c, scale, C, Cj2, Cj3, R7, 0, 1) + F3y) * inv_norm;
    PhononState dph_dt;
    phonon_derivatives(ph, t, F0x, F0y, dph_dt);
    dph_dt.to_array(D + spin_offset);

    // Extra modes
    double Ex, Ey;
    drive_params.E_field(t, Ex, Ey);
    size_t p = spin_offset + PhononState::N_DOF;
    for (size_t m = 1; m < modes.size(); ++m) {
        const LatticeMode& md = modes[m];
        if (md.frozen) continue;
        const int nc = md.ncoord();
        const double q1 = X[p], q2 = (nc == 2) ? X[p + 1] : 0.0;
        const double v1 = X[p + nc], v2 = (nc == 2) ? X[p + nc + 1] : 0.0;
        const double Qsq = q1 * q1 + q2 * q2;
        const double w2 = md.omega * md.omega;
        // Drive: only polar (weight-1) in-plane doublets carry an in-plane dipole; A2 modes
        // would need E_z, which is zero for the in-plane THz field used here.
        const bool polar = md.irrep == LatticeMode::Irrep::E && md.weight == 1;
        const double d1 = polar ? md.Zstar * Ex : 0.0;
        const double d2 = polar ? md.Zstar * Ey : 0.0;
        const double F1 = lattice_force_raw(c, scale, C, Cj2, Cj3, R7, m, 0) * inv_norm;
        D[p] = v1;
        D[p + nc] = -w2 * q1 - md.quartic * Qsq * q1 - md.gamma * v1 + d1 - F1;
        if (nc == 2) {
            const double F2 = lattice_force_raw(c, scale, C, Cj2, Cj3, R7, m, 1) * inv_norm;
            D[p + 1] = v2;
            D[p + nc + 1] = -w2 * q2 - md.quartic * Qsq * q2 - md.gamma * v2 + d2 - F2;
        }
        p += 2 * nc;
    }
    if (lattice_noise) {
        const size_t n_lat = phonon_dof();
        for (size_t k = 0; k < n_lat; ++k) D[spin_offset + k] += lattice_noise[k];
    }
}

// ============================================================
// ODE INTEGRATION
// ============================================================

namespace {
// Exact output grid t_k = t0 + k dt (k from an integer counter, odeint::integrate_n_steps),
// then one partial step to t1 when (t1 − t0)/dt is not integral. odeint::integrate_const
// decides the last step with an absolute epsilon and can stop one step short of t1, which
// made a chained segment restart from a stale state.
inline size_t whole_steps(double t0, double t1, double dt) {
    return static_cast<size_t>(std::floor((t1 - t0) / dt + 1e-9));
}

template <class Stepper, class Sys, class Obs>
void integrate_grid_fixed(Stepper stepper, Sys& sys, std::vector<double>& x,
                          double t0, double t1, double dt, Obs& obs) {
    const size_t n = whole_steps(t0, t1, dt);
    odeint::integrate_n_steps(stepper, sys, x, t0, dt, n, std::ref(obs));
    const double tn = t0 + double(n) * dt, rem = t1 - tn;
    if (rem > 1e-9 * dt) {
        odeint::runge_kutta4<std::vector<double>> rk4;
        rk4.do_step(sys, x, tn, rem);
        obs(x, t1);
    }
}

template <class Controlled, class Sys, class Obs>
void integrate_grid_controlled(Controlled stepper, Sys& sys, std::vector<double>& x,
                               double t0, double t1, double dt, Obs& obs) {
    const size_t n = whole_steps(t0, t1, dt);
    odeint::integrate_n_steps(stepper, sys, x, t0, dt, n, std::ref(obs));
    const double tn = t0 + double(n) * dt, rem = t1 - tn;
    if (rem > 1e-9 * dt) {
        odeint::integrate_adaptive(stepper, sys, x, tn, t1, rem);
        obs(x, t1);
    }
}
}  // namespace

/**
 * Generic ODE integrator with support for multiple methods
 *
 * Available methods:
 * - "euler": Explicit Euler (1st order, simple, inaccurate)
 * - "rk2" or "midpoint": Runge-Kutta 2nd order / modified midpoint
 * - "rk4": Classic Runge-Kutta 4th order (good balance, fixed step)
 * - "rk5" or "rkck54": Cash-Karp 5(4) (adaptive, good for smooth problems)
 * - "rk54" or "rkf54": Runge-Kutta-Fehlberg 5(4) (adaptive, equivalent to rkck54)
 * - "dopri5": Dormand-Prince 5(4) (default, recommended for general use)
 * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8) (high accuracy, expensive)
 * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (very high accuracy, expensive)
 * - "adams_bashforth" or "ab": Adams-Bashforth 5-step multistep (efficient for smooth problems)
 * - "adams_moulton" or "am": Adams-Bashforth-Moulton 5-step predictor-corrector (more accurate)
 * - "velocity_verlet", "verlet", "symplectic": NOT symplectic here — the flat spin–lattice
 *   state has no (q, p) split; mapped to fixed-step rk4 with a warning
 * Unknown names throw std::invalid_argument.
 */
template<typename System, typename Observer>
void PhononLattice::integrate_ode_system(
    System system_func, ODEState& state,
    double T_start, double T_end, double dt_step,
    Observer observer, const string& method,
    bool use_adaptive, double abs_tol, double rel_tol)
{
    namespace odeint = boost::numeric::odeint;
    if (!(dt_step > 0.0) || !std::isfinite(dt_step))
        throw std::invalid_argument("integrate_ode_system: time step must be positive and finite");
    if (!(T_end >= T_start))
        throw std::invalid_argument("integrate_ode_system: T_end must not precede T_start");
    if (!(abs_tol > 0.0) || !(rel_tol > 0.0))
        throw std::invalid_argument("integrate_ode_system: tolerances must be positive");

    if (method == "euler") {
        // Explicit Euler method (1st order, simple but inaccurate)
        integrate_grid_fixed(odeint::euler<ODEState>(), system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk2" || method == "midpoint") {
        // Modified midpoint method (2nd order)
        integrate_grid_fixed(odeint::modified_midpoint<ODEState>(), system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk4") {
        // Classic fixed-step RK4 (4th order, good balance)
        integrate_grid_fixed(odeint::runge_kutta4<ODEState>(), system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk5" || method == "rkck54" || method == "rk54" || method == "rkf54") {
        // Cash-Karp 5(4) adaptive method (Boost has no separate Fehlberg 5(4))
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            integrate_grid_controlled(odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                                      system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "dopri5") {
        // Dormand-Prince 5(4) adaptive method (default, recommended)
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            integrate_grid_controlled(odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                                      system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "rk78" || method == "rkf78") {
        // Runge-Kutta-Fehlberg 7(8) (very high accuracy)
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            integrate_grid_controlled(odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                                      system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "bulirsch_stoer" || method == "bs") {
        // Bulirsch-Stoer method (very high accuracy, expensive)
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            integrate_grid_controlled(odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                                      system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "adams_bashforth" || method == "ab") {
        // Adams-Bashforth 5-step multistep method (efficient for smooth problems)
        // Uses rk4 for initial steps, then switches to multistep
        integrate_grid_fixed(odeint::adams_bashforth<5, ODEState>(), system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "adams_moulton" || method == "am") {
        // Adams-Bashforth-Moulton predictor-corrector (higher accuracy multistep)
        integrate_grid_fixed(odeint::adams_bashforth_moulton<5, ODEState>(), system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rosenbrock4" || method == "rb4" || method == "implicit_euler" || method == "ie") {
        // Implicit methods with a dense finite-difference Jacobian: n RHS calls and 8 n² bytes
        // per Jacobian, so they are restricted to small systems.
        using ublas_state = boost::numeric::ublas::vector<double>;
        using ublas_matrix = boost::numeric::ublas::matrix<double>;
        const size_t N = state.size();
        constexpr size_t kMaxImplicitState = 1200;
        if (N > kMaxImplicitState)
            throw std::invalid_argument("integrate_ode_system: '" + method + "' builds a dense " + std::to_string(N) +
                                        "² finite-difference Jacobian; use an explicit method (dopri5, rk78) "
                                        "for states larger than " + std::to_string(kMaxImplicitState));
        const double eps_jac = 1e-8;

        ublas_state ublas_x(N);
        for (size_t i = 0; i < N; ++i) ublas_x(i) = state[i];
        auto ublas_system = [&system_func, N](const ublas_state& x, ublas_state& dxdt, double t) {
            ODEState x_vec(N), dxdt_vec(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            system_func(x_vec, dxdt_vec, t);
            for (size_t i = 0; i < N; ++i) dxdt(i) = dxdt_vec[i];
        };
        auto jacobian_columns = [&system_func, N, eps_jac](const ublas_state& x, ublas_matrix& J, double t,
                                                          ODEState& dxdt_base) {
            ODEState x_vec(N), dxdt_pert(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            system_func(x_vec, dxdt_base, t);
            J.resize(N, N);
            for (size_t j = 0; j < N; ++j) {
                const double x_orig = x_vec[j];
                const double h = eps_jac * std::max(1.0, std::abs(x_orig));
                x_vec[j] = x_orig + h;
                system_func(x_vec, dxdt_pert, t);
                x_vec[j] = x_orig;
                for (size_t i = 0; i < N; ++i) J(i, j) = (dxdt_pert[i] - dxdt_base[i]) / h;
            }
        };
        auto ublas_observer = [&observer, N](const ublas_state& x, double t) {
            ODEState x_vec(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            observer(x_vec, t);
        };

        if (method == "rosenbrock4" || method == "rb4") {
            auto ublas_jacobian = [&](const ublas_state& x, ublas_matrix& J, double t, ublas_state& dfdt) {
                ODEState base(N), pert(N), x_vec(N);
                jacobian_columns(x, J, t, base);
                // df/dt by finite differences in time
                const double h_t = eps_jac * std::max(1.0, std::abs(t));
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                system_func(x_vec, pert, t + h_t);
                for (size_t i = 0; i < N; ++i) dfdt(i) = (pert[i] - base[i]) / h_t;
            };
            auto implicit_system = std::make_pair(ublas_system, ublas_jacobian);
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::make_dense_output<odeint::rosenbrock4<double>>(abs_tol, rel_tol),
                    implicit_system, ublas_x, T_start, T_end, dt_step, ublas_observer);
            } else {
                // exact grid (see integrate_grid_fixed), dense output between grid points
                const size_t n = whole_steps(T_start, T_end, dt_step);
                odeint::integrate_n_steps(
                    odeint::make_dense_output<odeint::rosenbrock4<double>>(abs_tol, rel_tol),
                    implicit_system, ublas_x, T_start, dt_step, n, ublas_observer);
                const double tn = T_start + double(n) * dt_step, rem = T_end - tn;
                if (rem > 1e-9 * dt_step) {
                    odeint::integrate_adaptive(
                        odeint::make_dense_output<odeint::rosenbrock4<double>>(abs_tol, rel_tol),
                        implicit_system, ublas_x, tn, T_end, rem);
                    ublas_observer(ublas_x, T_end);
                }
            }
        } else {
            auto ublas_jacobian = [&](const ublas_state& x, ublas_matrix& J, double t) {
                ODEState base(N);
                jacobian_columns(x, J, t, base);
            };
            auto implicit_system = std::make_pair(ublas_system, ublas_jacobian);
            // Fixed-step implicit Euler on the exact grid t_k = T_start + k dt
            odeint::implicit_euler<double> stepper;
            const long n_steps = std::lround(std::ceil((T_end - T_start) / dt_step - 1e-9));
            ublas_observer(ublas_x, T_start);
            for (long k = 0; k < n_steps; ++k) {
                const double t0 = T_start + double(k) * dt_step;
                const double h = std::min(dt_step, T_end - t0);
                stepper.do_step(implicit_system, ublas_x, t0, h);
                ublas_observer(ublas_x, t0 + h);
            }
        }
        for (size_t i = 0; i < N; ++i) state[i] = ublas_x(i);
    } else if (method == "velocity_verlet" || method == "verlet" ||
               method == "symplectic_rkn" || method == "symrkn" ||
               method == "symplectic" || method == "sym") {
        // Symplectic methods require special pair<q,p> state type
        cout << "WARNING: '" << method << "' is not available for the coupled spin–lattice state "
             << "(no (q, p) split); integrating with fixed-step rk4 instead" << endl;
        integrate_grid_fixed(odeint::runge_kutta4<ODEState>(), system_func, state, T_start, T_end, dt_step, observer);
    } else {
        throw std::invalid_argument("PhononLattice: unknown integration method '" + method +
                                    "' (explicit: euler, rk2/midpoint, rk4, rk5/rkck54, rk54/rkf54, dopri5, "
                                    "rk78/rkf78, bulirsch_stoer/bs, adams_bashforth/ab, adams_moulton/am; "
                                    "implicit: rosenbrock4/rb4, implicit_euler/ie)");
    }
}

// ============================================================
// OBSERVABLES
// ============================================================

std::array<Eigen::Vector3d, 4> PhononLattice::magnetization_observables(const double* x) const {
    Vec3 M_local = Vec3::Zero(), M_af = Vec3::Zero(), M_global = Vec3::Zero();
    double O = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t atom = i % N_atoms;
        const CMap3 S(x + 3 * i);
        M_local += S;
        const Vec3 Sg = sublattice_frames[atom] * S;
        M_global += Sg;
        M_af += afm_sublattice_signs[atom] * Sg;
        if (has_ordering_pattern) O += S.dot(CMap3(ordering_pattern[i].data()));
    }
    const double invN = 1.0 / double(lattice_size);
    return {M_af * invN, M_local * invN, M_global * invN, Vec3(O * invN, 0.0, 0.0)};
}

// ============================================================
// MOLECULAR DYNAMICS
// ============================================================

void PhononLattice::molecular_dynamics(
    double T_start, double T_end, double dt_initial,
    string out_dir, size_t save_interval, string method,
    double abs_tol_in, double rel_tol_in)
{
#ifndef HDF5_ENABLED
    throw std::runtime_error("PhononLattice::molecular_dynamics needs HDF5 support (rebuild with HDF5)");
#else
    if (save_interval == 0) throw std::invalid_argument("molecular_dynamics: save_interval must be >= 1");
    if (!(dt_initial > 0.0)) throw std::invalid_argument("molecular_dynamics: dt must be positive");
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }

    cout << "Running PhononLattice spin-phonon dynamics: t=" << T_start << " → " << T_end << endl;
    cout << "Integration method: " << method << endl;
    cout << "Output grid step: " << dt_initial << " (saved every " << save_interval << " steps)" << endl;

    // Convert to flat state
    ODEState state = spins_to_state();

    // Create HDF5 writer with comprehensive metadata (like Lattice class)
    std::unique_ptr<HDF5MDWriter> hdf5_writer;

    // Storage for the zone-center E1 phonon trajectory.
    // The phonon has 4 DOF: (Q_x, Q_y, V_x, V_y).
    vector<double> times_phonon;
    vector<double> Qx_E1_traj, Qy_E1_traj;
    vector<double> Vx_E1_traj, Vy_E1_traj;
    vector<double> Ex_drive_traj, Ey_drive_traj;
    vector<double> energy_traj;

    if (!out_dir.empty()) {
        string hdf5_file = out_dir + "/trajectory.h5";
        cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
        hdf5_writer = std::make_unique<HDF5MDWriter>(
            hdf5_file, lattice_size, spin_dim, N_atoms,
            dim1, dim2, dim3, method,
            dt_initial, T_start, T_end, save_interval, spin_length,
            &site_positions, 10000);
    }

    size_t step_count = 0;
    size_t save_count = 0;
    PhononLattice snapshot(*this);   // scratch copy for energies of saved states

    auto observer = [&](const ODEState& x, double t) {
        if (step_count % save_interval == 0) {
            // M_antiferro (crystal frame, Néel signs), M_local (storage), M_global (crystal)
            const auto obs = magnetization_observables(x.data());
            const size_t p_idx = spin_dim * lattice_size;
            const double Qx_E1 = x[p_idx + 0], Qy_E1 = x[p_idx + 1];
            const double Vx_E1 = x[p_idx + 2], Vy_E1 = x[p_idx + 3];

            // Sample the THz drive at this snapshot for diagnostics.
            double Ex_t, Ey_t;
            drive_params.E_field(t, Ex_t, Ey_t);

            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, obs[0], obs[1], obs[2], x.data());

                times_phonon.push_back(t);
                Qx_E1_traj.push_back(Qx_E1); Qy_E1_traj.push_back(Qy_E1);
                Vx_E1_traj.push_back(Vx_E1); Vy_E1_traj.push_back(Vy_E1);
                Ex_drive_traj.push_back(Ex_t); Ey_drive_traj.push_back(Ey_t);

                // Energy (Hamiltonian at time t) for monitoring
                snapshot.state_to_spins(x);
                energy_traj.push_back(snapshot.total_energy(t) / double(lattice_size));
            }

            if (step_count % (save_interval * 10) == 0) {
                const double Qmag = std::sqrt(Qx_E1 * Qx_E1 + Qy_E1 * Qy_E1);
                cout << "t=" << t << ", |M|=" << obs[1].norm()
                     << ", |M_stag|=" << obs[0].norm()
                     << ", ε=(" << Qx_E1 << ", " << Qy_E1 << "), |ε|=" << Qmag
                     << ", E_drive=(" << Ex_t << ", " << Ey_t << ")" << endl;
            }

            save_count++;
        }
        step_count++;
    };

    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
        this->ode_system(x, dxdt, t);
    };

    // User overrides win when positive; otherwise fall back to the
    // method-aware defaults (1e-6, or 1e-8 for Bulirsch-Stoer).
    double abs_tol = (abs_tol_in > 0.0)
        ? abs_tol_in
        : ((method == "bulirsch_stoer") ? 1e-8 : 1e-6);
    double rel_tol = (rel_tol_in > 0.0)
        ? rel_tol_in
        : ((method == "bulirsch_stoer") ? 1e-8 : 1e-6);
    // Exact output grid: the observer sees T_start + k dt (adaptive steppers subdivide
    // between grid points), so the saved trajectory matches the dt·save_interval spacing
    // recorded in the metadata.
    integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                        observer, method, false, abs_tol, rel_tol);

    state_to_spins(state);

    // Write phonon trajectory data to HDF5 file
    if (hdf5_writer && !times_phonon.empty()) {
        // Close the main writer, then reopen to add the phonon group
        hdf5_writer->close();

        string hdf5_file = out_dir + "/trajectory.h5";
        H5::H5File h5file(hdf5_file, H5F_ACC_RDWR);
        H5::Group phonon_group = h5file.createGroup("/phonon_trajectory");

        hsize_t dims[1] = {times_phonon.size()};
        H5::DataSpace dataspace(1, dims);

        auto write_dataset = [&](const string& name, const vector<double>& data) {
            H5::DataSet ds = phonon_group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };

        // Zone-center E1 mode (single 2-component field).
        write_dataset("Qx_E1", Qx_E1_traj);
        write_dataset("Qy_E1", Qy_E1_traj);
        write_dataset("Vx_E1", Vx_E1_traj);
        write_dataset("Vy_E1", Vy_E1_traj);

        // THz drive samples on the saved grid.
        write_dataset("Ex_drive", Ex_drive_traj);
        write_dataset("Ey_drive", Ey_drive_traj);

        // Energy per site
        write_dataset("energy", energy_traj);

        // Write phonon parameters as metadata
        H5::Group meta_group = h5file.openGroup("/metadata");
        H5::DataSpace scalar_space(H5S_SCALAR);

        auto write_scalar = [&](const string& name, double val) {
            H5::Attribute attr = meta_group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &val);
        };

        // E1 phonon parameters
        write_scalar("omega_E1", phonon_params.omega_E1);
        write_scalar("gamma_E1", phonon_params.gamma_E1);
        write_scalar("lambda_E1_quartic", phonon_params.lambda_E1_quartic);
        write_scalar("Z_star", phonon_params.Z_star);
        write_scalar("alpha_gilbert", alpha_gilbert);
        write_scalar("legacy_kitaev_frame", frame() == Frame::Legacy ? 1.0 : 0.0);

        // E1 magnetoelastic couplings (one isotropic + one anisotropic per channel)
        write_scalar("lambda_E1_J_0", spin_phonon_params.lambda_E1_J_0);
        write_scalar("lambda_E1_J_2", spin_phonon_params.lambda_E1_J_2);
        write_scalar("lambda_E1_K_0", spin_phonon_params.lambda_E1_K_0);
        write_scalar("lambda_E1_K_2", spin_phonon_params.lambda_E1_K_2);
        write_scalar("lambda_E1_Gamma_0", spin_phonon_params.lambda_E1_Gamma_0);
        write_scalar("lambda_E1_Gamma_2", spin_phonon_params.lambda_E1_Gamma_2);
        write_scalar("lambda_E1_Gammap_0", spin_phonon_params.lambda_E1_Gammap_0);
        write_scalar("lambda_E1_Gammap_2", spin_phonon_params.lambda_E1_Gammap_2);
        write_scalar("lambda_E1_J7_0", spin_phonon_params.lambda_E1_J7_0);

        // Pump pulse 1 parameters
        write_scalar("pump_amplitude", drive_params.E0_1);
        write_scalar("pump_frequency", drive_params.omega_1);
        write_scalar("pump_time", drive_params.t_1);
        write_scalar("pump_width", drive_params.sigma_1);
        write_scalar("pump_phase", drive_params.phi_1);
        write_scalar("pump_polarization", drive_params.theta_1);

        // Pump pulse 2 parameters (probe)
        write_scalar("probe_amplitude", drive_params.E0_2);
        write_scalar("probe_frequency", drive_params.omega_2);
        write_scalar("probe_time", drive_params.t_2);
        write_scalar("probe_width", drive_params.sigma_2);
        write_scalar("probe_phase", drive_params.phi_2);
        write_scalar("probe_polarization", drive_params.theta_2);

        phonon_group.close();
        meta_group.close();
        h5file.close();

        cout << "HDF5 trajectory saved with " << save_count << " snapshots (full spin + phonon)" << endl;
    }

    cout << "Dynamics complete! (" << step_count << " steps, " << save_count << " saved)" << endl;
#endif
}

// ============================================================
// LANGEVIN DYNAMICS
// ============================================================

namespace {
// In-place iterative radix-2 complex FFT (forward: e^{-i}, inverse: e^{+i}; no 1/N scaling).
void fft_radix2(std::vector<std::complex<double>>& a, bool inverse) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * M_PI / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v; a[i + k + len / 2] = u - v; w *= wl;
            }
        }
    }
}

// Bose-coloured Gaussian noise for the semi-quantum thermostat.  Channel c of block b is a
// periodic Gaussian process of length Nb whose power spectrum is F(ω_k, T_b) times the classical
// white level 2 D_c T / dt (D_c = diffusion constant per unit temperature of channel c, the same
// value the classical thermostat uses, so F → 1 is the classical limit exactly); blocks overlap by
// half their length and are summed with sine windows (w_n = sin(π(n+½)/Nb), so that
// w_n² + w_{n+Nb/2}² = 1) — a stationary process with the target spectrum up to the 1/Nb spectral
// smearing of the window.  Each (block, channel) uses its own deterministic RNG stream derived
// from the run seed, so runs are reproducible and OpenMP-safe.
struct BoseNoise {
    int Nb, H; size_t nch; double dt; std::vector<double> D_per_T; uint64_t seed;
    std::vector<double> blk[2]; int blk_id[2] = {-1, -1};
    std::vector<double> win;
    BoseNoise(int Nb_, std::vector<double> D_per_T_, double dt_, uint64_t seed_)
        : Nb(Nb_), H(Nb_ / 2), nch(D_per_T_.size()), dt(dt_), D_per_T(std::move(D_per_T_)), seed(seed_) {
        blk[0].assign(static_cast<size_t>(Nb) * nch, 0.0); blk[1].assign(static_cast<size_t>(Nb) * nch, 0.0);
        win.resize(Nb);
        for (int n = 0; n < Nb; ++n) win[n] = std::sin(M_PI * (n + 0.5) / Nb);
    }
    // block b covers steps [(b-1)H, (b+1)H); fill slot b%2 for bath temperature T (meV)
    void generate(int b, double T) {
        const int slot = b & 1; blk_id[slot] = b;
        std::vector<double> F(Nb / 2 + 1);
        for (int k = 0; k <= Nb / 2; ++k) {
            const double w = 2.0 * M_PI * k / (Nb * dt);          // ħω in meV (time unit ħ/meV)
            const double x = (T > 0.0) ? w / T : 1e300;
            F[k] = (k == 0) ? 1.0 : (x < 700.0 ? x / std::expm1(x) : 0.0);
        }
        double* out = blk[slot].data();
        #pragma omp parallel for schedule(static)
        for (long c = 0; c < static_cast<long>(nch); ++c) {
            const double sigma = std::sqrt(2.0 * D_per_T[c] * std::max(T, 0.0) / dt);   // classical white level
            std::mt19937_64 rng(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(b + 1)) ^ (0xC2B2AE3D27D4EB4FULL * (uint64_t)(c + 1)));
            std::normal_distribution<double> g(0.0, 1.0);
            std::vector<std::complex<double>> Y(Nb);
            Y[0] = std::complex<double>(g(rng) * std::sqrt((double)Nb) * std::sqrt(F[0]), 0.0);
            Y[Nb / 2] = std::complex<double>(g(rng) * std::sqrt((double)Nb) * std::sqrt(F[Nb / 2]), 0.0);
            for (int k = 1; k < Nb / 2; ++k) {
                const double a = std::sqrt(Nb / 2.0) * std::sqrt(F[k]);
                Y[k] = std::complex<double>(a * g(rng), a * g(rng)); Y[Nb - k] = std::conj(Y[k]);
            }
            fft_radix2(Y, true);
            for (int n = 0; n < Nb; ++n) out[static_cast<size_t>(n) * nch + c] = sigma * Y[n].real() / Nb;
        }
    }
    // noise value for channel c at global step s (s >= 0)
    inline double sample(long s, size_t c) const {
        const int b1 = static_cast<int>(s / H); const int b2 = b1 + 1;
        const long n1 = s - static_cast<long>(b1 - 1) * H, n2 = s - static_cast<long>(b1) * H;
        const double* B1 = blk[b1 & 1].data(); const double* B2 = blk[b2 & 1].data();
        return win[n1] * B1[static_cast<size_t>(n1) * nch + c] + win[n2] * B2[static_cast<size_t>(n2) * nch + c];
    }
    // make sure blocks floor(s/H) and floor(s/H)+1 are present, generating with the bath T at the block centre
    template <class TF> void ensure(long s, const TF& T_at_step) {
        const int b1 = static_cast<int>(s / H);
        for (int b : {b1, b1 + 1}) if (blk_id[b & 1] != b) generate(b, T_at_step(static_cast<long>(b) * H));
    }
};

int bose_block_length(int requested) {
    int Nb = 16;
    while (Nb < requested) Nb <<= 1;     // power of two
    return Nb;
}

#ifdef HDF5_ENABLED
/// Appends float32 frames [n_frames, N, k] to an extendible dataset, one chunk per frame,
/// so long trajectories stream to disk instead of being buffered in memory.
class FrameStream {
public:
    FrameStream(H5::H5File& file, const std::string& name, size_t N, size_t k) : N_(N), k_(k) {
        hsize_t dims[3] = {0, N, k}, maxd[3] = {H5S_UNLIMITED, N, k}, chunk[3] = {1, N, k};
        H5::DataSpace space(3, dims, maxd);
        H5::DSetCreatPropList prop;
        prop.setChunk(3, chunk);
        ds_ = file.createDataSet(name, H5::PredType::NATIVE_FLOAT, space, prop);
    }
    void append(const std::vector<float>& frame) {
        hsize_t ext[3] = {n_ + 1, N_, k_};
        ds_.extend(ext);
        H5::DataSpace fs = ds_.getSpace();
        hsize_t off[3] = {n_, 0, 0}, cnt[3] = {1, N_, k_};
        fs.selectHyperslab(H5S_SELECT_SET, cnt, off);
        H5::DataSpace ms(3, cnt);
        ds_.write(frame.data(), H5::PredType::NATIVE_FLOAT, ms, fs);
        ++n_;
    }
private:
    H5::DataSet ds_;
    size_t N_, k_;
    hsize_t n_ = 0;
};
#endif
} // namespace

void PhononLattice::integrate_langevin(double t_start, double t_end, double dt,
                                       const string& output_dir,
                                       size_t save_every,
                                       uint64_t seed,
                                       const std::function<void(double)>& on_save) {
    if (!(langevin_temperature >= 0.0) || !std::isfinite(langevin_temperature))
        throw std::invalid_argument("integrate_langevin: langevin_temperature must be finite and >= 0, got " +
                                    std::to_string(langevin_temperature));
    if (!(alpha_gilbert > 0.0))
        throw std::invalid_argument("integrate_langevin: the spin thermostat needs alpha_gilbert > 0, got " +
                                    std::to_string(alpha_gilbert));
    if (!(dt > 0.0) || !std::isfinite(dt))
        throw std::invalid_argument("integrate_langevin: dt must be positive and finite");
    if (!(t_end > t_start))
        throw std::invalid_argument("integrate_langevin: t_end must exceed t_start");
    if (save_every == 0)
        throw std::invalid_argument("integrate_langevin: save_every must be >= 1");
    if (langevin_bath_C < 0.0)
        throw std::invalid_argument("integrate_langevin: langevin_bath_C must be >= 0");
    const bool finiteC = (langevin_bath_C > 0.0);
    if (finiteC && !time_dep_spin_phonon_params.is_constant())
        throw std::invalid_argument("integrate_langevin: the finite-capacity bath books energy changes as heat, "
                                    "which is wrong for a time-dependent magnetoelastic schedule (lambda_time_mode)");

    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }
    if (seed == 0) seed = derive_seed_from_master(0x4C414E47455649ULL);   // "LANGEVI"

    const double s_len = double(spin_length);
    const double alpha = alpha_gilbert;
    // FDT for the LL equation with noise in precession and damping (Stratonovich):
    // D = α T / (|S| (1 + α²)); see the class documentation and spin_integrators.h.
    const double D_spin_per_T = alpha / (s_len * (1.0 + alpha * alpha));
    const double invN = 1.0 / phonon_norm();
    std::mt19937_64 rng64(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Damped lattice velocity channels: (state index, D per unit T) with D = γ/N for the
    // zone-centre modes (inertia N) — δV ~ N(0, 2 D T dt) pairs with the friction −γV.
    std::vector<size_t> mode_ch;
    std::vector<double> mode_D;
    {
        const size_t off = spin_dim * lattice_size;
        if (phonon_params.gamma_E1 > 0.0) {
            mode_ch.push_back(off + 2); mode_D.push_back(phonon_params.gamma_E1 * invN);
            mode_ch.push_back(off + 3); mode_D.push_back(phonon_params.gamma_E1 * invN);
        }
        size_t p = off + PhononState::N_DOF;
        for (size_t m = 1; m < modes.size(); ++m) {
            const LatticeMode& md = modes[m];
            if (md.frozen) continue;
            const int nc = md.ncoord();
            if (md.gamma > 0.0)
                for (int c = 0; c < nc; ++c) { mode_ch.push_back(p + nc + c); mode_D.push_back(md.gamma * invN); }
            p += 2 * nc;
        }
    }

    // Semi-quantum thermostat: Bose-coloured noise blocks (see BoseNoise above)
    std::unique_ptr<BoseNoise> qnoise, mnoise, lnoise;
    if (langevin_quantum) {
        const int Nb = bose_block_length(langevin_block);
        qnoise = std::make_unique<BoseNoise>(Nb, std::vector<double>(3 * lattice_size, D_spin_per_T), dt, seed);
        if (!mode_ch.empty())
            mnoise = std::make_unique<BoseNoise>(Nb, mode_D, dt, seed ^ 0x6A09E667F3BCC909ULL);
        std::cout << "  SEMI-QUANTUM thermostat: Bose-coloured noise, block " << Nb
                  << " steps (" << Nb * dt << " code units), overlap-add sine windows; no zero-point term"
                  << std::endl;
        if (langevin_temperature > 0.0 && Nb * dt < 10.0 / langevin_temperature)
            std::cout << "  WARNING: block length " << Nb * dt << " < 10 ħ/k_B T = " << 10.0 / langevin_temperature
                      << ": the Bose spectrum is smeared by the window (raise langevin_block)" << std::endl;
    }
    // Finite-capacity bath: dynamical bath temperature T_dyn (meV), energy bookkeeping per step
    double T_dyn = langevin_temperature;
    const double bathN = static_cast<double>(lattice_size) * langevin_bath_C;   // meV per meV of T
    if (finiteC) {
        std::cout << "  FINITE-CAPACITY bath: C_l = " << langevin_bath_C << " k_B per spin; T_bath evolves by energy"
                  << " conservation (phonon dissipation + spin damping heat it, noise cools it)" << std::endl;
        if (langevin_quantum)
            std::cout << "  NOTE: Bose noise blocks use T_bath at block generation (lag up to "
                      << bose_block_length(langevin_block) * dt << " code units)" << std::endl;
    }
    auto T_at_step = [&](long s) {
        return finiteC ? T_dyn : langevin_bath_T(t_start + std::max(0.0, static_cast<double>(s)) * dt);
    };
    auto Tph_at_step = [&](long s) { return phonon_langevin_T >= 0.0 ? phonon_langevin_T : T_at_step(s); };

    // Spin–lattice dynamics: Langevin noise on the in-plane momenta, σ_l² = 2 γ_l m T_l / dt, classical or
    // Bose-coloured.  T_l follows sld_T if set, else the bath.
    auto Tl_at_step = [&](long s) { return sld_T >= 0.0 ? sld_T : T_at_step(s); };
    if (sld_enabled) {
        if (sld_relax_pending_) relax_sld_static(sld_relax, 1e-6);   // once per enable_sld(true)
        if (sld_init_T > 0.0) {
            bool cold = true;
            for (const auto& p : p_site) if (p.squaredNorm() > 0.0) { cold = false; break; }
            if (cold) {
                std::mt19937_64 rng0(seed ^ 0x5bd1e9955bd1e995ULL);
                std::normal_distribution<double> n0(0.0, std::sqrt(sld_mass * sld_init_T));
                for (auto& p : p_site) p = Eigen::Vector3d(n0(rng0), n0(rng0), 0.0);
                std::cout << "  SLD: momenta initialised from a Maxwell distribution at T = " << sld_init_T << std::endl;
            }
        }
        if (sld_quantum && sld_gamma > 0.0) {
            const int Nb = bose_block_length(langevin_block);
            lnoise = std::make_unique<BoseNoise>(Nb, std::vector<double>(2 * lattice_size, sld_gamma * sld_mass),
                                                 dt, seed ^ 0x9e3779b97f4a7c15ULL);
            std::cout << "  SLD: Bose-coloured lattice noise, block " << Nb << " steps" << std::endl;
        }
        std::cout << "  SLD: gamma_l = " << sld_gamma << ", T_l = " << Tl_at_step(0) << " (meV), m = " << sld_mass
                  << ", k = " << sld_k << ", k2 = " << sld_k2 << ", g = " << sld_g << ", v3 = " << sld_v3 << std::endl;
    }

    const long n_steps = static_cast<long>(std::ceil((t_end - t_start) / dt - 1e-9));
    std::cout << "PhononLattice Langevin dynamics (frozen-noise RK4, Stratonovich)"
              << std::endl;
    std::cout << "  t = " << t_start << " → " << t_end << ", dt = " << dt << " (" << n_steps << " steps)" << std::endl;
    std::cout << "  T (k_B T)            = " << langevin_temperature << std::endl;
    std::cout << "  Gilbert damping α    = " << alpha << "  (spin noise D = αT/(|S|(1+α²)))" << std::endl;
    std::cout << "  damped lattice DOF   = " << mode_ch.size() << " mode velocities"
              << (sld_enabled && sld_gamma > 0.0 ? " + SLD momenta" : "") << std::endl;
    if (langevin_dT != 0.0) {
        std::cout << "  Bath profile         : T0 + dT f(t), dT = " << langevin_dT
                  << ", t_step = " << langevin_t_step << ", tau_on = " << langevin_tau_on
                  << ", tau_off = " << langevin_tau_off << " (two-reservoir scenario)" << std::endl;
    }
    std::cout << "  RNG seed             = " << seed << std::endl;
    std::cout << "  Save every           = " << save_every << " steps" << std::endl;

    ODEState state = spins_to_state();
    auto sync_back = [&]() {
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i](0) = state[i * spin_dim + 0];
            spins[i](1) = state[i * spin_dim + 1];
            spins[i](2) = state[i * spin_dim + 2];
        }
        unpack_lattice(&state[spin_dim * lattice_size]);
    };

    // Streaming output: text observables, float32 spin (and displacement) frames in HDF5.
    std::ofstream traj_out;
#ifdef HDF5_ENABLED
    std::unique_ptr<H5::H5File> spin_file, lat_file;
    std::unique_ptr<FrameStream> spin_frames, u_frames;
    std::vector<double> frame_times;
#endif
    if (!output_dir.empty()) {
        traj_out.open(output_dir + "/langevin_trajectory.txt");
        if (!traj_out) throw std::runtime_error("integrate_langevin: cannot write " + output_dir + "/langevin_trajectory.txt");
        traj_out << "# t  Mx My Mz |M|  Mstag_x Mstag_y Mstag_z |M_stag|  "
                 << "E_total  Qx_E1 Qy_E1 Vx_E1 Vy_E1  T_bath"
                 << (sld_enabled ? "  T_lat E_kin_lat E_pot_lat E_striction\n" : "\n");
        traj_out << std::scientific << std::setprecision(10);
#ifdef HDF5_ENABLED
        spin_file = std::make_unique<H5::H5File>(output_dir + "/langevin_spins.h5", H5F_ACC_TRUNC);
        spin_frames = std::make_unique<FrameStream>(*spin_file, "spins", lattice_size, 3);
        if (sld_enabled) {
            lat_file = std::make_unique<H5::H5File>(output_dir + "/langevin_lattice.h5", H5F_ACC_TRUNC);
            u_frames = std::make_unique<FrameStream>(*lat_file, "u", lattice_size, 2);
        }
#endif
    }
    size_t n_frames = 0;
    std::vector<float> fbuf;

    // Frozen-noise (Wong–Zakai) RK4: the noise of a step is held constant over the four RK4
    // stages and enters the RHS like any other field/force, so the Stratonovich drift of the
    // multiplicative spin noise and its interplay with precession, damping and the lattice
    // are integrated together (a Lie split "deterministic step, then noise kick" carries an
    // O(α|H| dt) bias in <E>). Spin norms are restored after each step (RK4 conserves them to
    // O(dt^5)).
    using Stepper = boost::numeric::odeint::runge_kutta4<ODEState>;
    Stepper stepper;
    std::vector<double> xi_spin(3 * lattice_size, 0.0), xi_lat(phonon_dof(), 0.0);
    auto rhs_noisy = [&](const ODEState& x, ODEState& dxdt, double tt) {
        rhs(x, dxdt, tt, xi_spin.data(), xi_lat.empty() ? nullptr : xi_lat.data());
    };

    const size_t spin_offset = spin_dim * lattice_size;
    // Polar modes for the drive work of the finite bath: (state index of q1, Z*).
    std::vector<std::pair<size_t, double>> polar;
    {
        polar.push_back({spin_offset, phonon_params.Z_star});
        size_t p = spin_offset + PhononState::N_DOF;
        for (size_t m = 1; m < modes.size(); ++m) {
            const LatticeMode& md = modes[m];
            if (md.frozen) continue;
            if (md.irrep == LatticeMode::Irrep::E && md.weight == 1 && md.Zstar != 0.0) polar.push_back({p, md.Zstar});
            p += 2 * md.ncoord();
        }
    }
    double E_prev = finiteC ? total_energy() : 0.0;
    std::vector<double> q_prev(2 * polar.size(), 0.0);

    for (long step = 0; step < n_steps; ++step) {
        const double t = t_start + static_cast<double>(step) * dt;   // no accumulated round-off
        // Save observables BEFORE stepping
        if (step % static_cast<long>(save_every) == 0) {
            sync_back();
            if (on_save) on_save(t);
            if (traj_out.is_open()) {
                const Vec3 M = magnetization_local(), Ms = magnetization_local_antiferro();
                traj_out << t << ' ' << M(0) << ' ' << M(1) << ' ' << M(2) << ' ' << M.norm() << ' '
                         << Ms(0) << ' ' << Ms(1) << ' ' << Ms(2) << ' ' << Ms.norm() << ' '
                         << total_energy(t) << ' '
                         << state[spin_offset + 0] << ' ' << state[spin_offset + 1] << ' '
                         << state[spin_offset + 2] << ' ' << state[spin_offset + 3] << ' '
                         << (finiteC ? T_dyn : langevin_bath_T(t));
                if (sld_enabled)
                    traj_out << ' ' << sld_lattice_temperature() << ' ' << sld_kinetic_energy() << ' '
                             << sld_spring_energy() << ' ' << sld_striction_energy();
                traj_out << '\n';
#ifdef HDF5_ENABLED
                fbuf.resize(3 * lattice_size);
                for (size_t k = 0; k < 3 * lattice_size; ++k) fbuf[k] = static_cast<float>(state[k]);
                spin_frames->append(fbuf);
                if (u_frames) {
                    fbuf.resize(2 * lattice_size);
                    for (size_t i = 0; i < lattice_size; ++i) {
                        fbuf[2 * i] = static_cast<float>(u_site[i](0));
                        fbuf[2 * i + 1] = static_cast<float>(u_site[i](1));
                    }
                    u_frames->append(fbuf);
                }
                frame_times.push_back(t);
#endif
                ++n_frames;
            }
        }

        if (finiteC)
            for (size_t k = 0; k < polar.size(); ++k) {
                q_prev[2 * k] = state[polar[k].first];
                q_prev[2 * k + 1] = state[polar[k].first + 1];
            }

        // ── Noise of this step: spin fields h ~ N(0, 2D/dt) (D = αT/(|S|(1+α²))), mode
        //    forces η ~ N(0, 2γ_m T_ph/(N dt)), SLD forces N(0, 2γ_l m T_l/dt) — each the
        //    fluctuation–dissipation partner of the corresponding friction term ──
        const double Tb = T_at_step(step);
        const double sigma_s = std::sqrt(2.0 * D_spin_per_T * std::max(Tb, 0.0) / dt);
        if (qnoise) {
            qnoise->ensure(step, T_at_step);
            for (size_t k = 0; k < 3 * lattice_size; ++k) xi_spin[k] = qnoise->sample(step, k);
        } else {
            for (size_t k = 0; k < 3 * lattice_size; ++k) xi_spin[k] = sigma_s * normal(rng64);
        }
        if (!mode_ch.empty()) {
            if (mnoise) mnoise->ensure(step, Tph_at_step);
            const double Tph = std::max(Tph_at_step(step), 0.0);
            for (size_t k = 0; k < mode_ch.size(); ++k)
                xi_lat[mode_ch[k] - spin_offset] = mnoise ? mnoise->sample(step, k)
                                                          : std::sqrt(2.0 * mode_D[k] * Tph / dt) * normal(rng64);
        }
        if (sld_enabled && sld_gamma > 0.0) {
            const size_t poff = sld_offset() + 3 * lattice_size - spin_offset;
            if (lnoise) lnoise->ensure(step, Tl_at_step);
            const double sigma_l = std::sqrt(2.0 * sld_gamma * sld_mass * std::max(Tl_at_step(step), 0.0) / dt);
            for (size_t i = 0; i < lattice_size; ++i)
                for (int d = 0; d < 2; ++d)
                    xi_lat[poff + 3 * i + d] = lnoise ? lnoise->sample(step, 2 * i + d) : sigma_l * normal(rng64);
        }

        // ── RK4 step of the full system with the noise frozen over the step ──
        stepper.do_step(rhs_noisy, state, t, dt);
        for (size_t i = 0; i < lattice_size; ++i) {
            double* Sx = &state[3 * i];
            const double n = std::sqrt(Sx[0] * Sx[0] + Sx[1] * Sx[1] + Sx[2] * Sx[2]);
            if (n > 0.0) { const double f = s_len / n; Sx[0] *= f; Sx[1] *= f; Sx[2] *= f; }
        }
        if (finiteC && langevin_tau_off > 0.0) {
            // heat escape of the finite bath to the cryostat: T_dyn relaxes to T0 with langevin_tau_off
            T_dyn -= (T_dyn - langevin_temperature) * dt / langevin_tau_off;
        }

        if (finiteC) {
            // Energy conservation: whatever the system lost that was not supplied by the drive
            // went into the bath.  W_drive = N Σ_polar Z*_m E(t+dt/2)·ΔQ_m over every polar mode.
            sync_back();
            const double E_now = total_energy();
            double Ex = 0.0, Ey = 0.0;
            drive_params.E_field(t + 0.5 * dt, Ex, Ey);
            double W = 0.0;
            for (size_t k = 0; k < polar.size(); ++k)
                W += polar[k].second * (Ex * (state[polar[k].first] - q_prev[2 * k]) +
                                        Ey * (state[polar[k].first + 1] - q_prev[2 * k + 1]));
            W *= phonon_norm();
            T_dyn -= (E_now - E_prev - W) / bathN;
            if (T_dyn < 0.0) T_dyn = 0.0;
            E_prev = E_now;
        }
    }

    // Final sync
    sync_back();

    if (!output_dir.empty()) {
#ifdef HDF5_ENABLED
        auto write_times = [&](H5::H5File& f) {
            hsize_t d1[1] = {frame_times.size()};
            f.createDataSet("t", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, d1))
             .write(frame_times.data(), H5::PredType::NATIVE_DOUBLE);
        };
        write_times(*spin_file);
        spin_frames.reset();
        spin_file->close();
        if (lat_file) { write_times(*lat_file); u_frames.reset(); lat_file->close(); }
        std::cout << "Langevin spin frames written to " << output_dir << "/langevin_spins.h5 (/spins [frame, N, 3], /t)"
                  << (sld_enabled ? "; displacements to langevin_lattice.h5 (/u [frame, N, 2], /t)" : "") << std::endl;
#endif
        std::cout << "Langevin trajectory written to " << output_dir
                  << "/langevin_trajectory.txt (" << n_frames << " snapshots)" << std::endl;

        // Also save final state
        save_spin_config(output_dir + "/final_spins.txt");
#ifdef HDF5_ENABLED
        save_state_hdf5(output_dir + "/final_state.h5");
#endif
    }

    std::cout << "Langevin dynamics complete (" << n_steps << " steps, "
              << n_frames << " snapshots saved)." << std::endl;
}

// ============================================================
// MONTE CARLO METHODS
// ============================================================

Eigen::Vector3d PhononLattice::random_unit_vector() {
    // Uniform on S²: z uniform in [-1, 1), φ uniform (Archimedes).
    const double z = uniform_dist(rng) * 2.0 - 1.0;
    const double phi = uniform_dist(rng) * 2.0 * M_PI;
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    return Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

Eigen::Vector3d PhononLattice::gaussian_move3(const Eigen::Vector3d& current, double sigma) {
    const Vec3 v = current + sigma * spin_length * random_unit_vector();
    const double norm = v.norm();
    if (norm < 1e-12 * spin_length) return current;
    return v * (spin_length / norm);
}

double PhononLattice::metropolis(double T, bool gaussian_move, double sigma) {
    if (!(T > 0.0)) return 0.0;
    const double beta = 1.0 / T;
    size_t accepted = 0;
    std::uniform_int_distribution<size_t> site_dist(0, lattice_size - 1);

    for (size_t sweep_step = 0; sweep_step < lattice_size; ++sweep_step) {
        const size_t i = site_dist(rng);
        const Vec3 old_spin = CMap3(spins[i].data());
        const Vec3 new_spin = gaussian_move ? gaussian_move3(old_spin, sigma)
                                            : Vec3(random_unit_vector() * double(spin_length));
        // Exact: E is linear in S_i, so ΔE = −ΔS · H_i (includes every term, e.g. the striction).
        const double dE = -(new_spin - old_spin).dot(local_field(i));
        if (dE <= 0.0 || uniform_dist(rng) < std::exp(-beta * dE)) {
            spins[i] = new_spin;
            accepted++;
        }
    }
    if (mc_sample_lattice) lattice_mc_sweep(T);
    return static_cast<double>(accepted) / lattice_size;
}

double PhononLattice::heat_bath(double T) {
    if (!(T > 0.0)) {
        deterministic_sweep(1);
        return 1.0;
    }
    const double beta = 1.0 / T;
    const double s = spin_length;
    std::uniform_int_distribution<size_t> site_dist(0, lattice_size - 1);
    for (size_t step = 0; step < lattice_size; ++step) {
        const size_t i = site_dist(rng);
        const Vec3 H = local_field(i);
        const double h = H.norm();
        const double x = beta * s * h;
        Vec3 n;
        if (x < 1e-12) {
            n = random_unit_vector();
        } else {
            // P(cosθ) ∝ exp(x cosθ) on [−1, 1]; inverse CDF with v = 1 − u ∈ (0, 1]:
            //   cosθ = 1 + ln(1 − v (1 − e^{−2x})) / x   (stable for large x)
            const double v = 1.0 - uniform_dist(rng);
            const double c = std::clamp(1.0 + std::log1p(v * std::expm1(-2.0 * x)) / x, -1.0, 1.0);
            const double sn = std::sqrt(std::max(0.0, 1.0 - c * c));
            const double phi = 2.0 * M_PI * uniform_dist(rng);
            const Vec3 z = H / h;
            const Vec3 a = (std::abs(z(0)) < 0.9) ? Vec3::UnitX() : Vec3::UnitY();
            const Vec3 e1 = (a - a.dot(z) * z).normalized();
            const Vec3 e2 = z.cross(e1);
            n = c * z + sn * (std::cos(phi) * e1 + std::sin(phi) * e2);
        }
        spins[i] = n * s;
    }
    if (mc_sample_lattice) lattice_mc_sweep(T);
    return 1.0;
}

void PhononLattice::overrelaxation() {
    // Reflection about the local field: S' = 2(Ĥ·S)Ĥ − S conserves −S·H exactly.
    for (size_t i = 0; i < lattice_size; ++i) {
        const Vec3 H = local_field(i);
        const double norm = H.norm();
        if (norm > 1e-14) {
            const Vec3 h = H / norm;
            const Vec3 S = CMap3(spins[i].data());
            spins[i] = 2.0 * h.dot(S) * h - S;
        }
    }
}

// ============================================================
// SIMULATED ANNEALING
// ============================================================

void PhononLattice::simulated_annealing(
    double T_start, double T_end, size_t n_steps,
    size_t overrelax_rate, double cooling_rate,
    string out_dir, bool save_observables,
    bool T_zero, size_t n_deterministics,
    bool adiabatic_phonons, bool gaussian_move,
    bool preserve_initial_phonons)
{
    // Validates 0 < T_end <= T_start and 0 < cooling_rate < 1 (throws otherwise).
    const vector<double> schedule = mc::annealing_schedule(T_start, T_end, cooling_rate);

    cout << "Starting PhononLattice simulated annealing..." << endl;
    cout << "T: " << T_start << " → " << T_end << " (" << schedule.size() << " temperatures), sweeps per temp: "
         << n_steps << ", local update: " << (local_update == LocalUpdate::HeatBath ? "heat bath" :
                                              gaussian_move ? "adaptive small-angle Metropolis" : "uniform Metropolis")
         << endl;
    if (adiabatic_phonons) {
        cout << "Adiabatic phonons ENABLED: phonons will be relaxed at each temperature step" << endl;
    } else {
        cout << "Adiabatic phonons DISABLED: lattice coordinates held fixed during MC" << endl;
    }

    // Reset the lattice sector unless the caller intentionally prescribed frozen nonzero
    // coordinates for a fixed-epsilon basin diagnostic.
    if (!preserve_initial_phonons) {
        reset_lattice_sector();
    }

    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }

#ifdef HDF5_ENABLED
    std::unique_ptr<H5::H5File> h5file;
    vector<double> steps_data, temps_data, energies_data, acc_rates_data;
    vector<double> Qx_data, Qy_data;  ///< zone-center E1 components (if adiabatic)

    if (save_observables && !out_dir.empty()) {
        h5file = std::make_unique<H5::H5File>(out_dir + "/annealing.h5", H5F_ACC_TRUNC);
    }
#else
    (void)save_observables;
#endif

    mc::StepSizeController step_size;   // Robbins–Monro width of the small-angle moves
    const bool adapt = gaussian_move && local_update == LocalUpdate::Metropolis;

    // One MC "sweep": [overrelaxation] + local sweep every overrelax_rate-th time
    // (overrelax_rate > 0), the convention of mc::parallel_tempering.
    size_t sweep_counter = 0;
    auto sweep = [&](double T, double sigma, double& acc, size_t& n_loc) {
        if (overrelax_rate > 0) {
            overrelaxation();
            if (sweep_counter % overrelax_rate == 0) { acc += local_sweep(T, gaussian_move, sigma); ++n_loc; }
        } else {
            acc += local_sweep(T, gaussian_move, sigma); ++n_loc;
        }
        ++sweep_counter;
    };

    size_t temp_step = 0;
    for (const double T : schedule) {
        double accepted_rate = 0.0;
        size_t n_local = 0;
        size_t step = 0;
        if (adapt) {
            // Adapt during the first half of the sweeps at this temperature; then freeze σ
            // so the remaining sweeps form a time-homogeneous chain.
            step_size.restart();
            constexpr size_t block = 10;
            for (; step + block <= n_steps / 2; step += block) {
                double a = 0.0; size_t n = 0;
                for (size_t b = 0; b < block; ++b) sweep(T, step_size.sigma(), a, n);
                if (n > 0) step_size.update(a / double(n));
            }
        }
        for (; step < n_steps; ++step) sweep(T, step_size.sigma(), accepted_rate, n_local);

        // If using adiabatic phonons, relax phonons to equilibrium for current spin configuration
        if (adiabatic_phonons) {
            relax_phonons(1e-10, 1000, 1.0);
        }

        const double acceptance = n_local > 0 ? accepted_rate / double(n_local) : 0.0;

        if (temp_step % 10 == 0 || temp_step + 1 == schedule.size()) {
            double E = energy_density();
            Eigen::Vector3d M = magnetization_local();
            Eigen::Vector3d M_stag = magnetization_local_antiferro();
            cout << "T=" << std::scientific << std::setprecision(4) << T
                 << ", E/N=" << std::fixed << std::setprecision(6) << E
                 << ", acc=" << std::fixed << std::setprecision(4) << acceptance;
            if (adapt) cout << ", σ=" << step_size.sigma();
            cout << ", |M|=" << std::fixed << std::setprecision(4) << M.norm()
                 << ", |M_stag|=" << std::fixed << std::setprecision(4) << M_stag.norm();
            if (adiabatic_phonons) {
                cout << ", ε=(" << std::fixed << std::setprecision(4) << phonons.Q_x_E1
                     << ", " << phonons.Q_y_E1 << ")"
                     << ", |ε|=" << std::fixed << std::setprecision(4) << E1_amplitude();
            }
            cout << endl;
        }

#ifdef HDF5_ENABLED
        if (h5file) {
            steps_data.push_back(static_cast<double>(temp_step));
            temps_data.push_back(T);
            energies_data.push_back(energy_density());
            acc_rates_data.push_back(acceptance);
            if (adiabatic_phonons) {
                Qx_data.push_back(phonons.Q_x_E1);
                Qy_data.push_back(phonons.Q_y_E1);
            }
        }
#endif
        ++temp_step;
    }

    // Final report
    double E_final = energy_density();
    Eigen::Vector3d M_final = magnetization_local();
    Eigen::Vector3d M_stag_final = magnetization_local_antiferro();
    cout << "\n=== Simulated Annealing Complete ===" << endl;
    cout << "Temperature steps: " << temp_step << endl;
    cout << "Final energy density: " << E_final << endl;
    cout << "Final magnetization: [" << M_final.transpose() << "], |M|=" << M_final.norm() << endl;
    cout << "Final staggered M: [" << M_stag_final.transpose() << "], |M_stag|=" << M_stag_final.norm() << endl;
    if (adiabatic_phonons) {
        cout << "Zone-center E1 phonon equilibrium:" << endl;
        cout << "  ε = (" << phonons.Q_x_E1 << ", " << phonons.Q_y_E1 << ")"
             << ", |ε| = " << E1_amplitude() << endl;
    }
    cout << "====================================" << endl;

#ifdef HDF5_ENABLED
    if (h5file && !steps_data.empty()) {
        H5::Group ann_group = h5file->createGroup("/annealing");
        hsize_t dims[1] = {steps_data.size()};
        H5::DataSpace dataspace(1, dims);

        H5::DataSet ds = ann_group.createDataSet("steps", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(steps_data.data(), H5::PredType::NATIVE_DOUBLE);
        ds = ann_group.createDataSet("temperature", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(temps_data.data(), H5::PredType::NATIVE_DOUBLE);
        ds = ann_group.createDataSet("energy", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(energies_data.data(), H5::PredType::NATIVE_DOUBLE);
        ds = ann_group.createDataSet("acceptance_rate", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(acc_rates_data.data(), H5::PredType::NATIVE_DOUBLE);

        if (adiabatic_phonons && !Qx_data.empty()) {
            ds = ann_group.createDataSet("Qx_E1", H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(Qx_data.data(), H5::PredType::NATIVE_DOUBLE);
            ds = ann_group.createDataSet("Qy_E1", H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(Qy_data.data(), H5::PredType::NATIVE_DOUBLE);
        }

        h5file->close();
        cout << "Annealing data saved to " << out_dir << "/annealing.h5" << endl;
    }
#endif

    // Save spin config after annealing (before deterministic sweeps)
    if (!out_dir.empty()) {
        save_spin_config(out_dir + "/spins_T=" + std::to_string(T_end) + ".txt");
    }

    // T=0 deterministic sweeps if requested
    if (T_zero && n_deterministics > 0) {
        // Energy breakdown BEFORE deterministic sweeps
        cout << "\n=== Energy BEFORE deterministic sweeps ===" << endl;
        cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
        cout << "  Phonon energy:     " << phonon_energy() << endl;
        cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
        cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
        cout << "  |ε_E1| = " << E1_amplitude() << endl;

        cout << "\nPerforming at most " << n_deterministics << " deterministic sweeps at T=0..." << endl;
        const double dmax = deterministic_sweep(n_deterministics);
        cout << "  last sweep max |ΔS| = " << dmax << endl;

        // If using adiabatic phonons, relax phonons again after deterministic sweeps
        if (adiabatic_phonons) {
            relax_phonons(1e-10, 1000, 1.0);

            // Enforce a joint spin-phonon equilibrium so subsequent dynamics start from a stationary state
            cout << "Running joint spin-phonon relaxation to enforce equilibrium before dynamics..." << endl;
            relax_joint(1e-10, 100, 1, false);
        }

        // Energy breakdown AFTER deterministic sweeps
        cout << "\n=== Energy AFTER deterministic sweeps ===" << endl;
        cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
        cout << "  Phonon energy:     " << phonon_energy() << endl;
        cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
        cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
        cout << "  |ε_E1| = " << E1_amplitude() << endl;

        // Save final configuration after T=0 sweeps
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_T=0.txt");
            cout << "Final spin config saved to " << out_dir << "/spins_T=0.txt" << endl;
        }
    } else if (!out_dir.empty()) {
        // If no T=0 sweeps, just save the final config
        save_spin_config(out_dir + "/spins_final.txt");
        cout << "Final spin config saved to " << out_dir << "/spins_final.txt" << endl;
    }
}

// ============================================================
// LATTICE-SECTOR MONTE CARLO
// ============================================================

double PhononLattice::lattice_potential(const Coords& c, const Eigen::Matrix3d C[3], const double Cj2[2][3],
                                        const double Cj3[3], double R7) const {
    // Potential energy of the zone-centre coordinates at fixed spins (fixed correlations):
    //   V(q) = N Σ_m [½ω²|q_m|² + ¼λ4|q_m|⁴] + Σ_γ⟨δM_γ(q), C_γ⟩ + (J7_eff(q) − J7) R_7
    //          + Σ δJ2/3(q) C_J2/3 + H_anh(q)
    const double Nn = phonon_norm();
    double V = 0.0;
    for (size_t m = 0; m < modes.size(); ++m) {
        if (modes[m].frozen) continue;
        const double w2 = (m == 0) ? phonon_params.omega_E1 * phonon_params.omega_E1 : modes[m].omega * modes[m].omega;
        const double l4 = (m == 0) ? phonon_params.lambda_E1_quartic : modes[m].quartic;
        const double Q2 = c.q1[m] * c.q1[m] + c.q2[m] * c.q2[m];
        V += Nn * (0.5 * w2 * Q2 + 0.25 * l4 * Q2 * Q2);
    }
    Eigen::Matrix3d dM[3];
    bond_increments_global(c, 1.0, dM);
    for (int g = 0; g < 3; ++g) V += (dM[g].cwiseProduct(C[g])).sum();
    V += (effective_J7(c) - spin_phonon_params.J7) * R7;
    if (has_further_modulation) {
        for (int s = 0; s < 2; ++s)
            for (int k = 0; k < n_j2_cls; ++k)
                V += further_bond_modulation(c, j2_cs[k].first, j2_cs[k].second, 2, s) * Cj2[s][k];
        for (int k = 0; k < n_j3_cls; ++k)
            V += further_bond_modulation(c, j3_cs[k].first, j3_cs[k].second, 3, 0) * Cj3[k];
    }
    return V + anharmonic_energy(c);
}

double PhononLattice::lattice_mc_sweep(double T) {
    // Samples P(Q | S) ∝ exp(−V(Q)/T) for every non-frozen zone-centre coordinate by
    // Metropolis moves of the thermal width √(T/(N ω²)) (the spin correlations are fixed
    // during the update, so a proposal costs O(#tensors), not O(N)), and redraws the mode
    // velocities from their exact Maxwell distribution N(0, T/N). Together with the spin
    // moves this samples the joint Gibbs state of spins and lattice.
    if (!(T > 0.0)) return 0.0;
    Eigen::Matrix3d C[3];
    double Cj2[2][3], Cj3[3];
    correlations_of([this](size_t j) { return CMap3(spins[j].data()); }, C, Cj2, Cj3);
    const double R7 = ring_exchange_normalized();
    Coords c = coords_current();
    double V = lattice_potential(c, C, Cj2, Cj3, R7);
    const double beta = 1.0 / T, Nn = phonon_norm();
    size_t accepted = 0, proposed = 0;
    for (size_t m = 0; m < modes.size(); ++m) {
        if (modes[m].frozen) continue;
        const double w = (m == 0) ? phonon_params.omega_E1 : modes[m].omega;
        const double width = std::sqrt(T / (Nn * w * w));
        for (int comp = 0; comp < modes[m].ncoord(); ++comp) {
            double& q = (comp == 0 ? c.q1 : c.q2)[m];
            const double q_old = q;
            q = q_old + width * normal_dist(rng);
            const double V_new = lattice_potential(c, C, Cj2, Cj3, R7);
            ++proposed;
            if (V_new <= V || uniform_dist(rng) < std::exp(-beta * (V_new - V))) { V = V_new; ++accepted; }
            else q = q_old;
        }
    }
    const double sv = std::sqrt(T / Nn);
    phonons.Q_x_E1 = c.q1[0]; phonons.Q_y_E1 = c.q2[0];
    phonons.V_x_E1 = sv * normal_dist(rng); phonons.V_y_E1 = sv * normal_dist(rng);
    for (size_t m = 1; m < modes.size(); ++m) {
        LatticeMode& md = modes[m];
        if (md.frozen) continue;
        md.Q1 = c.q1[m]; md.Q2 = c.q2[m];
        md.V1 = sv * normal_dist(rng);
        md.V2 = (md.ncoord() == 2) ? sv * normal_dist(rng) : 0.0;
    }
    return proposed ? double(accepted) / double(proposed) : 0.0;
}

// ============================================================
// PHONON RELAXATION
// ============================================================

bool PhononLattice::relax_phonons(double tol, size_t max_iter, double damping) {
    if (!(tol > 0.0) || !(damping > 0.0) || damping > 1.0)
        throw std::invalid_argument("relax_phonons: need tol > 0 and 0 < damping <= 1");
    cout << "Relaxing lattice coordinates (all non-frozen modes) to equilibrium for the current spins..." << endl;
    // Minimise the lattice potential at fixed spins,
    //   V(q) = N Σ_m [½ω²|q_m|² + ¼λ4|q_m|⁴] + H_ME(q; C) + H_anh(q),
    // whose spin correlations C_γ, C_J2, C_J3, R_7 are constants during the relaxation.
    // Newton steps with the exact Hessian (central differences of the analytic force) and a
    // Levenberg shift when it is not positive definite, plus an Armijo backtracking line
    // search on V: robust near soft modes, where the old diagonal (harmonic-Jacobian)
    // fixed-point iteration diverged.
    std::vector<std::pair<size_t, int>> idx;              // (mode, component) of every free coordinate
    for (size_t m = 0; m < modes.size(); ++m)
        if (!modes[m].frozen)
            for (int comp = 0; comp < modes[m].ncoord(); ++comp) idx.push_back({m, comp});
    const size_t n = idx.size();
    const double Nn = phonon_norm();

    Eigen::Matrix3d C[3];
    double Cj2[2][3], Cj3[3];
    correlations_of([this](size_t j) { return CMap3(spins[j].data()); }, C, Cj2, Cj3);
    const double R7 = ring_exchange_normalized();

    Coords c = coords_current();
    auto set_q = [&](Coords& cc, const Eigen::VectorXd& q) {
        for (size_t k = 0; k < n; ++k) (idx[k].second == 0 ? cc.q1 : cc.q2)[idx[k].first] = q(k);
    };
    auto get_q = [&](const Coords& cc) {
        Eigen::VectorXd q(n);
        for (size_t k = 0; k < n; ++k) q(k) = (idx[k].second == 0 ? cc.q1 : cc.q2)[idx[k].first];
        return q;
    };
    auto omega2 = [&](size_t m) { return m == 0 ? phonon_params.omega_E1 * phonon_params.omega_E1 : modes[m].omega * modes[m].omega; };
    auto quartic = [&](size_t m) { return m == 0 ? phonon_params.lambda_E1_quartic : modes[m].quartic; };
    auto potential = [&](const Coords& cc) { return lattice_potential(cc, C, Cj2, Cj3, R7); };
    auto gradient = [&](const Coords& cc) {
        Eigen::VectorXd g(n);
        for (size_t k = 0; k < n; ++k) {
            const size_t m = idx[k].first;
            const int comp = idx[k].second;
            const double q = comp == 0 ? cc.q1[m] : cc.q2[m];
            const double Q2 = cc.q1[m] * cc.q1[m] + cc.q2[m] * cc.q2[m];
            g(k) = Nn * (omega2(m) * q + quartic(m) * Q2 * q) + lattice_force_raw(cc, 1.0, C, Cj2, Cj3, R7, m, comp);
        }
        return g;
    };

    bool converged = (n == 0);
    double residual = 0.0;
    size_t iter = 0;
    for (; !converged && iter < max_iter; ++iter) {
        Eigen::VectorXd q = get_q(c);
        const Eigen::VectorXd g = gradient(c);
        residual = g.norm() / Nn;                          // per-site force, as in the EOM
        if (residual < tol) { converged = true; break; }
        // Hessian by central differences of the analytic gradient
        Eigen::MatrixXd Hs(n, n);
        for (size_t k = 0; k < n; ++k) {
            const double h = 1e-6 * std::max(1.0, std::abs(q(k)));
            Coords cp = c, cm = c;
            Eigen::VectorXd qp = q, qm = q;
            qp(k) += h; qm(k) -= h;
            set_q(cp, qp); set_q(cm, qm);
            Hs.col(k) = (gradient(cp) - gradient(cm)) / (2.0 * h);
        }
        Hs = 0.5 * (Hs + Hs.transpose());
        // Levenberg shift until positive definite
        double mu = 0.0;
        Eigen::VectorXd dq;
        for (int tries = 0; tries < 60; ++tries) {
            Eigen::LLT<Eigen::MatrixXd> llt(Hs + mu * Eigen::MatrixXd::Identity(n, n));
            if (llt.info() == Eigen::Success) { dq = -llt.solve(g); break; }
            mu = (mu == 0.0) ? 1e-8 * std::max(1.0, Hs.diagonal().cwiseAbs().maxCoeff()) : 10.0 * mu;
        }
        if (dq.size() != Eigen::Index(n)) dq = -g / std::max(1.0, Hs.diagonal().cwiseAbs().maxCoeff());
        // Armijo backtracking on V
        const double V0 = potential(c);
        const double slope = g.dot(dq);
        double step = damping;
        Coords trial = c;
        for (int ls = 0; ls < 60; ++ls) {
            set_q(trial, q + step * dq);
            if (potential(trial) <= V0 + 1e-4 * step * slope) break;
            step *= 0.5;
        }
        c = trial;
        if (iter > 0 && iter % 200 == 0)
            cout << "  iter=" << iter << ", |F|/N=" << residual << endl;
    }
    // write back
    for (size_t k = 0; k < n; ++k) {
        const size_t m = idx[k].first;
        const double v = (idx[k].second == 0 ? c.q1 : c.q2)[m];
        if (m == 0) { if (idx[k].second == 0) phonons.Q_x_E1 = v; else phonons.Q_y_E1 = v; }
        else        { if (idx[k].second == 0) modes[m].Q1 = v;    else modes[m].Q2 = v; }
    }
    if (converged) cout << "  Lattice relaxation converged in " << iter << " iterations." << endl;
    phonons.V_x_E1 = 0.0;
    phonons.V_y_E1 = 0.0;
    for (size_t m = 1; m < modes.size(); ++m) { modes[m].V1 = 0.0; modes[m].V2 = 0.0; }

    cout << "  E1 equilibrium: ε=(" << phonons.Q_x_E1 << ", " << phonons.Q_y_E1
         << "), |ε|=" << E1_amplitude() << endl;
    for (size_t m = 1; m < modes.size(); ++m)
        cout << "  mode " << m << " (" << modes[m].name << "): Q=(" << modes[m].Q1 << ", " << modes[m].Q2 << ")"
             << (modes[m].frozen ? " [frozen]" : "") << endl;
    cout << "  --- Energy after lattice relaxation ---" << endl;
    cout << "    Spin energy:        " << spin_energy() << " (" << spin_energy() / lattice_size << " per site)" << endl;
    cout << "    Phonon energy:      " << phonon_energy() << endl;
    cout << "    Spin-phonon energy: " << spin_phonon_energy() << endl;
    cout << "    Anharmonic energy:  " << anharmonic_energy() << endl;
    cout << "    Total energy:       " << total_energy() << " (" << energy_density() << " per site)" << endl;
    if (!converged) cout << "  WARNING: lattice relaxation did not converge in " << max_iter
                         << " iterations (|F|/N = " << residual << ")." << endl;
    return converged;
}

bool PhononLattice::relax_joint(double tol, size_t max_iter, size_t spin_sweeps_per_iter, bool phonon_only) {
    if (!(tol > 0.0)) throw std::invalid_argument("relax_joint: tol must be positive");
    if (phonon_only) {
        cout << "Phonon-only relaxation (spins fixed)..." << endl;
    } else {
        cout << "Joint spin-phonon relaxation to find true steady state..." << endl;
    }
    const double N = double(lattice_size);
    double prev_energy = total_energy();
    double prev_Q_E = E1_amplitude();

    auto report = [&](const char* what) {
        cout << "\n=== Energy after " << what << " ===" << endl;
        cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
        cout << "  Phonon energy:     " << phonon_energy() << endl;
        cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
        cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
        cout << "  |Q_E1| = " << E1_amplitude() << endl;
    };

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Step 1: Relax phonons for current spin configuration
        relax_phonons(1e-10, 1000, 1.0);

        // Step 2: Relax spins for current phonon configuration (skip if phonon_only)
        if (!phonon_only) {
            deterministic_sweep(spin_sweeps_per_iter);
        }

        // Convergence on intensive quantities: energy per site and |Q| (both size independent)
        const double curr_energy = total_energy();
        const double curr_Q_E = E1_amplitude();
        const double dE = std::abs(curr_energy - prev_energy) / N;
        const double dQ = std::abs(curr_Q_E - prev_Q_E);

        if (iter % 10 == 0 || (dE < tol && dQ < tol)) {
            cout << "  " << (phonon_only ? "Phonon" : "Joint") << " relax iter " << iter
                 << ": E=" << curr_energy
                 << ", |Q_E|=" << curr_Q_E
                 << ", dE/N=" << dE
                 << ", dQ=" << dQ << endl;
        }

        if (dE < tol && dQ < tol) {
            // Final phonon relaxation: the last deterministic sweep moved the spins
            if (!phonon_only) {
                relax_phonons(1e-10, 1000, 1.0);
            }
            cout << "  " << (phonon_only ? "Phonon-only" : "Joint") << " relaxation converged in " << iter << " iterations!" << endl;
            report(phonon_only ? "phonon-only relaxation (equilibrium state)" : "joint relaxation (equilibrium state)");
            return true;
        }

        prev_energy = curr_energy;
        prev_Q_E = curr_Q_E;
    }

    // Final phonon relaxation even if not converged, to ensure V=0 and phonons at equilibrium
    if (!phonon_only) {
        relax_phonons(1e-10, 1000, 1.0);
    }
    cout << "  WARNING: " << (phonon_only ? "Phonon-only" : "Joint") << " relaxation did not fully converge after " << max_iter << " iterations" << endl;
    report("the last iteration");
    return false;
}

// ============================================================
// DETERMINISTIC SWEEP
// ============================================================

double PhononLattice::deterministic_sweep(size_t num_sweeps) {
    // Gauss–Seidel alignment S_i ← |S| Ĥ_i. The local energy is −S·H_i, so every update
    // lowers E (or leaves it unchanged); a site with a vanishing field keeps its spin.
    double max_change = 0.0;
    for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
        max_change = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const Vec3 H = local_field(i);
            const double norm = H.norm();
            if (norm < 1e-15) continue;
            const Vec3 S_new = H * (spin_length / norm);
            max_change = std::max(max_change, (S_new - CMap3(spins[i].data())).norm());
            spins[i] = S_new;
        }
        if (max_change < 1e-12 * spin_length) break;
    }
    return max_change;
}

// ============================================================
// I/O
// ============================================================

#ifdef HDF5_ENABLED
void PhononLattice::save_spin_config_hdf5(const string& filename) const {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    hsize_t dims[2] = {lattice_size, 3};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = file.createDataSet("spins", H5::PredType::NATIVE_DOUBLE, dataspace);

    vector<double> spin_data(lattice_size * 3);
    for (size_t i = 0; i < lattice_size; ++i) {
        spin_data[i*3 + 0] = spins[i](0);
        spin_data[i*3 + 1] = spins[i](1);
        spin_data[i*3 + 2] = spins[i](2);
    }
    dataset.write(spin_data.data(), H5::PredType::NATIVE_DOUBLE);

    file.close();
}

namespace {
/// Read a [n, 3] double dataset after checking its shape.
std::vector<double> read_n_by_3(H5::H5File& file, const std::string& name, size_t n, const std::string& filename) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[2] = {0, 0};
    if (sp.getSimpleExtentNdims() != 2 || (sp.getSimpleExtentDims(dims), dims[0] != n || dims[1] != 3))
        throw std::runtime_error(filename + ":" + name + " has the wrong shape (expected [" +
                                 std::to_string(n) + ", 3])");
    std::vector<double> data(3 * n);
    ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
    return data;
}
}  // namespace

void PhononLattice::load_spin_config_hdf5(const string& filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    const std::vector<double> spin_data = read_n_by_3(file, "spins", lattice_size, filename);
    for (size_t i = 0; i < lattice_size; ++i) {
        const Vec3 s(spin_data[3 * i], spin_data[3 * i + 1], spin_data[3 * i + 2]);
        if (!s.allFinite() || !(s.norm() > 1e-12)) throw std::runtime_error(filename + ": invalid spin at site " + std::to_string(i));
        spins[i] = s.normalized() * spin_length;
    }
    file.close();
}

void PhononLattice::save_state_hdf5(const string& filename) const {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    H5::Group spin_group = file.createGroup("/spins");
    H5::Group phonon_group = file.createGroup("/phonons");

    // Save spins
    {
        hsize_t dims[2] = {lattice_size, 3};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = spin_group.createDataSet("configuration", H5::PredType::NATIVE_DOUBLE, dataspace);

        vector<double> spin_data(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) {
            spin_data[i*3 + 0] = spins[i](0);
            spin_data[i*3 + 1] = spins[i](1);
            spin_data[i*3 + 2] = spins[i](2);
        }
        dataset.write(spin_data.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Save zone-center E1 phonon state: [Q_x, Q_y, V_x, V_y]
    {
        hsize_t dims[1] = {PhononState::N_DOF};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = phonon_group.createDataSet("state", H5::PredType::NATIVE_DOUBLE, dataspace);

        double ph_data[PhononState::N_DOF];
        phonons.to_array(ph_data);
        dataset.write(ph_data, H5::PredType::NATIVE_DOUBLE);
    }
    // Extra lattice modes: [Q1, Q2, V1, V2] per mode (modes 1..)
    if (modes.size() > 1) {
        hsize_t dims[1] = {4 * (modes.size() - 1)};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = phonon_group.createDataSet("extra_modes", H5::PredType::NATIVE_DOUBLE, dataspace);
        vector<double> data;
        for (size_t m = 1; m < modes.size(); ++m) {
            data.push_back(modes[m].Q1); data.push_back(modes[m].Q2);
            data.push_back(modes[m].V1); data.push_back(modes[m].V2);
        }
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    // Spin–lattice dynamics: site displacements and momenta [N, 3]
    if (sld_enabled) {
        H5::Group lg = file.createGroup("/lattice");
        hsize_t dims[2] = {lattice_size, 3};
        vector<double> ub(lattice_size * 3), pb(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) for (int d = 0; d < 3; ++d) { ub[3 * i + d] = u_site[i](d); pb[3 * i + d] = p_site[i](d); }
        lg.createDataSet("u", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(2, dims)).write(ub.data(), H5::PredType::NATIVE_DOUBLE);
        lg.createDataSet("p", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(2, dims)).write(pb.data(), H5::PredType::NATIVE_DOUBLE);
    }

    file.close();
}

void PhononLattice::load_state_hdf5(const string& filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // Load spins
    {
        const std::vector<double> spin_data = read_n_by_3(file, "/spins/configuration", lattice_size, filename);
        for (size_t i = 0; i < lattice_size; ++i) {
            const Vec3 s(spin_data[3 * i], spin_data[3 * i + 1], spin_data[3 * i + 2]);
            if (!s.allFinite() || !(s.norm() > 1e-12)) throw std::runtime_error(filename + ": invalid spin at site " + std::to_string(i));
            spins[i] = s.normalized() * spin_length;
        }
    }

    // Load zone-center E1 phonon state
    {
        H5::DataSet dataset = file.openDataSet("/phonons/state");
        double ph_data[PhononState::N_DOF];
        dataset.read(ph_data, H5::PredType::NATIVE_DOUBLE);
        phonons.from_array(ph_data);
    }
    if (sld_enabled && H5Lexists(file.getId(), "/lattice", H5P_DEFAULT) > 0) {
        const std::vector<double> ub = read_n_by_3(file, "/lattice/u", lattice_size, filename);
        const std::vector<double> pb = read_n_by_3(file, "/lattice/p", lattice_size, filename);
        for (size_t i = 0; i < lattice_size; ++i) for (int d = 0; d < 3; ++d) { u_site[i](d) = ub[3 * i + d]; p_site[i](d) = pb[3 * i + d]; }
        cout << "  loaded lattice displacements/momenta from " << filename << endl;
    }
    if (modes.size() > 1 && H5Lexists(file.getId(), "/phonons/extra_modes", H5P_DEFAULT) > 0) {
        H5::DataSet dataset = file.openDataSet("/phonons/extra_modes");
        vector<double> data(4 * (modes.size() - 1), 0.0);
        H5::DataSpace sp = dataset.getSpace();
        hsize_t n = 0;
        sp.getSimpleExtentDims(&n, nullptr);
        if (n != data.size())
            throw std::runtime_error(filename + ": /phonons/extra_modes holds " + std::to_string(n) +
                                     " values, the lattice has " + std::to_string(data.size()));
        dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);
        for (size_t m = 1; m < modes.size(); ++m) {
            modes[m].Q1 = data[4 * (m - 1)]; modes[m].Q2 = data[4 * (m - 1) + 1];
            modes[m].V1 = data[4 * (m - 1) + 2]; modes[m].V2 = data[4 * (m - 1) + 3];
        }
    }

    file.close();
}
#endif

void PhononLattice::save_spin_config(const string& filename) const {
    classical_spin::io::write_table(filename, lattice_size, 3, [&](size_t i, size_t j) { return spins[i](j); });
}

vector<Eigen::Vector3d> PhononLattice::read_vec3_file(const string& filename, size_t n) {
    const classical_spin::io::Table t = classical_spin::io::read_table(filename, n, 3);
    vector<Eigen::Vector3d> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(t.row(i)[0], t.row(i)[1], t.row(i)[2]);
    return out;
}

void PhononLattice::load_spin_config(const string& filename) {
    const classical_spin::io::Table t = classical_spin::io::read_table(filename, lattice_size, 3);
    vector<Eigen::Vector3d> v(lattice_size);
    for (size_t i = 0; i < lattice_size; ++i) {
        v[i] = Eigen::Vector3d(t.row(i)[0], t.row(i)[1], t.row(i)[2]);
        if (!(v[i].norm() > 1e-12))
            throw std::runtime_error(filename + ":" + std::to_string(t.lines[i]) + ": zero spin vector");
    }
    // Rescale to spin_length; a spin already of that length to round-off is kept bitwise.
    const double L = spin_length;
    for (size_t i = 0; i < lattice_size; ++i) {
        const double n = v[i].norm();
        spins[i] = (std::abs(n - L) <= 4.0 * std::numeric_limits<double>::epsilon() * L) ? v[i] : Eigen::Vector3d(v[i] * (L / n));
    }
}

void PhononLattice::save_positions(const string& filename) const {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("save_positions: cannot open " + filename + " for writing");
    file << std::scientific << std::setprecision(12);

    for (size_t i = 0; i < lattice_size; ++i) {
        file << site_positions[i](0) << " "
             << site_positions[i](1) << " "
             << site_positions[i](2) << "\n";
    }
}

namespace {
/// Line-oriented reader for the disorder / pinning files: strips '#' comments and skips
/// blank lines; any other line that does not parse as `expected` numbers throws.
template <class F>
size_t for_each_record(const std::string& filename, const char* what, F&& on_record) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error(std::string("Cannot open ") + what + " file: " + filename);
    std::string line;
    size_t line_no = 0, n = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::istringstream iss(line);
        on_record(iss, line_no);
        ++n;
    }
    return n;
}
}  // namespace

void PhononLattice::apply_plaquette_j7_disorder_from_file(const string& filename) {
    if (filename.empty()) return;
    if (plaquette_j7_offsets.size() != hexagons.size()) plaquette_j7_offsets.assign(hexagons.size(), 0.0);
    const size_t n = for_each_record(filename, "plaquette J7 disorder", [&](std::istringstream& iss, size_t line_no) {
        size_t plaquette; double dJ7;
        if (!(iss >> plaquette >> dJ7))
            throw std::runtime_error("Malformed plaquette J7 disorder line " + std::to_string(line_no) + " in " + filename);
        if (plaquette >= plaquette_j7_offsets.size())
            throw std::runtime_error("Plaquette J7 disorder index out of range at line " +
                                     std::to_string(line_no) + " in " + filename);
        plaquette_j7_offsets[plaquette] += dJ7;
    });
    invalidate_couplings();
    cout << "Applied " << n << " plaquette J7 disorder offsets from " << filename << endl;
}

void PhononLattice::add_pinning_fields_from_file(const string& filename) {
    if (filename.empty()) return;
    const size_t n = for_each_record(filename, "pinning field", [&](std::istringstream& iss, size_t line_no) {
        size_t site; double bx, by, bz;
        if (!(iss >> site >> bx >> by >> bz))
            throw std::runtime_error("Malformed pinning field line " + std::to_string(line_no) + " in " + filename);
        if (site >= lattice_size)
            throw std::runtime_error("Pinning field site index out of range at line " +
                                     std::to_string(line_no) + " in " + filename);
        field[site](0) += bx;
        field[site](1) += by;
        field[site](2) += bz;
    });
    cout << "Loaded " << n << " site-resolved pinning fields from " << filename << endl;
}

void PhononLattice::apply_nn_exchange_disorder_from_file(const string& filename) {
    if (filename.empty()) return;
    const size_t n = for_each_record(filename, "NN exchange disorder", [&](std::istringstream& iss, size_t line_no) {
        size_t site, partner; double scale;
        if (!(iss >> site >> partner >> scale))
            throw std::runtime_error("Malformed NN exchange disorder line " + std::to_string(line_no) + " in " + filename);
        if (site >= lattice_size || partner >= lattice_size)
            throw std::runtime_error("NN exchange disorder site index out of range at line " +
                                     std::to_string(line_no) + " in " + filename);
        if (!(scale > 0.0))
            throw std::runtime_error("NN exchange disorder scale must be positive at line " +
                                     std::to_string(line_no) + " in " + filename);
        const size_t nf = nn_index(site, partner, "NN exchange disorder (" + filename + ")");
        const size_t nr = nn_index(partner, site, "NN exchange disorder (" + filename + ")");
        nn_scale_[site][nf] *= scale;
        nn_scale_[partner][nr] *= scale;
    });
    rebuild_nn_exchange();
    cout << "Applied " << n << " NN exchange disorder bond scalings from " << filename << endl;
}

void PhononLattice::apply_nn_exchange_channel_disorder_from_file(const string& filename) {
    if (filename.empty()) return;
    const size_t n = for_each_record(filename, "NN exchange channel disorder", [&](std::istringstream& iss, size_t line_no) {
        size_t site, partner; double dJ, dK, dGamma, dGammap;
        if (!(iss >> site >> partner >> dJ >> dK >> dGamma >> dGammap))
            throw std::runtime_error("Malformed NN exchange channel disorder line " + std::to_string(line_no) +
                                     " in " + filename);
        if (site >= lattice_size || partner >= lattice_size)
            throw std::runtime_error("NN exchange channel disorder site index out of range at line " +
                                     std::to_string(line_no) + " in " + filename);
        const size_t nf = nn_index(site, partner, "NN exchange channel disorder (" + filename + ")");
        const size_t nr = nn_index(partner, site, "NN exchange channel disorder (" + filename + ")");
        // Kitaev-channel increment of this bond type, rotated into the storage frame; the
        // matrix is for the site→partner orientation (its transpose for the reverse entry).
        const Eigen::Matrix3d dM = spin_phonon_params.to_storage_frame(
            classical_spin::kitaev::make_J_local(nn_bond_types[site][nf], dJ, dK, dGamma, dGammap));
        nn_delta_[site][nf] += dM;
        nn_delta_[partner][nr] += dM.transpose();
    });
    rebuild_nn_exchange();
    cout << "Applied " << n << " NN exchange channel disorder increments from " << filename << endl;
}

// ============================================================
// 2DCS SPECTROSCOPY
// ============================================================

PhononLattice::MagTrajectory PhononLattice::single_pulse_drive(
    double polarization, double t_B,
    double pulse_amp, double pulse_width, double pulse_freq,
    double T_start, double T_end, double step_size, const string& method,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol) {
    // A single pulse is the double pulse with a zero second amplitude.
    if (!(pulse_width > 0.0)) throw std::invalid_argument("single_pulse_drive: pulse_width must be positive");
    drive_params.E0_1 = pulse_amp;
    drive_params.omega_1 = pulse_freq;
    drive_params.t_1 = t_B;
    drive_params.sigma_1 = pulse_width;
    drive_params.phi_1 = 0.0;
    drive_params.theta_1 = polarization;
    drive_params.E0_2 = 0.0;

    MagTrajectory trajectory;
    ODEState state = spins_to_state();
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) { this->ode_system(x, dxdt, t); };
    // Samples on the grid T_start + k step_size (k >= 1) and at T_end; segment
    // boundaries of the chunked integration are not sampled twice.
    double last_save_time = T_start;
    auto observer = [&](const ODEState& x, double t) {
        if (t - last_save_time >= step_size - 1e-10 || (t >= T_end - 1e-10 && t > last_save_time + 1e-10)) {
            trajectory.push_back({t, magnetization_observables(x.data())});
            last_save_time = t;
        }
    };

    // One integration on the exact grid T_start + k step_size. (The former pulse-window
    // chunking restarted the integrator at every window edge without ever enlarging the step
    // — every chunk used step_size — and a chunk end missed by integrate_const's epsilon
    // restarted the next chunk from a stale state; pulse_window_chunking is ignored.)
    (void)pulse_window_chunking;
    integrate_ode_system(system_func, state, T_start, T_end, step_size,
                         observer, method, false, abs_tol, rel_tol);

    drive_params.E0_1 = 0.0;
    state_to_spins(state);
    return trajectory;
}

PhononLattice::MagTrajectory PhononLattice::double_pulse_drive(
    double polarization_1, double t_B_1,
    double polarization_2, double t_B_2,
    double pulse_amp, double pulse_width, double pulse_freq,
    double T_start, double T_end, double step_size, const string& method,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol) {
    if (!(pulse_width > 0.0)) throw std::invalid_argument("double_pulse_drive: pulse_width must be positive");
    // Set up pump (pulse 1)
    drive_params.E0_1 = pulse_amp;
    drive_params.omega_1 = pulse_freq;
    drive_params.t_1 = t_B_1;
    drive_params.sigma_1 = pulse_width;
    drive_params.phi_1 = 0.0;
    drive_params.theta_1 = polarization_1;
    // Set up probe (pulse 2)
    drive_params.E0_2 = pulse_amp;
    drive_params.omega_2 = pulse_freq;
    drive_params.t_2 = t_B_2;
    drive_params.sigma_2 = pulse_width;
    drive_params.phi_2 = 0.0;
    drive_params.theta_2 = polarization_2;

    MagTrajectory trajectory;
    ODEState state = spins_to_state();
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) { this->ode_system(x, dxdt, t); };
    // Samples on the grid T_start + k step_size (k >= 1) and at T_end; segment
    // boundaries of the chunked integration are not sampled twice.
    double last_save_time = T_start;
    auto observer = [&](const ODEState& x, double t) {
        if (t - last_save_time >= step_size - 1e-10 || (t >= T_end - 1e-10 && t > last_save_time + 1e-10)) {
            trajectory.push_back({t, magnetization_observables(x.data())});
            last_save_time = t;
        }
    };

    // One integration on the exact grid T_start + k step_size. (The former pulse-window
    // chunking restarted the integrator at every window edge without ever enlarging the step
    // — every chunk used step_size — and a chunk end missed by integrate_const's epsilon
    // restarted the next chunk from a stale state; pulse_window_chunking is ignored.)
    (void)pulse_window_chunking;
    integrate_ode_system(system_func, state, T_start, T_end, step_size,
                         observer, method, false, abs_tol, rel_tol);

    drive_params.E0_1 = 0.0;
    drive_params.E0_2 = 0.0;
    state_to_spins(state);
    return trajectory;
}

// ============================================================
// 2DCS / pump-probe optimisation helpers (Ingredient XV).
// ============================================================

double PhononLattice::max_dSdt_norm_no_drive() const {
    // W1 requires exact time-translation invariance of the unperturbed evolution: the whole
    // right-hand side (spins AND lattice coordinates/velocities, SLD) must vanish. A spin-only
    // test passed a non-relaxed lattice (e.g. a linear λ1 coupling at Q = 0 accelerates Q
    // immediately), giving silently wrong synthesised M1.
    if (!time_dep_spin_phonon_params.is_constant()) return std::numeric_limits<double>::infinity();
    PhononLattice probe(*this);
    probe.drive_params.E0_1 = 0.0;
    probe.drive_params.E0_2 = 0.0;
    const ODEState state = spins_to_state();
    ODEState dsdt(state_size, 0.0);
    probe.ode_system(state, dsdt, 0.0);
    double max_norm = 0.0;
    for (double v : dsdt) max_norm = std::max(max_norm, std::abs(v));
    return max_norm;
}

PhononLattice::PumpProbeTrajectory PhononLattice::synthesize_M1_from_M0(
    const PumpProbeTrajectory& M_pulse_trajectory,
    const std::array<Eigen::Vector3d, 4>& M_ground,
    double tau, double T_step) const {
    PumpProbeTrajectory M1;
    M1.reserve(M_pulse_trajectory.size());
    if (M_pulse_trajectory.empty()) return M1;

    const double T_start_M0 = M_pulse_trajectory.front().first;
    const double tau_threshold = tau + T_start_M0;
    const ptrdiff_t n = static_cast<ptrdiff_t>(M_pulse_trajectory.size());

    for (const auto& [t_i, mag_i] : M_pulse_trajectory) {
        (void) mag_i;
        if (t_i < tau_threshold) {
            M1.push_back({t_i, M_ground});
        } else {
            const double rel = (t_i - tau - T_start_M0) / T_step;
            ptrdiff_t idx = static_cast<ptrdiff_t>(std::lround(rel));
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            M1.push_back({t_i, M_pulse_trajectory[static_cast<size_t>(idx)].second});
        }
    }
    return M1;
}

void PhononLattice::pack_trajectory(const MagTrajectory& traj, vector<double>& out) {
    // 13 doubles per sample: t, M_antiferro(3), M_local(3), M_global(3), (O_custom, 0, 0)
    out.reserve(out.size() + 13 * traj.size());
    for (const auto& [t, M] : traj) {
        out.push_back(t);
        for (int k = 0; k < 4; ++k)
            for (int d = 0; d < 3; ++d) out.push_back(M[k](d));
    }
}

PhononLattice::MagTrajectory PhononLattice::unpack_trajectory(const double* buf, size_t n_times) {
    MagTrajectory traj(n_times);
    for (size_t s = 0; s < n_times; ++s) {
        const double* r = buf + 13 * s;
        traj[s].first = r[0];
        for (int k = 0; k < 4; ++k) traj[s].second[k] = Vec3(r[1 + 3 * k], r[2 + 3 * k], r[3 + 3 * k]);
    }
    return traj;
}

void PhononLattice::write_pump_probe_hdf5(const string& filename, double polarization,
                                          double pulse_amp, double pulse_width, double pulse_freq,
                                          double E_ground, const vector<double>& tau_values,
                                          const MagTrajectory& M0,
                                          const vector<MagTrajectory>& M1,
                                          const vector<MagTrajectory>& M01) const {
#ifdef HDF5_ENABLED
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group params_group = file.createGroup("/parameters");
    H5::Group ref_group = file.createGroup("/reference");
    H5::Group scan_group = file.createGroup("/delay_scan");

    auto scalar = [](H5::Group& g, const char* name, double v) {
        hsize_t dims[1] = {1};
        g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims))
         .write(&v, H5::PredType::NATIVE_DOUBLE);
    };
    scalar(params_group, "pulse_amp", pulse_amp);
    scalar(params_group, "pulse_width", pulse_width);
    scalar(params_group, "pulse_freq", pulse_freq);
    scalar(params_group, "polarization", polarization);
    scalar(params_group, "E_ground", E_ground);
    scalar(params_group, "legacy_kitaev_frame", frame() == Frame::Legacy ? 1.0 : 0.0);
    {
        hsize_t dims[1] = {tau_values.size()};
        scan_group.createDataSet("tau_values", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims))
                  .write(tau_values.data(), H5::PredType::NATIVE_DOUBLE);
    }
    // time, M_antiferro, M_local, M_global, O_custom for one trajectory
    auto write_traj = [](H5::Group& g, const MagTrajectory& traj) {
        const size_t n = traj.size();
        vector<double> times(n), O(n);
        vector<double> M[3] = {vector<double>(3 * n), vector<double>(3 * n), vector<double>(3 * n)};
        for (size_t t = 0; t < n; ++t) {
            times[t] = traj[t].first;
            for (int k = 0; k < 3; ++k)
                for (int d = 0; d < 3; ++d) M[k][3 * t + d] = traj[t].second[k](d);
            O[t] = traj[t].second[3](0);
        }
        hsize_t d1[1] = {n}, d2[2] = {n, 3};
        H5::DataSpace s1(1, d1), s2(2, d2);
        g.createDataSet("time", H5::PredType::NATIVE_DOUBLE, s1).write(times.data(), H5::PredType::NATIVE_DOUBLE);
        static const char* names[3] = {"M_antiferro", "M_local", "M_global"};
        for (int k = 0; k < 3; ++k)
            g.createDataSet(names[k], H5::PredType::NATIVE_DOUBLE, s2).write(M[k].data(), H5::PredType::NATIVE_DOUBLE);
        g.createDataSet("O_custom", H5::PredType::NATIVE_DOUBLE, s1).write(O.data(), H5::PredType::NATIVE_DOUBLE);
    };
    write_traj(ref_group, M0);
    for (size_t i = 0; i < tau_values.size(); ++i) {
        H5::Group tau_group = file.createGroup("/delay_scan/tau_" + std::to_string(i));
        scalar(tau_group, "tau", tau_values[i]);
        H5::Group g1 = tau_group.createGroup("M1");
        write_traj(g1, M1[i]);
        H5::Group g01 = tau_group.createGroup("M01");
        write_traj(g01, M01[i]);
    }
    file.close();
    cout << "Wrote " << filename << endl;
#else
    (void)filename; (void)polarization; (void)pulse_amp; (void)pulse_width; (void)pulse_freq;
    (void)E_ground; (void)tau_values; (void)M0; (void)M1; (void)M01;
    cout << "Warning: HDF5 support not enabled; pump-probe data not written." << endl;
#endif
}

namespace {
int count_tau_points(double tau_start, double tau_end, double tau_step) {
    if (!(tau_step != 0.0) || !std::isfinite(tau_step))
        throw std::invalid_argument("pump_probe_spectroscopy: tau_step must be non-zero and finite");
    if ((tau_end - tau_start) * tau_step < 0.0)
        throw std::invalid_argument("pump_probe_spectroscopy: tau_step points away from tau_end");
    return static_cast<int>(std::floor(std::abs((tau_end - tau_start) / tau_step) + 1e-9)) + 1;
}
}  // namespace

void PhononLattice::pump_probe_spectroscopy(
    double polarization,
    double pulse_amp, double pulse_width, double pulse_freq,
    double tau_start, double tau_end, double tau_step,
    double T_start, double T_end, double T_step,
    const string& dir_name, const string& method,
    bool reuse_m0_for_m1,
    double stationarity_tol,
    int outer_omp_threads,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol) {

    const int tau_steps = count_tau_points(tau_start, tau_end, tau_step);
    std::filesystem::create_directories(dir_name);

    cout << "\n==========================================" << endl;
    cout << "Pump-Probe Spectroscopy (PhononLattice)" << endl;
    cout << "==========================================" << endl;
    cout << "Pulse parameters:" << endl;
    cout << "  Amplitude: " << pulse_amp << endl;
    cout << "  Width: " << pulse_width << endl;
    cout << "  Frequency: " << pulse_freq << endl;
    cout << "  Polarization: " << polarization << " rad" << endl;
    cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
    cout << "Integration time: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
    cout << "Optimisations: W1(reuse_m0_for_m1=" << (reuse_m0_for_m1 ? "on" : "off")
         << ", tol=" << stationarity_tol << "), "
         << "W2(outer_omp_threads=" << outer_omp_threads << "), "
         << "W3(pulse_window_chunking: ignored, exact-grid integration)" << endl;

    cout << "\n[1/3] Using current configuration as ground state..." << endl;
    const double E_ground = energy_density();
    cout << "  Ground state: E/N = " << E_ground << ", |M| = " << magnetization_local().norm() << endl;
    cout << "  Global (crystal-frame) magnetization: " << magnetization_global().transpose() << endl;

    set_ordering_pattern();
    save_positions(dir_name + "/positions.txt");
    save_spin_config(dir_name + "/spins_initial.txt");

    const SpinConfig ground_state = spins;
    const PhononState ground_phonons = phonons;
    const auto ground_modes = modes;
    const auto ground_u = u_site, ground_p = p_site;
    auto restore = [&](PhononLattice& L) {
        L.spins = ground_state; L.phonons = ground_phonons; L.modes = ground_modes;
        L.u_site = ground_u; L.p_site = ground_p;
    };

    // Ground-state observables in the layout of the drive observers
    const ODEState x_ground = spins_to_state();
    const std::array<Eigen::Vector3d, 4> M_ground_arr = magnetization_observables(x_ground.data());

    bool can_reuse_m0 = false;
    if (reuse_m0_for_m1) {
        const double max_rhs = max_dSdt_norm_no_drive();
        cout << "  [W1 guard] max |RHS|_inf at ground state = " << max_rhs
             << " (tol = " << stationarity_tol << ")" << endl;
        can_reuse_m0 = (max_rhs <= stationarity_tol);
        cout << (can_reuse_m0 ? "  [W1] State is stationary — synthesising M1(τ) from M0 by time-shift."
                              : "  [W1] State NOT stationary — integrating M1 for every τ.") << endl;
    }

    cout << "\n[2/3] Running reference single-pulse dynamics (M0)..." << endl;
    const MagTrajectory M0_trajectory = single_pulse_drive(polarization, 0.0,
                                            pulse_amp, pulse_width, pulse_freq,
                                            T_start, T_end, T_step, method,
                                            pulse_window_chunking, abs_tol, rel_tol);
    restore(*this);

    cout << "\n[3/3] Scanning delay times (" << tau_steps << " steps)..." << endl;
    vector<MagTrajectory> M1_trajectories(tau_steps);
    vector<MagTrajectory> M01_trajectories(tau_steps);
    vector<double> tau_values(tau_steps);
    for (int i = 0; i < tau_steps; ++i) tau_values[i] = tau_start + i * tau_step;

    auto run_tau = [&](PhononLattice& L, int i) {
        const double tau = tau_values[i];
        if (can_reuse_m0) {
            M1_trajectories[i] = synthesize_M1_from_M0(M0_trajectory, M_ground_arr, tau, T_step);
        } else {
            restore(L);
            M1_trajectories[i] = L.single_pulse_drive(polarization, tau, pulse_amp, pulse_width, pulse_freq,
                                                      T_start, T_end, T_step, method,
                                                      pulse_window_chunking, abs_tol, rel_tol);
        }
        restore(L);
        M01_trajectories[i] = L.double_pulse_drive(polarization, 0.0, polarization, tau,
                                                   pulse_amp, pulse_width, pulse_freq,
                                                   T_start, T_end, T_step, method,
                                                   pulse_window_chunking, abs_tol, rel_tol);
    };

    // ----- W2: outer OpenMP parallelism over τ (one deep copy of the lattice per thread) -----
    int n_outer = 1;
#ifdef _OPENMP
    n_outer = (outer_omp_threads <= 0) ? std::max(1, omp_get_max_threads()) : outer_omp_threads;
    n_outer = std::min(n_outer, std::max(1, tau_steps));
    const int saved_max_active = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
#else
    (void) outer_omp_threads;
#endif
    if (n_outer <= 1) {
        for (int i = 0; i < tau_steps; ++i) {
            cout << "--- Delay " << (i + 1) << "/" << tau_steps << ": tau = " << tau_values[i] << endl;
            run_tau(*this, i);
        }
    } else {
#ifdef _OPENMP
        cout << "  [W2] Distributing " << tau_steps << " τ points across " << n_outer << " OpenMP threads..." << endl;
        #pragma omp parallel num_threads(n_outer)
        {
            PhononLattice local_lat(*this);
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < tau_steps; ++i) run_tau(local_lat, i);
        }
#endif
    }
#ifdef _OPENMP
    omp_set_max_active_levels(saved_max_active);
#endif

    write_pump_probe_hdf5(dir_name + "/pump_probe_spectroscopy.h5", polarization, pulse_amp, pulse_width,
                          pulse_freq, E_ground, tau_values, M0_trajectory, M1_trajectories, M01_trajectories);
    restore(*this);

    cout << "\n==========================================" << endl;
    cout << "Pump-Probe Spectroscopy Complete!" << endl;
    cout << "Output directory: " << dir_name << endl;
    cout << "Total delay points: " << tau_steps << endl;
    cout << "==========================================" << endl;
}

void PhononLattice::broadcast_state(int root, MPI_Comm comm) {
    ODEState x = spins_to_state();
    unsigned long long n = x.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
    if (n != x.size()) {
        // Every rank must have the same lattice sector layout (modes, SLD) — abort the job
        // rather than leave the other ranks blocked in a later collective.
        std::cerr << "broadcast_state: state size mismatch across ranks (" << n << " vs " << x.size() << ")" << std::endl;
        MPI_Abort(comm, 2);
    }
    MPI_Bcast(x.data(), static_cast<int>(n), MPI_DOUBLE, root, comm);
    state_to_spins(x);
}

void PhononLattice::pump_probe_spectroscopy_mpi(
    double polarization,
    double pulse_amp, double pulse_width, double pulse_freq,
    double tau_start, double tau_end, double tau_step,
    double T_start, double T_end, double T_step,
    const string& dir_name, const string& method,
    bool reuse_m0_for_m1,
    double stationarity_tol,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol,
    MPI_Comm comm) {

    int rank = 0, mpi_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);

    // Every rank validates the same arguments, so either all ranks throw or none does.
    const int tau_steps = count_tau_points(tau_start, tau_end, tau_step);

    if (rank == 0) {
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy (MPI Parallel)" << endl;
        cout << "==========================================" << endl;
        cout << "MPI ranks: " << mpi_size << endl;
        cout << "Pulse parameters:" << endl;
        cout << "  Amplitude: " << pulse_amp << endl;
        cout << "  Width: " << pulse_width << endl;
        cout << "  Frequency: " << pulse_freq << endl;
        cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
        cout << "Total delay points: " << tau_steps << endl;
        cout << "Tau points per rank: ~" << (tau_steps + mpi_size - 1) / mpi_size << endl;
        cout << "Optimisations: W1(reuse_m0_for_m1=" << (reuse_m0_for_m1 ? "on" : "off")
             << ", tol=" << stationarity_tol << "), "
             << "W3(pulse_window_chunking: ignored, exact-grid integration)" << endl;
    }

    // Work done by rank 0 alone (files, the M0 reference) is wrapped so that a failure is
    // broadcast instead of leaving the other ranks blocked in the next collective.
    auto rank0_step = [&](const char* what, const std::function<void()>& f) {
        int ok = 1;
        std::string msg;
        if (rank == 0) {
            try { f(); } catch (const std::exception& e) { ok = 0; msg = e.what(); }
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
        if (!ok) throw std::runtime_error(std::string("pump_probe_spectroscopy_mpi: ") + what + " failed on rank 0" +
                                          (msg.empty() ? std::string() : ": " + msg));
    };

    const double E_ground = energy_density();
    rank0_step("writing the initial state", [&] {
        std::filesystem::create_directories(dir_name);
        cout << "\n[1/4] Using current configuration as ground state..." << endl;
        cout << "  Ground state: E/N = " << E_ground << endl;
        save_positions(dir_name + "/positions.txt");
        save_spin_config(dir_name + "/spins_initial.txt");
    });

    // Set ordering pattern from current config on every rank so that
    // local M0 / M1 / M01 observers compute identical custom O.
    set_ordering_pattern();

    const SpinConfig ground_state = spins;
    const PhononState ground_phonons = phonons;
    const auto ground_modes = modes;
    const auto ground_u = u_site, ground_p = p_site;
    auto restore = [&]() {
        spins = ground_state; phonons = ground_phonons; modes = ground_modes;
        u_site = ground_u; p_site = ground_p;
    };

    const ODEState x_ground = spins_to_state();
    const std::array<Eigen::Vector3d, 4> M_ground_arr = magnetization_observables(x_ground.data());

    // W1 verdict: evaluated on rank 0, broadcast so all ranks take the same branch.
    int can_reuse_m0 = 0;
    if (reuse_m0_for_m1) {
        rank0_step("the W1 stationarity check", [&] {
            const double max_rhs = max_dSdt_norm_no_drive();
            cout << "  [W1 guard] max |RHS|_inf at ground state = " << max_rhs
                 << " (tol = " << stationarity_tol << ")" << endl;
            can_reuse_m0 = (max_rhs <= stationarity_tol) ? 1 : 0;
        });
        MPI_Bcast(&can_reuse_m0, 1, MPI_INT, 0, comm);
        if (rank == 0)
            cout << (can_reuse_m0 ? "  [W1] State is stationary — M1(τ) will be synthesised from M0."
                                  : "  [W1] State NOT stationary — every rank will integrate fresh M1.") << endl;
    }

    MagTrajectory M0_trajectory;
    rank0_step("the reference trajectory M0", [&] {
        cout << "\n[2/4] Computing reference trajectory (M0)..." << endl;
        M0_trajectory = single_pulse_drive(polarization, 0.0,
                                           pulse_amp, pulse_width, pulse_freq,
                                           T_start, T_end, T_step, method,
                                           pulse_window_chunking, abs_tol, rel_tol);
        restore();
    });

    // If we will synthesise M1 from M0 on remote ranks, broadcast M0 (13 doubles per sample).
    if (can_reuse_m0) {
        vector<double> buf;
        if (rank == 0) pack_trajectory(M0_trajectory, buf);
        unsigned long long n_times = M0_trajectory.size();
        MPI_Bcast(&n_times, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
        buf.resize(13 * n_times);
        MPI_Bcast(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, 0, comm);
        if (rank != 0) M0_trajectory = unpack_trajectory(buf.data(), n_times);
    }

    // Contiguous block of delay indices per rank
    vector<int> tau_counts(mpi_size), tau_offsets(mpi_size);
    const int base_count = tau_steps / mpi_size;
    const int remainder = tau_steps % mpi_size;
    for (int r = 0; r < mpi_size; ++r) {
        tau_counts[r] = base_count + (r < remainder ? 1 : 0);
        tau_offsets[r] = (r == 0) ? 0 : tau_offsets[r - 1] + tau_counts[r - 1];
    }
    const int my_start = tau_offsets[rank];
    const int my_count = tau_counts[rank];

    if (rank == 0) cout << "\n[3/4] Parallel delay scan..." << endl;

    // Local work; an exception on one rank must not leave the others blocked in the
    // gather below, so failures are reduced to a flag first.
    vector<MagTrajectory> my_M1(my_count), my_M01(my_count);
    int local_fail = 0;
    std::string fail_msg;
    try {
        for (int i = 0; i < my_count; ++i) {
            const double tau = tau_start + (my_start + i) * tau_step;
            cout << "[Rank " << rank << "] tau = " << tau << " (" << (i + 1) << "/" << my_count << ")" << endl;
            if (can_reuse_m0) {
                my_M1[i] = synthesize_M1_from_M0(M0_trajectory, M_ground_arr, tau, T_step);
            } else {
                restore();
                my_M1[i] = single_pulse_drive(polarization, tau, pulse_amp, pulse_width, pulse_freq,
                                              T_start, T_end, T_step, method,
                                              pulse_window_chunking, abs_tol, rel_tol);
            }
            restore();
            my_M01[i] = double_pulse_drive(polarization, 0.0, polarization, tau,
                                           pulse_amp, pulse_width, pulse_freq,
                                           T_start, T_end, T_step, method,
                                           pulse_window_chunking, abs_tol, rel_tol);
        }
    } catch (const std::exception& e) {
        local_fail = 1;
        fail_msg = e.what();
    }
    restore();
    int any_fail = 0;
    MPI_Allreduce(&local_fail, &any_fail, 1, MPI_INT, MPI_MAX, comm);
    if (any_fail) {
        if (local_fail) std::cerr << "[Rank " << rank << "] pump_probe_spectroscopy_mpi failed: " << fail_msg << std::endl;
        throw std::runtime_error("pump_probe_spectroscopy_mpi: a rank failed during the delay scan" +
                                 (local_fail ? ": " + fail_msg : std::string()));
    }

    // ---- Gather the FULL records (t + M_antiferro + M_local + M_global + O_custom) ----
    // Per rank: lengths of its (M1, M01) trajectories, then the packed samples.
    vector<int> my_lengths;
    vector<double> my_buf;
    for (int i = 0; i < my_count; ++i) {
        my_lengths.push_back(static_cast<int>(my_M1[i].size()));
        my_lengths.push_back(static_cast<int>(my_M01[i].size()));
        pack_trajectory(my_M1[i], my_buf);
        pack_trajectory(my_M01[i], my_buf);
    }
    vector<int> len_counts(mpi_size), len_displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        len_counts[r] = 2 * tau_counts[r];
        len_displs[r] = 2 * tau_offsets[r];
    }
    vector<int> all_lengths(rank == 0 ? 2 * tau_steps : 0);
    MPI_Gatherv(my_lengths.data(), static_cast<int>(my_lengths.size()), MPI_INT,
                all_lengths.data(), len_counts.data(), len_displs.data(), MPI_INT, 0, comm);

    int my_n = static_cast<int>(my_buf.size());
    vector<int> buf_counts(rank == 0 ? mpi_size : 0), buf_displs(rank == 0 ? mpi_size : 0);
    MPI_Gather(&my_n, 1, MPI_INT, buf_counts.data(), 1, MPI_INT, 0, comm);
    vector<double> all_buf;
    if (rank == 0) {
        size_t total = 0;
        for (int r = 0; r < mpi_size; ++r) { buf_displs[r] = static_cast<int>(total); total += buf_counts[r]; }
        all_buf.resize(total);
    }
    MPI_Gatherv(my_buf.data(), my_n, MPI_DOUBLE, all_buf.data(), buf_counts.data(), buf_displs.data(),
                MPI_DOUBLE, 0, comm);

    rank0_step("writing the spectroscopy file", [&] {
        cout << "\n[4/4] Gathering results and writing output..." << endl;
        vector<double> tau_values(tau_steps);
        vector<MagTrajectory> M1(tau_steps), M01(tau_steps);
        size_t pos = 0;
        for (int g = 0; g < tau_steps; ++g) {
            tau_values[g] = tau_start + g * tau_step;
            const size_t n1 = static_cast<size_t>(all_lengths[2 * g]);
            const size_t n01 = static_cast<size_t>(all_lengths[2 * g + 1]);
            M1[g] = unpack_trajectory(all_buf.data() + pos, n1);
            pos += 13 * n1;
            M01[g] = unpack_trajectory(all_buf.data() + pos, n01);
            pos += 13 * n01;
        }
        write_pump_probe_hdf5(dir_name + "/pump_probe_spectroscopy.h5", polarization, pulse_amp, pulse_width,
                              pulse_freq, E_ground, tau_values, M0_trajectory, M1, M01);
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Complete!" << endl;
        cout << "==========================================" << endl;
    });
}

// Explicit template instantiation
template void PhononLattice::integrate_ode_system(
    std::function<void(const PhononLattice::ODEState&, PhononLattice::ODEState&, double)>,
    PhononLattice::ODEState&, double, double, double,
    std::function<void(const PhononLattice::ODEState&, double)>,
    const std::string&, bool, double, double);
