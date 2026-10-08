#ifndef UNITCELL_REFACTORED_H
#define UNITCELL_REFACTORED_H

#include "simple_linear_alg.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <map>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Bilinear interaction structure (replaces template version)
struct Bilinear {
    SpinMatrix interaction;  // N×N matrix
    size_t partner;
    Vector3i offset;  // Offset in lattice coordinates
    int bond_type;  // Bond type metadata (-1 = unspecified, 0/1/2 = x/y/z for Kitaev, etc.)
    
    Bilinear() : partner(SIZE_MAX), offset(Vector3i::Zero()), bond_type(-1) {}
    
    Bilinear(const SpinMatrix& J, size_t p) 
        : interaction(J), partner(p), offset(Vector3i::Zero()), bond_type(-1) {}
    
    Bilinear(const SpinMatrix& J, size_t p, const Vector3i& o)
        : interaction(J), partner(p), offset(o), bond_type(-1) {}
    
    Bilinear(const SpinMatrix& J, size_t p, const Vector3i& o, int bt)
        : interaction(J), partner(p), offset(o), bond_type(bt) {}
};

// Trilinear interaction structure (replaces template version)
struct Trilinear {
    SpinTensor3 interaction;  // N×N×N tensor stored as vector of matrices
    size_t partner1;
    size_t partner2;
    Vector3i offset1;
    Vector3i offset2;
    
    Trilinear() : partner1(SIZE_MAX), partner2(SIZE_MAX), 
                  offset1(Vector3i::Zero()), offset2(Vector3i::Zero()) {}
    
    Trilinear(const SpinTensor3& K, size_t p1, size_t p2)
        : interaction(K), partner1(p1), partner2(p2),
          offset1(Vector3i::Zero()), offset2(Vector3i::Zero()) {}
    
    Trilinear(const SpinTensor3& K, size_t p1, size_t p2, 
              const Vector3i& o1, const Vector3i& o2)
        : interaction(K), partner1(p1), partner2(p2), offset1(o1), offset2(o2) {}
};

// Mixed bilinear for different spin dimensions (e.g., SU2-SU3 coupling)
struct MixedBilinear {
    SpinMatrix interaction;  // N_SU2 × N_SU3 matrix (source dim × partner dim)
    size_t partner;          // Partner sublattice index (in the SU3 lattice)
    Vector3i offset;         // Cell offset from source to partner
    
    MixedBilinear() : partner(SIZE_MAX), offset(Vector3i::Zero()) {}
    
    MixedBilinear(const SpinMatrix& J, size_t p, const Vector3i& o)
        : interaction(J), partner(p), offset(o) {}
};

// Time-dependent (pulse-envelope-modulated) mixed bilinear, used for the
// field-assisted Fe-Tm exchange terms H_{E chi} and H_{B chi} of the TmFeO3
// model (tmfeo3_foundation.tex, Eqs. H_E_chi_reorg / H_B_chi_reorg).  The
// stored tensor is the static part chi^{assist}_{alpha a}; at runtime the
// coupling is multiplied by the corresponding Gaussian x carrier pulse
// envelope so that H = envelope(t) * S^alpha_Fe * chi^{assist}_{alpha a} *
// lambda^a_Tm.  `envelope` selects which pulse drives the modulation:
//   0 -> SU(3) pulse envelope (the electric / THz field E(t), for H_{E chi})
//   1 -> SU(2) pulse envelope (the magnetic field B(t),       for H_{B chi})
//   2, 3, 4 -> the B_x, B_y, B_z(t) component of the SU(2) pulse
struct MixedBilinearDrive {
    SpinMatrix interaction;  // N_SU2 × N_SU3 matrix (source dim × partner dim)
    size_t partner;          // Partner sublattice index (in the SU3 lattice)
    Vector3i offset;         // Cell offset from source to partner
    int envelope;            // 0 = E(t), 1 = |B(t)|, 2..4 = B_x/B_y/B_z(t)

    MixedBilinearDrive() : partner(SIZE_MAX), offset(Vector3i::Zero()), envelope(0) {}

    MixedBilinearDrive(const SpinMatrix& J, size_t p, const Vector3i& o, int env)
        : interaction(J), partner(p), offset(o), envelope(env) {}
};

// Mixed trilinear for different spin dimensions
struct MixedTrilinear {
    SpinTensor3 interaction;  // Tensor with mixed dimensions
    size_t partner1;
    size_t partner2;
    Vector3i offset1;
    Vector3i offset2;
    
    MixedTrilinear() : partner1(SIZE_MAX), partner2(SIZE_MAX),
                       offset1(Vector3i::Zero()), offset2(Vector3i::Zero()) {}
    
    MixedTrilinear(const SpinTensor3& K, size_t p1, size_t p2,
                   const Vector3i& o1, const Vector3i& o2)
        : interaction(K), partner1(p1), partner2(p2), offset1(o1), offset2(o2) {}
};

// Unit cell class (replaces template version)
class UnitCell {
public:
    size_t N;  // Spin dimension (3 for SU2, 8 for SU3, etc.)
    size_t N_atoms;  // Number of atoms in unit cell
    
    vector<Vector3d> lattice_pos;  // Atomic positions
    vector<Vector3d> lattice_vectors;  // 3 lattice vectors
    vector<SpinMatrix> sublattice_frames;  // Local coordinate frames for each atom
    vector<double> afm_sublattice_signs;  // Signs for staggered magnetization per sublattice
    
    vector<SpinVector> field;  // External field per atom
    vector<SpinMatrix> onsite_interaction;  // On-site interactions per atom

    // Coefficient c of the spin Poisson bracket, dS_a/dt = c f_abc (dE/dS_b) S_c:
    // 1 for SU(2) spins S = <S_op> ([S_a, S_b] = i eps_abc S_c), 2 for SU(3)
    // Gell-Mann expectations n = <lambda> ([lambda_a, lambda_b] = 2i f_abc lambda_c).
    // Part of the model definition because it fixes how the couplings of the
    // unit cell act in the dynamics (see core/su3_coherent_state.h).
    double poisson_bracket = 1.0;
    
    multimap<int, Bilinear> bilinear_interaction;
    multimap<int, Trilinear> trilinear_interaction;
    
    // Constructors
    UnitCell(size_t spin_dim, size_t num_atoms,
             const vector<Vector3d>& positions,
             const vector<Vector3d>& vectors)
        : N(spin_dim), N_atoms(num_atoms),
          lattice_pos(positions), lattice_vectors(vectors) {
        
        if (positions.size() != num_atoms) {
            throw std::invalid_argument("Number of positions must match N_atoms");
        }
        if (vectors.size() != 3) {
            throw std::invalid_argument("Must provide exactly 3 lattice vectors");
        }
        
        // Initialize fields and interactions to zero
        field.resize(N_atoms, SpinVector::Zero(N));
        onsite_interaction.resize(N_atoms, SpinMatrix::Zero(N, N));
        sublattice_frames.resize(N_atoms, SpinMatrix::Identity(N, N));
        afm_sublattice_signs.resize(N_atoms, 1.0);
        poisson_bracket = (N == 8) ? 2.0 : 1.0;
    }
    
    UnitCell(size_t spin_dim, size_t num_atoms)
        : N(spin_dim), N_atoms(num_atoms) {
        lattice_pos.resize(num_atoms, Vector3d::Zero());
        lattice_vectors.resize(3, Vector3d::Zero());
        field.resize(N_atoms, SpinVector::Zero(N));
        onsite_interaction.resize(N_atoms, SpinMatrix::Zero(N, N));
        sublattice_frames.resize(N_atoms, SpinMatrix::Identity(N, N));
        afm_sublattice_signs.resize(N_atoms, 1.0);
        poisson_bracket = (N == 8) ? 2.0 : 1.0;
    }
    
    // Setters
    void set_lattice_pos(const Vector3d& pos, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        lattice_pos[index] = pos;
    }
    
    void set_lattice_vector(const Vector3d& vec, size_t index) {
        if (index >= 3) throw out_of_range("Lattice vector index must be 0-2");
        lattice_vectors[index] = vec;
    }
    
    void set_field(const SpinVector& f, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        if (size_t(f.size()) != N) throw invalid_argument("Field dimension mismatch");
        require_finite(f, "field of atom " + std::to_string(index));
        field[index] = f;
    }
    
    void set_bilinear_interaction(const SpinMatrix& J, size_t source, 
                                  size_t partner, const Vector3i& offset) {
        set_bilinear_interaction(J, source, partner, offset, -1);
    }
    
    /**
     * Declare the bond E = S_source(R)^T J S_partner(R + offset). Each bond is
     * declared ONCE: the lattice adds the reverse bond with J^T itself, so
     * declaring both directions double counts (validate() rejects that).
     */
    void set_bilinear_interaction(const SpinMatrix& J, size_t source, 
                                  size_t partner, const Vector3i& offset,
                                  int bond_type) {
        if (size_t(J.rows()) != N || size_t(J.cols()) != N) {
            throw invalid_argument("Bilinear matrix dimension mismatch");
        }
        check_atom(source, "bilinear source");
        check_atom(partner, "bilinear partner");
        require_finite(J, "bilinear coupling " + bond_label(source, partner, offset));
        bilinear_interaction.insert(make_pair(source, Bilinear(J, partner, offset, bond_type)));
    }
    
    void set_trilinear_interaction(const SpinTensor3& K, size_t source,
                                   size_t partner1, size_t partner2,
                                   const Vector3i& offset1, const Vector3i& offset2) {
        if (K.size() != N) {
            throw invalid_argument("Trilinear tensor dimension mismatch");
        }
        for (const auto& slice : K) {
            if (size_t(slice.rows()) != N || size_t(slice.cols()) != N)
                throw invalid_argument("Trilinear tensor slice dimension mismatch (each slice must be N x N)");
            require_finite(slice, "trilinear coupling on atom " + std::to_string(source));
        }
        check_atom(source, "trilinear source");
        check_atom(partner1, "trilinear partner1");
        check_atom(partner2, "trilinear partner2");
        trilinear_interaction.insert(make_pair(source, 
            Trilinear(K, partner1, partner2, offset1, offset2)));
    }
    
    /**
     * On-site term S^T A S. Only the symmetric part of A contributes to the
     * energy (S^T A S = S^T A_s S with A_s = (A + A^T)/2), so A_s is stored:
     * the local-field and ΔE kernels may then use 2 A S = dE/dS.
     */
    void set_onsite_interaction(const SpinMatrix& A, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        if (size_t(A.rows()) != N || size_t(A.cols()) != N) {
            throw invalid_argument("Onsite matrix dimension mismatch");
        }
        require_finite(A, "on-site matrix of atom " + std::to_string(index));
        onsite_interaction[index] = 0.5 * (A + A.transpose());
    }
    
    void set_sublattice_frame(const SpinMatrix& frame, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        if (size_t(frame.rows()) != N || size_t(frame.cols()) != N) {
            throw invalid_argument("Frame matrix dimension mismatch");
        }
        sublattice_frames[index] = frame;
    }
    
    void set_afm_sublattice_signs(const vector<double>& signs) {
        if (signs.size() != N_atoms) throw invalid_argument("AFM signs size must match N_atoms");
        afm_sublattice_signs = signs;
    }

    // ------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------

    /// Real-space vector from atom `source` in the home cell to atom
    /// `partner` in the cell displaced by `offset` (lattice coordinates).
    Vector3d bond_vector(size_t source, size_t partner, const Vector3i& offset) const {
        return double(offset[0]) * lattice_vectors[0] + double(offset[1]) * lattice_vectors[1] +
               double(offset[2]) * lattice_vectors[2] + lattice_pos[partner] - lattice_pos[source];
    }

    /// One bond of a neighbour shell (see bonds_at_distance).
    struct BondGeometry {
        size_t source;
        size_t partner;
        Vector3i offset;
        Vector3d vector;
    };

    /**
     * Every bond (source, partner, offset) of length `distance` (to `tol`),
     * each unordered bond listed exactly once — source < partner, or
     * source == partner with the first nonzero offset component positive.
     * That is precisely the list a builder declares, since the lattice adds
     * the reverse of every declared bond itself. Generating shells this way
     * (instead of hand-written offset tables) keeps builders from wiring a
     * bond to the wrong neighbour.
     */
    vector<BondGeometry> bonds_at_distance(double distance, double tol = 1e-8) const {
        if (!(distance > 0.0)) throw invalid_argument("bonds_at_distance: distance must be > 0");
        // |n_d| <= (|d| + max |r_p - r_s|) |b_d|, b_d = rows of A^{-1} (A = [a1 a2 a3]).
        double r_max = 0.0;
        for (size_t a = 0; a < N_atoms; ++a)
            for (size_t b = 0; b < N_atoms; ++b) r_max = std::max(r_max, (lattice_pos[b] - lattice_pos[a]).norm());
        const Eigen::Matrix3d B = lattice_matrix().inverse();
        int n_max[3];
        for (int d = 0; d < 3; ++d)
            n_max[d] = int(std::ceil((distance + tol + r_max) * B.row(d).norm())) + 1;
        vector<BondGeometry> out;
        for (size_t s = 0; s < N_atoms; ++s)
            for (size_t p = s; p < N_atoms; ++p)
                for (int i = -n_max[0]; i <= n_max[0]; ++i)
                    for (int j = -n_max[1]; j <= n_max[1]; ++j)
                        for (int k = -n_max[2]; k <= n_max[2]; ++k) {
                            const Vector3i n(i, j, k);
                            if (s == p && !lexicographically_positive(n)) continue;
                            const Vector3d d = bond_vector(s, p, n);
                            if (std::abs(d.norm() - distance) <= tol) out.push_back({s, p, n, d});
                        }
        return out;
    }

    /// Sorted distinct bond lengths of the first `n_shells` neighbour shells.
    vector<double> neighbour_shells(size_t n_shells, double tol = 1e-8) const {
        const Eigen::Matrix3d B = lattice_matrix().inverse();
        vector<double> shells;
        for (int reach = 1; reach <= 64; ++reach) {
            vector<double> lengths;
            for (size_t s = 0; s < N_atoms; ++s)
                for (size_t p = 0; p < N_atoms; ++p)
                    for (int i = -reach; i <= reach; ++i)
                        for (int j = -reach; j <= reach; ++j)
                            for (int k = -reach; k <= reach; ++k) {
                                const double r = bond_vector(s, p, Vector3i(i, j, k)).norm();
                                if (r > tol) lengths.push_back(r);
                            }
            std::sort(lengths.begin(), lengths.end());
            shells.clear();
            for (double r : lengths)
                if (shells.empty() || r - shells.back() > tol) shells.push_back(r);
            // Complete once the search box holds a sphere beyond the last wanted shell
            // (minus the largest intra-cell separation).
            double r_max = 0.0;
            for (size_t a = 0; a < N_atoms; ++a)
                for (size_t b = 0; b < N_atoms; ++b) r_max = std::max(r_max, (lattice_pos[b] - lattice_pos[a]).norm());
            double box = 1e300;
            for (int d = 0; d < 3; ++d) box = std::min(box, double(reach) / B.row(d).norm());
            if (shells.size() >= n_shells && shells[n_shells - 1] < box - r_max - 1e-12) break;
        }
        if (shells.size() > n_shells) shells.resize(n_shells);
        return shells;
    }

    /// True if a lattice site (any atom, any cell) sits at real-space position r.
    bool has_site_at(const Vector3d& r, double tol = 1e-8) const {
        const Eigen::Matrix3d A = lattice_matrix();
        const Eigen::Matrix3d B = A.inverse();
        for (size_t a = 0; a < N_atoms; ++a) {
            const Vector3d f = B * (r - lattice_pos[a]);
            const Vector3d n = f.array().round().matrix();
            if ((A * (f - n)).norm() <= tol) return true;
        }
        return false;
    }

    /**
     * Consistency check, called by the lattice constructor. Throws
     * std::invalid_argument (std::out_of_range for indices) naming the
     * offending entry: atom indices out of range, wrong matrix/vector sizes,
     * non-finite entries, degenerate lattice vectors, and a bilinear bond
     * declared again as its own reverse (the lattice adds every reverse bond
     * with J^T, so declaring both directions silently doubles the coupling).
     * Several terms on one bond in the SAME orientation are fine: they add.
     */
    void validate() const {
        if (N == 0 || N_atoms == 0) throw invalid_argument("UnitCell: spin dimension and N_atoms must be >= 1");
        if (lattice_pos.size() != N_atoms || lattice_vectors.size() != 3)
            throw invalid_argument("UnitCell: need N_atoms positions and 3 lattice vectors");
        if (!(std::abs(lattice_matrix().determinant()) > 1e-12))
            throw invalid_argument("UnitCell: lattice vectors are linearly dependent");
        if (field.size() != N_atoms || onsite_interaction.size() != N_atoms || sublattice_frames.size() != N_atoms)
            throw invalid_argument("UnitCell: per-atom tables must have N_atoms entries");
        for (size_t a = 0; a < N_atoms; ++a) {
            const std::string atom = " of atom " + std::to_string(a);
            require_finite(lattice_pos[a], "position" + atom);
            if (size_t(field[a].size()) != N) throw invalid_argument("UnitCell: field" + atom + " has wrong size");
            if (size_t(onsite_interaction[a].rows()) != N || size_t(onsite_interaction[a].cols()) != N)
                throw invalid_argument("UnitCell: on-site matrix" + atom + " has wrong size");
            require_finite(field[a], "field" + atom);
            require_finite(onsite_interaction[a], "on-site matrix" + atom);
        }
        // Canonical unordered bond -> orientation it was first declared in.
        std::map<std::tuple<size_t, size_t, int, int, int>, bool> seen;
        for (const auto& [src, bi] : bilinear_interaction) {
            if (src < 0) throw out_of_range("UnitCell: negative bilinear source index");
            const size_t s = size_t(src);
            check_atom(s, "bilinear source");
            check_atom(bi.partner, "bilinear partner");
            const std::string label = bond_label(s, bi.partner, bi.offset);
            if (size_t(bi.interaction.rows()) != N || size_t(bi.interaction.cols()) != N)
                throw invalid_argument("UnitCell: bilinear matrix " + label + " has wrong size");
            require_finite(bi.interaction, "bilinear coupling " + label);
            // Canonical orientation of the unordered bond.
            size_t a = s, b = bi.partner;
            Vector3i n = bi.offset;
            const bool reversed = (a > b || (a == b && !lexicographically_positive(n)));
            if (reversed) {
                std::swap(a, b);
                n = -n;
            }
            const auto [it, inserted] = seen.insert({{a, b, n[0], n[1], n[2]}, reversed});
            if (!inserted && it->second != reversed)
                throw invalid_argument("UnitCell: bilinear bond " + label +
                                       " is also declared in the reverse direction; the lattice adds the "
                                       "reverse bond (with J^T) itself, so declare each bond in one direction");
        }
        for (const auto& [src, tri] : trilinear_interaction) {
            if (src < 0) throw out_of_range("UnitCell: negative trilinear source index");
            check_atom(size_t(src), "trilinear source");
            check_atom(tri.partner1, "trilinear partner1");
            check_atom(tri.partner2, "trilinear partner2");
            if (tri.interaction.size() != N)
                throw invalid_argument("UnitCell: trilinear tensor on atom " + std::to_string(src) + " has wrong size");
        }
    }

    // Print method for debugging
    void print() const {
        cout << "--- UnitCell Information ---" << endl;
        cout << "Number of atoms: " << N_atoms << endl;
        cout << "Spin dimension (N): " << N << endl;
        
        cout << "\nLattice Positions:" << endl;
        for (size_t i = 0; i < N_atoms; ++i) {
            cout << "  Atom " << i << ": " << lattice_pos[i].transpose() << endl;
        }
        
        cout << "\nLattice Vectors:" << endl;
        for (size_t i = 0; i < 3; ++i) {
            cout << "  v" << i + 1 << ": " << lattice_vectors[i].transpose() << endl;
        }
        
        cout << "\nFields:" << endl;
        for (size_t i = 0; i < N_atoms; ++i) {
            cout << "  Atom " << i << ": " << field[i].transpose() << endl;
        }
        
        cout << "\nBilinear Interactions: " << bilinear_interaction.size() << " total" << endl;
        cout << "Trilinear Interactions: " << trilinear_interaction.size() << " total" << endl;
        cout << "--- End of UnitCell Information ---" << endl;
    }

private:
    Eigen::Matrix3d lattice_matrix() const {
        Eigen::Matrix3d A;
        for (int d = 0; d < 3; ++d) A.col(d) = lattice_vectors[d];
        return A;
    }

    static bool lexicographically_positive(const Vector3i& n) {
        for (int d = 0; d < 3; ++d)
            if (n[d] != 0) return n[d] > 0;
        return false;
    }

    void check_atom(size_t index, const char* what) const {
        if (index >= N_atoms)
            throw out_of_range(std::string("UnitCell: ") + what + " index " + std::to_string(index) +
                               " out of range (N_atoms = " + std::to_string(N_atoms) + ")");
    }

    template<class M>
    static void require_finite(const M& m, const std::string& what) {
        if (!m.allFinite()) throw invalid_argument("UnitCell: " + what + " has non-finite entries");
    }

    static std::string bond_label(size_t s, size_t p, const Vector3i& n) {
        std::ostringstream os;
        os << s << " -> " << p << " offset (" << n[0] << "," << n[1] << "," << n[2] << ")";
        return os.str();
    }
};

// Mixed unit cell for systems with multiple spin types (e.g., TmFeO3 with Fe and Tm)
class MixedUnitCell {
public:
    UnitCell SU2_cell;
    UnitCell SU3_cell;
    
    multimap<int, MixedTrilinear> trilinear_SU2_SU3;
    multimap<int, MixedBilinear> bilinear_SU2_SU3;  // key = SU2 sublattice, partner = SU3 sublattice
    multimap<int, MixedBilinearDrive> bilinear_drive_SU2_SU3;  // pulse-modulated Fe-Tm exchange (H_{E chi}, H_{B chi})
    
    MixedUnitCell(const UnitCell& su2, const UnitCell& su3)
        : SU2_cell(su2), SU3_cell(su3) {}
    
    void set_mixed_trilinear(const SpinTensor3& K, size_t source,
                            size_t partner1, size_t partner2,
                            const Vector3i& offset1, const Vector3i& offset2) {
        check_index(source, SU2_cell.N_atoms, "mixed trilinear source (SU2)");
        check_index(partner1, SU2_cell.N_atoms, "mixed trilinear partner1 (SU2)");
        check_index(partner2, SU3_cell.N_atoms, "mixed trilinear partner2 (SU3)");
        trilinear_SU2_SU3.insert(make_pair(source,
            MixedTrilinear(K, partner1, partner2, offset1, offset2)));
    }
    
    // source = SU2 sublattice index (key), partner = SU3 sublattice index
    // J is N_SU2 × N_SU3 matrix, offset = cell displacement from source to partner
    void set_mixed_bilinear(const SpinMatrix& J, size_t source,
                           size_t partner, const Vector3i& offset) {
        check_mixed_bond(J, source, partner, "mixed bilinear");
        bilinear_SU2_SU3.insert(make_pair(source,
            MixedBilinear(J, partner, offset)));
    }

    // Pulse-envelope-modulated mixed bilinear (field-assisted Fe-Tm exchange).
    // Same geometry as set_mixed_bilinear; `envelope` picks the modulating
    // pulse (0 = SU(3)/E field, 1 = SU(2)/B field).
    void set_mixed_bilinear_drive(const SpinMatrix& J, size_t source,
                                  size_t partner, const Vector3i& offset,
                                  int envelope) {
        check_mixed_bond(J, source, partner, "mixed bilinear drive");
        if (envelope < 0 || envelope > 4)
            throw invalid_argument("mixed bilinear drive: envelope must be 0 (E), 1 (|B|) or 2..4 (B_x, B_y, B_z)");
        bilinear_drive_SU2_SU3.insert(make_pair(source,
            MixedBilinearDrive(J, partner, offset, envelope)));
    }

private:
    static void check_index(size_t index, size_t n, const char* what) {
        if (index >= n)
            throw out_of_range(std::string("MixedUnitCell: ") + what + " index " + std::to_string(index) +
                               " out of range (" + std::to_string(n) + " atoms)");
    }

    void check_mixed_bond(const SpinMatrix& J, size_t source, size_t partner, const char* what) const {
        check_index(source, SU2_cell.N_atoms, what);
        check_index(partner, SU3_cell.N_atoms, what);
        if (size_t(J.rows()) != SU2_cell.N || size_t(J.cols()) != SU3_cell.N)
            throw invalid_argument(std::string("MixedUnitCell: ") + what + " matrix must be N_SU2 x N_SU3 (" +
                                   std::to_string(SU2_cell.N) + " x " + std::to_string(SU3_cell.N) + ")");
        if (!J.allFinite()) throw invalid_argument(std::string("MixedUnitCell: ") + what + " has non-finite entries");
    }
};

// Predefined lattice structures

// Honeycomb lattice
class HoneyComb : public UnitCell {
public:
    HoneyComb(size_t spin_dim) 
        : UnitCell(spin_dim, 2,
                   {Vector3d(0, 0, 0), Vector3d(0, 1/sqrt(3.0), 0)},
                   {Vector3d(1, 0, 0), Vector3d(0.5, sqrt(3.0)/2, 0), Vector3d(0, 0, 1)}) {}
};


// Triangular Bravais lattice: ONE site per cell, a1=(1,0), a2=(1/2,sqrt3/2).
// Used for the NdMgAl11O19 anisotropic model (Jzz, Jpm, Jpmpm, Jzpm).
class Triangular : public UnitCell {
public:
    Triangular(size_t spin_dim)
        : UnitCell(spin_dim, 1,
                   {Vector3d(0, 0, 0)},
                   {Vector3d(1, 0, 0), Vector3d(0.5, sqrt(3.0)/2, 0), Vector3d(0, 0, 1)}) {}
};


class HoneyComb_alt : public UnitCell {
public:
    HoneyComb_alt(size_t spin_dim) 
        : UnitCell(spin_dim, 2,
                   {Vector3d(0, 0, 0), Vector3d(1/sqrt(3.0), 0, 0)},
                   {Vector3d(0, 1, 0), Vector3d(sqrt(3.0)/2, 0.5, 0), Vector3d(0, 0, 1)}) {}
};

// Pyrochlore lattice
class Pyrochlore : public UnitCell {
public:
    Pyrochlore(size_t spin_dim)
        : UnitCell(spin_dim, 4,
                   {Vector3d(0.125, 0.125, 0.125),
                    Vector3d(0.125, -0.125, -0.125),
                    Vector3d(-0.125, 0.125, -0.125),
                    Vector3d(-0.125, -0.125, 0.125)},
                   {Vector3d(0, 0.5, 0.5), Vector3d(0.5, 0, 0.5), Vector3d(0.5, 0.5, 0)}) {
        
        // Set local coordinate frames for pyrochlore sublattices
        if (spin_dim == 3) {  // Only for SU(2)
            Vector3d z1(1/sqrt(3.0), 1/sqrt(3.0), 1/sqrt(3.0));
            Vector3d z2(1/sqrt(3.0), -1/sqrt(3.0), -1/sqrt(3.0));
            Vector3d z3(-1/sqrt(3.0), 1/sqrt(3.0), -1/sqrt(3.0));
            Vector3d z4(-1/sqrt(3.0), -1/sqrt(3.0), 1/sqrt(3.0));
            
            Vector3d y1(0, -1/sqrt(2.0), 1/sqrt(2.0));
            Vector3d y2(0, 1/sqrt(2.0), -1/sqrt(2.0));
            Vector3d y3(0, -1/sqrt(2.0), -1/sqrt(2.0));
            Vector3d y4(0, 1/sqrt(2.0), 1/sqrt(2.0));
            
            Vector3d x1(-2/sqrt(6.0), 1/sqrt(6.0), 1/sqrt(6.0));
            Vector3d x2(-2/sqrt(6.0), -1/sqrt(6.0), -1/sqrt(6.0));
            Vector3d x3(2/sqrt(6.0), 1/sqrt(6.0), -1/sqrt(6.0));
            Vector3d x4(2/sqrt(6.0), -1/sqrt(6.0), 1/sqrt(6.0));
            
            SpinMatrix frame0(3, 3);
            frame0.col(0) = x1; frame0.col(1) = y1; frame0.col(2) = z1;
            set_sublattice_frame(frame0, 0);
            
            SpinMatrix frame1(3, 3);
            frame1.col(0) = x2; frame1.col(1) = y2; frame1.col(2) = z2;
            set_sublattice_frame(frame1, 1);
            
            SpinMatrix frame2(3, 3);
            frame2.col(0) = x3; frame2.col(1) = y3; frame2.col(2) = z3;
            set_sublattice_frame(frame2, 2);
            
            SpinMatrix frame3(3, 3);
            frame3.col(0) = x4; frame3.col(1) = y4; frame3.col(2) = z4;
            set_sublattice_frame(frame3, 3);
        }
    }
};

// TmFeO3 Iron sublattice
class TmFeO3_Fe : public UnitCell {
public:
    TmFeO3_Fe(size_t spin_dim)
        : UnitCell(spin_dim, 4,
                   {Vector3d(0, 0.5, 0.5), Vector3d(0.5, 0, 0.5),
                    Vector3d(0.5, 0, 0), Vector3d(0, 0.5, 0)},
                   {Vector3d(1, 0, 0), Vector3d(0, 1, 0), Vector3d(0, 0, 1)}) {
        
        if (spin_dim == 3) {  // Identity frames: storage == lab frame for Fe sites
            SpinMatrix I3 = SpinMatrix::Identity(3, 3);
            set_sublattice_frame(I3, 0);
            set_sublattice_frame(I3, 1);
            set_sublattice_frame(I3, 2);
            set_sublattice_frame(I3, 3);
        }
    }
};

// TmFeO3 Thulium sublattice
class TmFeO3_Tm : public UnitCell {
public:
    TmFeO3_Tm(size_t spin_dim)
        : UnitCell(spin_dim, 4,
                   {Vector3d(0.02111, 0.92839, 0.75), Vector3d(0.52111, 0.57161, 0.25),
                    Vector3d(0.47889, 0.42839, 0.75), Vector3d(0.97889, 0.07161, 0.25)},
                   {Vector3d(1, 0, 0), Vector3d(0, 1, 0), Vector3d(0, 0, 1)}) {}
};

#endif // UNITCELL_REFACTORED_H
