// test_unitcell_geometry.cpp — unit-cell builders wire every bond to the
// neighbour the model names, and the cell/lattice constructors reject
// inconsistent input.
//
// References are pure geometry: the neighbour shells of the pyrochlore
// (r_NN = sqrt(2)/4 in units of the cubic constant, coordination 6, then 12
// at sqrt(3) r_NN and 6 + 6 at 2 r_NN), of the honeycomb (3 at 1/sqrt(3),
// 6 at 1, 3 at 2/sqrt(3)) and of the triangular lattice (6 at 1).
#include "physics_test_util.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"

#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace phys_test;

namespace {

using Bond = std::tuple<size_t, size_t, int, int, int>;

// Canonical orientation of an unordered bond (same rule as bonds_at_distance).
Bond canonical(size_t s, size_t p, Eigen::Vector3i n) {
    auto positive = [](const Eigen::Vector3i& v) {
        for (int d = 0; d < 3; ++d)
            if (v[d] != 0) return v[d] > 0;
        return false;
    };
    if (s > p || (s == p && !positive(n))) { std::swap(s, p); n = -n; }
    return {s, p, n[0], n[1], n[2]};
}

// Declared bilinear bonds grouped by length (rounded to 1e-9).
std::map<long, std::set<Bond>> declared_by_length(const UnitCell& uc) {
    std::map<long, std::set<Bond>> out;
    for (const auto& [src, bi] : uc.bilinear_interaction) {
        const double r = uc.bond_vector(size_t(src), bi.partner, bi.offset).norm();
        out[std::lround(r * 1e9)].insert(canonical(size_t(src), bi.partner, bi.offset));
    }
    return out;
}

std::set<Bond> shell_bonds(const UnitCell& uc, double r) {
    std::set<Bond> out;
    for (const auto& b : uc.bonds_at_distance(r)) out.insert(canonical(b.source, b.partner, b.offset));
    return out;
}

// Number of bonds (counting both ends) touching each atom.
std::vector<int> coordination(const UnitCell& uc, const std::set<Bond>& bonds) {
    std::vector<int> z(uc.N_atoms, 0);
    for (const auto& [s, p, i, j, k] : bonds) { ++z[s]; ++z[p]; }
    return z;
}

bool all_equal(const std::vector<int>& z, int v) {
    for (int x : z) if (x != v) return false;
    return true;
}

// Neighbour count of every site of a periodic lattice (distinct partners).
bool lattice_coordination(const Lattice& lat, size_t z) {
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        std::set<size_t> partners(lat.bilinear_partners[i].begin(), lat.bilinear_partners[i].end());
        if (partners.size() != z || lat.bilinear_partners[i].size() != z) return false;
    }
    return true;
}

void test_shell_finder() {
    std::printf("\n== Neighbour shells from the geometry ==\n");
    Pyrochlore py(3);
    const auto sh = py.neighbour_shells(3);
    const double r = std::sqrt(2.0) / 4.0;
    check(sh.size() == 3, "pyrochlore: three shells found");
    check_close(sh[0], r, 1e-12, "pyrochlore r1 = sqrt(2)/4");
    check_close(sh[1], std::sqrt(3.0) * r, 1e-12, "pyrochlore r2 = sqrt(3) r1");
    check_close(sh[2], 2.0 * r, 1e-12, "pyrochlore r3 = 2 r1");
    check(all_equal(coordination(py, shell_bonds(py, sh[0])), 6), "pyrochlore z1 = 6");
    check(all_equal(coordination(py, shell_bonds(py, sh[1])), 12), "pyrochlore z2 = 12");
    check(all_equal(coordination(py, shell_bonds(py, sh[2])), 12), "pyrochlore z3 = 6 + 6");

    HoneyComb hc(3);
    const auto hs = hc.neighbour_shells(3);
    check_close(hs[0], 1.0 / std::sqrt(3.0), 1e-12, "honeycomb r1 = 1/sqrt(3)");
    check_close(hs[1], 1.0, 1e-12, "honeycomb r2 = 1 (in-plane, a3 = 1 is the same length)");
    check(all_equal(coordination(hc, shell_bonds(hc, hs[0])), 3), "honeycomb z1 = 3");
}

void test_pyrochlore() {
    std::printf("\n== Pyrochlore builders ==\n");
    const double r = std::sqrt(2.0) / 4.0;
    SpinConfig cfg;
    const UnitCell py = build_pyrochlore(cfg);
    const auto by_len = declared_by_length(py);
    check(by_len.size() == 1 && by_len.begin()->first == std::lround(r * 1e9),
          "every declared bond has length sqrt(2)/4 (the (2,3) inter-tetrahedron bond was 3 sqrt(2)/4)");
    check(by_len.begin()->second == shell_bonds(py, r),
          "declared bonds are exactly the 12 nearest-neighbour bonds of the cell");
    Lattice lat(py, 3, 3, 3, 1.0f);
    check(lattice_coordination(lat, 6), "every site of a 3x3x3 lattice has 6 distinct neighbours");

    SpinConfig nk;
    nk.set_param("J2", 0.3);
    nk.set_param("J3a", 0.2);
    nk.set_param("J3b", 0.1);
    const UnitCell pk = build_pyrochlore_non_kramer(nk);
    auto pk_len = declared_by_length(pk);
    const std::set<Bond> b1 = pk_len[std::lround(r * 1e9)];
    const std::set<Bond> b2 = pk_len[std::lround(std::sqrt(3.0) * r * 1e9)];
    const std::set<Bond> b3 = pk_len[std::lround(2.0 * r * 1e9)];
    check(pk_len.size() == 3, "non-Kramers: bonds only on the first three shells");
    check(b1 == shell_bonds(pk, r), "non-Kramers: complete nearest-neighbour shell");
    check(b2 == shell_bonds(pk, std::sqrt(3.0) * r) && all_equal(coordination(pk, b2), 12),
          "non-Kramers J2: complete second shell, 12 neighbours per site");
    check(b3 == shell_bonds(pk, 2.0 * r) && all_equal(coordination(pk, b3), 12),
          "non-Kramers J3a + J3b: complete third shell");
    // J3a sits along a bond chain (a site at the midpoint), J3b across a hexagon.
    int n_chain = 0, n_cross = 0;
    for (const auto& [src, bi] : pk.bilinear_interaction) {
        const Eigen::Vector3d d = pk.bond_vector(size_t(src), bi.partner, bi.offset);
        if (std::abs(d.norm() - 2.0 * r) > 1e-9) continue;
        const double J = bi.interaction(0, 0);
        const bool chain = pk.has_site_at(pk.lattice_pos[size_t(src)] + 0.5 * d);
        if (chain) n_chain += (std::abs(J - 0.2) < 1e-12);
        else n_cross += (std::abs(J - 0.1) < 1e-12);
    }
    check(n_chain == 12 && n_cross == 12, "J3a on the 12 chain bonds, J3b on the 12 hexagon bonds");
}

void test_honeycomb() {
    std::printf("\n== Honeycomb and triangular builders ==\n");
    const double r1 = 1.0 / std::sqrt(3.0), r3 = 2.0 / std::sqrt(3.0);
    SpinConfig cfg;
    cfg.set_param("J3", 0.4);
    cfg.set_param("J2_A", 0.1);
    cfg.set_param("J2_B", 0.1);
    const UnitCell k = build_kitaev_honeycomb(cfg);
    auto kl = declared_by_length(k);
    check(kl[std::lround(r1 * 1e9)] == shell_bonds(k, r1), "Kitaev: the three nearest-neighbour bonds");
    check(kl[std::lround(1.0 * 1e9)].size() == 6 && all_equal(coordination(k, kl[std::lround(1e9)]), 6),
          "Kitaev J2: six second neighbours per site");
    check(kl[std::lround(r3 * 1e9)] == shell_bonds(k, r3), "Kitaev J3: the three third neighbours");

    SpinConfig bc;
    const UnitCell b = build_bcao_honeycomb(bc);
    auto bl = declared_by_length(b);
    check(bl[std::lround(r1 * 1e9)] == shell_bonds(b, r1), "BCAO: nearest-neighbour shell");
    check(bl[std::lround(r3 * 1e9)] == shell_bonds(b, r3), "BCAO: third-neighbour shell");
    for (const auto& [len, bonds] : bl) {
        const double r = double(len) * 1e-9;
        check(bonds == shell_bonds(b, r), "BCAO: complete shell at r = " + std::to_string(r));
    }

    // A J3 bond (offset (1,-2,0)) crosses two periods along a2 on an L2 = 1
    // lattice; the single-step wrap used to index out of range.
    Lattice thin(k, 3, 1, 1, 1.0f);
    bool in_range = true;
    for (size_t i = 0; i < thin.lattice_size; ++i)
        for (size_t p : thin.bilinear_partners[i]) in_range = in_range && p < thin.lattice_size;
    check(in_range, "bonds longer than the lattice wrap into range (L2 = 1)");

    SpinConfig tc;
    const UnitCell t = build_triangular_anisotropic(tc);
    auto tl = declared_by_length(t);
    check(tl.size() == 1 && tl.begin()->second == shell_bonds(t, 1.0), "triangular: the three NN bonds");
    Lattice tri(t, 4, 4, 1, 1.0f);
    check(lattice_coordination(tri, 6), "triangular lattice: 6 neighbours per site");
}

void test_wrap() {
    std::printf("\n== Periodic wrap ==\n");
    bool ok = true;
    for (long L : {1L, 2L, 5L})
        for (long c = -3 * L - 1; c <= 3 * L + 1; ++c) {
            int n = 0;
            const size_t r = Lattice::wrap_coordinate(c, size_t(L), n);
            ok = ok && r < size_t(L) && long(r) + long(n) * L == c;
        }
    check(ok, "wrap_coordinate: c = r + n L with 0 <= r < L for all c");
}

template<class F>
bool throws(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; } catch (const std::out_of_range&) { return true; }
    return false;
}

void test_validation() {
    std::printf("\n== Input validation ==\n");
    const Eigen::Matrix3d J = Eigen::Matrix3d::Identity();
    check(throws([&] { Triangular t(3); t.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); }),
          "bilinear partner index out of range");
    check(throws([&] {
              Triangular t(3);
              Eigen::Matrix3d bad = J;
              bad(0, 1) = std::nan("");
              t.set_bilinear_interaction(bad, 0, 0, {1, 0, 0});
          }),
          "non-finite coupling");
    check(throws([&] {
              Triangular t(3);
              t.set_bilinear_interaction(J, 0, 0, {1, 0, 0});
              t.set_bilinear_interaction(J, 0, 0, {-1, 0, 0});  // its own reverse
              Lattice lat(t, 3, 3, 1, 1.0f);
          }),
          "a bond declared again as its own reverse is rejected (it would double count)");
    check(!throws([&] {
              Triangular t(3);
              t.set_bilinear_interaction(J, 0, 0, {1, 0, 0});
              t.set_bilinear_interaction(0.5 * J, 0, 0, {1, 0, 0});  // a second term on the same bond
              Lattice lat(t, 3, 3, 1, 1.0f);
          }),
          "two terms on the same bond in the same orientation add");
    check(throws([&] { Triangular t(3); Lattice lat(t, 0, 3, 1, 1.0f); }), "zero lattice dimension");
    check(throws([&] { Triangular t(3); Lattice lat(t, 3, 3, 1, -1.0f); }), "negative spin length");

    // On-site matrices: only the symmetric part enters S^T A S.
    Triangular t(3);
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    A(0, 1) = 1.0;
    t.set_onsite_interaction(A, 0);
    check_close((t.onsite_interaction[0] - t.onsite_interaction[0].transpose()).norm(), 0.0, 0.0,
                "on-site matrix stored symmetrised");
}

}  // namespace

int main() {
    test_shell_finder();
    test_pyrochlore();
    test_honeycomb();
    test_wrap();
    test_validation();
    return finish("test_unitcell_geometry");
}
