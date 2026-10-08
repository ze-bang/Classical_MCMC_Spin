// test_config_spin_io.cpp — spin-configuration files (initial_spin_config) of
// Lattice, MixedLattice and PhononLattice.
//
//   * save -> load reproduces the state bitwise (max_digits10 output);
//   * a missing file, a short file, a wrong column count, a non-finite value,
//     extra rows and a zero vector throw std::runtime_error naming the file
//     (and line), and leave the current state unchanged;
//   * loaded spins are rescaled to their length (SU(3): the pure-qutrit length
//     2/sqrt(3) is kept for states on that manifold).
#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/phonon_config.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "physics_test_util.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

using phys_test::check;
namespace fs = std::filesystem;

namespace {

const fs::path& scratch() {
    static const fs::path dir = [] {
        fs::path d = fs::temp_directory_path() / ("test_config_spin_io_" + std::to_string(::getpid()));
        fs::create_directories(d);
        return d;
    }();
    return dir;
}

std::string write_file(const std::string& name, const std::string& content) {
    const std::string path = (scratch() / name).string();
    std::ofstream(path) << content;
    return path;
}

/// load() must throw std::runtime_error whose message contains every needle.
template <class F>
bool throws_with(F&& load, std::initializer_list<std::string> needles) {
    try {
        load();
    } catch (const std::runtime_error& e) {
        const std::string m = e.what();
        for (const std::string& n : needles)
            if (m.find(n) == std::string::npos) {
                std::printf("      message lacks '%s': %s\n", n.c_str(), m.c_str());
                return false;
            }
        return true;
    } catch (const std::exception& e) {
        std::printf("      wrong exception type: %s\n", e.what());
        return false;
    }
    std::printf("      no exception\n");
    return false;
}

/// n lines of `row`.
std::string rows(size_t n, const std::string& row) {
    std::string s;
    for (size_t i = 0; i < n; ++i) s += row + "\n";
    return s;
}

template <class Spins>
bool same_spins(const Spins& a, const Spins& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].size() != b[i].size() || a[i] != b[i]) return false;
    return true;
}

// ---------------------------------------------------------------------------
void test_lattice() {
    SpinConfig cfg;
    Lattice lat(build_kitaev_honeycomb(cfg), 2, 2, 1, 1.0f);
    const size_t n = lat.lattice_size;
    lat.init_random();
    const std::string saved = (scratch() / "lattice_spins.txt").string();
    lat.save_spin_config(saved);

    Lattice other(build_kitaev_honeycomb(cfg), 2, 2, 1, 1.0f);
    other.load_spin_config(saved);
    check(same_spins(other.spins, lat.spins), "Lattice: save -> load is bitwise");

    const Lattice::SpinConfig before = other.spins;
    const std::string missing = (scratch() / "no_such_file.txt").string();
    bool ok = throws_with([&] { other.load_spin_config(missing); }, {"no_such_file.txt"});
    ok = throws_with([&] { other.load_spin_config(write_file("short.txt", rows(n - 1, "0 0 1"))); },
                     {"short.txt", std::to_string(n - 1) + " spins, expected " + std::to_string(n)}) && ok;
    ok = throws_with([&] { other.load_spin_config(write_file("cols2.txt", rows(n, "0 1"))); },
                     {"cols2.txt:1", "2 values, expected 3"}) && ok;
    ok = throws_with([&] { other.load_spin_config(write_file("cols4.txt", "0 0 1 0\n" + rows(n - 1, "0 0 1"))); },
                     {"cols4.txt:1", "more than 3"}) && ok;
    ok = throws_with([&] { other.load_spin_config(write_file("nan.txt", rows(2, "0 0 1") + "0 nan 1\n" +
                                                                         rows(n - 3, "0 0 1"))); },
                     {"nan.txt:3", "not a finite number"}) && ok;
    ok = throws_with([&] { other.load_spin_config(write_file("inf.txt", "inf 0 1\n" + rows(n - 1, "0 0 1"))); },
                     {"inf.txt:1"}) && ok;
    ok = throws_with([&] { other.load_spin_config(write_file("long.txt", rows(n + 1, "0 0 1"))); },
                     {"long.txt:" + std::to_string(n + 1), "more than"}) && ok;
    ok = throws_with([&] { other.load_spin_config(write_file("zero.txt", "0 0 1\n0 0 0\n" + rows(n - 2, "1 0 0"))); },
                     {"zero.txt:2", "zero spin"}) && ok;
    ok = throws_with([&] { other.load_spin_config(write_file("junk.txt", "0 0 1x\n" + rows(n - 1, "0 0 1"))); },
                     {"junk.txt:1"}) && ok;
    check(ok, "Lattice: missing / short / wrong-column / non-finite / extra-row / zero-vector files throw, naming "
              "file and line");
    check(same_spins(other.spins, before), "Lattice: a failed load leaves the spins unchanged");

    // Comments, blank lines and rescaling to spin_length.
    {
        Lattice l2(build_kitaev_honeycomb(cfg), 2, 2, 1, 0.5f);
        l2.load_spin_config(write_file("scaled.txt", "# header\n\n" + rows(n, "0 0 2   # comment")));
        bool scaled = true;
        for (const auto& s : l2.spins) scaled = scaled && std::abs(s.norm() - 0.5) < 1e-15 && s(2) > 0.0;
        check(scaled, "Lattice: comments allowed, every spin rescaled to spin_length");
    }
    // Spin dimension 8 (Tm-only TmFeO3 Lattice).
    {
        Lattice tm(build_tmfeo3_tm(cfg), 1, 1, 1, 1.0f);
        tm.init_random();
        const std::string p = (scratch() / "tm_spins.txt").string();
        tm.save_spin_config(p);
        Lattice tm2(build_tmfeo3_tm(cfg), 1, 1, 1, 1.0f);
        tm2.load_spin_config(p);
        const bool same = same_spins(tm.spins, tm2.spins);
        const bool rejected = throws_with([&] { tm2.load_spin_config(saved); }, {"3 values, expected 8"});
        check(same && rejected, "Lattice (spin_dim 8): round trip, and a 3-component file is rejected");
    }
}

// ---------------------------------------------------------------------------
void test_mixed() {
    SpinConfig cfg;
    MixedLattice lat(build_tmfeo3(cfg), 1, 1, 1, 1.0f, 1.0f);
    lat.init_random();
    const double pure = 2.0 / std::sqrt(3.0);
    // Two SU(3) states on the pure-qutrit manifold (one slightly off, as from a
    // 6-digit legacy file), the rest at spin_length_SU3.
    lat.spins_SU3[0] *= pure / lat.spins_SU3[0].norm();
    lat.spins_SU3[1] *= pure * (1.0 + 2e-6) / lat.spins_SU3[1].norm();
    const std::string base = (scratch() / "mixed").string();
    lat.save_spin_config(base);

    MixedLattice other(build_tmfeo3(cfg), 1, 1, 1, 1.0f, 1.0f);
    other.load_spin_config(base);
    const bool su2_same = same_spins(other.spins_SU2, lat.spins_SU2);
    bool su3_ok = std::abs(other.spins_SU3[0].norm() - pure) < 1e-15 && std::abs(other.spins_SU3[1].norm() - pure) < 1e-15;
    for (size_t i = 2; i < other.spins_SU3.size(); ++i) su3_ok = su3_ok && other.spins_SU3[i] == lat.spins_SU3[i];
    check(su2_same, "MixedLattice: SU(2) save -> load is bitwise (files were written with 6 digits)");
    check(su3_ok, "MixedLattice: SU(3) round trip; pure-qutrit states snap to |n| = 2/sqrt(3)");

    // A broken SU(3) file must not leave a half-loaded state (SU(2) parsed first).
    const auto before2 = other.spins_SU2;
    const auto before3 = other.spins_SU3;
    lat.init_random();
    lat.save_spin_config((scratch() / "broken").string());
    write_file("broken_SU3.txt", rows(lat.lattice_size_SU3 - 1, "0 0 1 0 0 0 0 0"));
    const bool short3 = throws_with([&] { other.load_spin_config((scratch() / "broken").string()); },
                                    {"broken_SU3.txt", "expected " + std::to_string(lat.lattice_size_SU3)});
    const bool missing = throws_with([&] { other.load_spin_config((scratch() / "absent").string()); },
                                     {"absent_SU2.txt"});
    const bool cols = throws_with(
        [&] {
            write_file("cols_SU2.txt", rows(lat.lattice_size_SU2, "0 0 1 0"));
            write_file("cols_SU3.txt", rows(lat.lattice_size_SU3, "0 0 1 0 0 0 0 0"));
            other.load_spin_config((scratch() / "cols").string());
        },
        {"cols_SU2.txt:1", "more than 3"});
    check(short3 && missing && cols, "MixedLattice: short / missing / wrong-column files throw, naming the file");
    check(same_spins(other.spins_SU2, before2) && same_spins(other.spins_SU3, before3),
          "MixedLattice: a failed load leaves both species unchanged");

    // Rescaling: SU(2) to spin_length_SU2, other SU(3) lengths to spin_length_SU3.
    write_file("scale_SU2.txt", rows(lat.lattice_size_SU2, "0 0 3"));
    write_file("scale_SU3.txt", rows(lat.lattice_size_SU3, "0 0 0.9 0 0 0 0 0"));
    MixedLattice s(build_tmfeo3(cfg), 1, 1, 1, 2.0f, 1.5f);
    s.load_spin_config((scratch() / "scale").string());
    bool scaled = true;
    for (const auto& v : s.spins_SU2) scaled = scaled && std::abs(v.norm() - 2.0) < 1e-15;
    for (const auto& v : s.spins_SU3) scaled = scaled && std::abs(v.norm() - 1.5) < 1e-15;
    check(scaled, "MixedLattice: spins rescaled to spin_length_SU2 / spin_length_SU3");
}

// ---------------------------------------------------------------------------
void test_phonon() {
    SpinConfig cfg;
    cfg.system = SystemType::NCTO;
    cfg.lattice_size = {2, 2, 1};
    PhononLattice lat = make_ncto_lattice(cfg);
    lat.init_random();
    const std::string p = (scratch() / "phonon_spins.txt").string();
    lat.save_spin_config(p);
    PhononLattice other = make_ncto_lattice(cfg);
    other.load_spin_config(p);
    bool same = true;
    for (size_t i = 0; i < lat.lattice_size; ++i) same = same && other.spins[i] == lat.spins[i];
    const auto before = other.spins;
    const bool errors =
        throws_with([&] { other.load_spin_config(write_file("ph_short.txt", rows(lat.lattice_size - 1, "0 0 1"))); },
                    {"ph_short.txt"}) &&
        throws_with([&] { other.load_spin_config(write_file("ph_nan.txt", "0 0 nan\n" + rows(lat.lattice_size - 1, "0 0 1"))); },
                    {"ph_nan.txt:1"}) &&
        throws_with([&] { other.load_spin_config(write_file("ph_zero.txt", rows(lat.lattice_size - 1, "0 0 1") + "0 0 0\n")); },
                    {"ph_zero.txt:" + std::to_string(lat.lattice_size), "zero spin"});
    bool unchanged = true;
    for (size_t i = 0; i < lat.lattice_size; ++i) unchanged = unchanged && other.spins[i] == before[i];
    check(same, "PhononLattice: save -> load is bitwise (was 12 digits)");
    check(errors && unchanged, "PhononLattice: malformed files throw naming file and line; state unchanged");
}

// The HDF5 pump-probe writers record the number of delays; the mixed writer
// truncated (0 -> 0.3 step 0.1 recorded 3 of the 4 delays written).
void test_delay_count() {
    check(hdf5_delay_count(0.0, 0.3, 0.1) == 4 && hdf5_delay_count(0.0, 0.7, 0.1) == 8 &&
              hdf5_delay_count(0.0, 0.6, 0.2) == 4,
          "HDF5 writers: delay count is round-off tolerant (truncation gave 3, 7, 3)");
    check(hdf5_delay_count(1.0, -1.0, -0.5) == 5 && hdf5_delay_count(2.0, 2.0, 0.1) == 1 &&
              hdf5_delay_count(0.0, 1.0, 0.3) == 4,
          "HDF5 writers: negative step, single delay, non-commensurate range (never overshoots)");
    check(hdf5_delay_count(-200.0, 200.0, 1.0) == int(classical_spin::dynamics::delay_grid(-200.0, 200.0, 1.0).size()),
          "HDF5 writers: count equals dynamics::delay_grid");
}

}  // namespace

int main() {
    test_lattice();
    test_mixed();
    test_phonon();
    test_delay_count();
    fs::remove_all(scratch());
    return phys_test::finish("test_config_spin_io");
}
