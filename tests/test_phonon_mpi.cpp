// test_phonon_mpi.cpp — MPI drivers of the spin–phonon model (run with 2 or more ranks).
//
//   1. pump_probe_spectroscopy_mpi (delay points distributed over the ranks) writes the
//      SAME file as the serial pump_probe_spectroscopy: every dataset — M_antiferro,
//      M_local, M_global and O_custom of M0, M1(τ) and M01(τ) — must agree. Before
//      2026-10 the gather sent 7 of the 13 doubles per sample, so for every τ not computed
//      on rank 0 M_global was written as zeros and O_custom as uninitialised memory.
//   2. Replica exchange of a PhononLattice swaps the lattice sector together with the
//      spins (mc::has_extra_dof), so the exchanged total energies are those of the
//      states actually exchanged.
#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"

#include <H5Cpp.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
int g_rank = 0;

void check(bool ok, const std::string& what) {
    if (g_rank == 0 || !ok) std::printf("[%s] (rank %d) %s\n", ok ? " OK " : "FAIL", g_rank, what.c_str());
    if (!ok) ++g_fail;
}

PhononLattice make_lattice() {
    std::streambuf* saved = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    SpinConfig config;
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice L(uc, 3, 3, 1, 1.0f);
    SpinPhononCouplingParams sp;              // NCTO exchange + ring
    sp.lambda_E1_K_1 = 2.0;                   // a linear E1 channel so the pulse reaches the spins
    sp.lambda_E1_Gamma_1 = -1.0;
    PhononParams ph;
    DriveParams dr;
    L.set_parameters(sp, ph, dr);
    for (size_t i = 0; i < L.lattice_size; ++i)          // identical, deterministic state on every rank
        L.spins[i] = Eigen::Vector3d(std::sin(0.3 * i + 0.1), std::cos(0.7 * i), 0.4 * std::sin(1.3 * i)).normalized();
    std::cout.rdbuf(saved);
    return L;
}

std::vector<double> read_dataset(H5::H5File& f, const std::string& name) {
    H5::DataSet ds = f.openDataSet(name);
    const hssize_t n = ds.getSpace().getSimpleExtentNpoints();
    std::vector<double> v(static_cast<size_t>(n));
    ds.read(v.data(), H5::PredType::NATIVE_DOUBLE);
    return v;
}

void test_pump_probe_mpi_equals_serial() {
    const std::string dir_mpi = "phonon_mpi_2dcs_par", dir_ser = "phonon_mpi_2dcs_ser";
    const double tau0 = 0.0, tau1 = 3.0, dtau = 1.0;      // 4 delay points: uneven on 3 ranks
    PhononLattice L = make_lattice();
    {
        std::streambuf* saved = std::cout.rdbuf();
        std::ostringstream sink;
        std::cout.rdbuf(sink.rdbuf());
        L.pump_probe_spectroscopy_mpi(0.3, 2.0, 0.3, 12.41, tau0, tau1, dtau, 0.0, 4.0, 0.05,
                                      dir_mpi, "rk4", /*reuse_m0_for_m1=*/false, 1e-6,
                                      /*pulse_window_chunking=*/true, 1e-8, 1e-8, MPI_COMM_WORLD);
        if (g_rank == 0) {
            PhononLattice S = make_lattice();
            S.pump_probe_spectroscopy(0.3, 2.0, 0.3, 12.41, tau0, tau1, dtau, 0.0, 4.0, 0.05,
                                      dir_ser, "rk4", false, 1e-6, /*outer_omp_threads=*/1, true, 1e-8, 1e-8);
        }
        std::cout.rdbuf(saved);
    }
    if (g_rank != 0) return;
    H5::H5File fp(dir_mpi + "/pump_probe_spectroscopy.h5", H5F_ACC_RDONLY);
    H5::H5File fs(dir_ser + "/pump_probe_spectroscopy.h5", H5F_ACC_RDONLY);
    const char* fields[] = {"time", "M_antiferro", "M_local", "M_global", "O_custom"};
    std::vector<std::string> groups = {"/reference"};
    for (int i = 0; i < 4; ++i) {
        groups.push_back("/delay_scan/tau_" + std::to_string(i) + "/M1");
        groups.push_back("/delay_scan/tau_" + std::to_string(i) + "/M01");
    }
    double worst = 0.0;
    bool nonzero_global = true;
    for (const auto& g : groups)
        for (const char* f : fields) {
            const auto a = read_dataset(fp, g + "/" + f), b = read_dataset(fs, g + "/" + f);
            if (a.size() != b.size() || a.empty()) {
                check(false, "dataset " + g + "/" + f + " has different sizes / is empty");
                return;
            }
            for (size_t k = 0; k < a.size(); ++k) worst = std::max(worst, std::abs(a[k] - b[k]));
            if (std::string(f) == "M_global") {
                double m = 0.0;
                for (double v : a) m = std::max(m, std::abs(v));
                nonzero_global = nonzero_global && m > 0.1;
            }
        }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "MPI 2DCS file == serial 2DCS file (max |diff| = %.3g over %zu groups)",
                  worst, groups.size());
    check(worst < 1e-10, buf);
    check(nonzero_global, "M_global is populated for every delay point");
    std::filesystem::remove_all(dir_mpi);
    std::filesystem::remove_all(dir_ser);
}

void test_replica_exchange_swaps_lattice() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;
    PhononLattice L = make_lattice();
    // distinct state per rank: rotate spins and displace the E1 coordinate
    for (size_t i = 0; i < L.lattice_size; ++i)
        L.spins[i] = Eigen::AngleAxisd(0.4 * g_rank, Eigen::Vector3d::UnitZ()) * Eigen::Vector3d(L.spins[i]);
    L.phonons.Q_x_E1 = 0.01 * (g_rank + 1);
    L.phonons.V_y_E1 = -0.02 * (g_rank + 1);
    const double E_before = L.total_energy();
    double E_partner_before = 0.0;
    const int partner = (g_rank % 2 == 0) ? g_rank + 1 : g_rank - 1;
    const bool paired = partner < size;
    std::vector<double> temps(size, 1.0);                 // equal temperatures: swap always accepted
    std::mt19937 rng(1);
    if (paired)
        MPI_Sendrecv(&E_before, 1, MPI_DOUBLE, partner, 9, &E_partner_before, 1, MPI_DOUBLE, partner, 9,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const int acc = mc::attempt_replica_exchange(L, rng, g_rank, size, temps, 1.0, 0, MPI_COMM_WORLD);
    if (!paired) return;
    const double expect_Q = 0.01 * (partner + 1), expect_V = -0.02 * (partner + 1);
    check(acc == 1, "equal-temperature swap accepted");
    check(L.phonons.Q_x_E1 == expect_Q && L.phonons.V_y_E1 == expect_V,
          "lattice sector travels with the spins in a replica exchange");
    check(std::abs(L.total_energy() - E_partner_before) < 1e-10 * std::abs(E_partner_before),
          "post-swap total energy equals the partner's pre-swap energy");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    try {
        test_pump_probe_mpi_equals_serial();
        test_replica_exchange_swaps_lattice();
    } catch (const std::exception& e) {
        std::printf("[FAIL] (rank %d) exception: %s\n", g_rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int total = 0;
    MPI_Allreduce(&g_fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) std::printf("\ntest_phonon_mpi: %s\n", total == 0 ? "all checks passed" : "FAILED");
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
