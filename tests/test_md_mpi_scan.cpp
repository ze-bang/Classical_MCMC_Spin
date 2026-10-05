// test_md_mpi_scan.cpp — MPI delay scan (run with 3 ranks).
//
//  1. pump_probe_spectroscopy_mpi with dynamic scheduling (rank 0 computing
//     too) writes exactly the file the single-process scan writes: every
//     dataset bitwise identical, for a stationary ground state (W1 on) and a
//     non-stationary one (every M1 integrated on the workers).
//  2. The scan runs on any communicator: two sub-communicators of
//     MPI_COMM_WORLD scan concurrently without touching each other.
//  3. A failure in the middle of the scan (the spherical-midpoint solver
//     diverging where two strong pulses overlap) and a setup failure (zero
//     delay step) make EVERY rank throw instead of leaving ranks blocked.
#include "physics_test_util.h"

#include <filesystem>
#include <unistd.h>

using namespace phys_test;
namespace fs = std::filesystem;

namespace {

int world_rank() { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }

// Bitwise comparison of every dataset of two pump-probe files.
bool same_file(const std::string& a, const std::string& b, size_t n_tau, std::string& why) {
    H5::H5File fa(a, H5F_ACC_RDONLY), fb(b, H5F_ACC_RDONLY);
    std::vector<std::string> paths = {"/reference/times", "/reference/M_antiferro", "/reference/M_local",
                                      "/reference/M_global", "/tau_scan/tau_values", "/tau_scan/m1_synthesized"};
    for (size_t i = 0; i < n_tau; ++i)
        for (const char* d : {"M1_antiferro", "M1_local", "M1_global", "M01_antiferro", "M01_local", "M01_global"})
            paths.push_back("/tau_scan/tau_" + std::to_string(i) + "/" + d);
    for (const auto& p : paths) {
        H5::DataSet da = fa.openDataSet(p), db = fb.openDataSet(p);
        const size_t na = size_t(da.getSpace().getSimpleExtentNpoints());
        const size_t nb = size_t(db.getSpace().getSimpleExtentNpoints());
        if (na != nb) { why = p + ": sizes differ"; return false; }
        std::vector<double> va(na), vb(nb);
        da.read(va.data(), H5::PredType::NATIVE_DOUBLE);
        db.read(vb.data(), H5::PredType::NATIVE_DOUBLE);
        if (va != vb) { why = p + ": values differ"; return false; }
    }
    return true;
}

Lattice make_chain() {
    UnitCell uc(3, 2, {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.5, 0, 0)},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    const Eigen::Matrix3d J = Eigen::Vector3d(-1.0, -0.9, -1.1).asDiagonal();   // XYZ ferromagnet
    uc.set_bilinear_interaction(J, 0, 1, Eigen::Vector3i(0, 0, 0));
    uc.set_bilinear_interaction(J, 1, 0, Eigen::Vector3i(1, 0, 0));
    uc.set_field(Eigen::Vector3d(0, 0, 0.4), 0);
    uc.set_field(Eigen::Vector3d(0, 0, 0.4), 1);
    uc.set_afm_sublattice_signs({1.0, -1.0});
    return Lattice(uc, 4, 1, 1, 1.0f);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const int rank = world_rank();
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long tag = (rank == 0) ? long(::getpid()) : 0;
    MPI_Bcast(&tag, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    const fs::path root = fs::temp_directory_path() / ("csmd_mpi_" + std::to_string(tag));
    if (rank == 0) { fs::remove_all(root); fs::create_directories(root); }
    MPI_Barrier(MPI_COMM_WORLD);

    Lattice lat = make_chain();
    const std::vector<SpinVector> field(lat.N_atoms, SpinVector(Eigen::Vector3d(1, 0.3, 0)));
    const size_t n_tau = 13;   // delays -3 .. 3 step 0.5

    // ---- 1. MPI scan == single-process scan, bitwise ----
    for (int stationary = 1; stationary >= 0; --stationary) {
        seed_lehman(77);
        lat.init_random();
        if (stationary) {
            for (auto& sp : lat.spins) sp = Eigen::Vector3d(0, 0, 1);   // exact equilibrium: W1 on
        }
        const std::string mpi_dir = (root / ("mpi_" + std::to_string(stationary))).string();
        const std::string ser_dir = (root / ("ser_" + std::to_string(stationary))).string();
        lat.pump_probe_spectroscopy_mpi(field, 0.3, 0.5, 2.0, -3.0, 3.0, 0.5, -6.0, 12.0, 0.05,
                                        0, 0, 0, false, 0, mpi_dir, "rk4", false, true, 1e-9);
        if (rank == 0) {
            lat.pump_probe_spectroscopy(field, 0.3, 0.5, 2.0, -3.0, 3.0, 0.5, -6.0, 12.0, 0.05,
                                        0, 0, 0, false, 0, ser_dir, "rk4", false, true, 1e-9, 1);
            std::string why;
            const bool same = same_file(mpi_dir + "/pump_probe_spectroscopy.h5",
                                        ser_dir + "/pump_probe_spectroscopy.h5", n_tau, why);
            check(same, std::string(stationary ? "stationary" : "non-stationary") +
                        " ground state: MPI scan file == serial scan file " + why);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ---- 2. two sub-communicators scan concurrently ----
    {
        MPI_Comm sub;
        const int color = (rank < 2) ? 0 : 1;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub);
        seed_lehman(5);
        lat.init_random();
        const std::string dir = (root / ("sub_" + std::to_string(color))).string();
        bool ok = true;
        try {
            lat.pump_probe_spectroscopy_mpi(field, 0.3, 0.5, 2.0, -3.0, 3.0, 0.5, -6.0, 12.0, 0.05,
                                            0, 0, 0, false, 0, dir, "rk4", false, true, 1e-9, true,
                                            1e-8, 1e-8, sub);
        } catch (const std::exception& e) {
            std::printf("rank %d: unexpected exception %s\n", rank, e.what());
            ok = false;
        }
        MPI_Comm_free(&sub);
        int all_ok = ok ? 1 : 0, global = 0;
        MPI_Allreduce(&all_ok, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            std::string why;
            check(global == 1 && size >= 3 &&
                      same_file((root / "sub_0/pump_probe_spectroscopy.h5").string(),
                                (root / "sub_1/pump_probe_spectroscopy.h5").string(), n_tau, why),
                  "scans on two sub-communicators agree " + why);
        }
    }

    // ---- 3. error propagation ----
    // Every rank must throw std::runtime_error; rank 0's message must contain `needle`.
    auto every_rank_threw = [&](const std::function<void()>& f, const std::string& needle) {
        int threw = 0;
        try {
            f();
        } catch (const std::runtime_error& e) {
            threw = (rank != 0) || std::string(e.what()).find(needle) != std::string::npos;
            if (rank == 0) std::printf("rank 0 caught: %s\n", e.what());
        } catch (...) {
        }
        int all = 0;
        MPI_Allreduce(&threw, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return all == 1;
    };
    for (auto& s : lat.spins) s = Eigen::Vector3d(0, 0, 1);
    // A = 6 with dt = 0.2: one pulse converges, two overlapping pulses (τ = 0) do not.
    const bool mid = every_rank_threw([&] {
        lat.pump_probe_spectroscopy_mpi(field, 6.0, 0.5, 0.0, -3.0, 3.0, 1.0, -5.0, 8.0, 0.2,
                                        0, 0, 0, false, 0, (root / "fail_mid").string(),
                                        "spherical_midpoint", false, false, 1e-9);
    }, "pump-probe scan failed: ");
    if (rank == 0) check(mid, "a failure in the middle of the scan reaches every rank");
    const bool setup = every_rank_threw([&] {
        lat.pump_probe_spectroscopy_mpi(field, 0.3, 0.5, 0.0, -3.0, 3.0, 0.0, -5.0, 8.0, 0.2,
                                        0, 0, 0, false, 0, (root / "fail_setup").string(), "rk4");
    }, "step");
    if (rank == 0) check(setup, "an invalid delay grid fails on every rank");

    MPI_Barrier(MPI_COMM_WORLD);
    int rc = 0;
    if (rank == 0) {
        fs::remove_all(root);
        rc = finish("test_md_mpi_scan");
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
