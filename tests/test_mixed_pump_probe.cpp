// test_mixed_pump_probe.cpp — MixedLattice pump-probe / 2DCS drivers.
//
// Run with any number of MPI ranks (ctest runs it with 1 and with 4):
//   * the serial driver writes M0, M1(tau), M01(tau) on one exact grid; every
//     trajectory has the same length and time stamps;
//   * M01 continued from stored M0 states equals M0 sample by sample before
//     the probe arrives and agrees with a full M01 integration afterwards;
//   * W1 (M1 = shifted M0) agrees with integrated M1;
//   * the MPI driver (dynamic scheduling, streamed results) writes the same
//     file contents as the serial driver, also with spin-state output;
//   * invalid input and an unwritable output directory raise an exception on
//     EVERY rank (no rank is left blocked in a collective).
#include "physics_test_util.h"

#include "classical_spin/core/su3_coherent_state.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <H5Cpp.h>
#include <mpi.h>
#include <unistd.h>

#include <filesystem>

using phys_test::check;
namespace su3 = classical_spin::su3;
using Complex = std::complex<double>;

namespace {

int g_rank = 0, g_size = 1;

// 1 Fe (spin 1/2, Zeeman along z) + 1 Tm (TmFeO3 CEF) with mixed exchange
// and a field-assisted bond; ground state: Fe along +z, Tm in the lowest level
// is NOT stationary with the exchange, so we relax it first.
MixedLattice make_lattice() {
    UnitCell fe(3, 1, {Eigen::Vector3d::Zero()},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    SpinConfig cfg;
    cfg.field_strength = 0.0;
    cfg.set_param("e1", 0.9);
    cfg.set_param("e2", 2.3);
    const UnitCell tm_full = build_tmfeo3_tm(cfg);
    UnitCell tm(8, 1, {Eigen::Vector3d(0.5, 0.5, 0.5)},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    tm.set_field(tm_full.field[0], 0);
    fe.set_field(Eigen::Vector3d(0.0, 0.0, 0.8), 0);
    MixedUnitCell cell(fe, tm);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(3, 8), Kd = Eigen::MatrixXd::Zero(3, 8);
    K(2, 2) = 0.15;  // S_z lambda_3: keeps the (Fe +z, Tm level 1) state stationary
    K(2, 7) = -0.1;
    Kd(0, 1) = 0.2;
    Kd(1, 4) = -0.1;
    cell.set_mixed_bilinear(K, 0, 0, Eigen::Vector3i::Zero());
    cell.set_mixed_bilinear_drive(Kd, 0, 0, Eigen::Vector3i::Zero(), 0);
    MixedLattice lat(cell, 1, 1, 1, 0.5f, 1.0f);
    lat.spins_SU2[0] = Eigen::Vector3d(0, 0, 0.5);
    lat.spins_SU3[0] = Eigen::VectorXd(su3::expectations_from_psi(su3::Vector3c(1.0, 0.0, 0.0)));
    return lat;
}

std::vector<double> read_ds(const std::string& file, const std::string& path) {
    H5::H5File f(file, H5F_ACC_RDONLY);
    H5::DataSet ds = f.openDataSet(path);
    H5::DataSpace sp = ds.getSpace();
    std::vector<hsize_t> dims(sp.getSimpleExtentNdims());
    sp.getSimpleExtentDims(dims.data());
    size_t n = 1;
    for (auto d : dims) n *= d;
    std::vector<double> v(n);
    ds.read(v.data(), H5::PredType::NATIVE_DOUBLE);
    return v;
}

bool has_ds(const std::string& file, const std::string& path) {
    H5::H5File f(file, H5F_ACC_RDONLY);
    return f.nameExists(path);
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

const char* kObs[] = {"antiferro_SU2", "local_SU2", "global_SU2", "antiferro_SU3", "local_SU3", "global_SU3"};

// Largest difference between two pump-probe files over every magnetisation
// (and spin-state) dataset; 1e300 if any shape differs.
double file_diff(const std::string& a, const std::string& b, size_t n_tau, bool spins) {
    double m = max_diff(read_ds(a, "/reference/times"), read_ds(b, "/reference/times"));
    m = std::max(m, max_diff(read_ds(a, "/tau_scan/tau_values"), read_ds(b, "/tau_scan/tau_values")));
    for (const char* o : kObs) {
        m = std::max(m, max_diff(read_ds(a, std::string("/reference/M_") + o), read_ds(b, std::string("/reference/M_") + o)));
        for (size_t j = 0; j < n_tau; ++j) {
            const std::string g = "/tau_scan/tau_" + std::to_string(j) + "/";
            m = std::max(m, max_diff(read_ds(a, g + "M1_" + o), read_ds(b, g + "M1_" + o)));
            m = std::max(m, max_diff(read_ds(a, g + "M01_" + o), read_ds(b, g + "M01_" + o)));
        }
    }
    if (spins) {
        m = std::max(m, max_diff(read_ds(a, "/reference/M0_spin_state"), read_ds(b, "/reference/M0_spin_state")));
        for (size_t j = 0; j < n_tau; ++j) {
            const std::string g = "/tau_scan/tau_" + std::to_string(j) + "/";
            m = std::max(m, max_diff(read_ds(a, g + "M1_spin_state"), read_ds(b, g + "M1_spin_state")));
            m = std::max(m, max_diff(read_ds(a, g + "M01_spin_state"), read_ds(b, g + "M01_spin_state")));
        }
    }
    return m;
}

struct Scan {
    double tau_start = 0.0, tau_end = 1.2, tau_step = 0.4;
    double T_start = -3.0, T_end = 8.0, T_step = 0.02;
    double amp = 0.4, width = 0.25, freq = 2.0;
};

void run(MixedLattice& lat, const Scan& s, const std::string& dir, bool mpi, bool w1, bool m01,
         bool spins, double tau_step_override = 0.0) {
    const std::vector<SpinVector> d2(1, Eigen::Vector3d(1.0, 0.3, 0.0));
    std::vector<SpinVector> d3(1, SpinVector::Zero(8));
    d3[0](1) = 1.0;
    d3[0](6) = 0.5;
    const double tau_step = tau_step_override != 0.0 ? tau_step_override : s.tau_step;
    if (mpi) {
        lat.pump_probe_spectroscopy_mpi(d2, d3, s.amp, s.width, s.freq, s.amp, s.width, s.freq,
                                        s.tau_start, s.tau_end, tau_step, s.T_start, s.T_end, s.T_step,
                                        1.0, 0.01, 0, false, 0, dir, "dopri5", false, spins, w1, 1e-8,
                                        true, 1e-11, 1e-11, {}, {}, m01);
    } else {
        lat.pump_probe_spectroscopy(d2, d3, s.amp, s.width, s.freq, s.amp, s.width, s.freq,
                                    s.tau_start, s.tau_end, tau_step, s.T_start, s.T_end, s.T_step,
                                    1.0, 0.01, 0, false, 0, dir, "dopri5", false, spins, w1, 1e-8,
                                    1, true, 1e-11, 1e-11, {}, {}, m01);
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    // One scratch directory per run, named after rank 0's pid.
    long pid = static_cast<long>(::getpid());
    MPI_Bcast(&pid, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    const std::string root = (std::filesystem::temp_directory_path() /
                              ("classical_spin_mixed_pp_test_" + std::to_string(pid))).string();
    const Scan scan;
    MixedLattice lat = make_lattice();
    const size_t n_tau = 4;
    const size_t n_t = static_cast<size_t>(std::llround((scan.T_end - scan.T_start) / scan.T_step)) + 1;

    if (g_rank == 0) {
        std::filesystem::remove_all(root);
        check(lat.relative_stationarity_residual(lat.spins_to_state()) < 1e-14, "test ground state is stationary");
        // Serial reference runs.
        run(lat, scan, root + "/serial_w1_m01", false, true, true, false);
        run(lat, scan, root + "/serial_plain", false, false, false, false);
        run(lat, scan, root + "/serial_spins", false, true, true, true);
        const std::string fa = root + "/serial_w1_m01/pump_probe_spectroscopy.h5";
        const std::string fb = root + "/serial_plain/pump_probe_spectroscopy.h5";
        const auto taus = read_ds(fa, "/tau_scan/tau_values");
        check(taus.size() == n_tau, "delay grid includes tau_end (0, 0.4, 0.8, 1.2)");
        const auto t0 = read_ds(fa, "/reference/times");
        bool exact = t0.size() == n_t;
        for (size_t k = 0; k < t0.size(); ++k) exact = exact && t0[k] == scan.T_start + static_cast<double>(k) * scan.T_step;
        check(exact, "M0 is sampled on the exact grid T_start + k T_step, k = 0..n");
        bool lengths = true;
        for (size_t j = 0; j < n_tau; ++j) {
            const std::string g = "/tau_scan/tau_" + std::to_string(j) + "/";
            lengths = lengths && read_ds(fa, g + "M1_local_SU2").size() == 3 * n_t &&
                      read_ds(fa, g + "M01_local_SU3").size() == 8 * n_t;
        }
        check(lengths, "every M1 / M01 trajectory has exactly n + 1 samples");
        // M01 continued from M0 states: identical to M0 before the probe lead.
        const auto m0 = read_ds(fa, "/reference/M_local_SU3");
        double prefix = 0.0;
        for (size_t j = 0; j < n_tau; ++j) {
            const auto m01 = read_ds(fa, "/tau_scan/tau_" + std::to_string(j) + "/M01_local_SU3");
            const double t_lead = taus[j] - 9.0 * scan.width;
            for (size_t k = 0; k < n_t && t0[k] < t_lead - scan.T_step; ++k)
                for (int a = 0; a < 8; ++a) prefix = std::max(prefix, std::abs(m01[8 * k + a] - m0[8 * k + a]));
        }
        check(prefix == 0.0, "M01 equals M0 exactly before the probe arrives");
        const double d_ab = file_diff(fa, fb, n_tau, false);
        check(d_ab < 1e-8, "W1 + continued M01 agree with fully integrated M1 / M01 (max diff " +
                           std::to_string(d_ab) + ")");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI driver (falls back to the serial driver on one rank).
    run(lat, scan, root + "/mpi_w1_m01", true, true, true, false);
    run(lat, scan, root + "/mpi_plain", true, false, false, false);
    run(lat, scan, root + "/mpi_spins", true, false, true, true);
    if (g_rank == 0) {
        const double d1 = file_diff(root + "/serial_w1_m01/pump_probe_spectroscopy.h5",
                                    root + "/mpi_w1_m01/pump_probe_spectroscopy.h5", n_tau, false);
        const double d2 = file_diff(root + "/serial_plain/pump_probe_spectroscopy.h5",
                                    root + "/mpi_plain/pump_probe_spectroscopy.h5", n_tau, false);
        const double d3 = file_diff(root + "/serial_spins/pump_probe_spectroscopy.h5",
                                    root + "/mpi_spins/pump_probe_spectroscopy.h5", n_tau, true);
        check(d1 < 1e-12 && d2 < 1e-12, "MPI driver reproduces the serial driver (" + std::to_string(g_size) +
                                        " ranks, diffs " + std::to_string(d1) + ", " + std::to_string(d2) + ")");
        check(d3 < 1e-8 && has_ds(root + "/mpi_spins/pump_probe_spectroscopy.h5", "/tau_scan/tau_3/M01_spin_state"),
              "MPI driver with spin-state output reproduces the serial file (diff " + std::to_string(d3) + ")");
    }

    // Errors must reach every rank.
    auto all_threw = [&](auto&& f) {
        int threw = 0;
        try { f(); } catch (const std::exception&) { threw = 1; }
        int all = 0;
        MPI_Allreduce(&threw, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return all == 1;
    };
    const bool e1 = all_threw([&] { run(lat, scan, root + "/bad_tau", true, true, true, false, -0.4); });
    const bool e2 = all_threw([&] { run(lat, scan, "/proc/no_such_dir/x", true, true, true, false); });
    if (g_rank == 0) {
        check(e1, "invalid delay grid raises on every rank");
        check(e2, "unwritable output directory raises on every rank");
        std::filesystem::remove_all(root);
    }
    const int failures = phys_test::failures();
    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int code = (g_rank == 0) ? phys_test::finish("test_mixed_pump_probe") : (total ? 1 : 0);
    MPI_Finalize();
    return code;
}
