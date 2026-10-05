// test_md_drivers.cpp — the Lattice spin-dynamics DRIVERS against exact results.
//
//  1. Exact time grid: double_pulse_drive samples t_k = T_start + k T_step
//     bitwise, with the same count for every delay, and follows the analytic
//     solution of a spin precessing in a pulsed longitudinal field. (The old
//     pulse-window chunking dropped the last step of a segment whenever
//     odeint's absolute time epsilon missed the seam: error ~ h T_step.)
//  2. A zero-amplitude probe changes nothing: M01 == M0 bitwise for every τ.
//  3. Drive frame: F dS_local/dt = S_global × B_global for a non-symmetric
//     (Kitaev) sublattice frame (set_pulse used F instead of F^T).
//  4. Integrator names are validated (parse, API entry, SpinConfig::validate),
//     delay / time grids are counted robustly.
//  5. molecular_dynamics writes a uniform output grid with energy and |S|
//     diagnostics; geometric integrators conserve both.
//  6. Langevin MD through the public molecular_dynamics entry point samples
//     the exact free-spin energy <E>/N = -h L(h/T).
//     Gilbert and Landau-Lifshitz damping forms both reproduce the analytic
//     damped precession and (with the matching noise strength) the Gibbs state.
//  7. 2DCS W1: M1 synthesised by time translation equals the integrated M1
//     in all three channels, including τ < 0, T_start inside the pump window
//     and a staggered channel with non-trivial frames and signs.
//  8. SU(3) Gilbert damping dissipates: dE/dt = -(α/|S|) Σ |H × S|² and
//     |S| is conserved (the damping used to be ignored for spin_dim = 8);
//     unsupported spin dimensions are rejected.
#include "physics_test_util.h"
#include "classical_spin/core/spin_config.h"
#include "classical_spin/dynamics/ode_method.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <unistd.h>

using namespace phys_test;
namespace fs = std::filesystem;
using classical_spin::dynamics::Pulse;
using classical_spin::dynamics::TimeGrid;

namespace {

using State = std::vector<double>;

fs::path scratch_dir(const std::string& name) {
    fs::path p = fs::temp_directory_path() / ("csmd_" + std::to_string(::getpid()) + "_" + name);
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

std::vector<double> h5_read(const std::string& file, const std::string& path) {
    H5::H5File f(file, H5F_ACC_RDONLY);
    H5::DataSet ds = f.openDataSet(path);
    std::vector<double> v(size_t(ds.getSpace().getSimpleExtentNpoints()));
    ds.read(v.data(), H5::PredType::NATIVE_DOUBLE);
    return v;
}

std::vector<int> h5_read_int(const std::string& file, const std::string& path) {
    H5::H5File f(file, H5F_ACC_RDONLY);
    H5::DataSet ds = f.openDataSet(path);
    std::vector<int> v(size_t(ds.getSpace().getSimpleExtentNpoints()));
    ds.read(v.data(), H5::PredType::NATIVE_INT);
    return v;
}

double h5_attr(const std::string& file, const std::string& group, const std::string& name) {
    H5::H5File f(file, H5F_ACC_RDONLY);
    double v = 0.0;
    f.openGroup(group).openAttribute(name).read(H5::PredType::NATIVE_DOUBLE, &v);
    return v;
}

template<class F>
bool throws_invalid_argument(F&& f, const std::string& must_contain = "") {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return must_contain.empty() || std::string(e.what()).find(must_contain) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

std::string sci(double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.2e", v);
    return b;
}

std::vector<SpinVector> uniform_dir(const Lattice& lat, const Eigen::Vector3d& d) {
    return std::vector<SpinVector>(lat.N_atoms, SpinVector(d));
}

// ----------------------------------------------------------- 1. exact grid
void test_exact_grid_and_seams() {
    std::printf("\n== Exact time grid, analytic pulsed precession, no seams ==\n");
    const double h = 1.0, A = 0.7, w = 0.5;
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(0, 0, h), 0);
    Lattice lat(uc, 1, 1, 1, 1.0f);
    lat.spins[0] = Eigen::Vector3d(1, 0, 0);
    const auto z = uniform_dir(lat, Eigen::Vector3d(0, 0, 1));
    const double T0 = 0.0, T1 = 40.0, dt = 0.01;
    const size_t n_expected = 4001;
    // B = b(t) z with b = h + Σ_p A exp(-((t - t_p)/2w)^2): S(t) = (cos φ, -sin φ, 0),
    // φ(t) = ∫ b = h (t - T0) + Σ_p A w √π [erf((t - t_p)/2w) - erf((T0 - t_p)/2w)].
    auto phi = [&](double t, double tau) {
        double p = h * (t - T0);
        for (double tp : {0.0, tau})
            p += A * w * std::sqrt(M_PI) * (std::erf((t - tp) / (2 * w)) - std::erf((T0 - tp) / (2 * w)));
        return p;
    };
    // 14.0 and 20.3 lost a step at a seam in the chunked driver.
    const std::vector<double> taus = {3.7, 14.0, 17.13, 20.3, 26.55};
    for (const char* method : {"dopri5", "rk4", "bulirsch_stoer", "spherical_midpoint"}) {
        bool grid_ok = true;
        double max_err = 0.0;
        for (double tau : taus) {
            auto traj = lat.double_pulse_drive(z, 0.0, z, tau, A, w, 0.0, T0, T1, dt, method,
                                               false, /*pulse_window_chunking=*/true, 1e-11, 1e-11);
            grid_ok = grid_ok && traj.size() == n_expected;
            for (size_t k = 0; k < traj.size(); ++k) {
                grid_ok = grid_ok && (traj[k].first == T0 + double(k) * dt);
                const auto& m = traj[k].second[1];
                const double p = phi(traj[k].first, tau);
                max_err = std::max(max_err, std::hypot(m(0) - std::cos(p), m(1) + std::sin(p)) + std::abs(m(2)));
            }
        }
        // spherical_midpoint is second order: phase error ~ (b dt)^2 t / 12 ≈ 1e-3 here.
        const double tol = (std::string(method) == "spherical_midpoint") ? 5e-3 : 1e-6;
        check(grid_ok, std::string(method) + ": every delay gives 4001 samples at T_start + k T_step exactly");
        check(max_err < tol, std::string(method) + ": max |M - exact| = " + sci(max_err));
    }
}

// ------------------------------------------------------ 2. zero probe
void test_zero_probe_is_inert() {
    std::printf("\n== Zero-amplitude probe: M01 == M0 for every delay ==\n");
    Eigen::Matrix3d J;
    J << 0.8, 0.3, -0.2, 0.3, 0.5, 0.4, -0.2, 0.4, -0.6;
    Lattice lat(chain_cell(J, Eigen::Vector3d(0.3, -0.2, 0.5), Eigen::Vector3d(0.0, 0.2, -0.9).asDiagonal()),
                6, 1, 1, 1.0f);
    seed_lehman(5);
    lat.init_random();
    const auto x = uniform_dir(lat, Eigen::Vector3d(1, 0, 0));
    const auto zero = uniform_dir(lat, Eigen::Vector3d(0, 0, 0));
    const auto M0 = lat.single_pulse_drive(x, 0.0, 0.4, 0.3, 1.5, -2.0, 12.0, 0.01, "dopri5");
    double worst = 0.0;
    bool same_len = true;
    for (double tau : {0.37, 2.0, 4.11, 7.5, 9.99}) {
        const auto M01 = lat.double_pulse_drive(x, 0.0, zero, tau, 0.4, 0.3, 1.5, -2.0, 12.0, 0.01, "dopri5");
        same_len = same_len && M01.size() == M0.size();
        for (size_t k = 0; k < std::min(M0.size(), M01.size()); ++k)
            for (int c = 0; c < 3; ++c) worst = std::max(worst, (M01[k].second[c] - M0[k].second[c]).cwiseAbs().maxCoeff());
    }
    check(same_len, "trajectory length independent of the delay");
    check(worst == 0.0, "M_NL = M01 - M0 vanishes identically (max " + sci(worst) + ")");
}

// ------------------------------------------------------- 3. drive frame
void test_drive_frame() {
    std::printf("\n== Drive frame: F dS/dt = S_g x B_g for a non-symmetric frame ==\n");
    Eigen::Matrix3d F;   // Kitaev frame (columns: local axes in global coordinates)
    F << 1 / std::sqrt(6.0), -1 / std::sqrt(2.0), 1 / std::sqrt(3.0),
         1 / std::sqrt(6.0),  1 / std::sqrt(2.0), 1 / std::sqrt(3.0),
        -2 / std::sqrt(6.0),  0.0,                1 / std::sqrt(3.0);
    UnitCell uc = simple_cubic_cell(1);
    uc.set_sublattice_frame(F, 0);
    Lattice lat(uc, 1, 1, 1, 1.0f);
    const Eigen::Vector3d S_g(0, 0, 1), B_g(1, 0, 0);
    const Eigen::Vector3d S_l = F.transpose() * S_g;
    lat.spins[0] = S_l;
    const double A = 0.8;
    lat.set_pulse(uniform_dir(lat, B_g), 2.0, uniform_dir(lat, Eigen::Vector3d::Zero()), 0.0, A, 1.0, 0.0);
    State xs = {S_l(0), S_l(1), S_l(2)}, dx(3);
    lat.landau_lifshitz_flat(xs.data(), dx.data(), 2.0);   // envelope = A at the pulse centre
    const Eigen::Vector3d dS_g = F * Eigen::Vector3d(dx[0], dx[1], dx[2]);
    const Eigen::Vector3d expected = S_g.cross(A * B_g);
    check_close((dS_g - expected).norm(), 0.0, 1e-14, "F dS_local/dt = S_g x B_g (= (0, 0.8, 0))");
    const auto pol = lat.local_polarisation(uniform_dir(lat, B_g));
    check_close((Eigen::Vector3d(pol[0], pol[1], pol[2]) - F.transpose() * B_g).norm(), 0.0, 1e-15,
                "local polarisation = F^T B_global");
    // End to end: a weak DC drive rotates M_global about B_g (to first order
    // in t: dM_g/dt = M_g x B_g, i.e. +y for M_g = z and B_g = x).
    lat.clear_pulse();
    Lattice::DriveSchedule d = lat.make_drive();
    lat.add_pulse(d, uniform_dir(lat, B_g), Pulse{0.0, A, 1e6, 0.0});
    auto traj = lat.drive_trajectory(d, TimeGrid::covering(0.0, 0.01, 0.01), {"dopri5", 1e-3, 1e-12, 1e-12, 0.0});
    const Eigen::Vector3d Mg = traj.back().second[2];
    check(Mg(1) > 0.0 && std::abs(Mg(1) - std::sin(A * 0.01)) < 1e-10,
          "global magnetisation precesses about the global drive axis");
}

// ------------------------------------------------- 4. names and grids
void test_validation() {
    std::printf("\n== Integrator names and grid validation ==\n");
    using classical_spin::dynamics::parse_ode_method;
    check(throws_invalid_argument([] { parse_ode_method("rk45"); }, "valid:"),
          "unknown name throws invalid_argument listing the valid names");
    check(throws_invalid_argument([] { parse_ode_method("rosenbrock4"); }, "spherical_midpoint"),
          "rosenbrock4 is rejected with a pointer to spherical_midpoint");
    check(parse_ode_method("rkf54") == classical_spin::dynamics::OdeMethod::CashKarp54 &&
              parse_ode_method("midpoint") == classical_spin::dynamics::OdeMethod::RK2 &&
              parse_ode_method("implicit_midpoint") == classical_spin::dynamics::OdeMethod::SphericalMidpoint,
          "aliases resolve");
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(0, 0, 1), 0);
    Lattice lat(uc, 2, 1, 1, 1.0f);
    const auto x = uniform_dir(lat, Eigen::Vector3d(1, 0, 0));
    check(throws_invalid_argument([&] { lat.single_pulse_drive(x, 0, 1, 1, 0, 0, 1, 0.1, "rk45"); }),
          "single_pulse_drive rejects an unknown method");
    check(throws_invalid_argument([&] { lat.molecular_dynamics(0, 1, 0.1, "", 1, "rk45"); }),
          "molecular_dynamics rejects an unknown method");
    check(throws_invalid_argument([&] { lat.molecular_dynamics(0, 1, 0.1, "", 0, "rk4"); }),
          "molecular_dynamics rejects save_interval = 0");
    check(throws_invalid_argument([&] { lat.single_pulse_drive(x, 0, 1, 1, 0, 0, 1, 0.0, "rk4"); }),
          "zero time step rejected");
    lat.langevin_temperature = 0.5;
    lat.alpha_gilbert = 0.1;
    check(throws_invalid_argument([&] { lat.single_pulse_drive(x, 0, 1, 1, 0, 0, 1, 0.1, "dopri5"); }, "geometric"),
          "Langevin dynamics with a non-geometric method rejected");
    lat.alpha_gilbert = 0.0;
    check(throws_invalid_argument([&] { lat.single_pulse_drive(x, 0, 1, 1, 0, 0, 1, 0.1, "depondt"); }, "alpha"),
          "Langevin temperature without damping rejected");
    lat.langevin_temperature = 0.0;

    SpinConfig cfg;
    cfg.system = SystemType::HONEYCOMB_KITAEV;
    cfg.simulation = SimulationType::MOLECULAR_DYNAMICS;
    cfg.md_integrator = "rk45";
    check(!cfg.validate(), "SpinConfig::validate rejects md_integrator = rk45 (config load)");
    cfg.md_integrator = "spherical_midpoint";
    check(cfg.validate(), "SpinConfig::validate accepts spherical_midpoint");
    cfg.set_param("langevin_temperature", 0.3);
    check(!cfg.validate(), "SpinConfig::validate rejects a bath without damping");
    cfg.set_param("alpha_gilbert", 0.1);
    check(cfg.validate(), "SpinConfig::validate accepts Langevin MD with damping");
    cfg.md_integrator = "dopri5";
    check(!cfg.validate(), "SpinConfig::validate rejects Langevin MD with dopri5");

    using classical_spin::dynamics::delay_grid;
    check(delay_grid(0.0, 0.3, 0.1).size() == 4, "0 -> 0.3 step 0.1 gives 4 delays (truncation gave 3)");
    check(delay_grid(1.0, -1.0, -0.5).size() == 5 && delay_grid(2.0, 2.0, 0.1).size() == 1,
          "negative step and degenerate range");
    check(delay_grid(0.0, 1.0, 0.3).back() <= 1.0 + 1e-12, "last delay never overshoots tau_end");
    check(throws_invalid_argument([] { delay_grid(0.0, 1.0, 0.0); }) &&
              throws_invalid_argument([] { delay_grid(0.0, 1.0, -0.1); }),
          "zero step and wrong-sign step rejected");
    const TimeGrid g = TimeGrid::covering(0.0, 3.3, 0.07);
    check(g.n == 48 && g[47] == 47 * 0.07, "covering grid: floor((T_end - T_start)/dt) + 1 samples");
}

// --------------------------------------------- 5. MD output grid + diagnostics
void test_md_output() {
    std::printf("\n== molecular_dynamics: uniform output grid and diagnostics ==\n");
    const fs::path dir = scratch_dir("md");
    Lattice lat(triangular_heisenberg_cell(1.0), 4, 4, 1, 1.0f);
    seed_lehman(31);
    lat.init_random();
    const Lattice::SpinConfig before = lat.spins;
    const double E0 = lat.energy_density();
    struct Case { const char* method; double dE_tol; double norm_tol; };
    for (const Case& c : {Case{"dopri5", 1e-5, 1e-5}, Case{"spherical_midpoint", 5e-4, 1e-12},
                          Case{"color_split", 1e-11, 1e-12}}) {
        const std::string out = (dir / c.method).string();
        const size_t save = 7;
        const double dt = 0.01, T1 = 3.3;
        lat.molecular_dynamics(0.0, T1, dt, out, save, c.method);
        const std::string file = out + "/trajectory.h5";
        const auto t = h5_read(file, "/trajectory/times");
        const auto E = h5_read(file, "/trajectory/energy_density");
        const auto ne = h5_read(file, "/trajectory/max_norm_error");
        const auto M = h5_read(file, "/trajectory/magnetization_global");
        const double dt_save = double(save) * dt;
        bool uniform = t.size() == 48 && E.size() == t.size() && ne.size() == t.size() && M.size() == 3 * t.size();
        for (size_t k = 0; k < t.size() && uniform; ++k) uniform = (t[k] == double(k) * dt_save);
        check(uniform, std::string(c.method) + ": 48 samples at k * dt_save exactly (with E and |S| datasets)");
        check(h5_attr(file, "/metadata", "dt_save") == dt_save, std::string(c.method) + ": dt_save attribute");
        double max_dE = 0.0, max_ne = 0.0;
        for (size_t k = 0; k < E.size(); ++k) {
            max_dE = std::max(max_dE, std::abs(E[k] - E0));
            max_ne = std::max(max_ne, ne[k]);
        }
        check(std::abs(E.front() - E0) < 1e-14, std::string(c.method) + ": first energy sample is E(0)");
        check(max_dE < c.dE_tol, std::string(c.method) + ": max |E(t) - E(0)|/N = " + sci(max_dE));
        check(max_ne < c.norm_tol, std::string(c.method) + ": max ||S_i| - s| = " + sci(max_ne));
        std::ifstream fin(out + "/final_spins.txt");
        size_t lines = 0;
        for (std::string line; std::getline(fin, line);) ++lines;
        check(lines == lat.lattice_size, std::string(c.method) + ": final configuration written");
    }
    bool unchanged = true;
    for (size_t i = 0; i < lat.lattice_size; ++i) unchanged = unchanged && (lat.spins[i] - before[i]).norm() == 0.0;
    check(unchanged, "Lattice::spins left unchanged by molecular_dynamics");
    fs::remove_all(dir);
}

// --------------------------------------------- 6. Langevin MD, exact <E>
void test_langevin_md_entry_point() {
    std::printf("\n== Langevin MD via molecular_dynamics: free-spin <E> ==\n");
    const fs::path dir = scratch_dir("langevin");
    const double h = 1.0, T = 0.5;
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(0, 0, h), 0);
    Lattice lat(uc, 16, 16, 1, 1.0f);
    lat.alpha_gilbert = 0.5;
    lat.langevin_temperature = T;
    seed_lehman(2718);
    lat.init_random();
    const double exact = -h * langevin(h / T);
    struct Case { double alpha; classical_spin::dynamics::DampingForm form; const char* label; };
    // Gilbert form at alpha = 1: the Landau-Lifshitz noise strength would sample T/2.
    for (const Case& c : {Case{0.5, classical_spin::dynamics::DampingForm::LandauLifshitz, "Landau-Lifshitz alpha=0.5"},
                          Case{1.0, classical_spin::dynamics::DampingForm::Gilbert, "Gilbert alpha=1"}}) {
        lat.alpha_gilbert = c.alpha;
        lat.damping_form = c.form;
        lat.init_random();
        lat.molecular_dynamics(0.0, 320.0, 0.01, dir.string(), 50, "spherical_midpoint");
        const auto E = h5_read((dir / "trajectory.h5").string(), "/trajectory/energy_density");
        std::vector<double> e(E.begin() + 40, E.end());   // drop t < 20 (relaxation)
        const auto r = batch_means(e);
        check_stat(r.mean, r.err, exact, std::string("spherical_midpoint ") + c.label + " T=0.5 <E>/N", 5.0,
                   0.01 * std::abs(exact));
    }
    fs::remove_all(dir);
}

// ------------------------------------- 6b. damping form, analytic relaxation
void test_damping_forms() {
    std::printf("\n== Damped precession: Landau-Lifshitz and Gilbert forms ==\n");
    // One spin in B = h z: dS/dt = g [S x B - alpha S x (S x B)] gives
    //   S_z(t) = tanh(g alpha h t + atanh S_z(0)),  azimuth phi(t) = phi(0) - g h t,
    // with g = 1 (Landau-Lifshitz) or 1/(1 + alpha^2) (Gilbert).
    const double h = 1.3, alpha = 0.4, Sz0 = -0.6;
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(0, 0, h), 0);
    Lattice lat(uc, 1, 1, 1, 1.0f);
    lat.alpha_gilbert = alpha;
    using classical_spin::dynamics::DampingForm;
    for (DampingForm form : {DampingForm::LandauLifshitz, DampingForm::Gilbert}) {
        lat.damping_form = form;
        const double g = (form == DampingForm::Gilbert) ? 1.0 / (1.0 + alpha * alpha) : 1.0;
        for (const char* method : {"rk4", "dopri5", "spherical_midpoint", "depondt"}) {
            lat.spins[0] = Eigen::Vector3d(std::sqrt(1 - Sz0 * Sz0), 0.0, Sz0);
            auto traj = lat.drive_trajectory(lat.make_drive(), TimeGrid::covering(0.0, 6.0, 0.05),
                                             {method, 0.002, 1e-11, 1e-11, 0.0});
            double err = 0.0;
            for (const auto& [t, M] : traj) {
                const double sz = std::tanh(g * alpha * h * t + std::atanh(Sz0));
                const double r = std::sqrt(1 - sz * sz), phi = -g * h * t;
                err = std::max(err, (M[1] - Eigen::Vector3d(r * std::cos(phi), r * std::sin(phi), sz)).norm());
            }
            const bool second_order = std::string(method) == "spherical_midpoint" || std::string(method) == "depondt";
            check(err < (second_order ? 2e-5 : 1e-8),
                  std::string(form == DampingForm::Gilbert ? "Gilbert" : "Landau-Lifshitz") + " form, " + method +
                      ": max error " + sci(err));
        }
    }
    lat.alpha_gilbert = 0.0;
    lat.damping_form = DampingForm::LandauLifshitz;
}

// --------------------------------------------- 7. 2DCS W1 synthesis
struct ScanData {
    std::vector<double> times, taus;
    std::vector<int> synth;
    std::vector<std::vector<double>> M1, M01;   // per delay: [3 channels][n][3]
    std::vector<double> M0;
};

ScanData read_scan(const std::string& file, size_t n_tau) {
    ScanData d;
    d.times = h5_read(file, "/reference/times");
    d.taus = h5_read(file, "/tau_scan/tau_values");
    d.synth = h5_read_int(file, "/tau_scan/m1_synthesized");
    for (const char* ch : {"M_antiferro", "M_local", "M_global"}) {
        const auto v = h5_read(file, std::string("/reference/") + ch);
        d.M0.insert(d.M0.end(), v.begin(), v.end());
    }
    for (size_t i = 0; i < n_tau; ++i) {
        std::vector<double> m1, m01;
        for (const char* ch : {"antiferro", "local", "global"}) {
            const std::string g = "/tau_scan/tau_" + std::to_string(i) + "/";
            const auto a = h5_read(file, g + "M1_" + ch), b = h5_read(file, g + "M01_" + ch);
            m1.insert(m1.end(), a.begin(), a.end());
            m01.insert(m01.end(), b.begin(), b.end());
        }
        d.M1.push_back(m1);
        d.M01.push_back(m01);
    }
    return d;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

void test_w1_synthesis() {
    std::printf("\n== 2DCS: synthesised M1 == integrated M1 ==\n");
    // Two-sublattice FM chain in a field: the all-z state is an exact
    // equilibrium. A non-symmetric frame on sublattice 1 and AFM signs (+,-)
    // make the three output channels distinct.
    UnitCell uc(3, 2, {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.5, 0, 0)},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    const Eigen::Matrix3d J = -1.0 * Eigen::Matrix3d::Identity();
    uc.set_bilinear_interaction(J, 0, 1, Eigen::Vector3i(0, 0, 0));
    uc.set_bilinear_interaction(J, 1, 0, Eigen::Vector3i(1, 0, 0));
    uc.set_field(Eigen::Vector3d(0, 0, 0.4), 0);
    uc.set_field(Eigen::Vector3d(0, 0, 0.4), 1);
    Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    uc.set_sublattice_frame(R, 1);
    uc.set_afm_sublattice_signs({1.0, -1.0});
    Lattice lat(uc, 4, 1, 1, 1.0f);
    for (auto& s : lat.spins) s = Eigen::Vector3d(0, 0, 1);
    check(lat.stationarity_residual() < 1e-15, "ground state is an exact equilibrium");
    const auto field = uniform_dir(lat, Eigen::Vector3d(1, 0.3, 0));
    const fs::path dir = scratch_dir("w1");
    // Scenario A: pump window inside the run (T_start = -10 <= -9 w), delays
    // down to -4 (the reference extends past T_end). Scenario B: T_start = 0
    // cuts the pump in half; only τ >= 9 w = 4.5 may be synthesised.
    struct Scenario { double T0, T1, tau0, tau1; std::vector<int> synth; };
    const std::vector<Scenario> scen = {
        {-10.0, 20.0, -4.0, 6.0, std::vector<int>(21, 1)},
        {0.0, 15.0, 0.0, 6.0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}}};
    for (size_t sc = 0; sc < scen.size(); ++sc) {
        const auto& S = scen[sc];
        ScanData d[2];
        for (int reuse = 0; reuse < 2; ++reuse) {
            const std::string out = (dir / ("s" + std::to_string(sc) + "_" + std::to_string(reuse))).string();
            lat.pump_probe_spectroscopy(field, 0.3, 0.5, 2.0, S.tau0, S.tau1, 0.5, S.T0, S.T1, 0.05,
                                        0, 0, 0, false, 0, out, "rk4", false, reuse == 1, 1e-10, 1);
            d[reuse] = read_scan(out + "/pump_probe_spectroscopy.h5", S.synth.size());
        }
        const std::string tag = "scenario " + std::string(sc == 0 ? "A" : "B") + ": ";
        check(d[1].synth == S.synth, tag + "W1 applied exactly to the qualifying delays");
        check(std::all_of(d[0].synth.begin(), d[0].synth.end(), [](int v) { return v == 0; }),
              tag + "reuse_m0_for_m1 = false integrates every M1");
        const size_t n = d[0].times.size();
        bool grid_ok = n == size_t(std::llround((S.T1 - S.T0) / 0.05)) + 1;
        for (size_t k = 0; k < n; ++k) grid_ok = grid_ok && d[0].times[k] == S.T0 + double(k) * 0.05;
        check(grid_ok, tag + "reference sampled on the exact grid");
        double worst_m1 = 0.0, worst_m01 = 0.0;
        for (size_t i = 0; i < S.synth.size(); ++i) {
            worst_m1 = std::max(worst_m1, max_diff(d[0].M1[i], d[1].M1[i]));
            worst_m01 = std::max(worst_m01, max_diff(d[0].M01[i], d[1].M01[i]));
        }
        // 1e-9: the drive is truncated where its envelope is 1.6e-9 relative (9 w).
        check(worst_m1 < 1e-9, tag + "max |M1_synth - M1_integrated| (3 channels) = " + sci(worst_m1));
        check(worst_m01 == 0.0 && max_diff(d[0].M0, d[1].M0) == 0.0, tag + "M0 and M01 unaffected by W1");
    }
    fs::remove_all(dir);
}

// --------------------------------------------- 8. SU(3) damping, dimensions
void test_su3_damping_and_dims() {
    std::printf("\n== SU(3) damping dissipates; unsupported dimensions rejected ==\n");
    std::mt19937 gen(99);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    UnitCell uc(8, 1, {Eigen::Vector3d::Zero()},
                {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    Eigen::MatrixXd J(8, 8), A(8, 8);
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b) { J(a, b) = U(gen); A(a, b) = U(gen); }
    Eigen::VectorXd B(8);
    for (int a = 0; a < 8; ++a) B(a) = U(gen);
    uc.set_bilinear_interaction(J, 0, 0, Eigen::Vector3i(1, 0, 0));
    uc.set_onsite_interaction(0.5 * (A + A.transpose()), 0);
    uc.set_field(B, 0);
    Lattice lat(uc, 5, 1, 1, 1.0f);
    seed_lehman(8);
    lat.init_random();
    lat.alpha_gilbert = 0.3;
    State x = lat.spins_to_state(lat.spins), f(x.size());
    lat.landau_lifshitz_flat(x.data(), f.data(), 0.0);
    double dEdt = 0.0, predicted = 0.0, s_dot = 0.0;
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        double H[8], P[8], n2 = 0.0;
        lat.get_local_field_flat(x.data(), i, H);
        cross_prod_SU3_flat(H, &x[8 * i], P, false);
        double p2 = 0.0;
        for (int a = 0; a < 8; ++a) {
            dEdt += H[a] * f[8 * i + a];
            p2 += P[a] * P[a];
            n2 += x[8 * i + a] * x[8 * i + a];
            s_dot += x[8 * i + a] * f[8 * i + a];
        }
        predicted -= lat.alpha_gilbert / std::sqrt(n2) * p2;
    }
    // The RHS is only as good as its field: H must be ∂E/∂S also across the
    // periodic wrap of the chain (boundary bonds once dropped components 3..7).
    double fd_err = 0.0;
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        double H[8];
        lat.get_local_field_flat(x.data(), i, H);
        for (int a = 0; a < 8; ++a) {
            State xp = x, xm = x;
            const double eps = 1e-6;
            xp[8 * i + a] += eps;
            xm[8 * i + a] -= eps;
            const double fd = (lat.total_energy_flat(xp.data()) - lat.total_energy_flat(xm.data())) / (2 * eps);
            fd_err = std::max(fd_err, std::abs(fd - H[a]));
        }
    }
    check(fd_err < 1e-7, "SU(3) field = finite-difference gradient incl. wrapped bonds (" + sci(fd_err) + ")");
    check(predicted < -1e-3, "random SU(3) state has a non-zero dissipation rate");
    check_close(dEdt, predicted, 1e-12 * std::abs(predicted) + 1e-14, "dE/dt = -(alpha/|S|) sum |H x S|^2");
    check_close(s_dot, 0.0, 1e-13, "S . dS/dt = 0 (|S| conserved)");
    // Along a damped trajectory the energy decreases monotonically.
    double E_prev = lat.total_energy_flat(x.data());
    bool monotone = true;
    lat.integrate_on_grid(x, TimeGrid::covering(0.0, 10.0, 0.05), lat.make_drive(), {"rk4", 0.005},
                          [&](const double* s, size_t, double) {
                              const double E = lat.total_energy_flat(s);
                              monotone = monotone && E <= E_prev + 1e-10;
                              E_prev = E;
                          });
    check(monotone, "SU(3) energy non-increasing along the damped trajectory");

    UnitCell uc4(4, 1, {Eigen::Vector3d::Zero()},
                 {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)});
    uc4.set_bilinear_interaction(Eigen::MatrixXd::Identity(4, 4), 0, 0, Eigen::Vector3i(1, 0, 0));
    Lattice lat4(uc4, 3, 1, 1, 1.0f);
    State x4 = lat4.spins_to_state(lat4.spins);
    check(throws_invalid_argument([&] {
              lat4.integrate_on_grid(x4, TimeGrid::covering(0.0, 1.0, 0.1), lat4.make_drive(), {"rk4", 0.1},
                                     [](const double*, size_t, double) {});
          }, "spin_dim"),
          "spin_dim = 4 rejected with invalid_argument (was a stack overflow / odeint throw)");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    test_exact_grid_and_seams();
    test_zero_probe_is_inert();
    test_drive_frame();
    test_validation();
    test_md_output();
    test_langevin_md_entry_point();
    test_damping_forms();
    test_w1_synthesis();
    test_su3_damping_and_dims();
    const int rc = finish("test_md_drivers");
    MPI_Finalize();
    return rc;
}
