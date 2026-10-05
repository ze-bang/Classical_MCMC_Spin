// test_md_exact.cpp — spin-dynamics equations of motion vs. exact results.
//
// The Landau-Lifshitz right-hand side is checked independently of any
// library integrator (a reference RK4 lives in this file), so a failure
// isolates the physics of the RHS from the numerics of the stepper:
//
//  1. Single spin in a field precesses at omega = h with the sense of
//     dS/dt = S x B  (B = -dE/dS).
//  2. On-site anisotropy E = D Sz^2 gives omega = 2 D Sz (factor 2 in H_eff).
//  3. The flow is Hamiltonian: energy, every |S_i| and (for isotropic
//     exchange) total M are conserved, including with trilinear couplings.
//  4. FM chain magnons follow omega(k) = 2|J|S(1 - cos k) (linear spin waves).
//  5. Gilbert damping dissipates: dE/dt <= 0 for random configurations.
//  6. Library integrators (via single_pulse_drive with zero drive) reproduce
//     the analytic precession.
#include "physics_test_util.h"

#include <random>

using namespace phys_test;

namespace {

using State = std::vector<double>;

// Reference RK4 on the library's LL right-hand side.
void rk4_integrate(const Lattice& lat, State& x, double t0, double t1, double dt,
                   const std::function<void(const State&, double)>& obs = nullptr) {
    const size_t n = x.size();
    State k1(n), k2(n), k3(n), k4(n), tmp(n);
    const long steps = std::lround((t1 - t0) / dt);
    double t = t0;
    if (obs) obs(x, t);
    for (long s = 0; s < steps; ++s) {
        lat.landau_lifshitz_flat(x.data(), k1.data(), t);
        for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + 0.5 * dt * k1[i];
        lat.landau_lifshitz_flat(tmp.data(), k2.data(), t + 0.5 * dt);
        for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + 0.5 * dt * k2[i];
        lat.landau_lifshitz_flat(tmp.data(), k3.data(), t + 0.5 * dt);
        for (size_t i = 0; i < n; ++i) tmp[i] = x[i] + dt * k3[i];
        lat.landau_lifshitz_flat(tmp.data(), k4.data(), t + dt);
        for (size_t i = 0; i < n; ++i) x[i] += dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        t += dt;
        if (obs) obs(x, t);
    }
}

double energy_of(Lattice& lat, const State& x) {
    for (size_t i = 0; i < lat.lattice_size; ++i)
        for (size_t d = 0; d < lat.spin_dim; ++d) lat.spins[i](d) = x[i * lat.spin_dim + d];
    return lat.total_energy();
}

// ------------------------------------------------------------ precession
void test_single_spin_precession() {
    std::printf("\n== Single spin precession ==\n");
    const double h = 1.3;
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(0, 0, h), 0);
    Lattice lat(uc, 1, 1, 1, 1.0f);
    State x = {1.0, 0.0, 0.0};
    const double t1 = 10.0;
    rk4_integrate(lat, x, 0.0, t1, 1e-3);
    // dS/dt = S x B with B = h z  =>  S(t) = (cos ht, -sin ht, 0)
    const double err = std::hypot(x[0] - std::cos(h * t1), x[1] + std::sin(h * t1)) + std::abs(x[2]);
    check_close(err, 0.0, 1e-9, "S(t) = (cos ht, -sin ht, 0) for B = h z");
}

void test_onsite_precession() {
    std::printf("\n== On-site anisotropy precession (factor 2) ==\n");
    const double D = 0.7, Sz = 0.6;
    UnitCell uc = simple_cubic_cell(1);
    uc.set_onsite_interaction(Eigen::Vector3d(0, 0, D).asDiagonal(), 0);
    Lattice lat(uc, 1, 1, 1, 1.0f);
    const double st = std::sqrt(1 - Sz * Sz);
    State x = {st, 0.0, Sz};
    const double t1 = 8.0;
    rk4_integrate(lat, x, 0.0, t1, 1e-3);
    // B_eff = -dE/dS = -2 D Sz z  =>  phi(t) = +2 D Sz t
    const double w = 2.0 * D * Sz;
    const double err = std::hypot(x[0] - st * std::cos(w * t1), x[1] - st * std::sin(w * t1)) +
                       std::abs(x[2] - Sz);
    check_close(err, 0.0, 1e-9, "omega = 2 D Sz about z");
}

// -------------------------------------------------------- conservation laws
void check_conservation(Lattice& lat, const std::string& label, bool isotropic, double t1) {
    std::mt19937 gen(7);
    State x(lat.lattice_size * 3);
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        SpinVector s = lat.gen_random_spin(lat.spin_length);
        for (int d = 0; d < 3; ++d) x[i * 3 + d] = s(d);
    }
    const double E0 = energy_of(lat, x);
    Eigen::Vector3d M0 = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < lat.lattice_size; ++i) M0 += Eigen::Vector3d(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    rk4_integrate(lat, x, 0.0, t1, 2e-3);
    const double E1 = energy_of(lat, x);
    Eigen::Vector3d M1 = Eigen::Vector3d::Zero();
    double max_norm_err = 0.0;
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        Eigen::Vector3d s(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        M1 += s;
        max_norm_err = std::max(max_norm_err, std::abs(s.norm() - lat.spin_length));
    }
    check_close((E1 - E0) / double(lat.lattice_size), 0.0, 1e-8, label + " energy conserved");
    check_close(max_norm_err, 0.0, 1e-8, label + " |S_i| conserved");
    if (isotropic) check_close((M1 - M0).norm() / double(lat.lattice_size), 0.0, 1e-8,
                               label + " total M conserved");
}

void test_conservation() {
    std::printf("\n== Hamiltonian flow: conservation laws ==\n");
    seed_lehman(11);
    {
        Lattice lat(chain_cell(-1.0 * Eigen::Matrix3d::Identity()), 32, 1, 1, 1.0f);
        check_conservation(lat, "FM Heisenberg chain", true, 10.0);
    }
    {
        Lattice lat(triangular_heisenberg_cell(1.0), 6, 6, 1, 1.0f);
        check_conservation(lat, "AFM triangular", true, 5.0);
    }
    {
        Eigen::Matrix3d J;
        J << 0.8, 0.3, -0.2, -0.1, 0.5, 0.4, 0.25, -0.35, -0.6;
        Lattice lat(chain_cell(J, Eigen::Vector3d(0.3, -0.2, 0.5),
                               Eigen::Vector3d(0.0, 0.2, -0.9).asDiagonal()), 16, 1, 1, 1.0f);
        check_conservation(lat, "anisotropic chain + field + on-site", false, 5.0);
    }
    {
        std::mt19937 gen(2024);
        std::uniform_real_distribution<double> U(-0.5, 0.5);
        SpinTensor3 Tt(3, Eigen::MatrixXd::Zero(3, 3));
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int c = 0; c < 3; ++c) Tt[a](b, c) = U(gen);
        UnitCell uc = chain_cell(0.4 * Eigen::Matrix3d::Identity());
        uc.set_trilinear_interaction(Tt, 0, 0, 0, Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(2, 0, 0));
        Lattice lat(uc, 8, 1, 1, 1.0f);
        check_conservation(lat, "chain + trilinear", false, 5.0);
    }
}

// -------------------------------------------------------------- magnons
void test_magnon_dispersion() {
    std::printf("\n== FM chain magnon dispersion ==\n");
    const double J = -1.0, S = 1.0, eps = 1e-3;
    const size_t N = 32;
    Lattice lat(chain_cell(J * Eigen::Matrix3d::Identity()), N, 1, 1, float(S));
    for (int m : {1, 4, 9}) {
        const double k = 2.0 * M_PI * m / N;
        State x(3 * N);
        for (size_t i = 0; i < N; ++i) {
            x[3 * i + 0] = eps * std::cos(k * i);
            x[3 * i + 1] = eps * std::sin(k * i);
            x[3 * i + 2] = std::sqrt(S * S - eps * eps);
        }
        std::vector<double> ts, ph;
        rk4_integrate(lat, x, 0.0, 6.0, 1e-3, [&](const State& s, double t) {
            std::complex<double> mk = 0;
            for (size_t i = 0; i < N; ++i)
                mk += std::complex<double>(s[3 * i], s[3 * i + 1]) * std::exp(std::complex<double>(0, -k * i));
            ts.push_back(t);
            ph.push_back(std::arg(mk));
        });
        for (size_t i = 1; i < ph.size(); ++i) {  // unwrap
            while (ph[i] - ph[i - 1] > M_PI) ph[i] -= 2 * M_PI;
            while (ph[i] - ph[i - 1] < -M_PI) ph[i] += 2 * M_PI;
        }
        double st = 0, sp = 0, stt = 0, stp = 0;
        for (size_t i = 0; i < ts.size(); ++i) { st += ts[i]; sp += ph[i]; stt += ts[i] * ts[i]; stp += ts[i] * ph[i]; }
        const double n = double(ts.size());
        const double slope = (n * stp - st * sp) / (n * stt - st * st);
        const double w_lswt = 2.0 * std::abs(J) * S * (1.0 - std::cos(k));
        check_close(-slope, w_lswt, 1e-5 * std::max(1.0, w_lswt),
                    "omega(k=2pi*" + std::to_string(m) + "/32) = 2|J|S(1-cos k)");
    }
}

// --------------------------------------------------------------- damping
void test_damping_dissipates() {
    std::printf("\n== Gilbert damping dissipates energy ==\n");
    Eigen::Matrix3d J;
    J << 0.8, 0.3, -0.2, -0.1, 0.5, 0.4, 0.25, -0.35, -0.6;
    Lattice lat(chain_cell(J, Eigen::Vector3d(0.3, -0.2, 0.5),
                           Eigen::Vector3d(0.0, 0.2, -0.9).asDiagonal()), 16, 1, 1, 1.0f);
    lat.alpha_gilbert = 0.5;
    seed_lehman(3);
    State x(lat.lattice_size * 3);
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        SpinVector s = lat.gen_random_spin(1.0f);
        for (int d = 0; d < 3; ++d) x[i * 3 + d] = s(d);
    }
    double E_prev = energy_of(lat, x);
    bool monotone = true;
    rk4_integrate(lat, x, 0.0, 120.0, 2e-3, [&](const State& s, double) {
        const double E = energy_of(lat, s);
        if (E > E_prev + 1e-10) monotone = false;
        E_prev = E;
    });
    check(monotone, "energy non-increasing along damped trajectory");
    // At the end the configuration should be (close to) stationary.
    State dx(x.size());
    lat.landau_lifshitz_flat(x.data(), dx.data(), 0.0);
    double mx = 0;
    for (double v : dx) mx = std::max(mx, std::abs(v));
    check(mx < 1e-3, "damped dynamics reaches a stationary point (max|dS/dt| = " + std::to_string(mx) + ")");
}

// ------------------------------------------------- library integrators
void test_library_integrators() {
    std::printf("\n== Library integrators: single spin precession ==\n");
    const double h = 1.0;
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(0, 0, h), 0);
    Lattice lat(uc, 1, 1, 1, 1.0f);
    std::vector<SpinVector> no_drive(1, SpinVector::Zero(3));
    for (const char* method : {"rk4", "dopri5", "rk78", "bulirsch_stoer"}) {
        lat.spins[0] = Eigen::Vector3d(1, 0, 0);
        auto traj = lat.single_pulse_drive(no_drive, 0.0, 0.0, 1.0, 0.0, 0.0, 20.0, 0.01,
                                           method, false, false, 1e-10, 1e-10);
        double max_err = 0.0;
        for (const auto& [t, M] : traj) {
            const auto& m = M[1];
            max_err = std::max(max_err, std::hypot(m(0) - std::cos(h * t), m(1) + std::sin(h * t)) +
                                            std::abs(m(2)));
        }
        check(!traj.empty() && max_err < 1e-6,
              std::string(method) + " precession max error " + std::to_string(max_err));
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    test_single_spin_precession();
    test_onsite_precession();
    test_conservation();
    test_magnon_dispersion();
    test_damping_dissipates();
    test_library_integrators();
    const int rc = finish("test_md_exact");
    MPI_Finalize();
    return rc;
}
