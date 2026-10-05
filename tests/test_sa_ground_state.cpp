// test_sa_ground_state.cpp — simulated annealing reaches known ground states.
//
//  1. Triangular Heisenberg AFM (L = 6): 120-degree order, E/N = -3/2 J S^2.
//  2. Pyrochlore Heisenberg AFM (L = 2): every tetrahedron has zero total
//     spin, E/N = -J S^2 (massively degenerate manifold; no metastable trap).
//  3. Chain in a field with easy-plane anisotropy: the T = 0 quench must
//     reach the analytic single-spin optimum (checks the quench handles
//     quadratic on-site terms).
#include "physics_test_util.h"

using namespace phys_test;

namespace {

void anneal(Lattice& lat) {
    lat.simulated_annealing(/*T_start=*/3.0, /*T_end=*/0.005, /*n_anneal=*/200,
                            /*overrelaxation_rate=*/0, /*boundary_update=*/false,
                            /*gaussian_move=*/true, /*cooling_rate=*/0.9,
                            /*out_dir=*/"", /*save_observables=*/false,
                            /*T_zero=*/true, /*n_deterministics=*/2000);
}

void test_triangular() {
    std::printf("\n== Triangular AFM ground state ==\n");
    Lattice lat(triangular_heisenberg_cell(1.0), 6, 6, 1, 1.0f);
    seed_lehman(1);
    anneal(lat);
    check_close(lat.energy_density(), -1.5, 1e-6, "triangular AFM E/N = -3/2");
}

void test_pyrochlore() {
    std::printf("\n== Pyrochlore AFM ground state ==\n");
    Lattice lat(pyrochlore_heisenberg_cell(1.0), 2, 2, 2, 1.0f);
    seed_lehman(2);
    anneal(lat);
    check_close(lat.energy_density(), -1.0, 1e-6, "pyrochlore AFM E/N = -1");
}

void test_heat_bath_anneal() {
    std::printf("\n== Heat-bath annealing (pyrochlore AFM, L = 3) ==\n");
    Lattice lat(pyrochlore_heisenberg_cell(1.0), 3, 3, 3, 1.0f);
    lat.local_update = Lattice::LocalUpdate::HeatBath;
    seed_lehman(5);
    lat.simulated_annealing(3.0, 0.005, 200, 0, false, false, 0.85, "", false, true, 5000);
    check_close(lat.energy_density(), -1.0, 1e-6, "heat-bath SA reaches E/N = -1");
}

void test_schedule_validation() {
    std::printf("\n== Schedule validation ==\n");
    auto sched = mc::annealing_schedule(2.0, 0.3, 0.5);
    check(!sched.empty() && sched.back() == 0.3 && sched.front() == 2.0,
          "schedule starts at T_start and ends exactly at T_end");
    bool threw = false;
    try { mc::annealing_schedule(1.0, 0.1, 1.0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "cooling_rate >= 1 rejected (no infinite loop)");
    threw = false;
    try { mc::annealing_schedule(1.0, 0.0, 0.9); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "T_end <= 0 rejected");
}

void test_easy_plane_field() {
    std::printf("\n== Easy-plane anisotropy + tilted field (quench with on-site terms) ==\n");
    // Decoupled spins: E = D Sz^2 - h.S with D = 1, h = (0.5, 0, 0.8).
    // Minimise on the sphere: S = (sin t, 0, cos t),
    // e(t) = D cos^2 t - hx sin t - hz cos t.
    const double D = 1.0, hx = 0.5, hz = 0.8;
    UnitCell uc = simple_cubic_cell(1);
    uc.set_field(Eigen::Vector3d(hx, 0, hz), 0);
    uc.set_onsite_interaction(Eigen::Vector3d(0, 0, D).asDiagonal(), 0);
    Lattice lat(uc, 4, 4, 1, 1.0f);
    double e_min = 1e300;
    for (int i = 0; i <= 2000000; ++i) {
        const double t = M_PI * i / 2000000.0;
        e_min = std::min(e_min, D * std::cos(t) * std::cos(t) - hx * std::sin(t) - hz * std::cos(t));
    }
    seed_lehman(3);
    anneal(lat);
    check_close(lat.energy_density(), e_min, 1e-8, "decoupled easy-plane spins reach analytic minimum");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    test_triangular();
    test_pyrochlore();
    test_easy_plane_field();
    test_heat_bath_anneal();
    test_schedule_validation();
    const int rc = finish("test_sa_ground_state");
    MPI_Finalize();
    return rc;
}
