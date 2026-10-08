// test_observables_mpi.cpp — RealSpaceCorrelationAccumulator::mpi_reduce on
// several ranks: rank 0 must end with exactly the statistics of one
// accumulator fed with every rank's samples (each rank regenerates all
// configurations from per-rank seeds, so the reference is computed locally),
// and a geometry mismatch must make EVERY rank throw instead of deadlocking.
#include "physics_test_util.h"

#include "classical_spin/lattice/correlation_accumulator.h"

#include <mpi.h>
#include <random>
#include <stdexcept>

using namespace phys_test;
using Acc = RealSpaceCorrelationAccumulator;

namespace {

void fill_random(Lattice& lat, std::mt19937_64& g) {
    std::normal_distribution<double> n(0.0, 1.0);
    for (auto& s : lat.spins) {
        Eigen::Vector3d v(n(g), n(g), n(g));
        s = v / v.norm();
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t L = 3;
    Lattice lat(pyrochlore_heisenberg_cell(1.0), L, L, L, 1.0);
    Acc mine = lat.create_correlation_accumulator(Acc::Options{4}, 1);
    Acc reference = lat.create_correlation_accumulator(Acc::Options{4}, 1);
    // Rank r contributes 3 + r samples from seed 100 + r; bins are merged
    // bin by bin, so the reference adds each rank's samples into its own
    // accumulator and merges them the same way.
    for (int r = 0; r < size; ++r) {
        std::mt19937_64 g(100 + r);
        Acc part = lat.create_correlation_accumulator(Acc::Options{4}, 1);
        for (int k = 0; k < 3 + r; ++k) {
            fill_random(lat, g);
            part.add_sample(lat.spins);
            if (r == rank) mine.add_sample(lat.spins);
        }
        reference.merge(part);
    }
    mine.mpi_reduce(MPI_COMM_WORLD);

    if (rank == 0) {
        const Eigen::Vector3d q = mine.grid_wavevector(1, 2, -1);
        const double dS = (mine.structure_factor(q) - reference.structure_factor(q)).cwiseAbs().maxCoeff();
        const double dC = (mine.structure_factor(Eigen::Vector3d::Zero(), true) -
                           reference.structure_factor(Eigen::Vector3d::Zero(), true)).cwiseAbs().maxCoeff();
        const auto e1 = mine.structure_factor_estimate(q), e2 = reference.structure_factor_estimate(q);
        const double dE = (e1.error - e2.error).cwiseAbs().maxCoeff();
        double dD = 0.0;
        for (size_t c = 0; c < mine.bond_classes().size(); ++c)
            for (size_t a = 0; a < 3; ++a) dD = std::max(dD, std::abs(mine.dimer_mean(c, a) - reference.dimer_mean(c, a)));
        check(mine.n_samples() == reference.n_samples() && mine.n_dimer_samples() == reference.n_dimer_samples(),
              "mpi_reduce: sample counts of all ranks on rank 0");
        check(dS < 1e-12 && dC < 1e-12, "mpi_reduce: S(q) and connected S(0) equal the merged reference");
        check(dE < 1e-12, "mpi_reduce: jackknife errors equal the merged reference (bins reduced bin by bin)");
        check(dD < 1e-14, "mpi_reduce: dimer means equal the merged reference");
    }

    // Mismatched geometry on one rank: every rank must throw (collective check).
    Lattice other(pyrochlore_heisenberg_cell(1.0), (rank == size - 1) ? 4 : 3, 3, 3, 1.0);
    Acc odd = other.create_correlation_accumulator(Acc::Options{4}, 1);
    int threw = 0;
    try {
        odd.mpi_reduce(MPI_COMM_WORLD);
    } catch (const std::invalid_argument&) {
        threw = 1;
    }
    int all_threw = 0;
    MPI_Allreduce(&threw, &all_threw, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) check(size == 1 || all_threw == 1, "mpi_reduce: a geometry mismatch throws on every rank");

    int failed = failures();
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int rc = (rank == 0) ? finish("test_observables_mpi") : 0;
    MPI_Finalize();
    return (failed == 0) ? rc : 1;
}
