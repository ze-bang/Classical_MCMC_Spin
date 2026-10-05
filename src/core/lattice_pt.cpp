/**
 * @file lattice_pt.cpp
 * @brief Lattice adapter for the shared replica-exchange engine.
 *
 * Parallel tempering and temperature-ladder tuning are implemented once in
 * mc/parallel_tempering.h. This file only supplies what is specific to
 * Lattice: the order parameters it records (global-frame and, when the unit
 * cell defines AFM signs, staggered magnetisation), the optional pyrochlore
 * order parameters and real-space correlation accumulator, and the
 * per-temperature files.
 */

#include "classical_spin/lattice/lattice.h"
#include "classical_spin/mc/parallel_tempering.h"

#include <mpi.h>
#include <algorithm>
#include <string>
#include <vector>

namespace {

/// Lattice replica: SpinLatticeReplica (state, MC step, "m", HDF5 time series)
/// plus the Lattice-only observables.
class LatticeReplica : public mc::SpinLatticeReplica<Lattice> {
    using Base = mc::SpinLatticeReplica<Lattice>;

public:
    LatticeReplica(Lattice& lat, size_t overrelaxation_rate, bool gaussian_move, bool adaptive,
                   bool accumulate_correlations, size_t n_bond_types)
        : Base(lat, overrelaxation_rate, gaussian_move, adaptive),
          accumulate_correlations_(accumulate_correlations),
          pyrochlore_(lat.is_pyrochlore() && lat.N_atoms >= 4) {
        for (double s : lat.afm_sublattice_signs) staggered_ = staggered_ || s != 1.0;
        if (accumulate_correlations_) corr_ = lat.create_correlation_accumulator(n_bond_types);
    }

    void measure(mc::MeasurementRecorder& rec) {
        Base::measure(rec);
        if (staggered_) rec.record("m_staggered", staggered_magnetization(), lat_.lattice_size);
        if (pyrochlore_) {
            const auto p = lat_.compute_pyrochlore_order_parameters_fast();
            scalar_chi_.push_back(p.scalar_chirality);
            vector_chi_.push_back(p.vector_chirality);
            nematic_.push_back(p.nematic_order);
            monopole_.push_back(p.monopole_density);
            monopole_sub_.push_back(p.monopole_by_sublattice);
            scalar_chi_g_.push_back(p.scalar_chirality_global);
            vector_chi_g_.push_back(p.vector_chirality_global);
            nematic_g_.push_back(p.nematic_order_global);
        }
        if (accumulate_correlations_) lat_.accumulate_correlations_internal(corr_);
    }

    void write_rank_outputs(const mc::PTRankContext& ctx) const {
        Base::write_rank_outputs(ctx);
        const double T = ctx.result->temperature;
        if (pyrochlore_ && !scalar_chi_.empty())
            lat_.save_kagome_order_parameters(ctx.rank_dir, T, scalar_chi_, vector_chi_, nematic_, monopole_,
                                              monopole_sub_, scalar_chi_g_, vector_chi_g_, nematic_g_);
#ifdef HDF5_ENABLED
        if (accumulate_correlations_)
            corr_.save_hdf5(ctx.rank_dir + "/correlations_T" + std::to_string(T) + ".h5");
#endif
    }

private:
    /// m_s = (1/N) sum_i eps_{a(i)} R_{a(i)} S_i with the unit cell's AFM signs.
    SpinVector staggered_magnetization() const {
        const size_t d = lat_.spin_dim, na = lat_.N_atoms;
        std::vector<SpinVector> sum(na, SpinVector::Zero(d));
        for (size_t i = 0; i < lat_.lattice_size; ++i) sum[i % na] += lat_.spins[i];
        SpinVector ms = SpinVector::Zero(d);
        for (size_t a = 0; a < na; ++a) ms += lat_.afm_sublattice_signs[a] * (lat_.sublattice_frames[a] * sum[a]);
        return ms / double(lat_.lattice_size);
    }

    bool accumulate_correlations_ = false, pyrochlore_ = false, staggered_ = false;
    RealSpaceCorrelationAccumulator corr_;
    std::vector<double> scalar_chi_, monopole_, scalar_chi_g_;
    std::vector<Eigen::Vector3d> vector_chi_, vector_chi_g_;
    std::vector<Eigen::Matrix3d> nematic_, nematic_g_;
    std::vector<Eigen::Vector4d> monopole_sub_;
};

/// Gaussian proposals with an adapted width unless the heat bath is selected.
bool adaptive_proposals(const Lattice& lat, bool gaussian_move) {
    const bool heat_bath = lat.local_update == Lattice::LocalUpdate::HeatBath && lat.spin_dim == 3;
    return !heat_bath && (gaussian_move || lat.local_update == Lattice::LocalUpdate::Gaussian);
}

}  // namespace

// ---- Lattice::parallel_tempering ----
mc::PTResult Lattice::parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                                         size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                                         string dir_name, const vector<int>& rank_to_write,
                                         bool gaussian_move, MPI_Comm comm, bool /*verbose*/,
                                         bool accumulate_correlations, size_t n_bond_types) {
    LatticeReplica replica(*this, overrelaxation_rate, gaussian_move,
                           adaptive_proposals(*this, gaussian_move), accumulate_correlations, n_bond_types);
    mc::PTOptions o;
    o.temperatures = std::move(temp);
    o.n_equilibration = n_anneal;
    o.n_measurement = n_measure;
    o.exchange_every = swap_rate;
    o.probe_every = probe_rate;
    o.output_dir = std::move(dir_name);
    o.ranks_to_write = rank_to_write;
    return mc::run_parallel_tempering(replica, o, comm);
}

// ---- Lattice::tune_temperature_ladder ----
mc::LadderTuningResult Lattice::tune_temperature_ladder(const mc::LadderTuningOptions& options,
                                                        size_t overrelaxation_rate, bool gaussian_move,
                                                        MPI_Comm comm) {
    mc::SpinLatticeReplica<Lattice> replica(*this, overrelaxation_rate, gaussian_move,
                                            adaptive_proposals(*this, gaussian_move));
    return mc::tune_temperature_ladder(replica, options, comm);
}
