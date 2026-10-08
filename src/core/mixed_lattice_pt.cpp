/**
 * @file mixed_lattice_pt.cpp
 * @brief MixedLattice adapter for the shared replica-exchange engine.
 *
 * Parallel tempering and ladder tuning live in mc/parallel_tempering.h; this
 * adapter defines the MixedLattice replica: its state (SU(2) then SU(3)
 * components, so both species travel together on an accepted swap), its MC
 * step (MixedLattice::local_sweep with the configured local_update, SU(3)
 * sites on su3_mc_manifold, cell-interleaved when mixed couplings exist), the SU(2)
 * and SU(3) magnetisations as order parameters, and the MixedLattice
 * per-temperature HDF5 file.
 */

#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/mc/parallel_tempering.h"

#include <mpi.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

class MixedLatticeReplica {
public:
    MixedLatticeReplica(MixedLattice& lat, size_t overrelaxation_rate, bool gaussian_move, bool use_interleaved,
                        bool save_spins = false)
        : lat_(lat), or_rate_(overrelaxation_rate), gaussian_(gaussian_move),
          interleaved_(use_interleaved && (lat.num_bi_SU2_SU3 > 0 || lat.num_tri_SU2_SU3 > 0)),
          save_spins_(save_spins) {}

    // One step = one overrelaxation sweep at T (Metropolis-corrected where the
    // local energy is not linear in the site's own spin) when or_rate_ > 0,
    // plus a local sweep (policy local_update) every or_rate_-th step.
    double mc_step(double T, size_t step, double sigma) {
        if (or_rate_ > 0) {
            lat_.overrelaxation_sweep(T, interleaved_);
            if (step % or_rate_ != 0) return std::numeric_limits<double>::quiet_NaN();
        }
        return lat_.local_sweep(T, gaussian_, sigma, interleaved_);
    }

    double energy() const { return lat_.total_energy(); }
    size_t n_sites() const { return lat_.lattice_size_SU2 + lat_.lattice_size_SU3; }
    bool uses_step_size() const { return lat_.uses_adaptive_step(gaussian_); }
    size_t state_size() const {
        return lat_.lattice_size_SU2 * lat_.spin_dim_SU2 + lat_.lattice_size_SU3 * lat_.spin_dim_SU3;
    }

    void pack_state(double* out) const {
        const size_t d2 = lat_.spin_dim_SU2, d3 = lat_.spin_dim_SU3;
        for (size_t i = 0; i < lat_.lattice_size_SU2; ++i)
            for (size_t a = 0; a < d2; ++a) *out++ = lat_.spins_SU2[i](a);
        for (size_t i = 0; i < lat_.lattice_size_SU3; ++i)
            for (size_t a = 0; a < d3; ++a) *out++ = lat_.spins_SU3[i](a);
    }

    void unpack_state(const double* in) {
        const size_t d2 = lat_.spin_dim_SU2, d3 = lat_.spin_dim_SU3;
        for (size_t i = 0; i < lat_.lattice_size_SU2; ++i)
            for (size_t a = 0; a < d2; ++a) lat_.spins_SU2[i](a) = *in++;
        for (size_t i = 0; i < lat_.lattice_size_SU3; ++i)
            for (size_t a = 0; a < d3; ++a) lat_.spins_SU3[i](a) = *in++;
    }

    void measure(mc::MeasurementRecorder& rec) {
        SpinVector m2 = lat_.magnetization_SU2(), m3 = lat_.magnetization_SU3();
        rec.record("m_SU2", m2, lat_.lattice_size_SU2);
        rec.record("m_SU3", m3, lat_.lattice_size_SU3);
        magnetizations_.emplace_back(std::move(m2), std::move(m3));
        measurements_.push_back(lat_.measure_all_observables());
    }

    void write_rank_outputs(const mc::PTRankContext& ctx) const {
        const mc::PTResult& r = *ctx.result;
        const mc::PTOptions& o = *ctx.options;
#ifdef HDF5_ENABLED
        const auto obs = lat_.compute_thermodynamic_observables(measurements_, r.temperature);
        lat_.save_thermodynamic_observables_hdf5(ctx.rank_dir, obs, r.energies, magnetizations_, measurements_,
                                                 o.n_equilibration, o.n_measurement, o.probe_every,
                                                 o.exchange_every, or_rate_, r.local_acceptance,
                                                 r.swap_acceptance);
#endif
        if (save_spins_) lat_.save_spin_config_to_dir(ctx.rank_dir, "spins_T=" + std::to_string(r.temperature));
    }

private:
    MixedLattice& lat_;
    size_t or_rate_;
    bool gaussian_, interleaved_, save_spins_;
    std::vector<std::pair<SpinVector, SpinVector>> magnetizations_;
    std::vector<MixedLattice::MixedMeasurement> measurements_;
};

}  // namespace

// ---- MixedLattice::parallel_tempering ----
mc::PTResult MixedLattice::parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                                              size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                                              string dir_name, const vector<int>& rank_to_write,
                                              bool gaussian_move, bool use_interleaved, MPI_Comm comm,
                                              bool verbose) {
    project_SU3_to_manifold();   // sample SU(3) sites on su3_mc_manifold from the first step
    MixedLatticeReplica replica(*this, overrelaxation_rate, gaussian_move, use_interleaved, verbose);
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

// ---- MixedLattice::tune_temperature_ladder ----
mc::LadderTuningResult MixedLattice::tune_temperature_ladder(const mc::LadderTuningOptions& options,
                                                             size_t overrelaxation_rate, bool gaussian_move,
                                                             bool use_interleaved, MPI_Comm comm) {
    project_SU3_to_manifold();
    MixedLatticeReplica replica(*this, overrelaxation_rate, gaussian_move, use_interleaved);
    return mc::tune_temperature_ladder(replica, options, comm);
}
