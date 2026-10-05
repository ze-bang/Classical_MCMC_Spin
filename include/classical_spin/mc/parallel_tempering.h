#pragma once
/**
 * parallel_tempering.h — the replica-exchange (parallel tempering) engine and
 * temperature-ladder tuning shared by Lattice, MixedLattice and PhononLattice.
 *
 * Layout
 * ------
 * One MPI rank holds one replica at the fixed temperature T_k = temperatures[k]
 * (k = rank, strictly increasing: rank 0 is the coldest). A model is plugged in
 * through the `ReplicaModel` concept (one MC step, total energy, state packing,
 * measurement); thin adapters exist for the three lattice classes.
 *
 * Exchange (Okabe et al., CPL 335, 435 (2001); Syed, Bouchard-Cote,
 * Deligiannidis, Doucet, JRSS-B 84, 321 (2022)):
 *  - deterministic even/odd (DEO, non-reversible) rounds: round t proposes the
 *    edges (k, k+1) with k = t mod 2, so round trips scale linearly in the
 *    number of replicas instead of quadratically;
 *  - acceptance min(1, exp[(beta_k - beta_{k+1})(E_k - E_{k+1})]). Both
 *    partners evaluate the same expression on the same operands and compare it
 *    with the same counter-based uniform u(seed, round, edge), so they reach the
 *    identical decision without a second message;
 *  - on acceptance the configurations are exchanged through one persistent
 *    buffer (MPI_Sendrecv_replace) together with the replica label, its
 *    direction (last visited the cold or the hot end) and its round-trip phase,
 *    so real round trips, the up-fraction f(T) (Katzgraber, Trebst, Huse,
 *    Troyer, J. Stat. Mech. P03018 (2006)) and per-edge acceptance are measured;
 *  - the energy of the current configuration is cached: it is recomputed only
 *    after an MC step, and an accepted swap hands over the partner's energy.
 *
 * Proposal widths are adapted per temperature (StepSizeController) only while
 * equilibrating and are frozen while measuring. Inputs are validated on every
 * rank and failures are reported collectively (every rank throws the same
 * exception), so a bad argument or an unwritable output directory never leaves
 * ranks blocked in a collective.
 *
 * Ladder tuning: the default is the non-reversible PT schedule update of Syed
 * et al. (2022): estimate the per-edge rejection rates r_e, build the
 * cumulative communication barrier Lambda(beta) = sum_{e<k} r_e, interpolate it
 * monotonically (Fritsch-Carlson PCHIP) and place the new inverse temperatures
 * at equal Lambda spacing, over rounds of doubling length. The flow-based
 * feedback of Katzgraber et al. (2006) (density of temperatures proportional to
 * sqrt(|df/dT| / dT), with f measured from the replica labels) is available as
 * `katzgraber`.
 */

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "classical_spin/core/simple_linear_alg.h"  // derive_seed_from_master, SpinVector
#include "classical_spin/core/spin_config.h"        // should_rank_write
#include "classical_spin/mc/mc_common.h"
#include "classical_spin/mc/statistics.h"

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#include "classical_spin/io/hdf5_io.h"
#endif

namespace mc {

// =====================================================================
// MEASUREMENT RECORDER
// =====================================================================

/**
 * Collects the per-sample order parameters a model reports from measure().
 * The first sample defines the list; every later sample must record the same
 * names in the same order (checked). `energy()` is the total energy of the
 * configuration being measured (already known to the engine).
 */
class MeasurementRecorder {
public:
    double energy() const { return energy_; }
    size_t n_samples() const { return n_samples_; }

    /// Record vector order parameter `name` (per-site normalised, averaged
    /// over `n_sites` sites; n_sites enters chi = beta N Var(|m|)).
    void record(const std::string& name, const double* v, size_t dim, size_t n_sites) {
        if (n_samples_ == 0) {
            for (const auto& s : series_)
                if (s.name == name)
                    throw std::logic_error("MeasurementRecorder: duplicate order parameter '" + name + "'");
            OrderParameterSeries s;
            s.name = name;
            s.dim = dim;
            s.n_sites = std::max<size_t>(n_sites, 1);
            series_.push_back(std::move(s));
        } else if (cursor_ >= series_.size() || series_[cursor_].name != name ||
                   series_[cursor_].dim != dim) {
            throw std::logic_error("MeasurementRecorder: order parameter '" + name +
                                   "' recorded out of order or with a different dimension");
        }
        OrderParameterSeries& s = series_[cursor_++];
        s.data.insert(s.data.end(), v, v + dim);
    }

    template <class Vec>
    void record(const std::string& name, const Vec& v, size_t n_sites) {
        record(name, v.data(), size_t(v.size()), n_sites);
    }

    const std::vector<OrderParameterSeries>& series() const { return series_; }

    // ---- engine side
    void begin_sample(double E) { energy_ = E; cursor_ = 0; }
    void end_sample() {
        if (cursor_ != series_.size())
            throw std::logic_error("MeasurementRecorder: an order parameter was not recorded");
        ++n_samples_;
    }

private:
    std::vector<OrderParameterSeries> series_;
    size_t cursor_ = 0;
    size_t n_samples_ = 0;
    double energy_ = 0.0;
};

// =====================================================================
// MODEL CONCEPT
// =====================================================================

/**
 * What the engine needs from a replica:
 *  - mc_step(T, step, sigma): one MC step at temperature T (step counts the
 *    calls, so a model can interleave overrelaxation and local sweeps); returns
 *    the acceptance of the local update, or NaN if no local update ran;
 *  - energy(): total energy of the current state;
 *  - n_sites(): number of sites used for per-site normalisation;
 *  - state_size() / pack_state(double*) / unpack_state(const double*): the full
 *    configuration exchanged on an accepted swap (everything energy() depends
 *    on that differs between replicas);
 *  - measure(MeasurementRecorder&): record order parameters (and anything
 *    model-specific the adapter keeps for its own output);
 *  - uses_step_size(): whether `sigma` matters (adaptive Gaussian proposals).
 * Optional:
 *  - write_rank_outputs(const PTRankContext&): per-temperature files;
 *  - replica_invariants(): values that must be identical on every rank for the
 *    exchange to be valid (e.g. frozen non-spin coordinates), checked at start.
 */
template <class M>
concept ReplicaModel = requires(M& m, const M& cm, double T, size_t step, double sigma,
                                double* out, const double* in, MeasurementRecorder& rec) {
    { m.mc_step(T, step, sigma) } -> std::convertible_to<double>;
    { m.energy() } -> std::convertible_to<double>;
    { cm.n_sites() } -> std::convertible_to<size_t>;
    { cm.state_size() } -> std::convertible_to<size_t>;
    cm.pack_state(out);
    m.unpack_state(in);
    m.measure(rec);
    { cm.uses_step_size() } -> std::convertible_to<bool>;
};

struct PTRankContext;

template <class M>
concept HasRankOutputs = requires(M& m, const PTRankContext& c) { m.write_rank_outputs(c); };

template <class M>
concept HasReplicaInvariants = requires(const M& m) {
    { m.replica_invariants() } -> std::convertible_to<std::vector<double>>;
};

// =====================================================================
// OPTIONS AND RESULTS
// =====================================================================

struct PTOptions {
    std::vector<double> temperatures;  ///< one per rank, strictly increasing (rank 0 coldest)
    size_t n_equilibration = 0;        ///< MC steps before measuring (sigma adapts here)
    size_t n_measurement = 0;          ///< MC steps while measuring (sigma frozen)
    size_t exchange_every = 1;         ///< MC steps between exchange rounds (0 = no exchanges)
    size_t probe_every = 1;            ///< MC steps between measurements
    bool adapt_step_size = true;       ///< Robbins-Monro on the Gaussian width while equilibrating
    double initial_step_size = 2.0;
    double target_acceptance = 0.45;
    std::string output_dir;            ///< "" = no files
    std::vector<int> ranks_to_write = {-1};  ///< per-temperature files (-1 = all)
    int verbosity = 1;                 ///< 0 quiet, 1 summary table
};

/// Per-temperature table of one order parameter (cold -> hot).
struct OrderParameterTable {
    std::string name;
    std::vector<double> abs, abs_error, m2, m2_error;
    std::vector<double> susceptibility, susceptibility_error, binder, binder_error;
};

struct PTResult {
    // ---- this rank (temperature index k = rank)
    int rank = 0;
    double temperature = 0.0;
    ThermodynamicObservables thermo;
    std::vector<double> energies;                       ///< total energy per sample
    std::vector<OrderParameterSeries> order_parameters; ///< per-sample order parameters
    double local_acceptance = 0.0;  ///< mean local-update acceptance while measuring
    double step_size = 0.0;         ///< frozen Gaussian width (0 if not adaptive)
    double swap_acceptance = 0.0;   ///< over the edges of this rank while measuring

    // ---- whole ladder (identical on every rank), indexed cold -> hot
    std::vector<double> temperatures;
    std::vector<double> energy, energy_error, specific_heat, specific_heat_error;
    std::vector<double> tau_int_energy;  ///< in MC steps
    std::vector<double> acceptance;      ///< local moves
    std::vector<double> step_sizes;
    std::vector<double> up_fraction;     ///< f(T_k) = n_up / (n_up + n_down); NaN if unvisited
    std::vector<size_t> n_samples;
    std::vector<OrderParameterTable> order_parameter_tables;
    std::vector<uint64_t> edge_attempts, edge_accepts;  ///< edge k = (T_k, T_{k+1})
    std::vector<double> edge_acceptance;
    bool bookkeeping_consistent = true;  ///< both partners recorded identical decisions
    uint64_t exchange_rounds = 0;        ///< while measuring
    uint64_t round_trips = 0;            ///< cold -> hot -> cold, all replicas, while measuring
    std::vector<uint64_t> round_trips_per_replica;
    double round_trip_rate = 0.0;            ///< measured, per exchange round
    double predicted_round_trip_rate = 0.0;  ///< 1 / (2 + 2 sum_e r_e / (1 - r_e))
    double barrier = 0.0;                    ///< Lambda = sum_e r_e
};

/// What a model's write_rank_outputs() receives (rank_dir already exists).
struct PTRankContext {
    int rank = 0;
    int size = 1;
    std::string rank_dir;
    const PTOptions* options = nullptr;
    const PTResult* result = nullptr;
};

// =====================================================================
// DETAIL: RNG, MPI HELPERS, VALIDATION
// =====================================================================

namespace detail {

inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// Counter-based uniform in [0, 1) shared by the two partners of an exchange:
/// a function of (run seed, round, edge) only, hence identical on both ranks.
inline double exchange_uniform(uint64_t seed, uint64_t round, uint64_t edge) {
    const uint64_t z = mix64(seed ^ mix64(round ^ mix64(edge + 0x632BE59BD9B4E019ULL)));
    return double(z >> 11) * 0x1.0p-53;
}

class CommGuard {
public:
    explicit CommGuard(MPI_Comm parent) { MPI_Comm_dup(parent, &comm_); }
    ~CommGuard() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Comm_free(&comm_);
    }
    CommGuard(const CommGuard&) = delete;
    CommGuard& operator=(const CommGuard&) = delete;
    operator MPI_Comm() const { return comm_; }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
};

inline void require_mpi(const char* who) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
        throw std::runtime_error(std::string(who) +
                                 ": MPI is not initialised; call MPI_Init in main() first");
}

/**
 * Collective failure: every rank passes its local error (empty = fine); if any
 * rank failed, all ranks throw the message of the lowest failing rank.
 */
template <class Exception = std::invalid_argument>
inline void throw_if_any(const std::string& local_error, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int mine = local_error.empty() ? INT_MAX : rank, first = INT_MAX;
    MPI_Allreduce(&mine, &first, 1, MPI_INT, MPI_MIN, comm);
    if (first == INT_MAX) return;
    int len = (rank == first) ? int(local_error.size()) : 0;
    MPI_Bcast(&len, 1, MPI_INT, first, comm);
    std::string msg(size_t(len), '\0');
    if (rank == first) msg = local_error;
    MPI_Bcast(msg.data(), len, MPI_CHAR, first, comm);
    throw Exception(msg + " [rank " + std::to_string(first) + "]");
}

/// Ladder checks shared by the engine and the tuner (empty = valid).
inline std::string check_ladder(const std::vector<double>& T) {
    if (T.empty()) return "temperature ladder is empty";
    for (size_t k = 0; k < T.size(); ++k) {
        if (!std::isfinite(T[k]) || !(T[k] > 0.0))
            return "temperature " + std::to_string(k) + " = " + std::to_string(T[k]) +
                   " is not a finite positive number";
        if (k > 0 && !(T[k] > T[k - 1]))
            return "temperatures must be strictly increasing with rank (T[" + std::to_string(k - 1) +
                   "] = " + std::to_string(T[k - 1]) + ", T[" + std::to_string(k) + "] = " +
                   std::to_string(T[k]) + ")";
    }
    return {};
}

/// Collective comparison of a vector with rank 0's copy (bitwise).
inline bool same_on_all_ranks(const std::vector<double>& v, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    unsigned long long n = v.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    std::vector<double> ref(n);
    if (rank == 0) ref = v;
    MPI_Bcast(ref.data(), int(n), MPI_DOUBLE, 0, comm);
    int ok = (n == v.size() && std::equal(ref.begin(), ref.end(), v.begin())) ? 1 : 0, all = 0;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, comm);
    return all == 1;
}

/// Seed of the exchange decisions: one draw from rank 0's stream (so it is
/// reproducible from the run seed, yet differs between successive runs such
/// as tuning, production and further trials), broadcast to all ranks.
inline uint64_t shared_exchange_seed(MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    unsigned long long s = 0;
    if (rank == 0) s = mix64(lehman_next() ^ 0x5054455843484731ULL);
    MPI_Bcast(&s, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    return s;
}

/// Validation common to everything that runs a replica chain.
template <ReplicaModel M>
inline void validate_chain_setup(const M& model, const std::vector<double>& T, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::string err;
    if (T.size() != size_t(size))
        err = "parallel tempering needs one temperature per MPI rank (got " +
              std::to_string(T.size()) + " temperatures for " + std::to_string(size) + " ranks)";
    if (err.empty()) err = check_ladder(T);
    throw_if_any(err, comm);
    if (!same_on_all_ranks(T, comm))
        throw std::invalid_argument("parallel tempering: ranks were given different temperature ladders");
    unsigned long long n = model.state_size(), lo = 0, hi = 0;
    MPI_Allreduce(&n, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(&n, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    if (lo != hi || lo == 0)
        throw std::invalid_argument("parallel tempering: replicas must have equal, non-empty states "
                                    "(state sizes " + std::to_string(lo) + ".." + std::to_string(hi) + ")");
    // Replicas drawing identical random numbers are still valid chains but are
    // strongly correlated: warn if the process seeds were not made rank-specific.
    unsigned long long seed = lehman_master_seed_value(), s_lo = 0, s_hi = 0;
    MPI_Allreduce(&seed, &s_lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(&seed, &s_hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    if (size > 1 && s_lo == s_hi && rank == 0)
        std::cerr << "[PT] WARNING: all ranks share one RNG seed; call seed_lehman_from_rank(rank) "
                     "after seed_lehman(seed) so the replicas use independent streams" << std::endl;
    if constexpr (HasReplicaInvariants<M>) {
        const std::vector<double> inv = model.replica_invariants();
        if (!same_on_all_ranks(inv, comm))
            throw std::invalid_argument(
                "parallel tempering: the configuration outside the exchanged state differs between "
                "ranks (e.g. frozen phonon coordinates); the swap acceptance would be wrong");
    }
}

template <class F>
inline std::string guarded_io(F&& f) {
    try {
        f();
        return {};
    }
#ifdef HDF5_ENABLED
    catch (const H5::Exception& e) {
        return "HDF5 error: " + e.getDetailMsg();
    }
#endif
    catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "unknown exception during output";
    }
}

}  // namespace detail

// =====================================================================
// REPLICA-EXCHANGE CHAIN
// =====================================================================

/**
 * The per-rank state of a replica-exchange run: the model, the ladder, the
 * cached energy, the proposal-width controller of this temperature, the label
 * and direction of the replica currently held, and all exchange counters.
 * Every rank must call step() / exchange_round() the same number of times.
 */
template <ReplicaModel M>
class ReplicaExchange {
public:
    enum Direction : int { kNone = 0, kUp = 1, kDown = -1 };

    ReplicaExchange(M& model, MPI_Comm comm, std::vector<double> temperatures, uint64_t seed,
                    double sigma0 = 2.0, double target_acceptance = 0.45)
        : model_(model), comm_(comm), seed_(seed), ctrl_(sigma0, target_acceptance) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
        set_temperatures(std::move(temperatures));
        label_ = rank_;
        // A replica starting at the cold end has visited it (heading up); one
        // starting at the hot end is heading down but has not been cold yet.
        dir_ = (rank_ == 0) ? kUp : (rank_ == size_ - 1 ? kDown : kNone);
        phase_ = (rank_ == 0) ? 1 : 0;
        buf_.resize(model_.state_size());
        trips_by_label_.assign(size_t(size_), 0);
    }

    /// Install a new ladder (tuning); the controller restarts its gain sequence.
    void set_temperatures(std::vector<double> T) {
        temps_ = std::move(T);
        betas_.resize(temps_.size());
        for (size_t k = 0; k < temps_.size(); ++k) betas_[k] = 1.0 / temps_[k];
        ctrl_.restart();
    }

    const std::vector<double>& temperatures() const { return temps_; }
    double temperature() const { return temps_[size_t(rank_)]; }
    double step_size() const { return model_.uses_step_size() ? ctrl_.sigma() : 0.0; }
    int rank() const { return rank_; }
    int size() const { return size_; }

    /// One MC step; adapts sigma from the local acceptance when `adapt`.
    void step(bool adapt) {
        const double acc = model_.mc_step(temps_[size_t(rank_)], steps_++, ctrl_.sigma());
        e_valid_ = false;
        if (std::isfinite(acc)) {
            acc_sum_ += acc;
            ++acc_n_;
            if (adapt && model_.uses_step_size()) ctrl_.update(acc);
        }
    }

    double energy() {
        if (!e_valid_) {
            e_ = model_.energy();
            e_valid_ = true;
        }
        return e_;
    }

    /// One DEO round: edges (k, k+1) with k = round mod 2 attempt a swap.
    void exchange_round() {
        const uint64_t round = round_++;
        if (size_ > 1) {
            const int parity = int(round & 1u);
            const int partner = (rank_ % 2 == parity) ? rank_ + 1 : rank_ - 1;
            if (partner >= 0 && partner < size_) swap_with(partner, round);
        }
        if (dir_ == kUp) ++n_up_;
        else if (dir_ == kDown) ++n_down_;
        ++rounds_;
    }

    /// Zero the statistics (acceptance, edges, f(T), round trips); keeps state.
    void reset_statistics() {
        acc_sum_ = 0.0;
        acc_n_ = 0;
        lo_att_ = lo_acc_ = hi_att_ = hi_acc_ = 0;
        n_up_ = n_down_ = 0;
        rounds_ = 0;
        trips_ = 0;
        std::fill(trips_by_label_.begin(), trips_by_label_.end(), 0);
    }

    double local_acceptance() const { return acc_n_ ? acc_sum_ / double(acc_n_) : 0.0; }
    uint64_t lower_attempts() const { return lo_att_; }   ///< edge (rank, rank+1)
    uint64_t lower_accepts() const { return lo_acc_; }
    uint64_t upper_attempts() const { return hi_att_; }   ///< edge (rank-1, rank)
    uint64_t upper_accepts() const { return hi_acc_; }
    uint64_t n_up() const { return n_up_; }
    uint64_t n_down() const { return n_down_; }
    uint64_t rounds() const { return rounds_; }
    uint64_t round_trips() const { return trips_; }       ///< counted on rank 0
    const std::vector<uint64_t>& round_trips_by_label() const { return trips_by_label_; }
    int label() const { return label_; }

private:
    static constexpr int kTagEnergy = 0x5054;
    static constexpr int kTagState = 0x5055;

    void swap_with(int partner, uint64_t round) {
        const bool lower = rank_ < partner;
        const size_t lo = size_t(lower ? rank_ : partner);
        double mine[4] = {energy(), double(label_), double(dir_), double(phase_)};
        double theirs[4];
        MPI_Sendrecv(mine, 4, MPI_DOUBLE, partner, kTagEnergy, theirs, 4, MPI_DOUBLE, partner,
                     kTagEnergy, comm_, MPI_STATUS_IGNORE);
        const double E_lo = lower ? mine[0] : theirs[0];
        const double E_hi = lower ? theirs[0] : mine[0];
        // Swap acceptance pi_lo(x_hi) pi_hi(x_lo) / (pi_lo(x_lo) pi_hi(x_hi))
        //   = exp[(beta_lo - beta_hi)(E_lo - E_hi)], evaluated identically on both sides.
        const double log_a = (betas_[lo] - betas_[lo + 1]) * (E_lo - E_hi);
        const bool accept = log_a >= 0.0 || detail::exchange_uniform(seed_, round, lo) < std::exp(log_a);
        if (lower) { ++lo_att_; lo_acc_ += accept ? 1 : 0; }
        else { ++hi_att_; hi_acc_ += accept ? 1 : 0; }
        if (!accept) return;

        model_.pack_state(buf_.data());
        MPI_Sendrecv_replace(buf_.data(), int(buf_.size()), MPI_DOUBLE, partner, kTagState, partner,
                             kTagState, comm_, MPI_STATUS_IGNORE);
        model_.unpack_state(buf_.data());
        e_ = theirs[0];
        e_valid_ = true;
        label_ = int(theirs[1]);
        dir_ = int(theirs[2]);
        phase_ = int(theirs[3]);
        // Direction labels (Katzgraber et al. 2006) and round trips: phase 1 =
        // has been cold, phase 2 = has been cold and then hot; arriving cold
        // in phase 2 completes a round trip cold -> hot -> cold.
        if (rank_ == 0) {
            if (phase_ == 2) {
                ++trips_;
                if (label_ >= 0 && label_ < size_) ++trips_by_label_[size_t(label_)];
            }
            dir_ = kUp;
            phase_ = 1;
        }
        if (rank_ == size_ - 1) {
            dir_ = kDown;
            if (phase_ == 1) phase_ = 2;
        }
    }

    M& model_;
    MPI_Comm comm_;
    int rank_ = 0, size_ = 1;
    uint64_t seed_ = 0;
    StepSizeController ctrl_;
    std::vector<double> temps_, betas_, buf_;
    double e_ = 0.0;
    bool e_valid_ = false;
    size_t steps_ = 0;
    uint64_t round_ = 0;
    int label_ = 0, dir_ = kNone, phase_ = 0;
    double acc_sum_ = 0.0;
    uint64_t acc_n_ = 0;
    uint64_t lo_att_ = 0, lo_acc_ = 0, hi_att_ = 0, hi_acc_ = 0;
    uint64_t n_up_ = 0, n_down_ = 0, rounds_ = 0, trips_ = 0;
    std::vector<uint64_t> trips_by_label_;
};

// =====================================================================
// LADDER ARITHMETIC (pure functions, unit-testable without MPI)
// =====================================================================

/// Round-trip rate of DEO parallel tempering predicted from the per-edge
/// rejection rates (Syed et al. 2022, Thm. 3): round trips per exchange round,
/// summed over replicas.
inline double nrpt_round_trip_rate(const std::vector<double>& rejection) {
    double s = 0.0;
    for (double r : rejection) {
        if (r >= 1.0) return 0.0;
        s += std::max(r, 0.0) / (1.0 - r);
    }
    return 1.0 / (2.0 + 2.0 * s);
}

namespace detail {

/// Fritsch-Carlson monotone cubic (PCHIP) through (x_i, y_i), x strictly increasing.
struct Pchip {
    std::vector<double> x, y, d;

    Pchip(std::vector<double> xs, std::vector<double> ys) : x(std::move(xs)), y(std::move(ys)) {
        const size_t n = x.size();
        d.assign(n, 0.0);
        if (n < 2) return;
        std::vector<double> h(n - 1), del(n - 1);
        for (size_t k = 0; k + 1 < n; ++k) {
            h[k] = x[k + 1] - x[k];
            del[k] = (y[k + 1] - y[k]) / h[k];
        }
        if (n == 2) { d[0] = d[1] = del[0]; return; }
        for (size_t k = 1; k + 1 < n; ++k) {
            if (del[k - 1] * del[k] <= 0.0) { d[k] = 0.0; continue; }
            const double w1 = 2.0 * h[k] + h[k - 1], w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / del[k - 1] + w2 / del[k]);  // weighted harmonic mean
        }
        auto edge = [](double h0, double h1, double m0, double m1) {
            double dd = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
            if (dd * m0 <= 0.0) dd = 0.0;
            else if (m0 * m1 <= 0.0 && std::abs(dd) > 3.0 * std::abs(m0)) dd = 3.0 * m0;
            return dd;
        };
        d[0] = edge(h[0], h[1], del[0], del[1]);
        d[n - 1] = edge(h[n - 2], h[n - 3], del[n - 2], del[n - 3]);
    }

    double eval_on(size_t k, double xv) const {
        const double hk = x[k + 1] - x[k], t = (xv - x[k]) / hk;
        const double t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * hk * d[k] +
               (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * hk * d[k + 1];
    }

    /// x with p(x) = target for nondecreasing data (bisection on the bracketing
    /// interval; the interpolant is monotone there).
    double inverse(double target) const {
        const size_t n = x.size();
        if (target <= y.front()) return x.front();
        if (target >= y.back()) return x.back();
        size_t k = size_t(std::upper_bound(y.begin(), y.end(), target) - y.begin());
        k = std::clamp<size_t>(k, 1, n - 1) - 1;
        double a = x[k], b = x[k + 1];
        for (int it = 0; it < 200 && b - a > 1e-15 * (std::abs(a) + std::abs(b)); ++it) {
            const double m = 0.5 * (a + b);
            (eval_on(k, m) < target ? a : b) = m;
        }
        return 0.5 * (a + b);
    }
};

/// Pool-adjacent-violators: weighted least-squares non-increasing fit
/// (all weights > 0).
inline std::vector<double> isotonic_nonincreasing(const std::vector<double>& v, const std::vector<double>& w) {
    std::vector<double> mean, weight;
    std::vector<size_t> count;
    for (size_t i = 0; i < v.size(); ++i) {
        mean.push_back(v[i]);
        weight.push_back(w[i]);
        count.push_back(1);
        while (mean.size() > 1 && mean[mean.size() - 2] < mean.back()) {
            const size_t b = mean.size() - 1, a = b - 1;
            mean[a] = (mean[a] * weight[a] + mean[b] * weight[b]) / (weight[a] + weight[b]);
            weight[a] += weight[b];
            count[a] += count[b];
            mean.pop_back();
            weight.pop_back();
            count.pop_back();
        }
    }
    std::vector<double> out;
    out.reserve(v.size());
    for (size_t b = 0; b < mean.size(); ++b) out.insert(out.end(), count[b], mean[b]);
    return out;
}

}  // namespace detail

/**
 * Non-reversible PT schedule update (Syed et al. 2022, Sec. 5.3): given the
 * ladder (ascending temperatures) and the rejection rate of each edge
 * (T_k, T_{k+1}), return the ladder with the same end points whose edges have
 * equal estimated rejection. In beta = 1/T the cumulative barrier
 * Lambda(beta_j) = sum of the rejections of the edges between beta_min and
 * beta_j is interpolated by a monotone PCHIP and inverted at
 * Lambda_i = i Lambda_total / (R - 1). Rejections are floored at a tiny
 * positive value so Lambda is strictly increasing (an edge measured at zero
 * rejection is simply wide).
 */
inline std::vector<double> nrpt_schedule_update(const std::vector<double>& temperatures,
                                                const std::vector<double>& rejection) {
    const size_t R = temperatures.size();
    if (R < 3) return temperatures;
    if (rejection.size() != R - 1)
        throw std::invalid_argument("nrpt_schedule_update: need one rejection rate per edge");
    std::vector<double> beta(R), lambda(R, 0.0);
    for (size_t j = 0; j < R; ++j) beta[j] = 1.0 / temperatures[R - 1 - j];  // ascending
    for (size_t j = 0; j + 1 < R; ++j) {
        const double r = rejection[R - 2 - j];  // edge (beta_j, beta_{j+1}) = T edge R-2-j
        lambda[j + 1] = lambda[j] + std::max(std::isfinite(r) ? r : 1.0, 1e-9);
    }
    const detail::Pchip p(beta, lambda);
    std::vector<double> nb(R);
    nb.front() = beta.front();
    nb.back() = beta.back();
    for (size_t i = 1; i + 1 < R; ++i) nb[i] = p.inverse(lambda.back() * double(i) / double(R - 1));
    std::vector<double> T(R);
    for (size_t j = 0; j < R; ++j) T[R - 1 - j] = 1.0 / nb[j];
    T.front() = temperatures.front();
    T.back() = temperatures.back();
    for (size_t k = 1; k < R; ++k)  // guard against round-off ties
        if (!(T[k] > T[k - 1])) T[k] = std::nextafter(T[k - 1], INFINITY);
    return T;
}

/**
 * Flow-feedback update of Katzgraber, Trebst, Huse, Troyer (J. Stat. Mech.
 * P03018 (2006)): with f(T_k) the fraction of replicas at T_k that last
 * visited the coldest rather than the hottest temperature (f(T_min) = 1,
 * f(T_max) = 0), the optimal density of temperatures is
 * eta(T) ∝ sqrt(|df/dT| / dT). With f piecewise linear the integral of eta
 * over edge k is proportional to sqrt(f_k - f_{k+1}), so the new ladder puts
 * equal shares of sum_k sqrt(f_k - f_{k+1}) between consecutive temperatures.
 * f is first made non-increasing by weighted isotonic regression (weights =
 * number of labelled observations; unobserved points are interpolated).
 */
inline std::vector<double> katzgraber_schedule_update(const std::vector<double>& temperatures,
                                                      const std::vector<double>& up_fraction,
                                                      const std::vector<double>& weights = {}) {
    const size_t R = temperatures.size();
    if (R < 3) return temperatures;
    if (up_fraction.size() != R || (!weights.empty() && weights.size() != R))
        throw std::invalid_argument("katzgraber_schedule_update: need f and weights per temperature");
    std::vector<double> f(R), w(R);
    for (size_t k = 0; k < R; ++k) {
        const bool ok = std::isfinite(up_fraction[k]) && (weights.empty() || weights[k] > 0.0);
        f[k] = ok ? std::clamp(up_fraction[k], 0.0, 1.0) : 0.0;
        w[k] = ok ? (weights.empty() ? 1.0 : weights[k]) : 0.0;
    }
    f.front() = 1.0; f.back() = 0.0;  // exact by definition of the labels
    w.front() = w.back() = std::max(1.0, *std::max_element(w.begin(), w.end()));
    std::vector<size_t> obs;
    std::vector<double> fv, wv;
    for (size_t k = 0; k < R; ++k)
        if (w[k] > 0.0) { obs.push_back(k); fv.push_back(f[k]); wv.push_back(w[k]); }
    fv = detail::isotonic_nonincreasing(fv, wv);
    for (size_t j = 0; j + 1 < obs.size(); ++j) {  // unobserved points: linear in T
        const size_t a = obs[j], b = obs[j + 1];
        for (size_t k = a; k < b; ++k) {
            const double t = (temperatures[k] - temperatures[a]) / (temperatures[b] - temperatures[a]);
            f[k] = fv[j] + t * (fv[j + 1] - fv[j]);
        }
    }
    f.back() = 0.0;
    std::vector<double> G(R, 0.0);
    for (size_t k = 0; k + 1 < R; ++k) G[k + 1] = G[k] + std::sqrt(std::max(f[k] - f[k + 1], 1e-8));
    std::vector<double> T(R);
    T.front() = temperatures.front();
    T.back() = temperatures.back();
    for (size_t i = 1; i + 1 < R; ++i) {
        const double g = G.back() * double(i) / double(R - 1);
        size_t k = size_t(std::upper_bound(G.begin(), G.end(), g) - G.begin());
        k = std::clamp<size_t>(k, 1, R - 1) - 1;
        const double t = (g - G[k]) / (G[k + 1] - G[k]);
        T[i] = temperatures[k] + t * (temperatures[k + 1] - temperatures[k]);
    }
    for (size_t k = 1; k < R; ++k)
        if (!(T[k] > T[k - 1])) T[k] = std::nextafter(T[k - 1], INFINITY);
    return T;
}

/// Largest move of an interior temperature in units of its local spacing.
inline double ladder_change(const std::vector<double>& old_T, const std::vector<double>& new_T) {
    double c = 0.0;
    for (size_t k = 1; k + 1 < old_T.size(); ++k) {
        const double h = std::min(old_T[k] - old_T[k - 1], old_T[k + 1] - old_T[k]);
        c = std::max(c, std::abs(new_T[k] - old_T[k]) / h);
    }
    return c;
}

// =====================================================================
// OUTPUT
// =====================================================================

namespace detail {

inline void write_pt_summary_text(const std::string& filename, const PTResult& r, const PTOptions& o) {
    std::ofstream f(filename);
    if (!f) throw std::runtime_error("cannot open " + filename + " for writing");
    f << std::setprecision(10);
    f << "# Parallel tempering summary (DEO replica exchange)\n";
    f << "# replicas " << r.temperatures.size() << ", equilibration " << o.n_equilibration
      << " steps, measurement " << o.n_measurement << " steps, exchange every " << o.exchange_every
      << ", probe every " << o.probe_every << "\n";
    f << "# exchange rounds " << r.exchange_rounds << ", round trips " << r.round_trips
      << ", measured rate " << r.round_trip_rate << " / round, predicted (Syed et al. 2022) "
      << r.predicted_round_trip_rate << " / round, barrier Lambda " << r.barrier << "\n";
    f << "# acceptance bookkeeping " << (r.bookkeeping_consistent ? "consistent" : "INCONSISTENT") << "\n";
    f << "#\n# k  T  e  de  c  dc  tau_int_E[steps]  n_samples  acc_local  sigma  f_up";
    for (const auto& op : r.order_parameter_tables)
        f << "  |" << op.name << "|  d|" << op.name << "|  chi_" << op.name << "  dchi_" << op.name
          << "  U4_" << op.name << "  dU4_" << op.name;
    f << "\n";
    for (size_t k = 0; k < r.temperatures.size(); ++k) {
        f << k << "  " << r.temperatures[k] << "  " << r.energy[k] << "  " << r.energy_error[k] << "  "
          << r.specific_heat[k] << "  " << r.specific_heat_error[k] << "  " << r.tau_int_energy[k] << "  "
          << r.n_samples[k] << "  " << r.acceptance[k] << "  " << r.step_sizes[k] << "  " << r.up_fraction[k];
        for (const auto& op : r.order_parameter_tables)
            f << "  " << op.abs[k] << "  " << op.abs_error[k] << "  " << op.susceptibility[k] << "  "
              << op.susceptibility_error[k] << "  " << op.binder[k] << "  " << op.binder_error[k];
        f << "\n";
    }
    f << "#\n# edge  T_k  T_k+1  attempts  accepts  acceptance\n";
    for (size_t k = 0; k < r.edge_attempts.size(); ++k)
        f << "# " << k << "  " << r.temperatures[k] << "  " << r.temperatures[k + 1] << "  "
          << r.edge_attempts[k] << "  " << r.edge_accepts[k] << "  " << r.edge_acceptance[k] << "\n";
    f.close();
    if (!f) throw std::runtime_error("write error on " + filename);
}

#ifdef HDF5_ENABLED
inline void write_pt_aggregated_hdf5(const std::string& filename, const PTResult& r, const PTOptions& o) {
    namespace fs = std::filesystem;
    const std::string tmp = filename + ".tmp";
    {
        H5::H5File file(tmp, H5F_ACC_TRUNC);
        auto dvec = [](H5::Group& g, const std::string& name, const std::vector<double>& v) {
            hsize_t n = v.size();
            H5::DataSpace sp(1, &n);
            g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, sp).write(v.data(), H5::PredType::NATIVE_DOUBLE);
        };
        auto uvec = [](H5::Group& g, const std::string& name, const std::vector<uint64_t>& v) {
            hsize_t n = v.size();
            H5::DataSpace sp(1, &n);
            g.createDataSet(name, H5::PredType::NATIVE_UINT64, sp).write(v.data(), H5::PredType::NATIVE_UINT64);
        };
        auto dattr = [](H5::Group& g, const std::string& name, double v) {
            H5::DataSpace sc(H5S_SCALAR);
            g.createAttribute(name, H5::PredType::NATIVE_DOUBLE, sc).write(H5::PredType::NATIVE_DOUBLE, &v);
        };
        auto uattr = [](H5::Group& g, const std::string& name, uint64_t v) {
            H5::DataSpace sc(H5S_SCALAR);
            g.createAttribute(name, H5::PredType::NATIVE_UINT64, sc).write(H5::PredType::NATIVE_UINT64, &v);
        };
        auto sattr = [](H5::Group& g, const std::string& name, const std::string& v) {
            H5::DataSpace sc(H5S_SCALAR);
            H5::StrType st(H5::PredType::C_S1, v.size() + 1);  // with the terminating NUL
            g.createAttribute(name, st, sc).write(st, v.c_str());
        };
        H5::Group scan = file.createGroup("/temperature_scan");
        dvec(scan, "temperature", r.temperatures);
        dvec(scan, "energy_per_site", r.energy);
        dvec(scan, "energy_per_site_error", r.energy_error);
        dvec(scan, "specific_heat", r.specific_heat);
        dvec(scan, "specific_heat_error", r.specific_heat_error);
        dvec(scan, "tau_int_energy", r.tau_int_energy);
        dvec(scan, "local_acceptance", r.acceptance);
        dvec(scan, "step_size", r.step_sizes);
        dvec(scan, "up_fraction", r.up_fraction);
        std::vector<uint64_t> ns(r.n_samples.begin(), r.n_samples.end());
        uvec(scan, "n_samples", ns);

        H5::Group ex = file.createGroup("/exchange");
        uvec(ex, "edge_attempts", r.edge_attempts);
        uvec(ex, "edge_accepts", r.edge_accepts);
        dvec(ex, "edge_acceptance", r.edge_acceptance);
        uvec(ex, "round_trips_per_replica", r.round_trips_per_replica);
        uattr(ex, "exchange_rounds", r.exchange_rounds);
        uattr(ex, "round_trips", r.round_trips);
        dattr(ex, "round_trip_rate", r.round_trip_rate);
        dattr(ex, "predicted_round_trip_rate", r.predicted_round_trip_rate);
        dattr(ex, "barrier", r.barrier);
        uattr(ex, "bookkeeping_consistent", r.bookkeeping_consistent ? 1 : 0);

        H5::Group ops = file.createGroup("/order_parameters");
        for (const auto& op : r.order_parameter_tables) {
            H5::Group g = ops.createGroup(op.name);
            dvec(g, "abs_mean", op.abs);
            dvec(g, "abs_mean_error", op.abs_error);
            dvec(g, "m2_mean", op.m2);
            dvec(g, "m2_mean_error", op.m2_error);
            dvec(g, "susceptibility", op.susceptibility);
            dvec(g, "susceptibility_error", op.susceptibility_error);
            dvec(g, "binder", op.binder);
            dvec(g, "binder_error", op.binder_error);
        }

        H5::Group meta = file.createGroup("/metadata");
        uattr(meta, "n_temperatures", r.temperatures.size());
        uattr(meta, "n_equilibration", o.n_equilibration);
        uattr(meta, "n_measurement", o.n_measurement);
        uattr(meta, "exchange_every", o.exchange_every);
        uattr(meta, "probe_every", o.probe_every);
        std::time_t now = std::time(nullptr);
        char ts[64];
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        sattr(meta, "creation_time", ts);
        sattr(meta, "file_format", "HDF5_PT_aggregated_v2");
        sattr(meta, "exchange_scheme", "deterministic even-odd (non-reversible), configuration swap");
    }
    fs::rename(tmp, filename);
}
#endif

}  // namespace detail

// =====================================================================
// PARALLEL TEMPERING DRIVER
// =====================================================================

namespace detail {

template <ReplicaModel M>
PTResult summarize(ReplicaExchange<M>& chain, std::vector<double>&& energies, const MeasurementRecorder& rec,
                   size_t n_sites, const PTOptions& o, MPI_Comm comm) {
    PTResult res;
    const int rank = chain.rank(), R = chain.size();
    res.rank = rank;
    res.temperature = chain.temperature();
    res.energies = std::move(energies);
    res.order_parameters = rec.series();
    res.thermo = compute_thermodynamics(res.energies, res.order_parameters, res.temperature, n_sites);
    res.local_acceptance = chain.local_acceptance();
    res.step_size = chain.step_size();
    const uint64_t att = chain.lower_attempts() + chain.upper_attempts();
    res.swap_acceptance = att ? double(chain.lower_accepts() + chain.upper_accepts()) / double(att) : 0.0;

    // Every rank must report the same order parameters.
    int K = int(res.order_parameters.size()), K_lo = 0, K_hi = 0;
    MPI_Allreduce(&K, &K_lo, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&K, &K_hi, 1, MPI_INT, MPI_MAX, comm);
    if (K_lo != K_hi)
        throw std::logic_error("parallel tempering: replicas reported different order parameters");

    constexpr size_t kBase = 15, kPerOp = 8;
    const size_t row = kBase + kPerOp * size_t(K);
    std::vector<double> mine(row), all(row * size_t(R));
    const ThermodynamicObservables& th = res.thermo;
    double vals[kBase] = {res.temperature, th.energy.value, th.energy.error, th.specific_heat.value,
                          th.specific_heat.error, th.energy_tau_int * double(o.probe_every),
                          double(res.energies.size()), res.local_acceptance, res.step_size,
                          double(chain.n_up()), double(chain.n_down()), double(chain.lower_attempts()),
                          double(chain.lower_accepts()), double(chain.upper_attempts()),
                          double(chain.upper_accepts())};
    std::copy(vals, vals + kBase, mine.begin());
    for (int q = 0; q < K; ++q) {
        const OrderParameterStats& s = th.order_parameters[size_t(q)];
        const double v[kPerOp] = {s.abs.value, s.abs.error, s.m2.value, s.m2.error,
                                  s.susceptibility.value, s.susceptibility.error, s.binder.value, s.binder.error};
        std::copy(v, v + kPerOp, mine.begin() + long(kBase + kPerOp * size_t(q)));
    }
    MPI_Allgather(mine.data(), int(row), MPI_DOUBLE, all.data(), int(row), MPI_DOUBLE, comm);

    const size_t Rs = size_t(R);
    auto col = [&](size_t c) {
        std::vector<double> v(Rs);
        for (int k = 0; k < R; ++k) v[size_t(k)] = all[size_t(k) * row + c];
        return v;
    };
    res.temperatures = col(0);
    res.energy = col(1);
    res.energy_error = col(2);
    res.specific_heat = col(3);
    res.specific_heat_error = col(4);
    res.tau_int_energy = col(5);
    const std::vector<double> ns = col(6), up = col(9), down = col(10);
    res.n_samples.assign(ns.begin(), ns.end());
    res.acceptance = col(7);
    res.step_sizes = col(8);
    res.up_fraction.resize(size_t(R));
    for (int k = 0; k < R; ++k) {
        const double tot = up[size_t(k)] + down[size_t(k)];
        res.up_fraction[size_t(k)] = tot > 0 ? up[size_t(k)] / tot : std::nan("");
    }
    const std::vector<double> lo_att = col(11), lo_acc = col(12), hi_att = col(13), hi_acc = col(14);
    std::vector<double> rejection;
    for (int k = 0; k + 1 < R; ++k) {
        res.edge_attempts.push_back(uint64_t(lo_att[size_t(k)]));
        res.edge_accepts.push_back(uint64_t(lo_acc[size_t(k)]));
        if (lo_att[size_t(k)] != hi_att[size_t(k) + 1] || lo_acc[size_t(k)] != hi_acc[size_t(k) + 1])
            res.bookkeeping_consistent = false;
        const double a = lo_att[size_t(k)] > 0 ? lo_acc[size_t(k)] / lo_att[size_t(k)] : 0.0;
        res.edge_acceptance.push_back(a);
        rejection.push_back(1.0 - a);
    }
    for (int q = 0; q < K; ++q) {
        OrderParameterTable t;
        t.name = res.order_parameters[size_t(q)].name;
        const size_t c0 = kBase + kPerOp * size_t(q);
        t.abs = col(c0); t.abs_error = col(c0 + 1); t.m2 = col(c0 + 2); t.m2_error = col(c0 + 3);
        t.susceptibility = col(c0 + 4); t.susceptibility_error = col(c0 + 5);
        t.binder = col(c0 + 6); t.binder_error = col(c0 + 7);
        res.order_parameter_tables.push_back(std::move(t));
    }

    // Round trips are counted where they complete (rank 0).
    std::vector<unsigned long long> trips(size_t(R) + 1, 0);
    if (rank == 0) {
        trips[0] = chain.round_trips();
        for (int k = 0; k < R; ++k) trips[size_t(k) + 1] = chain.round_trips_by_label()[size_t(k)];
    }
    MPI_Bcast(trips.data(), R + 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    res.round_trips = trips[0];
    res.round_trips_per_replica.assign(trips.begin() + 1, trips.end());
    res.exchange_rounds = chain.rounds();
    res.round_trip_rate = res.exchange_rounds ? double(res.round_trips) / double(res.exchange_rounds) : 0.0;
    if (R > 1 && o.exchange_every > 0) {
        res.predicted_round_trip_rate = nrpt_round_trip_rate(rejection);
        for (double r : rejection) res.barrier += r;
    }
    return res;
}

template <ReplicaModel M>
void write_outputs(M& model, const PTResult& res, const PTOptions& o, MPI_Comm comm) {
    if (o.output_dir.empty()) return;
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::string err;
    if (rank == 0) err = guarded_io([&] { std::filesystem::create_directories(o.output_dir); });
    throw_if_any<std::runtime_error>(err.empty() ? err : "parallel tempering output: " + err, comm);

    err.clear();
    if (should_rank_write(rank, o.ranks_to_write)) {
        err = guarded_io([&] {
            PTRankContext ctx;
            ctx.rank = rank;
            ctx.size = size;
            ctx.rank_dir = o.output_dir + "/rank_" + std::to_string(rank);
            ctx.options = &o;
            ctx.result = &res;
            std::filesystem::create_directories(ctx.rank_dir);
            if constexpr (HasRankOutputs<M>) model.write_rank_outputs(ctx);
        });
    }
    if (rank == 0) {  // the text summary first: it does not depend on HDF5
        for (const std::string& e : {
                 guarded_io([&] { write_pt_summary_text(o.output_dir + "/pt_summary.txt", res, o); }),
#ifdef HDF5_ENABLED
                 guarded_io([&] {
                     write_pt_aggregated_hdf5(o.output_dir + "/parallel_tempering_aggregated.h5", res, o);
                 }),
#endif
             })
            if (!e.empty()) err += (err.empty() ? "" : "; ") + e;
    }
    if (!err.empty()) std::cerr << "[PT] rank " << rank << ": output failed: " << err << std::endl;
    throw_if_any<std::runtime_error>(err.empty() ? err : "parallel tempering output failed: " + err, comm);
}

inline void print_summary(const PTResult& r) {
    const std::ios_base::fmtflags flags = std::cout.flags();
    const std::streamsize prec = std::cout.precision();
    std::cout << "\n=== Parallel tempering: " << r.temperatures.size() << " replicas, "
              << r.exchange_rounds << " exchange rounds ===\n";
    std::cout << "  k            T        <E>/N       err          c        err   tau_E  acc_loc   sigma   f_up  A(k,k+1)\n";
    for (size_t k = 0; k < r.temperatures.size(); ++k) {
        std::cout << std::setw(3) << k << std::scientific << std::setprecision(4) << std::setw(13)
                  << r.temperatures[k] << std::setw(13) << r.energy[k] << std::setw(10) << std::setprecision(2)
                  << r.energy_error[k] << std::setw(11) << std::setprecision(4) << r.specific_heat[k]
                  << std::setw(10) << std::setprecision(2) << r.specific_heat_error[k] << std::fixed
                  << std::setprecision(1) << std::setw(8) << r.tau_int_energy[k] << std::setprecision(3)
                  << std::setw(9) << r.acceptance[k] << std::setw(8) << r.step_sizes[k] << std::setw(7)
                  << r.up_fraction[k];
        if (k < r.edge_acceptance.size()) std::cout << std::setw(10) << r.edge_acceptance[k];
        std::cout << "\n";
    }
    std::cout << "Round trips: " << r.round_trips << " (rate " << std::scientific << std::setprecision(3)
              << r.round_trip_rate << " per round; DEO prediction from rejections "
              << r.predicted_round_trip_rate << "), barrier Lambda = " << std::fixed
              << std::setprecision(3) << r.barrier << "\n";
    if (!r.bookkeeping_consistent)
        std::cout << "[PT] WARNING: exchange partners recorded different decisions\n";
    for (size_t k = 0; k < r.edge_attempts.size(); ++k)
        if (r.edge_attempts[k] > 0 && r.edge_accepts[k] == 0)
            std::cout << "[PT] WARNING: no swap was accepted between T = " << r.temperatures[k] << " and "
                      << r.temperatures[k + 1] << "; replicas cannot cross this edge (add temperatures "
                      << "or narrow the range)\n";
    std::cout << std::flush;
    std::cout.flags(flags);
    std::cout.precision(prec);
}

}  // namespace detail

/**
 * Run parallel tempering on the replica `model` (one per rank of `comm`).
 * Equilibrates for n_equilibration MC steps (adapting the proposal width),
 * then measures for n_measurement steps, attempting a DEO exchange round
 * every exchange_every steps throughout. Returns per-rank and whole-ladder
 * statistics (identical ladder tables on every rank) and writes them to
 * output_dir when given. Collective; throws on every rank on invalid input.
 */
template <ReplicaModel M>
PTResult run_parallel_tempering(M& model, const PTOptions& o, MPI_Comm comm = MPI_COMM_WORLD) {
    detail::require_mpi("mc::run_parallel_tempering");
    detail::CommGuard pt(comm);
    int rank = 0, size = 1;
    MPI_Comm_rank(pt, &rank);
    MPI_Comm_size(pt, &size);

    std::string err;
    if (o.probe_every == 0) err = "probe_every (probe_rate) must be >= 1";
    else if (o.n_measurement < o.probe_every)
        err = "n_measurement (" + std::to_string(o.n_measurement) + ") must be >= probe_every (" +
              std::to_string(o.probe_every) + ") to record at least one sample";
    else if (!(o.initial_step_size > 0.0) || !(o.target_acceptance > 0.0 && o.target_acceptance < 1.0))
        err = "initial_step_size must be > 0 and target_acceptance in (0, 1)";
    detail::throw_if_any(err, pt);
    int writes = o.output_dir.empty() ? 0 : 1, w_lo = 0, w_hi = 0;  // output is collective
    MPI_Allreduce(&writes, &w_lo, 1, MPI_INT, MPI_MIN, pt);
    MPI_Allreduce(&writes, &w_hi, 1, MPI_INT, MPI_MAX, pt);
    if (w_lo != w_hi)
        throw std::invalid_argument("parallel tempering: output_dir must be set on all ranks or on none");
    detail::validate_chain_setup(model, o.temperatures, pt);

    const uint64_t seed = detail::shared_exchange_seed(pt);
    ReplicaExchange<M> chain(model, pt, o.temperatures, seed, o.initial_step_size, o.target_acceptance);
    const size_t ex = o.exchange_every;
    if (rank == 0 && o.verbosity > 0) {
        std::cout << "Parallel tempering: " << size << " replicas, T in [" << o.temperatures.front() << ", "
                  << o.temperatures.back() << "], " << o.n_equilibration << " + " << o.n_measurement
                  << " MC steps, exchange every " << ex << ", probe every " << o.probe_every
                  << (model.uses_step_size() && o.adapt_step_size ? ", adaptive proposal width" : "")
                  << std::endl;
        if (ex == 0) std::cout << "[PT] exchange_every = 0: replicas run independently" << std::endl;
    }

    for (size_t i = 0; i < o.n_equilibration; ++i) {
        chain.step(o.adapt_step_size);
        if (ex > 0 && (i + 1) % ex == 0) chain.exchange_round();
    }

    chain.reset_statistics();
    MeasurementRecorder rec;
    std::vector<double> energies;
    energies.reserve(o.n_measurement / o.probe_every + 1);
    for (size_t i = 0; i < o.n_measurement; ++i) {
        chain.step(false);
        if (ex > 0 && (i + 1) % ex == 0) chain.exchange_round();
        if ((i + 1) % o.probe_every == 0) {
            const double E = chain.energy();
            energies.push_back(E);
            rec.begin_sample(E);
            model.measure(rec);
            rec.end_sample();
        }
    }

    PTResult res = detail::summarize(chain, std::move(energies), rec, model.n_sites(), o, pt);
    if (rank == 0 && o.verbosity > 0) detail::print_summary(res);
    detail::write_outputs(model, res, o, pt);
    return res;
}

// =====================================================================
// TEMPERATURE-LADDER TUNING
// =====================================================================

enum class LadderMethod { NRPT, Katzgraber };

/**
 * Optimiser names: "nrpt" (default; also "syed", "equi_rejection"),
 * "katzgraber" (also "feedback"). The former "gradient" optimiser was removed:
 * its acceptance gradient dropped the pathwise term, it used a stale energy
 * after a first-parity swap, and its stopping rule compared a variance with a
 * mean-deviation tolerance; the name now maps to nrpt with a warning.
 */
inline LadderMethod parse_ladder_method(const std::string& name, bool warn = true) {
    if (name == "nrpt" || name == "syed" || name == "equi_rejection" || name.empty()) return LadderMethod::NRPT;
    if (name == "katzgraber" || name == "feedback") return LadderMethod::Katzgraber;
    if (name == "gradient") {
        if (warn)
            std::cerr << "[PT] pt_temperature_optimizer = gradient has been removed; using the "
                         "non-reversible PT schedule (nrpt) of Syed et al. (2022) instead" << std::endl;
        return LadderMethod::NRPT;
    }
    throw std::invalid_argument("unknown temperature optimizer '" + name + "' (valid: nrpt, katzgraber)");
}

inline const char* ladder_method_name(LadderMethod m) {
    return m == LadderMethod::NRPT ? "nrpt (Syed et al. 2022)" : "katzgraber (KTHT 2006 flow feedback)";
}

struct LadderTuningOptions {
    double T_min = 0.0, T_max = 0.0;
    LadderMethod method = LadderMethod::NRPT;
    std::vector<double> initial_temperatures;  ///< empty: geometric ladder in [T_min, T_max]
    size_t warmup_steps = 500;
    size_t steps_per_round = 500;  ///< first round; doubles every round up to 16x
    size_t max_rounds = 20;
    size_t exchange_every = 1;
    double tolerance = 0.05;       ///< stop when no interior T moves more than this x its spacing
    bool adapt_step_size = true;
    double initial_step_size = 2.0;
    double target_acceptance = 0.45;
    int verbosity = 1;
};

struct LadderTuningResult {
    std::vector<double> temperatures;           ///< tuned ladder, cold -> hot
    std::vector<double> acceptance_rates;       ///< edge k = (T_k, T_k+1) of the last round's ladder
    std::vector<double> up_fraction;            ///< f(T_k) in the last round
    std::vector<double> autocorrelation_times;  ///< tau_int(E) at fixed T, MC steps, last round
    double mean_acceptance_rate = 0.0;
    double barrier = 0.0;                       ///< Lambda = sum of the edge rejections
    double predicted_round_trip_rate = 0.0;     ///< per exchange round
    size_t recommended_replicas = 0;            ///< ~ 2 Lambda + 1 (Syed et al. 2022)
    bool barrier_is_lower_bound = false;        ///< some edge never accepted: Lambda saturated
    uint64_t round_trips = 0;                   ///< in the last round
    size_t rounds_used = 0;
    double last_change = 0.0;                   ///< ladder_change of the last update
    bool converged = false;
};

/// Former name of the tuning result.
using OptimizedTempGridResult = LadderTuningResult;

/**
 * Tune the temperature ladder with the replica chain itself (collective).
 * Rounds of steps_per_round * 2^min(r, 4) MC steps measure per-edge
 * acceptance (and f(T) for katzgraber); after each round the ladder is
 * updated (end points fixed) and broadcast. Stops when the largest interior
 * move is below `tolerance` (after at least two rounds) or after max_rounds.
 * Each rank's replica ends equilibrated near its final temperature, so the
 * production run can start from it.
 */
template <ReplicaModel M>
LadderTuningResult tune_temperature_ladder(M& model, const LadderTuningOptions& o,
                                           MPI_Comm comm = MPI_COMM_WORLD) {
    detail::require_mpi("mc::tune_temperature_ladder");
    detail::CommGuard pt(comm);
    int rank = 0, size = 1;
    MPI_Comm_rank(pt, &rank);
    MPI_Comm_size(pt, &size);
    const size_t R = size_t(size);

    std::string err;
    if (!(o.T_min > 0.0) || !std::isfinite(o.T_max) || !(o.T_max > o.T_min || (R == 1 && o.T_max >= o.T_min)))
        err = "ladder tuning needs 0 < T_min < T_max (got T_min=" + std::to_string(o.T_min) +
              ", T_max=" + std::to_string(o.T_max) + ")";
    else if (o.steps_per_round == 0 || o.max_rounds == 0 || o.exchange_every == 0)
        err = "ladder tuning needs steps_per_round, max_rounds and exchange_every >= 1";
    else if (!(o.tolerance > 0.0))
        err = "ladder tuning tolerance must be > 0";
    else if (!o.initial_temperatures.empty() &&
             (o.initial_temperatures.size() != R || o.initial_temperatures.front() != o.T_min ||
              o.initial_temperatures.back() != o.T_max))
        err = "initial_temperatures must have one entry per rank and end points T_min, T_max";
    detail::throw_if_any(err, pt);

    LadderTuningResult res;
    std::vector<double> ladder = o.initial_temperatures.empty()
                                     ? generate_geometric_temperature_ladder(o.T_min, o.T_max, R)
                                     : o.initial_temperatures;
    res.temperatures = ladder;
    if (R == 1) {
        res.converged = true;
        return res;
    }
    detail::validate_chain_setup(model, ladder, pt);
    const uint64_t seed = detail::shared_exchange_seed(pt);
    ReplicaExchange<M> chain(model, pt, ladder, seed, o.initial_step_size, o.target_acceptance);
    const size_t ex = o.exchange_every;
    if (rank == 0 && o.verbosity > 0)
        std::cout << "Tuning the temperature ladder: " << ladder_method_name(o.method) << ", R = " << R
                  << ", T in [" << o.T_min << ", " << o.T_max << "], " << o.warmup_steps
                  << " warm-up steps, rounds of " << o.steps_per_round << " x 2^r steps (max "
                  << o.max_rounds << ")" << std::endl;

    for (size_t i = 0; i < o.warmup_steps; ++i) {
        chain.step(o.adapt_step_size);
        if ((i + 1) % ex == 0) chain.exchange_round();
    }

    for (size_t r = 0; r < o.max_rounds; ++r) {
        const size_t n_steps = o.steps_per_round << std::min<size_t>(r, 4);
        chain.reset_statistics();
        std::vector<double> e_series;
        e_series.reserve(n_steps / ex + 1);
        for (size_t i = 0; i < n_steps; ++i) {
            chain.step(o.adapt_step_size);
            if ((i + 1) % ex == 0) {
                chain.exchange_round();
                e_series.push_back(chain.energy());
            }
        }
        const double tau = gamma_method(e_series).tau_int * double(ex);
        double mine[5] = {double(chain.lower_attempts()), double(chain.lower_accepts()), double(chain.n_up()),
                          double(chain.n_down()), tau};
        std::vector<double> all(5 * R);
        MPI_Allgather(mine, 5, MPI_DOUBLE, all.data(), 5, MPI_DOUBLE, pt);

        res.acceptance_rates.assign(R - 1, 0.0);
        res.up_fraction.assign(R, std::nan(""));
        res.autocorrelation_times.assign(R, 0.0);
        std::vector<double> rejection(R - 1), weights(R);
        for (size_t k = 0; k < R; ++k) {
            const double up = all[5 * k + 2], down = all[5 * k + 3];
            weights[k] = up + down;
            if (up + down > 0) res.up_fraction[k] = up / (up + down);
            res.autocorrelation_times[k] = all[5 * k + 4];
            if (k + 1 < R) {
                res.acceptance_rates[k] = all[5 * k] > 0 ? all[5 * k + 1] / all[5 * k] : 0.0;
                rejection[k] = 1.0 - res.acceptance_rates[k];
            }
        }
        res.mean_acceptance_rate = 0.0;
        res.barrier = 0.0;
        for (size_t k = 0; k + 1 < R; ++k) {
            res.mean_acceptance_rate += res.acceptance_rates[k] / double(R - 1);
            res.barrier += rejection[k];
        }
        res.predicted_round_trip_rate = nrpt_round_trip_rate(rejection);
        res.recommended_replicas = size_t(std::ceil(2.0 * res.barrier)) + 1;
        // An edge without a single accepted swap carries no information on
        // its barrier (r = 1 is only a bound); equal saturated rejections are
        // a spurious fixed point, so the ladder cannot count as converged.
        res.barrier_is_lower_bound = false;
        for (double a : res.acceptance_rates) res.barrier_is_lower_bound |= (a <= 0.0);
        unsigned long long trips = chain.round_trips();
        MPI_Bcast(&trips, 1, MPI_UNSIGNED_LONG_LONG, 0, pt);
        res.round_trips = trips;

        std::vector<double> next = ladder;
        int flags[2] = {0, 0};  // converged, fell back to nrpt
        double change = 0.0;
        if (rank == 0) {
            if (o.method == LadderMethod::Katzgraber && trips > 0) {
                next = katzgraber_schedule_update(ladder, res.up_fraction, weights);
            } else {
                flags[1] = (o.method == LadderMethod::Katzgraber) ? 1 : 0;
                next = nrpt_schedule_update(ladder, rejection);
            }
            change = ladder_change(ladder, next);
            flags[0] = (r >= 1 && change < o.tolerance && !res.barrier_is_lower_bound) ? 1 : 0;
        }
        MPI_Bcast(next.data(), int(R), MPI_DOUBLE, 0, pt);
        MPI_Bcast(flags, 2, MPI_INT, 0, pt);
        MPI_Bcast(&change, 1, MPI_DOUBLE, 0, pt);
        if (rank == 0 && o.verbosity > 0) {
            double a_min = 1.0, a_max = 0.0;
            for (double a : res.acceptance_rates) { a_min = std::min(a_min, a); a_max = std::max(a_max, a); }
            std::cout << "  round " << r + 1 << " (" << n_steps << " steps): A in [" << std::fixed
                      << std::setprecision(3) << a_min << ", " << a_max << "], Lambda = " << res.barrier
                      << ", round trips " << trips << ", ladder change " << change
                      << (flags[1] ? " (no round trip yet: rejection-based step)" : "") << std::endl;
        }
        ladder = next;
        chain.set_temperatures(ladder);
        res.rounds_used = r + 1;
        res.last_change = change;
        if (flags[0]) { res.converged = true; break; }
    }
    res.temperatures = ladder;
    if (rank == 0 && res.barrier_is_lower_bound)
        std::cerr << "[PT] WARNING: some neighbouring temperatures never exchanged during tuning; the "
                     "barrier Lambda = " << res.barrier << " is only a lower bound. Use more replicas "
                     "(at least " << res.recommended_replicas << ") or a narrower temperature range." << std::endl;
    if (rank == 0 && o.verbosity > 0) {
        std::cout << "Ladder " << (res.converged ? "converged" : "NOT converged") << " after " << res.rounds_used
                  << " rounds; mean acceptance " << std::fixed << std::setprecision(3) << res.mean_acceptance_rate
                  << ", barrier Lambda = " << res.barrier << " (about " << res.recommended_replicas
                  << " replicas would be optimal)" << std::endl;
    }
    return res;
}

// =====================================================================
// ADAPTER FOR SINGLE-SPECIES SPIN LATTICES (Lattice, PhononLattice)
// =====================================================================

/**
 * Replica adapter for lattices exposing `spins`, `lattice_size`, `spin_dim`,
 * `N_atoms`, total_energy(), magnetization_global(), magnetization_sublattice(),
 * save_spin_config() and save_positions(). One MC step follows
 * perform_mc_sweeps: with overrelaxation_rate = k > 0 an overrelaxation sweep
 * every step and a local sweep every k-th step, otherwise a local sweep. The
 * lattice's local_sweep()/overrelaxation_sweep(T) policy (heat bath, Gaussian,
 * coloured OpenMP kernels) is used when available, metropolis()/
 * overrelaxation() otherwise. Extra DOF (extra_dof_size/pack/unpack) travel
 * with the spins.
 */
template <class L>
class SpinLatticeReplica {
public:
    SpinLatticeReplica(L& lat, size_t overrelaxation_rate, bool gaussian_move, bool adaptive)
        : lat_(lat), or_rate_(overrelaxation_rate), gaussian_(gaussian_move), adaptive_(adaptive) {}

    double mc_step(double T, size_t step, double sigma) {
        if (or_rate_ > 0) {
            if constexpr (requires { lat_.overrelaxation_sweep(T); }) lat_.overrelaxation_sweep(T);
            else lat_.overrelaxation();
            if (step % or_rate_ != 0) return std::numeric_limits<double>::quiet_NaN();
        }
        if constexpr (requires { lat_.local_sweep(T, gaussian_, sigma); })
            return lat_.local_sweep(T, gaussian_, sigma);
        else
            return lat_.metropolis(T, gaussian_ || adaptive_, sigma);
    }

    double energy() const { return lat_.total_energy(); }
    size_t n_sites() const { return lat_.lattice_size; }
    bool uses_step_size() const { return adaptive_; }

    size_t state_size() const { return lat_.lattice_size * lat_.spin_dim + extra_size(); }

    void pack_state(double* out) const {
        const size_t d = lat_.spin_dim;
        for (size_t i = 0; i < lat_.lattice_size; ++i)
            for (size_t a = 0; a < d; ++a) out[i * d + a] = lat_.spins[i](a);
        if constexpr (has_extra_dof<L>::value)
            if (extra_size() > 0) lat_.pack_extra_dof(out + lat_.lattice_size * d);
    }

    void unpack_state(const double* in) {
        const size_t d = lat_.spin_dim;
        for (size_t i = 0; i < lat_.lattice_size; ++i)
            for (size_t a = 0; a < d; ++a) lat_.spins[i](a) = in[i * d + a];
        if constexpr (has_extra_dof<L>::value)
            if (extra_size() > 0) lat_.unpack_extra_dof(in + lat_.lattice_size * d);
    }

    /// Global-frame magnetisation per site as order parameter "m"; the
    /// sublattice magnetisations are kept for the per-rank HDF5 file.
    void measure(MeasurementRecorder& rec) {
        SpinVector m = lat_.magnetization_global();
        rec.record("m", m, lat_.lattice_size);
        magnetizations_.push_back(std::move(m));
        sublattice_.push_back(lat_.magnetization_sublattice());
    }

    /// parallel_tempering_data.h5 (time series + observables), spins, positions.
    void write_rank_outputs(const PTRankContext& ctx) const {
        const PTResult& r = *ctx.result;
        const PTOptions& o = *ctx.options;
#ifdef HDF5_ENABLED
        ThermodynamicObservables obs = compute_thermodynamic_observables<SpinVector>(
            r.energies, sublattice_, r.temperature, lat_.lattice_size);
        // "magnetization" is the global-frame order parameter m (the legacy
        // estimator averaged local-frame sublattice vectors).
        for (const auto& op : r.thermo.order_parameters)
            if (op.name == "m") obs.magnetization = op.mean;
        HDF5PTWriter writer(ctx.rank_dir + "/parallel_tempering_data.h5", r.temperature, lat_.lattice_size,
                            lat_.spin_dim, lat_.N_atoms, r.energies.size(), o.n_equilibration,
                            o.n_measurement, o.probe_every, o.exchange_every, or_rate_,
                            r.local_acceptance, r.swap_acceptance);
        writer.write_timeseries(r.energies, magnetizations_, sublattice_);
        std::vector<std::vector<double>> sm, se, cm, ce;
        for (size_t a = 0; a < obs.sublattice_magnetization.size(); ++a) {
            sm.push_back(obs.sublattice_magnetization[a].values);
            se.push_back(obs.sublattice_magnetization[a].errors);
            cm.push_back(obs.energy_sublattice_cross[a].values);
            ce.push_back(obs.energy_sublattice_cross[a].errors);
        }
        writer.write_observables(obs.energy.value, obs.energy.error, obs.specific_heat.value,
                                 obs.specific_heat.error, obs.magnetization.values, obs.magnetization.errors,
                                 sm, se, cm, ce);
        writer.close();
#endif
        lat_.save_spin_config(ctx.rank_dir + "/spins_T=" + std::to_string(r.temperature) + ".txt");
        lat_.save_positions(ctx.rank_dir + "/positions.txt");
    }

protected:
    size_t extra_size() const {
        if constexpr (has_extra_dof<L>::value) return lat_.extra_dof_size();
        else return 0;
    }

    L& lat_;
    size_t or_rate_;
    bool gaussian_, adaptive_;
    std::vector<SpinVector> magnetizations_;
    std::vector<std::vector<SpinVector>> sublattice_;
};

}  // namespace mc
