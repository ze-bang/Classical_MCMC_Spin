#pragma once
/**
 * population_annealing.h — population annealing (PA): simulated annealing of a
 * population of replicas with Boltzmann resampling, which turns annealing into
 * an equilibrium sampler at every temperature of the schedule and yields the
 * free energy.
 *
 * Algorithm (Hukushima & Iba, AIP Conf. Proc. 690, 200 (2003); Machta,
 * PRE 82, 026704 (2010); Wang, Machta & Katzgraber, PRE 92, 063307 (2015)):
 *
 *   R replicas start from independent uniformly random configurations, i.e.
 *   in equilibrium at beta_0 = 0. For every step beta_i -> beta_{i+1}:
 *     1. reweight  w_j = exp[-(beta_{i+1} - beta_i) E_j],  Q_i = (1/R) sum_j w_j;
 *     2. resample  replica j gets n_j copies with E[n_j] = R w_j / sum_k w_k.
 *        Systematic resampling (one shared uniform U, n_j = floor(c_j + U) -
 *        floor(c_{j-1} + U) with c_j the cumulative normalised weights) keeps
 *        the population at exactly R and has the smallest variance among the
 *        unbiased schemes;
 *     3. equilibrate: every copy runs `sweeps` MC steps at beta_{i+1} with its
 *        own random stream, so copies of one parent decorrelate.
 *   ln Z(beta_{i+1}) = ln Z(beta_i) + ln Q_i, so ln Z(beta)/N - ln Z(0)/N is
 *   reported at every temperature (Z itself is estimated without bias).
 *
 * Errors and diagnostics. A family is the set of replicas descending from one
 * initial replica. Replicas of one family are correlated, different families
 * nearly independent, so errors come from the delete-m_j jackknife over
 * families (Busing, Meijer & van der Leeden, Stat. Comput. 9, 3 (1999); groups
 * of unequal size m_j). Reported with them: rho_t = R sum_f nu_f^2 (nu_f the
 * fraction of the population in family f; err ~ sqrt(rho_t Var / R), Wang et
 * al. 2015), the family entropy S_f = -sum_f nu_f ln nu_f, the effective sample
 * size of each reweighting ESS = (sum w)^2 / (R sum w^2) and the culled
 * fraction. R / rho_t >> 1 (well above 100) indicates a population large
 * enough for the sweeps used.
 *
 * Schedules: linear in beta (default; the natural variable of the reweighting),
 * geometric in T, or adaptive — each step is the largest one whose ESS stays at
 * or above `target_ess`, the criterion of adaptive sequential Monte Carlo (Del
 * Moral, Doucet & Jasra, Stat. Comput. 22, 1009 (2012)).
 *
 * Parallel layout. The population is split over the MPI ranks in contiguous
 * blocks of the global replica index; each rank evolves its block with one
 * worker model per OpenMP thread (states are stored packed and swapped into a
 * worker for its sweeps). Every rank derives the same resampling from the
 * gathered energies; copies are moved with one MPI_Alltoallv. Replica k at step
 * i draws from a stream keyed on (run seed, i, k), so for a fixed seed the
 * results do not depend on the number of ranks or threads.
 */

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "classical_spin/mc/parallel_tempering.h"  // detail:: MPI helpers, SpinLatticeReplica

namespace mc {

/**
 * A model evolved by population annealing: the parallel-tempering replica
 * operations (one MC step, total energy, state packing) plus a uniformly random
 * initial state and per-replica scalar observables.
 */
template <class W>
concept PopulationWorker = requires(W& w, const W& cw, double T, size_t step, double sigma,
                                    double* out, const double* in) {
    { w.mc_step(T, step, sigma) } -> std::convertible_to<double>;
    { w.energy() } -> std::convertible_to<double>;
    { cw.n_sites() } -> std::convertible_to<size_t>;
    { cw.state_size() } -> std::convertible_to<size_t>;
    cw.pack_state(out);
    w.unpack_state(in);
    w.randomize();
    { cw.uses_step_size() } -> std::convertible_to<bool>;
    { cw.observable_names() } -> std::convertible_to<std::vector<std::string>>;
    cw.observables(out);
};

enum class PASchedule { LinearBeta, GeometricT, Adaptive };

inline PASchedule parse_pa_schedule(const std::string& name) {
    if (name == "linear_beta" || name == "linear") return PASchedule::LinearBeta;
    if (name == "geometric" || name == "geometric_T") return PASchedule::GeometricT;
    if (name == "adaptive") return PASchedule::Adaptive;
    throw std::invalid_argument("unknown population-annealing schedule '" + name +
                                "' (valid: linear_beta, geometric, adaptive)");
}

inline const char* pa_schedule_name(PASchedule s) {
    switch (s) {
        case PASchedule::LinearBeta: return "linear_beta";
        case PASchedule::GeometricT: return "geometric";
        case PASchedule::Adaptive: return "adaptive";
    }
    return "?";
}

struct PAOptions {
    size_t population = 1000;        ///< R, total over all ranks (fixed)
    double T_start = 10.0;           ///< first temperature after beta = 0 (fixed schedules)
    double T_end = 0.1;              ///< last temperature
    size_t n_temperatures = 100;     ///< fixed schedules: points from T_start to T_end inclusive
    PASchedule schedule = PASchedule::LinearBeta;
    double target_ess = 0.9;         ///< adaptive schedule: ESS fraction kept by every step
    size_t max_temperatures = 100000;///< adaptive schedule: safety cap on the number of steps
    size_t sweeps = 10;              ///< MC steps per replica and temperature
    double sigma0 = 2.0;             ///< initial proposal width (adaptive Gaussian moves)
    double target_acceptance = 0.45;
    std::string output_dir;          ///< pa_summary.txt is written here (rank 0); empty: none
    int verbosity = 1;
};

/// Population statistics at one temperature (after the sweeps at that temperature).
struct PAStep {
    double beta = 0.0, T = 0.0;
    double energy = 0.0, energy_error = 0.0;       ///< <E>/N
    double specific_heat = 0.0, specific_heat_error = 0.0;  ///< beta^2 Var(E)/N
    double ln_Z = 0.0;                             ///< ln Z(beta)/N - ln Z(0)/N
    double energy_min = 0.0;                       ///< lowest E/N in the population
    double rho_t = 0.0, family_entropy = 0.0;
    size_t n_families = 0;
    double ess_fraction = 1.0, culled_fraction = 0.0;  ///< of the reweighting into this beta
    double acceptance = 0.0, sigma = 0.0;
    std::vector<double> obs_mean, obs_error;
};

struct PAResult {
    std::vector<PAStep> steps;
    std::vector<std::string> observable_names;
    size_t population = 0;
    double best_energy = 0.0;          ///< lowest total energy at the final temperature
    std::vector<double> best_state;    ///< its packed state (every rank)
};

namespace detail {

/// Inclusive-exclusive block [lo, hi) of the global replica index owned by `rank`.
inline size_t pa_block_begin(size_t R, int rank, int size) {
    return size_t((unsigned __int128)R * unsigned(rank) / unsigned(size));
}

inline int pa_owner(size_t k, size_t R, int size) {
    // Largest r with block_begin(r) <= k.
    int r = int((unsigned __int128)k * unsigned(size) / R);
    while (r + 1 < size && pa_block_begin(R, r + 1, size) <= k) ++r;
    while (r > 0 && pa_block_begin(R, r, size) > k) --r;
    return r;
}

/// ESS fraction (sum w)^2 / (R sum w^2) of reweighting energies by d_beta.
inline double pa_ess(const std::vector<double>& E, double e_min, double d_beta) {
    double s1 = 0.0, s2 = 0.0;
    for (double e : E) {
        const double w = std::exp(-d_beta * (e - e_min));
        s1 += w;
        s2 += w * w;
    }
    return (s1 * s1) / (double(E.size()) * s2);
}

/**
 * Delete-m_j grouped jackknife (Busing et al. 1999) of a statistic of the
 * population, given per-group sums. `stat(n, sums)` evaluates the statistic
 * from a count and the vector of sums; returns (estimate, error).
 */
template <class Stat>
std::pair<double, double> grouped_jackknife(const std::vector<double>& group_n,
                                            const std::vector<std::vector<double>>& group_sums,
                                            Stat&& stat) {
    const size_t G = group_n.size();
    const size_t P = group_sums.empty() ? 0 : group_sums[0].size();
    double n = 0.0;
    std::vector<double> tot(P, 0.0);
    for (size_t g = 0; g < G; ++g) {
        n += group_n[g];
        for (size_t p = 0; p < P; ++p) tot[p] += group_sums[g][p];
    }
    const double full = stat(n, tot);
    if (G < 2) return {full, std::numeric_limits<double>::infinity()};
    std::vector<double> loo(G), sums(P);
    double theta_j = double(G) * full;
    for (size_t g = 0; g < G; ++g) {
        for (size_t p = 0; p < P; ++p) sums[p] = tot[p] - group_sums[g][p];
        loo[g] = stat(n - group_n[g], sums);
        theta_j -= (1.0 - group_n[g] / n) * loo[g];
    }
    double var = 0.0;
    for (size_t g = 0; g < G; ++g) {
        const double h = n / group_n[g];
        const double pseudo = h * full - (h - 1.0) * loo[g];
        var += (pseudo - theta_j) * (pseudo - theta_j) / (h - 1.0);
    }
    return {full, std::sqrt(var / double(G))};
}

inline uint64_t pa_stream(uint64_t seed, uint64_t step, uint64_t replica) {
    return mix64(mix64(seed ^ mix64(step + 0x5041ULL)) ^ replica);
}

}  // namespace detail

/**
 * Run population annealing. `workers` holds one model per OpenMP thread (all
 * describing the same Hamiltonian; their states are overwritten). Collective
 * over `comm`; the result is identical on every rank.
 */
template <PopulationWorker W>
PAResult run_population_annealing(std::vector<W*> workers, const PAOptions& o, MPI_Comm parent_comm) {
    detail::require_mpi("population annealing");
    detail::CommGuard comm(parent_comm);
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // ---- validation (identical options on every rank, so every rank agrees) ----
    std::string err;
    if (workers.empty() || std::find(workers.begin(), workers.end(), nullptr) != workers.end())
        err = "population annealing: need at least one (non-null) worker per rank";
    else if (o.population < size_t(size))
        err = "population annealing: population (" + std::to_string(o.population) +
              ") must be at least the number of MPI ranks (" + std::to_string(size) + ")";
    else if (!(o.T_end > 0.0) || !std::isfinite(o.T_end))
        err = "population annealing: T_end must be positive and finite";
    else if (o.schedule != PASchedule::Adaptive &&
             (!(o.T_start >= o.T_end) || !std::isfinite(o.T_start) || o.n_temperatures < 1))
        err = "population annealing: need T_start >= T_end > 0 and n_temperatures >= 1";
    else if (o.schedule == PASchedule::Adaptive && !(o.target_ess > 0.0 && o.target_ess < 1.0))
        err = "population annealing: target_ess must lie in (0, 1)";
    else if (o.sweeps == 0)
        err = "population annealing: sweeps per temperature must be >= 1";
    detail::throw_if_any(err, comm);
    if (!detail::same_on_all_ranks({double(o.population), o.T_start, o.T_end, double(o.n_temperatures),
                                    double(int(o.schedule)), o.target_ess, double(o.sweeps)},
                                   comm))
        throw std::invalid_argument("population annealing: ranks were given different options");

    const size_t R = o.population;
    const size_t S = workers[0]->state_size();
    const double N = double(workers[0]->n_sites());
    const std::vector<std::string> obs_names = workers[0]->observable_names();
    const size_t n_obs = obs_names.size();
    const size_t lo = detail::pa_block_begin(R, rank, size);
    const size_t n_local = detail::pa_block_begin(R, rank + 1, size) - lo;
    const uint64_t seed = detail::shared_exchange_seed(comm);
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = std::max(1, std::min<int>(int(workers.size()), omp_get_max_threads()));
#endif

    // Global block layout (counts / displacements in replicas).
    std::vector<int> cnt(size), dsp(size);
    for (int r = 0; r < size; ++r) {
        dsp[r] = int(detail::pa_block_begin(R, r, size));
        cnt[r] = int(detail::pa_block_begin(R, r + 1, size)) - dsp[r];
    }

    std::vector<double> states(n_local * S), energy(n_local), obs(n_local * n_obs);
    std::vector<double> family(n_local), accept(n_local);
    for (size_t j = 0; j < n_local; ++j) family[j] = double(lo + j);

    // Evolve the local replicas at temperature T (step index `step`): random
    // initial states when `init`, `sweeps` MC steps otherwise.
    auto evolve = [&](size_t step, double T, double sigma, bool init) {
        std::string local_err;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
#endif
        for (long jj = 0; jj < long(n_local); ++jj) {
            const size_t j = size_t(jj);
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            W& w = *workers[size_t(tid)];
            try {
                seed_lehman_stream(detail::pa_stream(seed, step, lo + j));
                double acc_sum = 0.0;
                size_t acc_n = 0;
                if (init) {
                    w.randomize();
                } else {
                    w.unpack_state(states.data() + j * S);
                    for (size_t s = 0; s < o.sweeps; ++s) {
                        const double a = w.mc_step(T, s, sigma);
                        if (std::isfinite(a)) { acc_sum += a; ++acc_n; }
                    }
                }
                energy[j] = w.energy();
                w.observables(obs.data() + j * n_obs);
                w.pack_state(states.data() + j * S);
                accept[j] = acc_n ? acc_sum / double(acc_n) : std::numeric_limits<double>::quiet_NaN();
                if (!std::isfinite(energy[j])) throw std::runtime_error("non-finite energy");
            } catch (const std::exception& e) {
#ifdef _OPENMP
                #pragma omp critical(pa_error)
#endif
                if (local_err.empty())
                    local_err = "population annealing: replica " + std::to_string(lo + j) + ": " + e.what();
            }
        }
        detail::throw_if_any<std::runtime_error>(local_err, comm);
    };

    // ---- beta schedule ----
    const double beta_end = 1.0 / o.T_end;
    std::vector<double> fixed_betas;
    if (o.schedule != PASchedule::Adaptive) {
        const size_t n = o.n_temperatures;
        for (size_t i = 0; i < n; ++i) {
            const double x = (n == 1) ? 1.0 : double(i) / double(n - 1);
            const double b = (o.schedule == PASchedule::LinearBeta)
                                 ? (1.0 - x) / o.T_start + x * beta_end
                                 : 1.0 / (o.T_start * std::pow(o.T_end / o.T_start, x));
            fixed_betas.push_back(b);
        }
        fixed_betas.back() = beta_end;
    }

    StepSizeController ctrl(o.sigma0, o.target_acceptance);
    const bool adapt_sigma = workers[0]->uses_step_size();
    PAResult res;
    res.observable_names = obs_names;
    res.population = R;

    evolve(0, std::numeric_limits<double>::infinity(), ctrl.sigma(), true);
    double beta = 0.0, ln_Z = 0.0;
    std::vector<double> E_all(R), payload, recv;

    for (size_t step = 1;; ++step) {
        const bool done = (o.schedule == PASchedule::Adaptive) ? !(beta < beta_end)
                                                               : step > fixed_betas.size();
        if (done) break;
        if (o.schedule == PASchedule::Adaptive && step > o.max_temperatures)
            throw std::runtime_error("population annealing: more than max_temperatures steps; lower "
                                     "target_ess or raise max_temperatures");

        // 1. Reweight into the next beta.
        MPI_Allgatherv(energy.data(), int(n_local), MPI_DOUBLE, E_all.data(), cnt.data(), dsp.data(),
                       MPI_DOUBLE, comm);
        const double e_min = *std::min_element(E_all.begin(), E_all.end());
        double beta_next;
        if (o.schedule == PASchedule::Adaptive) {
            double hi = beta_end - beta;
            if (detail::pa_ess(E_all, e_min, hi) < o.target_ess) {
                double a = 0.0;
                for (int it = 0; it < 60; ++it) {
                    const double m = 0.5 * (a + hi);
                    (detail::pa_ess(E_all, e_min, m) >= o.target_ess ? a : hi) = m;
                }
                hi = std::max(a, 1e-12 * (beta + 1.0));  // always make progress
            }
            beta_next = (beta + hi >= beta_end * (1.0 - 1e-12)) ? beta_end : beta + hi;
        } else {
            beta_next = fixed_betas[step - 1];
        }
        const double d_beta = beta_next - beta;
        std::vector<double> w(R);
        double w_sum = 0.0, w2_sum = 0.0;
        for (size_t k = 0; k < R; ++k) {
            w[k] = std::exp(-d_beta * (E_all[k] - e_min));
            w_sum += w[k];
            w2_sum += w[k] * w[k];
        }
        ln_Z += std::log(w_sum / double(R)) - d_beta * e_min;

        // 2. Systematic resampling: parent[k] of every new replica k.
        const double U = detail::exchange_uniform(seed, step, 0x5245534D504C45ULL);
        std::vector<size_t> parent;
        parent.reserve(R);
        size_t culled = 0;
        double cum = 0.0;
        long prev = 0;
        for (size_t j = 0; j < R; ++j) {
            cum += w[j];
            const double c = (j + 1 == R) ? double(R) : double(R) * cum / w_sum;
            const long upto = long(std::floor(c + U));
            const long n_j = std::max(0L, upto - prev);
            prev = std::max(prev, upto);
            if (n_j == 0) ++culled;
            for (long c2 = 0; c2 < n_j && parent.size() < R; ++c2) parent.push_back(j);
        }
        while (parent.size() < R) parent.push_back(parent.empty() ? 0 : parent.back());  // round-off guard

        // 3. Move the copies to the ranks that own their new index.
        const size_t stride = S + 1;  // packed state + family id
        std::vector<int> scount(size, 0), sdisp(size, 0), rcount(size, 0), rdisp(size, 0);
        for (size_t k = 0; k < R; ++k) {
            const int src = detail::pa_owner(parent[k], R, size), dst = detail::pa_owner(k, R, size);
            if (src == rank) scount[dst] += int(stride);
            if (dst == rank) rcount[src] += int(stride);
        }
        for (int r = 1; r < size; ++r) {
            sdisp[r] = sdisp[r - 1] + scount[r - 1];
            rdisp[r] = rdisp[r - 1] + rcount[r - 1];
        }
        payload.assign(size_t(sdisp[size - 1] + scount[size - 1]), 0.0);
        {
            std::vector<int> fill(sdisp);
            for (size_t k = 0; k < R; ++k) {
                const size_t p = parent[k];
                if (detail::pa_owner(p, R, size) != rank) continue;
                const int dst = detail::pa_owner(k, R, size);
                double* out = &payload[size_t(fill[dst])];
                std::copy_n(&states[(p - lo) * S], S, out);
                out[S] = family[p - lo];
                fill[dst] += int(stride);
            }
        }
        recv.assign(size_t(rdisp[size - 1] + rcount[size - 1]), 0.0);
        MPI_Alltoallv(payload.data(), scount.data(), sdisp.data(), MPI_DOUBLE, recv.data(), rcount.data(),
                      rdisp.data(), MPI_DOUBLE, comm);
        {
            // Messages from each source arrive in increasing new index k.
            std::vector<int> take(rdisp);
            for (size_t j = 0; j < n_local; ++j) {
                const int src = detail::pa_owner(parent[lo + j], R, size);
                const double* in = &recv[size_t(take[src])];
                std::copy_n(in, S, &states[j * S]);
                family[j] = in[S];
                take[src] += int(stride);
            }
        }

        // 4. Equilibrate at the new temperature.
        beta = beta_next;
        const double T = 1.0 / beta;
        const double sigma = ctrl.sigma();
        evolve(step, T, sigma, false);

        // 5. Statistics: gather (E, family, acceptance, observables) to every rank.
        const size_t rec = 3 + n_obs;
        std::vector<double> mine(n_local * rec), all(R * rec);
        for (size_t j = 0; j < n_local; ++j) {
            mine[j * rec + 0] = energy[j];
            mine[j * rec + 1] = family[j];
            mine[j * rec + 2] = accept[j];
            std::copy_n(obs.data() + j * n_obs, n_obs, mine.data() + j * rec + 3);
        }
        std::vector<int> c2(size), d2(size);
        for (int r = 0; r < size; ++r) { c2[r] = cnt[r] * int(rec); d2[r] = dsp[r] * int(rec); }
        MPI_Allgatherv(mine.data(), int(mine.size()), MPI_DOUBLE, all.data(), c2.data(), d2.data(), MPI_DOUBLE,
                       comm);

        PAStep st;
        st.beta = beta;
        st.T = T;
        st.ln_Z = ln_Z / N;
        st.ess_fraction = (w_sum * w_sum) / (double(R) * w2_sum);
        st.culled_fraction = double(culled) / double(R);
        st.sigma = adapt_sigma ? sigma : 0.0;
        // Per-family sums: [E, E^2, O_1, O_1^2, ...].
        std::map<long long, size_t> fam_index;
        std::vector<double> gn;
        std::vector<std::vector<double>> gs;
        double acc_sum = 0.0, e_lo = std::numeric_limits<double>::infinity();
        size_t acc_n = 0;
        for (size_t k = 0; k < R; ++k) {
            const double* r = &all[k * rec];
            const auto [it, fresh] = fam_index.emplace((long long)r[1], gn.size());
            if (fresh) { gn.push_back(0.0); gs.emplace_back(2 + 2 * n_obs, 0.0); }
            std::vector<double>& g = gs[it->second];
            gn[it->second] += 1.0;
            g[0] += r[0];
            g[1] += r[0] * r[0];
            for (size_t q = 0; q < n_obs; ++q) {
                g[2 + 2 * q] += r[3 + q];
                g[3 + 2 * q] += r[3 + q] * r[3 + q];
            }
            if (std::isfinite(r[2])) { acc_sum += r[2]; ++acc_n; }
            e_lo = std::min(e_lo, r[0]);
        }
        st.energy_min = e_lo / N;
        st.acceptance = acc_n ? acc_sum / double(acc_n) : std::numeric_limits<double>::quiet_NaN();
        st.n_families = gn.size();
        for (double n_f : gn) {
            const double nu = n_f / double(R);
            st.rho_t += double(R) * nu * nu;
            st.family_entropy -= nu * std::log(nu);
        }
        auto mean_of = [](size_t p) {
            return [p](double n, const std::vector<double>& s) { return s[p] / n; };
        };
        std::tie(st.energy, st.energy_error) = detail::grouped_jackknife(gn, gs, mean_of(0));
        st.energy /= N;
        st.energy_error /= N;
        std::tie(st.specific_heat, st.specific_heat_error) =
            detail::grouped_jackknife(gn, gs, [&](double n, const std::vector<double>& s) {
                const double m = s[0] / n;
                return beta * beta * (s[1] / n - m * m) / N;
            });
        for (size_t q = 0; q < n_obs; ++q) {
            const auto [m, e] = detail::grouped_jackknife(gn, gs, mean_of(2 + 2 * q));
            st.obs_mean.push_back(m);
            st.obs_error.push_back(e);
        }
        if (adapt_sigma && std::isfinite(st.acceptance)) ctrl.update(st.acceptance);

        if (rank == 0 && o.verbosity > 0 && (o.verbosity > 1 || step % 10 == 0 || beta == beta_end)) {
            detail::CoutFormatGuard guard;
            std::cout << "[PA] step " << std::setw(4) << step << "  T = " << std::setw(11) << std::setprecision(5)
                      << std::scientific << T << "  E/N = " << std::setprecision(8) << std::fixed << st.energy
                      << " +- " << std::setprecision(2) << std::scientific << st.energy_error
                      << "  ESS = " << std::fixed << std::setprecision(3) << st.ess_fraction
                      << "  R/rho_t = " << std::setprecision(1) << double(R) / st.rho_t << std::endl;
        }
        res.steps.push_back(std::move(st));
    }

    // Lowest-energy replica at the final temperature, broadcast from its owner.
    {
        struct { double e; int r; } loc{std::numeric_limits<double>::infinity(), rank}, best{};
        size_t j_best = 0;
        for (size_t j = 0; j < n_local; ++j)
            if (energy[j] < loc.e) { loc.e = energy[j]; j_best = j; }
        MPI_Allreduce(&loc, &best, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
        res.best_energy = best.e;
        res.best_state.assign(S, 0.0);
        if (rank == best.r) std::copy_n(&states[j_best * S], S, res.best_state.begin());
        MPI_Bcast(res.best_state.data(), int(S), MPI_DOUBLE, best.r, comm);
    }

    if (!res.steps.empty() && rank == 0 && o.verbosity > 0) {
        const PAStep& f = res.steps.back();
        if (double(R) / f.rho_t < 100.0)
            std::cerr << "[PA] WARNING: R / rho_t = " << double(R) / f.rho_t
                      << " < 100 at T_end: the population is dominated by few families; raise the "
                         "population or the sweeps per temperature" << std::endl;
    }

    // Summary file (rank 0); a failure is reported on every rank.
    std::string io_err;
    if (rank == 0 && !o.output_dir.empty()) {
        io_err = detail::guarded_io([&] {
            std::filesystem::create_directories(o.output_dir);
            const std::string path = o.output_dir + "/pa_summary.txt";
            std::ofstream out(path + ".tmp");
            out << std::setprecision(17);
            out << "# population annealing: R = " << R << ", sweeps = " << o.sweeps << ", schedule = "
                << pa_schedule_name(o.schedule) << ", N = " << N << ", MPI ranks = " << size << "\n";
            out << "# columns: step beta T E/N dE/N C/N dC/N lnZ/N-lnZ0/N E_min/N rho_t family_entropy "
                   "n_families ess culled acceptance sigma";
            for (const auto& name : obs_names) out << " " << name << " d" << name;
            out << "\n";
            for (size_t i = 0; i < res.steps.size(); ++i) {
                const PAStep& s = res.steps[i];
                out << i + 1 << " " << s.beta << " " << s.T << " " << s.energy << " " << s.energy_error << " "
                    << s.specific_heat << " " << s.specific_heat_error << " " << s.ln_Z << " " << s.energy_min
                    << " " << s.rho_t << " " << s.family_entropy << " " << s.n_families << " "
                    << s.ess_fraction << " " << s.culled_fraction << " " << s.acceptance << " " << s.sigma;
                for (size_t q = 0; q < n_obs; ++q) out << " " << s.obs_mean[q] << " " << s.obs_error[q];
                out << "\n";
            }
            out.close();
            if (!out) throw std::runtime_error("cannot write " + path);
            std::filesystem::rename(path + ".tmp", path);
        });
    }
    detail::throw_if_any<std::runtime_error>(io_err, comm);
    return res;
}

/**
 * Population-annealing worker for the lattice classes (Lattice, PhononLattice):
 * the parallel-tempering adapter (same MC step policy) plus a uniformly random
 * initial state and the global-frame order parameter |m| and m^2 per site.
 */
template <class L>
class LatticePopulationWorker : public SpinLatticeReplica<L> {
public:
    LatticePopulationWorker(L& lat, size_t overrelaxation_rate, bool gaussian_move)
        : SpinLatticeReplica<L>(lat, overrelaxation_rate, gaussian_move, gaussian_move) {
        // Replicas are evolved in parallel; a replica's own sweeps stay serial
        // so its trajectory depends only on its random stream.
        if constexpr (requires { lat.parallel_sweep_min_sites; })
            lat.parallel_sweep_min_sites = std::numeric_limits<size_t>::max();
    }

    void randomize() { this->lat_.init_random(); }
    std::vector<std::string> observable_names() const { return {"m_abs", "m2"}; }
    void observables(double* out) const {
        const SpinVector m = this->lat_.magnetization_global();
        out[0] = m.norm();
        out[1] = m.squaredNorm();
    }
};

}  // namespace mc
