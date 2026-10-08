/**
 * spin_config.cpp - configuration parsing, validation and serialisation.
 *
 * One table (typed_key_table) describes every typed key: its canonical name,
 * accepted aliases, a strict text parser and a lossless formatter. The file
 * parser, SpinConfig::set (used by parameter sweeps) and to_file all go
 * through it, so a key the parser accepts can be swept and written back, and
 * from_file(to_file(c)) reproduces c. Keys that are not typed fields are
 * Hamiltonian / model parameters read through get_param; they are accepted
 * only if registered below (hamiltonian_key_registry), so a misspelt key is
 * an error with a suggestion instead of a silently ignored line.
 */

#include "classical_spin/core/spin_config.h"
#include "classical_spin/dynamics/ode_method.h"
#include "classical_spin/dynamics/time_grid.h"
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <unordered_map>

using namespace std;

namespace {

// ============================================================================
// Strict value parsing. Errors are std::invalid_argument naming the key.
// ============================================================================

[[noreturn]] void bad_value(const string& key, const string& value, const string& why) {
    throw invalid_argument(key + " = '" + value + "': " + why);
}

/// Finite double; the whole (trimmed) text must be consumed.
double parse_real(const string& key, const string& text) {
    const string v = trim(text);
    if (v.empty()) bad_value(key, text, "expected a number");
    errno = 0;
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') bad_value(key, text, "not a number");
    if (!std::isfinite(x) || errno == ERANGE) bad_value(key, text, "not a finite number");
    return x;
}

/**
 * Integer of type I. Plain integer text is parsed exactly (all 64 bits);
 * otherwise the value is read as a double ("1e5", "100000.0") and accepted
 * only if it is integral and within the range of I.
 */
template <class I>
I parse_integer(const string& key, const string& text) {
    const string v = trim(text);
    if (v.empty()) bad_value(key, text, "expected an integer");
    const char* first = v.data();
    const char* last = v.data() + v.size();
    if (*first == '+') ++first;
    I out{};
    auto [ptr, ec] = std::from_chars(first, last, out);
    if (ec == std::errc() && ptr == last) return out;
    if (ec == std::errc::result_out_of_range) bad_value(key, text, "integer out of range");
    // Not plain integer text: allow a double that holds an integral value.
    const double x = parse_real(key, v);
    if (x != std::floor(x)) bad_value(key, text, "expected an integer (got a fractional value)");
    if (std::is_unsigned_v<I> && x < 0.0) bad_value(key, text, "must be non-negative");
    // 2^64 and 2^63 are exactly representable, so these bounds are exact.
    const double lo = static_cast<double>(std::numeric_limits<I>::min());
    const double hi = std::ldexp(1.0, std::numeric_limits<I>::digits);   // max + 1
    if (x < lo || x >= hi) bad_value(key, text, "integer out of range");
    return static_cast<I>(x);
}

bool parse_flag(const string& key, const string& text) {
    string s = trim(text);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (s == "true" || s == "yes" || s == "on" || s == "1") return true;
    if (s == "false" || s == "no" || s == "off" || s == "0") return false;
    bad_value(key, text, "expected true/false (also yes/no, on/off, 1/0)");
}

/// Comma-separated items with surrounding [] or () removed; empty items skipped.
vector<string> split_list(const string& text) {
    string s = trim(text);
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '[' || c == ']' || c == '(' || c == ')'; }),
            s.end());
    vector<string> items;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) items.push_back(item);
    }
    return items;
}

vector<double> parse_real_list(const string& key, const string& text) {
    vector<double> out;
    for (const string& item : split_list(text)) out.push_back(parse_real(key, item));
    return out;
}

/// Flat list grouped into vectors of `dim` values when the count is a multiple
/// of dim, else one vector of everything (the historic pump_direction rule).
vector<vector<double>> parse_grouped(const string& key, const string& text, size_t dim) {
    const vector<double> v = parse_real_list(key, text);
    vector<vector<double>> out;
    if (v.empty()) return out;
    if (v.size() % dim != 0) {
        out.push_back(v);
        return out;
    }
    for (size_t i = 0; i < v.size(); i += dim) out.emplace_back(v.begin() + i, v.begin() + i + dim);
    return out;
}

/// Exactly `dim` values per vector (dssf_q_points).
vector<vector<double>> parse_vectors_of(const string& key, const string& text, size_t dim) {
    const vector<double> v = parse_real_list(key, text);
    if (v.size() % dim != 0)
        bad_value(key, text, "needs a multiple of " + to_string(dim) + " values (got " + to_string(v.size()) + ")");
    vector<vector<double>> out;
    for (size_t i = 0; i < v.size(); i += dim) out.emplace_back(v.begin() + i, v.begin() + i + dim);
    return out;
}

// ============================================================================
// Lossless formatting (read back exactly by the parsers above).
// ============================================================================

string fmt_real(double x) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", x);
    return buf;
}

string fmt_float(float x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(x));
    return buf;
}

string fmt_real_list(const vector<double>& v) {
    string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + fmt_real(v[i]);
    return s;
}

string fmt_grouped(const vector<vector<double>>& vv) {
    vector<double> flat;
    for (const auto& v : vv) flat.insert(flat.end(), v.begin(), v.end());
    return fmt_real_list(flat);
}

string fmt_flag(bool b) { return b ? "true" : "false"; }

string simulation_name(SimulationType s) {
    switch (s) {
        case SimulationType::SIMULATED_ANNEALING: return "simulated_annealing";
        case SimulationType::POPULATION_ANNEALING: return "population_annealing";
        case SimulationType::PARALLEL_TEMPERING: return "parallel_tempering";
        case SimulationType::MOLECULAR_DYNAMICS: return "molecular_dynamics";
        case SimulationType::PUMP_PROBE: return "pump_probe";
        case SimulationType::TWOD_COHERENT_SPECTROSCOPY: return "2dcs";
        case SimulationType::PARAMETER_SWEEP: return "parameter_sweep";
        case SimulationType::KINETIC_BARRIER_ANALYSIS: return "kinetic_barrier";
        case SimulationType::CUSTOM: return "custom";
    }
    return "unknown";
}

// ============================================================================
// Typed key table
// ============================================================================

using Setter = std::function<void(SpinConfig&, const string& key, const string& value)>;
using Getter = std::function<string(const SpinConfig&)>;

struct KeySpec {
    string name;              // canonical name (written by to_file)
    vector<string> aliases;   // also accepted on input
    Setter set;
    Getter get;
    string deprecated;        // non-empty: warning shown when an alias in `deprecated_aliases` is used
    vector<string> deprecated_aliases;
    string no_effect;         // non-empty: the key is accepted but not used (warning shown)
};

/// Mark a key as accepted for compatibility but not read by any simulation.
KeySpec unused(KeySpec k, const string& why) {
    k.no_effect = why;
    return k;
}

/// Spec for a plain field: parser and formatter chosen from the field type.
template <class T>
KeySpec field(const string& name, T SpinConfig::*m, vector<string> aliases = {}) {
    KeySpec k;
    k.name = name;
    k.aliases = std::move(aliases);
    if constexpr (std::is_same_v<T, double>) {
        k.set = [m](SpinConfig& c, const string& key, const string& v) { c.*m = parse_real(key, v); };
        k.get = [m](const SpinConfig& c) { return fmt_real(c.*m); };
    } else if constexpr (std::is_same_v<T, float>) {
        k.set = [m](SpinConfig& c, const string& key, const string& v) {
            const double x = parse_real(key, v);
            if (std::abs(x) > std::numeric_limits<float>::max()) bad_value(key, v, "out of range");
            c.*m = static_cast<float>(x);
        };
        k.get = [m](const SpinConfig& c) { return fmt_float(c.*m); };
    } else if constexpr (std::is_same_v<T, bool>) {
        k.set = [m](SpinConfig& c, const string& key, const string& v) { c.*m = parse_flag(key, v); };
        k.get = [m](const SpinConfig& c) { return fmt_flag(c.*m); };
    } else if constexpr (std::is_integral_v<T>) {
        k.set = [m](SpinConfig& c, const string& key, const string& v) { c.*m = parse_integer<T>(key, v); };
        k.get = [m](const SpinConfig& c) { return to_string(c.*m); };
    } else if constexpr (std::is_same_v<T, string>) {
        k.set = [m](SpinConfig& c, const string&, const string& v) { c.*m = trim(v); };
        k.get = [m](const SpinConfig& c) { return c.*m; };
    } else if constexpr (std::is_same_v<T, vector<double>>) {
        k.set = [m](SpinConfig& c, const string& key, const string& v) { c.*m = parse_real_list(key, v); };
        k.get = [m](const SpinConfig& c) { return fmt_real_list(c.*m); };
    } else if constexpr (std::is_same_v<T, vector<string>>) {
        k.set = [m](SpinConfig& c, const string&, const string& v) { c.*m = split_list(v); };
        k.get = [m](const SpinConfig& c) {
            string s;
            for (size_t i = 0; i < (c.*m).size(); ++i) s += (i ? "," : "") + (c.*m)[i];
            return s;
        };
    } else {
        static_assert(sizeof(T) == 0, "no parser for this field type");
    }
    return k;
}

/// Spec for a list of vectors grouped by `dim` (pump directions).
KeySpec grouped(const string& name, vector<vector<double>> SpinConfig::*m, size_t dim) {
    KeySpec k;
    k.name = name;
    k.set = [m, dim](SpinConfig& c, const string& key, const string& v) { c.*m = parse_grouped(key, v, dim); };
    k.get = [m](const SpinConfig& c) { return fmt_grouped(c.*m); };
    return k;
}

/// Legacy single-axis sweep keys: element 0 of the N-dimensional lists.
KeySpec legacy_sweep_real(const string& name, double SpinConfig::*single, vector<double> SpinConfig::*list) {
    KeySpec k;
    k.name = name;
    k.set = [single, list](SpinConfig& c, const string& key, const string& v) {
        const double x = parse_real(key, v);
        c.*single = x;
        if ((c.*list).empty()) (c.*list).push_back(x);
        else (c.*list)[0] = x;
    };
    k.get = nullptr;   // input only: to_file writes the list form
    return k;
}

vector<KeySpec> build_typed_key_table() {
    using C = SpinConfig;
    vector<KeySpec> t;
    auto add = [&](KeySpec k) { t.push_back(std::move(k)); };

    // --- System -----------------------------------------------------------
    {
        KeySpec k;
        k.name = "system";
        k.aliases = {"system_type"};
        k.set = [](C& c, const string&, const string& v) { c.system = parse_system(v); };
        k.get = [](const C& c) { return system_type_to_string(c.system); };
        add(k);
    }
    {
        KeySpec k;
        k.name = "lattice_size";
        k.set = [](C& c, const string& key, const string& v) {
            const vector<string> items = split_list(v);
            if (items.size() != 3) bad_value(key, v, "needs 3 values (cells along a1, a2, a3)");
            for (int d = 0; d < 3; ++d) c.lattice_size[d] = parse_integer<size_t>(key, items[d]);
        };
        k.get = [](const C& c) {
            return to_string(c.lattice_size[0]) + "," + to_string(c.lattice_size[1]) + "," +
                   to_string(c.lattice_size[2]);
        };
        add(k);
    }
    add(field("spin_length", &C::spin_length));
    add(field("spin_length_su3", &C::spin_length_su3));
    {
        KeySpec k;
        k.name = "simulation_mode";
        k.aliases = {"simulation"};   // the name the old to_file wrote
        k.set = [](C& c, const string&, const string& v) { c.simulation = parse_simulation(v); };
        k.get = [](const C& c) { return simulation_name(c.simulation); };
        add(k);
    }

    // --- General ----------------------------------------------------------
    add(field("num_trials", &C::num_trials));
    add(field("seed", &C::seed));
    add(field("local_update", &C::local_update));
    add(field("su3_mc_manifold", &C::su3_mc_manifold));
    add(field("output_dir", &C::output_dir));
    add(unused(field("initial_step_size", &C::initial_step_size),
               "the Gaussian proposal width is adapted during equilibration"));
    add(field("use_twist_boundary", &C::use_twist_boundary, {"tbc"}));
    add(field("twist_sweep_count", &C::twist_sweep_count, {"tbc_sweeps"}));
    add(field("allow_unknown_keys", &C::allow_unknown_keys));

    // --- Temperatures / annealing ----------------------------------------
    add(field("T_start", &C::T_start, {"temperature_start"}));
    add(field("T_end", &C::T_end, {"temperature_end"}));
    add(field("annealing_steps", &C::annealing_steps));
    add(unused(field("equilibration_steps", &C::equilibration_steps),
               "use pt_equilibration_steps for parallel tempering"));
    add(field("cooling_rate", &C::cooling_rate));
    add(field("gaussian_move", &C::gaussian_move));
    add(field("save_observables", &C::save_observables));
    add(unused(field("deterministic", &C::deterministic), "use T_zero / n_deterministics"));
    add(field("T_zero", &C::T_zero));
    add(field("n_deterministics", &C::n_deterministics));
    add(field("adiabatic_phonons", &C::adiabatic_phonons));
    add(field("relax_phonons", &C::relax_phonons));
    add(field("phonon_only_relax", &C::phonon_only_relax));

    // --- Molecular dynamics -----------------------------------------------
    add(field("md_time_start", &C::md_time_start));
    add(field("md_time_end", &C::md_time_end));
    add(field("md_timestep", &C::md_timestep, {"dt"}));
    add(field("md_save_interval", &C::md_save_interval));
    add(field("md_integrator", &C::md_integrator, {"integrator"}));
    add(field("damping_form", &C::damping_form));
    add(field("dssf_samples", &C::dssf_samples));
    {
        KeySpec k;
        k.name = "dssf_q_points";
        k.set = [](C& c, const string& key, const string& v) { c.dssf_q_points = parse_vectors_of(key, v, 3); };
        k.get = [](const C& c) { return fmt_grouped(c.dssf_q_points); };
        add(k);
    }
    add(field("dssf_temperature", &C::dssf_temperature));
    add(field("dssf_t_equilibrate", &C::dssf_t_equilibrate));
    add(field("dssf_t_decorrelate", &C::dssf_t_decorrelate));
    add(field("dssf_alpha", &C::dssf_alpha));
    add(field("dssf_hann_window", &C::dssf_hann_window));
    add(field("md_abs_tol", &C::md_abs_tol));
    add(field("md_rel_tol", &C::md_rel_tol));
    add(field("pump_probe_abs_tol", &C::pump_probe_abs_tol));
    add(field("pump_probe_rel_tol", &C::pump_probe_rel_tol));
    add(field("use_gpu", &C::use_gpu, {"gpu"}));

    // --- Parallel tempering -----------------------------------------------
    add(unused(field("num_replicas", &C::num_replicas),
               "parallel tempering runs one replica per MPI rank (pt_ranks_per_point in sweeps)"));
    {
        KeySpec k = field("pt_exchange_frequency", &C::pt_exchange_frequency, {"pt_sweeps_per_exchange"});
        // pt_sweeps_per_exchange was parsed but never read; it is now an alias.
        auto plain = k.set;
        k.set = [plain](C& c, const string& key, const string& v) {
            plain(c, key, v);
            c.pt_sweeps_per_exchange = c.pt_exchange_frequency;
        };
        k.deprecated = "use pt_exchange_frequency";
        k.deprecated_aliases = {"pt_sweeps_per_exchange"};
        add(k);
    }
    add(field("pt_equilibration_steps", &C::pt_equilibration_steps));
    add(field("pt_measurement_steps", &C::pt_measurement_steps));
    // --- Population annealing ---------------------------------------------
    add(field("pa_population", &C::pa_population));
    add(field("pa_sweeps", &C::pa_sweeps));
    add(field("pa_temperatures", &C::pa_temperatures));
    add(field("pa_schedule", &C::pa_schedule));
    add(field("pa_target_ess", &C::pa_target_ess));
    add(field("pa_max_temperatures", &C::pa_max_temperatures));
    add(field("overrelaxation_rate", &C::overrelaxation_rate));
    add(field("probe_rate", &C::probe_rate));
    {
        KeySpec k;
        k.name = "ranks_to_write";
        k.set = [](C& c, const string& key, const string& v) {
            string up = trim(v);
            std::transform(up.begin(), up.end(), up.begin(), [](unsigned char ch) { return char(std::toupper(ch)); });
            if (up == "ALL" || up == "FULL") {
                c.ranks_to_write = {-1};
                return;
            }
            vector<int> r;
            for (const string& item : split_list(v)) r.push_back(parse_integer<int>(key, item));
            c.ranks_to_write = r;
        };
        k.get = [](const C& c) {
            if (!c.ranks_to_write.empty() && c.ranks_to_write[0] == -1) return string("ALL");
            string s;
            for (size_t i = 0; i < c.ranks_to_write.size(); ++i) s += (i ? "," : "") + to_string(c.ranks_to_write[i]);
            return s;
        };
        add(k);
    }
    add(field("pt_ranks_per_point", &C::pt_ranks_per_point));
    add(field("pt_accumulate_correlations", &C::pt_accumulate_correlations));
    add(field("pt_n_bond_types", &C::pt_n_bond_types));
    add(field("pt_optimize_temperatures", &C::pt_optimize_temperatures));
    add(field("pt_temperature_optimizer", &C::pt_temperature_optimizer));
    add(unused(field("pt_target_acceptance", &C::pt_target_acceptance),
               "the ladder tuners equalise the swap rejection instead"));
    add(field("pt_optimization_warmup", &C::pt_optimization_warmup));
    add(field("pt_optimization_sweeps", &C::pt_optimization_sweeps));
    add(field("pt_optimization_iterations", &C::pt_optimization_iterations));
    add(field("pt_optimization_tolerance", &C::pt_optimization_tolerance));

    // --- Pulses -------------------------------------------------------------
    add(field("pump_table_file", &C::pump_table_file));
    add(field("pump_amplitude", &C::pump_amplitude));
    add(field("pump_width", &C::pump_width));
    add(field("pump_frequency", &C::pump_frequency));
    add(field("pump_time", &C::pump_time));
    add(grouped("pump_direction", &C::pump_directions, 3));
    add(grouped("pump_direction_2", &C::pump_directions_2, 3));
    add(field("pump_amplitude_su3", &C::pump_amplitude_su3));
    add(field("pump_width_su3", &C::pump_width_su3));
    add(field("pump_frequency_su3", &C::pump_frequency_su3));
    add(field("pump_frequency_su3_2", &C::pump_frequency_su3_2));
    add(grouped("pump_direction_su3", &C::pump_directions_su3, 8));
    add(grouped("pump_direction_su3_2", &C::pump_directions_su3_2, 8));
    add(field("auto_su3_pump", &C::auto_su3_pump));
    add(field("probe_amplitude", &C::probe_amplitude));
    add(field("probe_width", &C::probe_width));
    add(field("probe_frequency", &C::probe_frequency));
    add(field("probe_time", &C::probe_time));
    add(field("probe_direction", &C::probe_direction));
    add(field("tau_start", &C::tau_start));
    add(field("tau_end", &C::tau_end));
    add(field("tau_step", &C::tau_step));
    add(field("parallel_tau", &C::parallel_tau));
    add(field("reuse_m0_for_m1", &C::reuse_m0_for_m1));
    add(field("stationarity_tol", &C::stationarity_tol));
    add(field("pulse_window_chunking", &C::pulse_window_chunking));
    add(field("pump_probe_omp_threads", &C::pump_probe_omp_threads));

    // --- Parameter sweeps ---------------------------------------------------
    {
        KeySpec k;
        k.name = "sweep_parameter";
        k.set = [](C& c, const string&, const string& v) {
            c.sweep_parameter = trim(v);
            if (c.sweep_parameter.empty()) return;
            if (c.sweep_parameters.empty()) c.sweep_parameters.push_back(c.sweep_parameter);
            else c.sweep_parameters[0] = c.sweep_parameter;
        };
        k.get = nullptr;   // input only (legacy 1-D form of sweep_parameters)
        add(k);
    }
    add(legacy_sweep_real("sweep_start", &C::sweep_start, &C::sweep_starts));
    add(legacy_sweep_real("sweep_end", &C::sweep_end, &C::sweep_ends));
    add(legacy_sweep_real("sweep_step", &C::sweep_step, &C::sweep_steps));
    add(field("sweep_parameters", &C::sweep_parameters));
    add(field("sweep_starts", &C::sweep_starts));
    add(field("sweep_ends", &C::sweep_ends));
    add(field("sweep_steps", &C::sweep_steps));
    {
        KeySpec k;
        k.name = "sweep_base_simulation";
        k.set = [](C& c, const string&, const string& v) { c.sweep_base_simulation = parse_simulation(v); };
        k.get = [](const C& c) { return simulation_name(c.sweep_base_simulation); };
        add(k);
    }

    // --- Fields and initial state -------------------------------------------
    add(field("field_strength", &C::field_strength, {"h"}));
    {
        KeySpec k;
        k.name = "field_direction";
        k.set = [](C& c, const string& key, const string& v) {
            vector<double> d = parse_real_list(key, v);
            double n2 = 0.0;
            for (double x : d) n2 += x * x;
            const double n = std::sqrt(n2);
            // Normalised on input; a unit vector is left untouched so that
            // writing and re-reading the config is exact.
            if (n > 1e-10 && std::abs(n - 1.0) > 4.0 * std::numeric_limits<double>::epsilon())
                for (double& x : d) x /= n;
            c.field_direction = d;
        };
        k.get = [](const C& c) { return fmt_real_list(c.field_direction); };
        add(k);
    }
    add(field("g_factor", &C::g_factor));
    add(field("initial_spin_config", &C::initial_spin_config));
    add(field("local_field_config", &C::local_field_config));
    add(field("pinning_field_config", &C::pinning_field_config));
    add(field("nn_exchange_disorder_config", &C::nn_exchange_disorder_config));
    add(field("nn_exchange_channel_disorder_config", &C::nn_exchange_channel_disorder_config));
    add(field("plaquette_j7_disorder_config", &C::plaquette_j7_disorder_config));
    add(field("use_ferromagnetic_init", &C::use_ferromagnetic_init));
    add(field("ferromagnetic_direction", &C::ferromagnetic_direction));
    add(unused(field("use_mpi", &C::use_mpi), "spin_solver always runs under MPI"));

    // --- GNEB / strain -------------------------------------------------------
    add(field("gneb_n_images", &C::gneb_n_images));
    add(field("gneb_spring_constant", &C::gneb_spring_constant));
    add(field("gneb_max_iterations", &C::gneb_max_iterations));
    add(field("gneb_force_tolerance", &C::gneb_force_tolerance));
    add(field("gneb_step_size", &C::gneb_step_size));
    add(field("gneb_fire_dtmax", &C::gneb_fire_dtmax));
    add(field("gneb_max_strain_amplitude", &C::gneb_max_strain_amplitude));
    add(field("gneb_climbing_start", &C::gneb_climbing_start));
    add(field("gneb_redistribution_freq", &C::gneb_redistribution_freq));
    add(field("gneb_use_climbing_image", &C::gneb_use_climbing_image));
    add(field("gneb_climbing_threshold", &C::gneb_climbing_threshold));
    add(field("gneb_analysis_steps", &C::gneb_analysis_steps));
    add(field("gneb_phonon_amplitude_max", &C::gneb_phonon_amplitude_max));
    add(field("gneb_save_path_evolution", &C::gneb_save_path_evolution));
    add(field("gneb_initial_state_file", &C::gneb_initial_state_file));
    add(field("gneb_final_state_file", &C::gneb_final_state_file));
    add(field("gneb_strain_sweep", &C::gneb_strain_sweep));
    add(field("gneb_n_strain_steps", &C::gneb_n_strain_steps));
    add(field("gneb_strain_max", &C::gneb_strain_max));
    add(field("gneb_strain_direction", &C::gneb_strain_direction));
    add(field("gneb_zigzag_domain", &C::gneb_zigzag_domain));
    add(field("gneb_fixed_strain", &C::gneb_fixed_strain));
    add(field("gneb_dynamic_strain", &C::gneb_dynamic_strain));
    add(field("gneb_adiabatic_strain", &C::gneb_adiabatic_strain));
    add(field("gneb_pin_Eg2_zero", &C::gneb_pin_Eg2_zero));
    add(field("gneb_polish_endpoints", &C::gneb_polish_endpoints));
    add(field("gneb_polish_max_iter", &C::gneb_polish_max_iter));
    add(field("gneb_polish_force_tol", &C::gneb_polish_force_tol));
    add(field("gneb_weight_strain", &C::gneb_weight_strain));
    add(field("gneb_initial_path_dir", &C::gneb_initial_path_dir));
    add(field("fix_strain", &C::fix_strain));
    add(field("external_strain_Eg1", &C::external_strain_Eg1));
    add(field("external_strain_Eg2", &C::external_strain_Eg2));
    add(field("drive_F_Eg1", &C::drive_F_Eg1));
    add(field("drive_F_Eg2", &C::drive_F_Eg2));
    return t;
}

const vector<KeySpec>& typed_key_table() {
    static const vector<KeySpec> table = build_typed_key_table();
    return table;
}

/// Name or alias -> table index.
const unordered_map<string, size_t>& typed_key_index() {
    static const unordered_map<string, size_t> index = [] {
        unordered_map<string, size_t> m;
        const auto& t = typed_key_table();
        for (size_t i = 0; i < t.size(); ++i) {
            m.emplace(t[i].name, i);
            for (const string& a : t[i].aliases) m.emplace(a, i);
        }
        return m;
    }();
    return index;
}

// ============================================================================
// Hamiltonian / model parameter registry
//
// Every key read with get_param / has_param / was_set by a unit-cell builder
// (src/core/unitcell_builders.cpp), the model factories (src/core/phonon_config.cpp),
// the runners or the tools in src/apps. tests/test_config.cpp scans the sources
// for literal keys and fails if one is missing here; key families built at run
// time (mode<i>_..., Kminus<orbit>_...) are matched by the patterns below.
// ============================================================================

const std::set<string>& hamiltonian_key_names() {
    static const std::set<string> names = {
        // Generic honeycomb / Kitaev / BCAO / triangular / pyrochlore exchange
        "D", "D1", "D2", "E", "F", "G", "Gamma", "Gammap", "J", "J1ab", "J1c", "J1xy", "J1z", "J2",
        "J2_A", "J2_B", "J2ab", "J2c", "J2xy", "J2z", "J3", "J3a", "J3b", "J3xy", "J3z", "J7", "J_perp",
        "Jpm", "Jpmpm", "Jxx", "Jyy", "Jzpm", "Jzz", "K", "Ka", "Kb", "Kc", "delta1", "delta2",
        "gxx", "gyy", "gzz", "theta", "pyrochlore_legacy_J2",
        // TmFeO3 (Tm CEF, Tm-Tm, Fe-Tm orbits, frames)
        "e1", "e2", "Jtm_1", "Jtm_2", "Jtm_3", "Jtm_4", "Jtm_5", "Jtm_6", "Jtm_7", "Jtm_8",
        "mu_2x", "mu_2y", "mu_2z", "mu_5x", "mu_5y", "mu_5z", "mu_7x", "mu_7y", "mu_7z",
        "g_ratio_tm", "tm_alpha_scale", "tm_beta_scale", "su3_legacy_convention", "su3_init_component",
        "Kminus_orbit1_scale", "Kminus_orbit2_scale", "Kminus_orbit3_scale", "Kminus_orbit4_scale",
        "W_orbit1_scale", "W_orbit2_scale", "W_orbit3_scale", "W_orbit4_scale",
        "kappaB_orbit1_scale", "kappaB_orbit2_scale", "kappaB_orbit3_scale", "kappaB_orbit4_scale",
        "kappaE_orbit1_scale", "kappaE_orbit2_scale", "kappaE_orbit3_scale", "kappaE_orbit4_scale",
        "use_global_frame", "use_local_frame", "tm_trilinear_reference",
        // MixedLattice dynamics
        "alpha_su3", "gamma_su3", "thermal_cap", "thermal_cool", "thermal_heat", "linear_drive_torque",
        "save_spin_trajectories", "reuse_m0_for_m01",
        // Lattice / generic dynamics
        "alpha_gilbert", "langevin_temperature",
        // NCTO spin-phonon model (phonon_config.cpp, PhononLattice runners and tools)
        "legacy_kitaev_frame", "Z_star", "gamma_E", "gamma_E1", "omega_E", "omega_E1", "lambda_E",
        "lambda_E1_Gamma_0", "lambda_E1_Gamma_1", "lambda_E1_Gamma_2", "lambda_E1_Gammap_0",
        "lambda_E1_Gammap_1", "lambda_E1_Gammap_2", "lambda_E1_J7_0", "lambda_E1_J_0", "lambda_E1_J_1",
        "lambda_E1_J_2", "lambda_E1_K_0", "lambda_E1_K_1", "lambda_E1_K_2", "lambda_E1_quartic",
        "lambda_E1_t_end", "lambda_E1_t_start", "lambda_E1_target", "lambda_Gamma_0", "lambda_Gamma_2",
        "lambda_Gammap_0", "lambda_Gammap_2", "lambda_J7_0", "lambda_J_0", "lambda_J_2", "lambda_K_0",
        "lambda_K_2", "lambda_time_mode", "langevin_bath_C", "langevin_block", "langevin_dT",
        "langevin_quantum", "langevin_seed", "langevin_t_step", "langevin_tau_off", "langevin_tau_on",
        "local_field_h", "mc_sample_lattice", "n_anharmonic", "n_extra_modes", "phonon_langevin_T",
        "phonon_per_site", "preserve_initial_phonons", "probe_phase", "probe_polarization",
        "pulse_polarization", "pump_phase", "pump_polarization", "initial_eps_x", "initial_eps_y",
        "initial_v_x", "initial_v_y", "sld_T", "sld_enabled", "sld_g", "sld_gamma", "sld_init_T", "sld_k",
        "sld_k2", "sld_mass", "sld_quantum", "sld_relax", "sld_v3",
    };
    return names;
}

const vector<std::regex>& hamiltonian_key_patterns() {
    static const vector<std::regex> patterns = [] {
        const char* src[] = {
            // TmFeO3 per-generator SU(3) field and Bloch rates
            R"(h_tm_[1-8])",
            R"(gamma_su3_lambda[1-8])",
            // Fe-Tm couplings: common key or per-orbit override (orbit 1..4)
            R"(Kminus[1-4]?_[257][xyz])",
            R"(kappaE[1-4]?_[257][xyz])",
            R"(kappaB[1-4]?_[xyz]?[13468][xyz])",
            R"(W[13468]_(xx|yy|zz|xy|xz|yz))",
            R"(W[1-4]_[13468]_(xx|yy|zz|xy|xz|yz))",
            // NCTO extra lattice modes (mode0 = the primary E1 mode) and anharmonic terms
            R"(mode[0-9]+_(cE[1-9]|aA1_[1-5]|dA2_[1-4]|bEsq[1-9]|aA1sq_[1-5]|lamJ7|lamJ2A|lamJ2B|lamJ3|)"
            R"(lamJ7sq|lamJ2Asq|lamJ2Bsq|lamJ3sq|irrep|omega|gamma|quartic|Zstar|frozen|Q1|Q2))",
            R"(anh[0-9]+_(target|lam|lamp|g))",
        };
        vector<std::regex> out;
        for (const char* s : src) out.emplace_back(s, std::regex::ECMAScript | std::regex::optimize);
        return out;
    }();
    return patterns;
}

size_t edit_distance(const string& a, const string& b) {
    vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, sub});
        }
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

string unknown_key_message(const string& key) {
    string msg = "unknown key '" + key + "'";
    const string s = SpinConfig::suggest_key(key);
    if (!s.empty()) msg += " (did you mean '" + s + "'?)";
    return msg;
}

}  // namespace

// ============================================================================
// Key/value interface
// ============================================================================

bool SpinConfig::is_typed_key(const string& key) {
    return typed_key_index().count(key) > 0;
}

bool SpinConfig::is_hamiltonian_key(const string& key) {
    if (is_typed_key(key)) return false;
    if (hamiltonian_key_names().count(key)) return true;
    for (const std::regex& re : hamiltonian_key_patterns())
        if (std::regex_match(key, re)) return true;
    return false;
}

string SpinConfig::suggest_key(const string& key) {
    string best;
    size_t best_d = std::numeric_limits<size_t>::max();
    auto consider = [&](const string& cand) {
        const size_t d = edit_distance(key, cand);
        if (d < best_d) {
            best_d = d;
            best = cand;
        }
    };
    for (const auto& [name, idx] : typed_key_index()) consider(name);
    for (const string& name : hamiltonian_key_names()) consider(name);
    // Close enough to be a typo: at most 2 edits, or a third of the key.
    const size_t limit = std::max<size_t>(2, key.size() / 3);
    return (best_d <= limit) ? best : string();
}

vector<string> SpinConfig::typed_keys() {
    vector<string> out;
    for (const KeySpec& k : typed_key_table())
        if (k.get) out.push_back(k.name);
    return out;
}

bool SpinConfig::set(const string& key_in, const string& value) {
    const string key = trim(key_in);
    const auto& index = typed_key_index();
    const auto it = index.find(key);
    if (it != index.end()) {
        const KeySpec& spec = typed_key_table()[it->second];
        spec.set(*this, key, value);
        explicit_keys.insert(spec.name);
        if (key != spec.name) explicit_keys.insert(key);
        return true;
    }
    if (is_hamiltonian_key(key)) {
        hamiltonian_params[key] = parse_real(key, value);
        return true;
    }
    return false;
}

void SpinConfig::set_value(const string& key, double value) {
    if (!set(key, fmt_real(value))) throw invalid_argument(unknown_key_message(key));
}

string SpinConfig::get(const string& key) const {
    const auto& index = typed_key_index();
    const auto it = index.find(key);
    if (it != index.end()) {
        const KeySpec& spec = typed_key_table()[it->second];
        if (!spec.get) throw invalid_argument("'" + key + "' is an input-only alias");
        return spec.get(*this);
    }
    const auto h = hamiltonian_params.find(key);
    if (h != hamiltonian_params.end()) return fmt_real(h->second);
    throw invalid_argument(unknown_key_message(key));
}

SpinConfig SpinConfig::from_file(const string& filename, bool verbose) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open config file: " + filename);
    }
    stringstream ss;
    ss << file.rdbuf();
    return from_string(ss.str(), filename, verbose);
}

SpinConfig SpinConfig::from_string(const string& text, const string& source, bool verbose) {
    SpinConfig config;
    istringstream in(text);
    string line;
    int line_num = 0;
    struct Unknown {
        string key, value;
        int line;
    };
    vector<Unknown> unknown;
    map<string, int> first_line;   // canonical key -> line where it was set

    while (getline(in, line)) {
        line_num++;
        const string raw = line;

        // Skip comments and empty lines
        size_t comment_pos = line.find('#');
        if (comment_pos != string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty() || line[0] == '/' || line[0] == ';') continue;

        // Find the delimiter (= or :)
        size_t eq_pos = line.find('=');
        size_t colon_pos = line.find(':');
        size_t delim_pos = (eq_pos != string::npos) ? eq_pos : colon_pos;
        if (delim_pos == string::npos) {
            throw runtime_error(source + ":" + to_string(line_num) + ": expected 'key = value', got '" +
                                trim(raw) + "'");
        }

        string key = trim(line.substr(0, delim_pos));
        string value = trim(line.substr(delim_pos + 1));

        try {
            const auto idx = typed_key_index().find(key);
            const string canonical = (idx != typed_key_index().end()) ? typed_key_table()[idx->second].name : key;
            if (config.set(key, value)) {
                if (idx != typed_key_index().end()) {
                    const KeySpec& spec = typed_key_table()[idx->second];
                    const bool deprecated = std::find(spec.deprecated_aliases.begin(), spec.deprecated_aliases.end(),
                                                      key) != spec.deprecated_aliases.end();
                    if (deprecated && verbose)
                        cerr << "Warning: " << source << ":" << line_num << ": '" << key << "' is deprecated; "
                             << spec.deprecated << endl;
                    if (!spec.no_effect.empty() && verbose)
                        cerr << "Warning: " << source << ":" << line_num << ": '" << key << "' has no effect ("
                             << spec.no_effect << ")" << endl;
                }
                const auto [pos, fresh] = first_line.emplace(canonical, line_num);
                if (!fresh && verbose)
                    cerr << "Warning: " << source << ":" << line_num << ": '" << canonical
                         << "' was already set on line " << pos->second << "; the last value is used" << endl;
                if (!fresh) pos->second = line_num;
            } else {
                unknown.push_back({key, value, line_num});
            }
        } catch (const exception& e) {
            throw runtime_error(source + ":" + to_string(line_num) + ": " + e.what());
        }
    }

    if (!unknown.empty()) {
        if (!config.allow_unknown_keys) {
            string msg;
            for (const Unknown& u : unknown)
                msg += "\n  " + source + ":" + to_string(u.line) + ": " + unknown_key_message(u.key);
            throw runtime_error("configuration " + source + " has " + to_string(unknown.size()) +
                                " unknown key(s):" + msg +
                                "\n(set allow_unknown_keys = true to keep them as Hamiltonian parameters)");
        }
        for (const Unknown& u : unknown) {
            try {
                config.hamiltonian_params[u.key] = parse_real(u.key, u.value);
            } catch (const exception& e) {
                throw runtime_error(source + ":" + to_string(u.line) + ": " + e.what());
            }
            if (verbose)
                cerr << "Warning: " << source << ":" << u.line << ": " << unknown_key_message(u.key)
                     << " kept as a Hamiltonian parameter (allow_unknown_keys = true)" << endl;
        }
    }
    return config;
}

void SpinConfig::write(std::ostream& out) const {
    out << "# Spin solver configuration (every key at full precision; reload with from_file)\n";
    for (const KeySpec& k : typed_key_table()) {
        if (!k.get) continue;
        out << k.name << " = " << k.get(*this) << "\n";
    }
    if (!hamiltonian_params.empty()) out << "# Hamiltonian / model parameters\n";
    for (const auto& [key, value] : hamiltonian_params) out << key << " = " << fmt_real(value) << "\n";
}

void SpinConfig::to_file(const string& filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot write config file: " + filename);
    }
    write(file);
    file.close();
    if (!file) throw runtime_error("Writing config file failed: " + filename);
}

// ============================================================================
// Sweeps
// ============================================================================

vector<double> sweep_grid(double start, double end, double step, const string& name) {
    if (!std::isfinite(start) || !std::isfinite(end) || !std::isfinite(step))
        throw invalid_argument(name + ": non-finite sweep start, end or step");
    if (step == 0.0) throw invalid_argument(name + ": sweep step must be non-zero");
    const double span = end - start;
    if (span != 0.0 && (span > 0.0) != (step > 0.0))
        throw invalid_argument(name + ": sweep step " + fmt_real(step) + " points away from the end value " +
                               fmt_real(end));
    const double r = span / step;
    const long long k = std::llround(r);
    if (std::abs(r - static_cast<double>(k)) > 1e-6)
        throw invalid_argument(name + ": the range " + fmt_real(start) + " -> " + fmt_real(end) +
                               " is not a whole number of steps of " + fmt_real(step) + " (" + fmt_real(r) +
                               " steps)");
    if (k > 10000000) throw invalid_argument(name + ": more than 1e7 sweep points");
    vector<double> v(static_cast<size_t>(k) + 1);
    for (size_t i = 0; i < v.size(); ++i) v[i] = start + static_cast<double>(i) * step;
    return v;
}

vector<SpinConfig::SweepAxis> SpinConfig::sweep_axes() const {
    const size_t n = sweep_parameters.size();
    if (n == 0) throw invalid_argument("parameter sweep: no sweep_parameters (or sweep_parameter) given");
    if (sweep_starts.size() != n || sweep_ends.size() != n || sweep_steps.size() != n)
        throw invalid_argument("parameter sweep: sweep_parameters has " + to_string(n) + " entries but sweep_starts/" +
                               "sweep_ends/sweep_steps have " + to_string(sweep_starts.size()) + "/" +
                               to_string(sweep_ends.size()) + "/" + to_string(sweep_steps.size()));
    vector<SweepAxis> axes;
    for (size_t p = 0; p < n; ++p) {
        const string& key = sweep_parameters[p];
        if (!is_known_key(key)) throw invalid_argument("parameter sweep: " + unknown_key_message(key));
        axes.push_back({key, sweep_grid(sweep_starts[p], sweep_ends[p], sweep_steps[p], "sweep of " + key)});
    }
    return axes;
}

// ============================================================================
// Validation
// ============================================================================

bool simulation_supported(SystemType system, SimulationType simulation) {
    if (system == SystemType::CUSTOM) return false;
    switch (simulation) {
        case SimulationType::SIMULATED_ANNEALING:
        case SimulationType::MOLECULAR_DYNAMICS:
        case SimulationType::PUMP_PROBE:
        case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
        case SimulationType::PARAMETER_SWEEP:
            return true;
        case SimulationType::PARALLEL_TEMPERING:
            return system != SystemType::NCTO;
        case SimulationType::POPULATION_ANNEALING:   // Lattice systems
            return system != SystemType::NCTO && system != SystemType::TMFEO3;
        case SimulationType::KINETIC_BARRIER_ANALYSIS:
        case SimulationType::CUSTOM:
            return false;
    }
    return false;
}

vector<string> SpinConfig::validation_errors() const {
    vector<string> err;
    auto fail = [&](const string& m) { err.push_back(m); };

    if (num_trials < 1) fail("num_trials must be >= 1");
    for (int d = 0; d < 3; ++d)
        if (lattice_size[d] == 0) fail("lattice_size dimensions must be > 0");
    if (lattice_size[0] > 0 && lattice_size[1] > 0 && lattice_size[2] > 0 &&
        double(lattice_size[0]) * double(lattice_size[1]) * double(lattice_size[2]) > 1e9)
        fail("lattice_size: more than 1e9 unit cells");
    if (!(spin_length > 0.0f) || !(spin_length_su3 > 0.0f)) fail("spin_length and spin_length_su3 must be > 0");
    if (field_direction.size() != 3) fail("field_direction needs 3 components");
    if (!su3_mc_manifold.empty() && su3_mc_manifold != "auto" && su3_mc_manifold != "cp2" &&
        su3_mc_manifold != "sphere")
        fail("su3_mc_manifold must be cp2 or sphere (got '" + su3_mc_manifold + "')");
    if (g_factor.size() != 3) fail("g_factor needs 3 components");

    // System / mode combination (the driver dispatch table).
    if (system == SystemType::CUSTOM) fail("system = custom is not implemented");
    const bool sweep = (simulation == SimulationType::PARAMETER_SWEEP);
    const SimulationType run = sweep ? sweep_base_simulation : simulation;
    if (sweep && run == SimulationType::PARAMETER_SWEEP)
        fail("sweep_base_simulation cannot be parameter_sweep");
    else if (!simulation_supported(system, run))
        fail("simulation '" + simulation_name(run) + "' is not supported for system '" +
             system_type_to_string(system) + "'" +
             (system == SystemType::NCTO ? " (NCTO supports simulated_annealing, molecular_dynamics, pump_probe, 2dcs)"
                                         : ""));
    if (sweep && run != SimulationType::PARAMETER_SWEEP) {
        // Every point must be a valid run of the base simulation (a sweep that
        // reaches T_end = 0 or md_timestep = 0 is rejected before anything runs).
        try {
            const vector<SweepAxis> axes = sweep_axes();
            vector<size_t> k(axes.size(), 0);
            size_t n_bad = 0;
            for (bool more = true; more && n_bad < 5;) {
                SpinConfig point = *this;
                point.simulation = run;
                string where = "sweep point";
                for (size_t p = 0; p < axes.size(); ++p) {
                    point.set_value(axes[p].name, axes[p].values[k[p]]);   // throws for a non-numeric key
                    where += (p ? ", " : " ") + axes[p].name + " = " + fmt_real(axes[p].values[k[p]]);
                }
                const vector<string> e = point.validation_errors();
                if (!e.empty()) {
                    ++n_bad;
                    for (const string& m : e) fail(where + ": " + m);
                }
                // next index (odometer)
                more = false;
                for (size_t p = axes.size(); p-- > 0;) {
                    if (++k[p] < axes[p].values.size()) {
                        more = true;
                        break;
                    }
                    k[p] = 0;
                }
            }
        } catch (const exception& e) {
            fail(e.what());
        }
    }

    auto is_dynamics = [](SimulationType s) {
        return s == SimulationType::MOLECULAR_DYNAMICS || s == SimulationType::PUMP_PROBE ||
               s == SimulationType::TWOD_COHERENT_SPECTROSCOPY;
    };

    // Annealing: SA itself (annealing_steps = 0 only evaluates the start), and
    // the ground-state preparation of the dynamics modes, which runs the
    // cooling schedule whenever no configuration is loaded (the TmFeO3 2DCS
    // runner anneals a loaded one too). A schedule that never reaches T_end
    // would run forever.
    const bool mixed_2dcs = (system == SystemType::TMFEO3 && run == SimulationType::TWOD_COHERENT_SPECTROSCOPY);
    const bool anneals = (run == SimulationType::SIMULATED_ANNEALING && annealing_steps > 0) ||
                         (is_dynamics(run) && (initial_spin_config.empty() || mixed_2dcs));
    if (anneals) {
        if (!(T_end > 0.0)) fail("T_end must be > 0 for annealing (use T_zero = true for a T = 0 quench)");
        if (T_start < T_end) fail("T_start must be >= T_end for annealing");
        if (!(cooling_rate > 0.0 && cooling_rate < 1.0)) fail("cooling_rate must lie in (0, 1)");
    }
    if (run == SimulationType::POPULATION_ANNEALING) {
        const bool adaptive = (pa_schedule == "adaptive");
        if (!adaptive && pa_schedule != "linear_beta" && pa_schedule != "linear" && pa_schedule != "geometric" &&
            pa_schedule != "geometric_T")
            fail("pa_schedule must be linear_beta, geometric or adaptive");
        if (!(T_end > 0.0)) fail("T_end must be > 0 for population annealing");
        if (!adaptive && T_start < T_end) fail("T_start must be >= T_end for population annealing");
        if (pa_population < 1 || pa_sweeps < 1 || pa_temperatures < 1)
            fail("pa_population, pa_sweeps and pa_temperatures must be >= 1");
        if (adaptive && !(pa_target_ess > 0.0 && pa_target_ess < 1.0)) fail("pa_target_ess must lie in (0, 1)");
    }
    if (run == SimulationType::PARALLEL_TEMPERING) {
        if (!(T_end > 0.0)) fail("T_end (the coldest replica) must be > 0 for parallel tempering");
        if (T_start < T_end) fail("T_start (the hottest replica) must be >= T_end");
        if (probe_rate == 0) fail("probe_rate must be >= 1");
        if (pt_exchange_frequency == 0) fail("pt_exchange_frequency must be >= 1");
    }

    if (is_dynamics(run)) {
        if (!(md_timestep > 0.0)) fail("md_timestep must be > 0");
        if (md_save_interval == 0) fail("md_save_interval must be >= 1");
        if (md_time_end < md_time_start) fail("md_time_end must be >= md_time_start");
    }
    if (run == SimulationType::TWOD_COHERENT_SPECTROSCOPY) {
        // NCTO keeps its historic default scan (0 -> 100 step 5) for unset keys.
        const bool ncto = (system == SystemType::NCTO);
        const double t0 = (ncto && !was_set("tau_start")) ? 0.0 : tau_start;
        const double t1 = (ncto && !was_set("tau_end")) ? 100.0 : tau_end;
        const double ts = (ncto && !was_set("tau_step")) ? 5.0 : tau_step;
        try {
            classical_spin::dynamics::delay_grid(t0, t1, ts, "tau scan (tau_start, tau_end, tau_step)");
        } catch (const std::invalid_argument& e) {
            fail(e.what());
        }
    }

    // Spin dynamics of the Lattice family (every system except the mixed
    // TmFeO3 and the NCTO spin-phonon models, which have their own integrator
    // tables): parse the integrator name once, so a typo fails here instead
    // of silently running another method, and reject parameters the drivers
    // would otherwise reject mid-run.
    const bool lattice_family = (system != SystemType::TMFEO3 && system != SystemType::NCTO);
    if (lattice_family && is_dynamics(run)) {
        try {
            classical_spin::dynamics::parse_ode_method(md_integrator);
        } catch (const std::invalid_argument& e) {
            fail(string("md_integrator: ") + e.what());
        }
        try {
            classical_spin::dynamics::parse_damping_form(damping_form);
        } catch (const std::invalid_argument& e) {
            fail(e.what());
        }
        const double alpha = get_param("alpha_gilbert", 0.0);
        const double T_bath = get_param("langevin_temperature", 0.0);
        if (alpha < 0.0 || T_bath < 0.0) fail("alpha_gilbert and langevin_temperature must be >= 0");
        if (T_bath > 0.0 && alpha == 0.0)
            fail("langevin_temperature > 0 needs alpha_gilbert > 0 (the bath acts through the damping)");
        bool stochastic_ok = false;
        try {
            stochastic_ok = classical_spin::dynamics::is_geometric_method(md_integrator) &&
                            classical_spin::dynamics::parse_geometric_method(md_integrator) !=
                                classical_spin::dynamics::Method::ColorSplit &&
                            classical_spin::dynamics::parse_geometric_method(md_integrator) !=
                                classical_spin::dynamics::Method::ColorSplit4;
        } catch (const std::invalid_argument&) {
            stochastic_ok = false;
        }
        if (T_bath > 0.0 && !stochastic_ok)
            fail("langevin_temperature > 0 needs md_integrator = spherical_midpoint or depondt");
        if (run == SimulationType::MOLECULAR_DYNAMICS && dssf_samples > 0) {
            if (dssf_q_points.empty()) fail("dssf_samples > 0 needs dssf_q_points");
            if (!(dssf_alpha > 0.0) || dssf_t_equilibrate < 0.0 || dssf_t_decorrelate < 0.0)
                fail("dssf_alpha must be > 0 and the dssf times >= 0");
        }
        if (run == SimulationType::TWOD_COHERENT_SPECTROSCOPY && T_bath > 0.0)
            fail("2DCS needs deterministic dynamics (langevin_temperature = 0)");
    }
    return err;
}

bool SpinConfig::validate() const {
    const vector<string> errors = validation_errors();
    for (const string& e : errors) cerr << "Error: " << e << "\n";
    return errors.empty();
}

void SpinConfig::print() const {
    cout << "==================== Spin Solver Configuration ====================\n";
    cout << "System: ";
    switch (system) {
        case SystemType::HONEYCOMB_BCAO: cout << "BCAO Honeycomb"; break;
        case SystemType::HONEYCOMB_KITAEV: cout << "Kitaev Honeycomb"; break;
        case SystemType::PYROCHLORE: cout << "Pyrochlore"; break;
        case SystemType::PYROCHLORE_NON_KRAMER: cout << "Pyrochlore (Non-Kramers)"; break;
        case SystemType::TMFEO3: cout << "TmFeO3"; break;
        case SystemType::TMFEO3_FE: cout << "TmFeO3 (Fe only)"; break;
        case SystemType::TMFEO3_TM: cout << "TmFeO3 (Tm only)"; break;
        case SystemType::NCTO: cout << "NCTO (Na2Co2TeO6) Spin-Phonon"; break;
        case SystemType::TRIANGULAR_ANISOTROPIC: cout << "Triangular (anisotropic)"; break;
        case SystemType::CUSTOM: cout << "Custom"; break;
    }
    cout << "\n";
    cout << "Lattice size: " << lattice_size[0] << " x " << lattice_size[1] << " x " << lattice_size[2] << "\n";
    cout << "Simulation: ";
    switch (simulation) {
        case SimulationType::SIMULATED_ANNEALING: cout << "Simulated Annealing"; break;
        case SimulationType::POPULATION_ANNEALING: cout << "Population Annealing"; break;
        case SimulationType::PARALLEL_TEMPERING: cout << "Parallel Tempering"; break;
        case SimulationType::MOLECULAR_DYNAMICS: cout << "Molecular Dynamics"; break;
        case SimulationType::PUMP_PROBE: cout << "Pump-Probe"; break;
        case SimulationType::TWOD_COHERENT_SPECTROSCOPY: cout << "2DCS Spectroscopy"; break;
        case SimulationType::PARAMETER_SWEEP: cout << "Parameter Sweep"; break;
        case SimulationType::KINETIC_BARRIER_ANALYSIS: cout << "Kinetic Barrier (GNEB)"; break;
        case SimulationType::CUSTOM: cout << "Custom"; break;
    }
    cout << "\n";
    cout << "Trials: " << num_trials << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Temperature: " << T_start << " -> " << T_end << "\n";
    cout << "Field: " << field_strength << " along [" << fmt_real_list(field_direction) << "]\n";

    // Parallel tempering specific output
    if (simulation == SimulationType::PARALLEL_TEMPERING) {
        cout << "\nParallel Tempering Settings:\n";
        cout << "  Exchange frequency: " << pt_exchange_frequency << "\n";
        cout << "  Equilibration / measurement steps: "
             << (pt_equilibration_steps ? pt_equilibration_steps : annealing_steps) << " / "
             << (pt_measurement_steps ? pt_measurement_steps : annealing_steps) << "\n";
        cout << "  Overrelaxation rate: " << overrelaxation_rate << "\n";
        cout << "  Probe rate: " << probe_rate << "\n";
        cout << "  Optimize temperatures: " << (pt_optimize_temperatures ? "yes" : "no (geometric)") << "\n";
        if (pt_optimize_temperatures) {
            cout << "  Temperature optimizer: " << pt_temperature_optimizer << "\n";
            cout << "  Optimization warmup: " << pt_optimization_warmup << " steps\n";
            cout << "  Optimization rounds: " << pt_optimization_sweeps << " x 2^r steps, at most "
                 << pt_optimization_iterations << " rounds, tolerance " << pt_optimization_tolerance << "\n";
        }
    }

    if (!hamiltonian_params.empty()) {
        cout << "\nHamiltonian Parameters:\n";
        for (const auto& [key, value] : hamiltonian_params) {
            cout << "  " << key << " = " << value << "\n";
        }
    }
    cout << "=========================================================================\n";
}
