// test_config.cpp — SpinConfig parsing, validation, serialisation and the
// parameter-sweep grid (no MPI, no lattice).
//
//   * strict numbers: integers accept "1e5" / "100000.0" only when integral
//     and in range; fractional / negative / overflowing values are errors;
//   * unknown keys are errors with an edit-distance suggestion, registered
//     Hamiltonian keys (literal names and run-time key families) are not;
//   * set() is the one entry point (file parser and sweeps), typed fields are
//     reachable through it, sweep values are applied losslessly;
//   * validate() rejects values that hang or crash a run;
//   * from_file(to_file(c)) reproduces c for every typed key;
//   * every example configuration in the repository parses and validates;
//   * every literal key read with get_param/has_param/was_set anywhere in
//     src/ or include/ is registered (so the registry cannot fall behind).
#include "classical_spin/core/spin_config.h"
#include "physics_test_util.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <unistd.h>

using phys_test::check;
namespace fs = std::filesystem;

namespace {

template <class F>
bool throws(F&& f, const std::string& needle = "") {
    try {
        f();
    } catch (const std::exception& e) {
        if (needle.empty() || std::string(e.what()).find(needle) != std::string::npos) return true;
        std::printf("      (threw, but without '%s': %s)\n", needle.c_str(), e.what());
        return false;
    }
    return false;
}

SpinConfig parse(const std::string& text) { return SpinConfig::from_string(text, "<test>", false); }

bool has_error(const SpinConfig& c, const std::string& needle) {
    for (const std::string& e : c.validation_errors())
        if (e.find(needle) != std::string::npos) return true;
    return false;
}

void test_numbers() {
    SpinConfig c = parse("annealing_steps = 1e5\nmd_save_interval = 100000.0\nnum_trials = 3\n");
    check(c.annealing_steps == 100000, "annealing_steps = 1e5 is 100000 (stoull gave 1)");
    check(c.md_save_interval == 100000, "md_save_interval = 100000.0 is accepted as an integer");
    check(throws([] { parse("annealing_steps = 2.7"); }, "fractional"), "fractional integer rejected");
    check(throws([] { parse("annealing_steps = -1"); }, "non-negative"),
          "negative unsigned rejected (stoull gave 2^64 - 1)");
    check(throws([] { parse("annealing_steps = 1e30"); }, "out of range"), "out-of-range integer rejected");
    check(throws([] { parse("num_trials = 1e10"); }, "out of range"), "int overflow rejected");
    check(throws([] { parse("T_start = 1.5abc"); }, "not a number"), "trailing garbage rejected (stod took 1.5)");
    check(throws([] { parse("T_start = nan"); }), "non-finite value rejected");
    check(throws([] { parse("K = 0.1x"); }, "K"), "malformed Hamiltonian value rejected, naming the key");
    c = parse("seed = 18446744073709551615\n");
    check(c.seed == 18446744073709551615ull, "seed parsed exactly to 64 bits");
    check(throws([] { parse("lattice_size = 4,4"); }, "3 values"), "lattice_size needs three values");
    check(parse("lattice_size = [6, 5, 4]").lattice_size == std::array<size_t, 3>{6, 5, 4}, "lattice_size list");
    check(parse("use_gpu = YES\nT_zero = on\ngaussian_move = 0").use_gpu, "booleans: yes/on/0");
    check(throws([] { parse("use_gpu = ture"); }, "true/false"), "misspelt boolean rejected (was silently false)");
    check(throws([] { parse("simulation_mode = annealling"); }), "unknown simulation type rejected");
    check(throws([] { parse("no delimiter here"); }, ":1:"), "a line without '=' is an error naming the line");
}

void test_unknown_keys_and_registry() {
    check(throws([] { parse("T_start = 1\nanealing_steps = 5000\n"); }, "did you mean 'annealing_steps'"),
          "typo in a typed key: error with suggestion");
    check(throws([] { parse("T_start = 1\nanealing_steps = 5000\n"); }, "<test>:2"), "the error names file:line");
    check(throws([] { parse("Gama = 0.25"); }, "did you mean 'Gamma'"), "typo in a Hamiltonian key: suggestion");
    check(throws([] { parse("field_scan_enabled = true"); }), "unregistered key rejected");
    const char* known[] = {"K", "Gamma", "J1xy", "alpha_gilbert", "langevin_temperature", "mode2_omega",
                           "mode0_lamJ7sq", "anh3_lamp", "Kminus_2y", "Kminus3_5x", "kappaE_7z", "kappaB_z1y",
                           "kappaB2_3x", "W3_xy", "W1_4_zz", "h_tm_8", "gamma_su3_lambda1", "legacy_kitaev_frame"};
    bool all = true;
    for (const char* k : known) all = all && SpinConfig::is_hamiltonian_key(k);
    check(all, "registered Hamiltonian keys (literal names and key families)");
    check(!SpinConfig::is_hamiltonian_key("mode2_omegaa") && !SpinConfig::is_hamiltonian_key("h_tm_9") &&
              !SpinConfig::is_hamiltonian_key("Kminus5_2x") && !SpinConfig::is_hamiltonian_key("T_start"),
          "near misses of key families and typed keys are not Hamiltonian keys");
    SpinConfig c = parse("my_tool_key = 2.5\nallow_unknown_keys = true\n");
    check(c.get_param("my_tool_key") == 2.5, "allow_unknown_keys keeps an unknown key (order-independent)");
    check(throws([] { parse("allow_unknown_keys = true\nmy_tool_key = abc\n"); }), "...but it must be numeric");
}

void test_set_and_aliases() {
    SpinConfig c = parse("dt = 0.02\ntemperature_start = 7\nh = 0.3\ntbc = true\nintegrator = rk4\n"
                         "gpu = false\nsystem_type = pyrochlore\nsimulation = molecular_dynamics\n");
    check(c.md_timestep == 0.02 && c.T_start == 7 && c.field_strength == 0.3 && c.use_twist_boundary &&
              c.md_integrator == "rk4" && c.system == SystemType::PYROCHLORE &&
              c.simulation == SimulationType::MOLECULAR_DYNAMICS,
          "aliases (dt, temperature_start, h, tbc, integrator, system_type, simulation)");
    check(c.was_set("md_timestep") && c.was_set("T_start") && !c.was_set("T_end"),
          "was_set uses canonical names for aliases");
    c = parse("pt_sweeps_per_exchange = 7");
    check(c.pt_exchange_frequency == 7, "deprecated pt_sweeps_per_exchange is an alias of pt_exchange_frequency");

    SpinConfig s;
    check(!s.set("pump_amplitud", "1"), "set() returns false for an unknown key");
    check(s.hamiltonian_params.empty() && !s.was_set("pump_amplitud"), "...and changes nothing");
    s.set_value("pump_amplitude", 0.5);
    check(s.pump_amplitude == 0.5, "set_value reaches a typed field (sweeps of pump_amplitude were no-ops)");
    s.set_value("probe_time", 30.0);
    s.set_value("T_end", 0.1 + 0.2);
    check(s.probe_time == 30.0 && s.T_end == 0.1 + 0.2, "set_value is lossless (%.17g)");
    s.set_value("annealing_steps", 2000.0);
    check(s.annealing_steps == 2000, "set_value on an integer key");
    check(throws([&] { s.set_value("annealing_steps", 20.5); }, "fractional"), "fractional value for an integer key");
    check(throws([&] { s.set_value("pump_amplitud", 1.0); }, "did you mean 'pump_amplitude'"),
          "set_value with a typo throws with a suggestion");
    s.set_value("K", -1.5);
    check(s.get_param("K") == -1.5, "set_value on a Hamiltonian key");
}

void test_sweep_grid() {
    std::vector<double> g = sweep_grid(0.0, 2.0, 0.1);
    check(g.size() == 21 && std::abs(g.back() - 2.0) < 1e-12,
          "0 -> 2 step 0.1: 21 points including the endpoint (repeated addition gave 20)");
    check(g[7] == 0.0 + 7 * 0.1, "grid values built by index");
    check(sweep_grid(0.0, 0.3, 0.1).size() == 4, "0 -> 0.3 step 0.1: 4 points");
    g = sweep_grid(2.0, 0.0, -0.5);
    check(g.size() == 5 && g.back() == 0.0, "negative step");
    check(sweep_grid(1.0, 1.0, 0.5).size() == 1, "start == end: one point");
    check(throws([] { sweep_grid(0.0, 1.0, 0.0); }, "non-zero"), "step 0 rejected (looped forever)");
    check(throws([] { sweep_grid(0.0, 1.0, -0.1); }, "points away"), "wrong-sign step rejected (gave 0 points)");
    check(throws([] { sweep_grid(1.5, 3.0, 0.2); }, "whole number"), "range not a whole number of steps rejected");

    SpinConfig c = parse("simulation_mode = parameter_sweep\nsweep_parameter = pump_amplitude\nsweep_start = 0.5\n"
                         "sweep_end = 2.5\nsweep_step = 0.5\n");
    auto axes = c.sweep_axes();
    check(axes.size() == 1 && axes[0].name == "pump_amplitude" && axes[0].values.size() == 5,
          "legacy 1-D sweep keys fill the N-D lists");
    c = parse("simulation_mode = parameter_sweep\nsweep_parameters = K, Gamma\nsweep_starts = 0\n"
              "sweep_ends = 1, 1\nsweep_steps = 0.5, 0.5\n");
    check(throws([&] { c.sweep_axes(); }, "sweep_starts"), "inconsistent N-D sweep lists rejected");
    c = parse("simulation_mode = parameter_sweep\nsweep_parameters = Gama\nsweep_starts = 0\n"
              "sweep_ends = 1\nsweep_steps = 0.5\n");
    check(throws([&] { c.sweep_axes(); }, "did you mean 'Gamma'"), "unknown sweep parameter rejected");
}

void test_validation() {
    SpinConfig d;
    check(d.validation_errors().empty(), "the default configuration is valid");
    auto with = [](const std::string& text) { return parse(text); };
    check(has_error(with("cooling_rate = 1.0"), "cooling_rate"), "SA: cooling_rate >= 1 rejected (infinite loop)");
    check(has_error(with("cooling_rate = 0"), "cooling_rate"), "SA: cooling_rate <= 0 rejected");
    check(has_error(with("T_end = 0"), "T_end"), "SA: T_end = 0 rejected (T underflow loop)");
    check(has_error(with("T_start = 0.01\nT_end = 0.1"), "T_start"), "SA: T_start < T_end rejected");
    check(with("annealing_steps = 0\nT_end = 0").validation_errors().empty(),
          "SA with annealing_steps = 0 (energy evaluation) ignores the schedule");
    check(has_error(with("simulation_mode = PT\nprobe_rate = 0"), "probe_rate"), "PT: probe_rate = 0 rejected");
    check(has_error(with("simulation_mode = PT\nT_end = 0"), "T_end"), "PT: T_end = 0 rejected (NaN ladder)");
    check(has_error(with("simulation_mode = MD\nmd_timestep = 0"), "md_timestep"), "MD: md_timestep = 0 rejected");
    check(has_error(with("system = tmfeo3\nsimulation_mode = MD\nmd_timestep = -0.1"), "md_timestep"),
          "MD (mixed): negative md_timestep rejected");
    check(has_error(with("simulation_mode = MD\nmd_save_interval = 0"), "md_save_interval"),
          "MD: md_save_interval = 0 rejected");
    check(has_error(with("simulation_mode = MD\nT_end = 0"), "T_end"),
          "MD: the annealing preparation is validated too");
    check(with("simulation_mode = MD\nT_end = 0\ninitial_spin_config = seed.txt").validation_errors().empty(),
          "MD from a loaded configuration does not anneal");
    check(has_error(with("lattice_size = 0,4,4"), "lattice_size"), "lattice size 0 rejected");
    check(has_error(with("simulation_mode = 2dcs\ntau_step = 0"), "non-zero"), "2DCS: tau_step = 0 rejected");
    check(has_error(with("system = tmfeo3\nsimulation_mode = 2dcs\ntau_start = 0\ntau_end = 10\ntau_step = -1"),
                    "points away"),
          "2DCS (mixed): wrong-sign tau_step rejected");
    check(has_error(with("system = ncto\nsimulation_mode = 2dcs\ntau_step = -5"), "points away"),
          "2DCS (NCTO): wrong-sign tau_step rejected");
    check(has_error(with("system = ncto\nsimulation_mode = PT"), "not supported"), "NCTO + PT rejected up front");
    check(has_error(with("simulation_mode = gneb"), "not supported"), "kinetic_barrier mode rejected (never ran)");
    check(has_error(with("system = custom"), "custom"), "system = custom rejected");
    check(has_error(with("simulation_mode = parameter_sweep"), "sweep_parameters"), "sweep without parameters");
    check(has_error(with("simulation_mode = parameter_sweep\nsweep_parameter = system\nsweep_start = 0\n"
                         "sweep_end = 1\nsweep_step = 1"),
                    "system"),
          "sweep of a non-numeric key rejected");
    check(has_error(with("simulation_mode = parameter_sweep\nsweep_parameter = K\nsweep_start = 0\n"
                         "sweep_end = 1\nsweep_step = 0"),
                    "non-zero"),
          "sweep with step 0 rejected");
    check(has_error(with("simulation_mode = parameter_sweep\nsweep_parameter = T_end\nsweep_start = 0.1\n"
                         "sweep_end = 0\nsweep_step = -0.05"),
                    "sweep point T_end = 0: T_end must be > 0"),
          "every sweep point is validated (a point with T_end = 0 is rejected up front)");
    check(with("simulation_mode = parameter_sweep\nsweep_parameter = T_end\nsweep_start = 0.1\n"
               "sweep_end = 0.05\nsweep_step = -0.05").validation_errors().empty(),
          "...a sweep whose points are all valid passes");
    check(has_error(with("field_direction = 0,1"), "field_direction"), "2-component field_direction rejected");
    check(has_error(with("num_trials = 0"), "num_trials"), "num_trials = 0 rejected");
    check(has_error(with("simulation_mode = MD\nmd_integrator = rk45"), "md_integrator"),
          "unknown integrator rejected");
}

/// A different, valid text for a key (for the round-trip test).
std::string perturb(const std::string& key, const std::string& text) {
    if (key == "system") return "pyrochlore_non_kramer";
    if (key == "simulation_mode") return "molecular_dynamics";
    if (key == "sweep_base_simulation") return "2dcs";
    if (key == "ranks_to_write") return "ALL";
    if (text == "true") return "false";
    if (text == "false") return "true";
    if (text.empty()) {
        if (key.find("sweep_parameters") != std::string::npos) return "K,Gamma";
        if (key.find("direction") != std::string::npos || key.find("q_points") != std::string::npos ||
            key.find("sweep_") == 0)
            return "0.25,-0.5,0.125";
        return "some_file.txt";
    }
    static const std::regex number(R"([-+0-9.eE]+)");
    std::stringstream in(text);
    std::string item, out;
    bool numeric = true;
    std::vector<std::string> items;
    while (std::getline(in, item, ',')) {
        items.push_back(item);
        numeric = numeric && std::regex_match(item, number);
    }
    if (!numeric) return text + "_x";
    for (size_t i = 0; i < items.size(); ++i) {
        const double x = std::stod(items[i]);
        const bool integral = items[i].find_first_of(".eE") == std::string::npos;
        char buf[40];
        if (integral) std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(x) + 3);
        else std::snprintf(buf, sizeof buf, "%.17g", x * 1.37 + 0.1 / 3.0);
        out += (i ? "," : "") + std::string(buf);
    }
    return out;
}

void test_round_trip() {
    const SpinConfig d;
    SpinConfig c;
    bool every_key_changes = true;
    for (const std::string& key : SpinConfig::typed_keys()) {
        const std::string before = d.get(key);
        const std::string v = perturb(key, before);
        try {
            if (!c.set(key, v)) throw std::runtime_error("rejected");
        } catch (const std::exception& e) {
            std::printf("      %s = '%s': %s\n", key.c_str(), v.c_str(), e.what());
            every_key_changes = false;
            continue;
        }
        if (c.get(key) == before) {
            std::printf("      %s did not change (%s)\n", key.c_str(), before.c_str());
            every_key_changes = false;
        }
    }
    check(every_key_changes, "every typed key (" + std::to_string(SpinConfig::typed_keys().size()) +
                                 ") is settable through set() and read back by get()");
    c.set_param("K", -1.0 / 3.0);
    c.set_value("Kminus2_5y", 0.1);

    const fs::path dir = fs::temp_directory_path() / ("test_config_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const std::string path = (dir / "round_trip.param").string();
    c.to_file(path);
    SpinConfig r = SpinConfig::from_file(path, false);
    bool same = true;
    for (const std::string& key : SpinConfig::typed_keys()) {
        if (r.get(key) != c.get(key)) {
            std::printf("      %s: wrote '%s', read '%s'\n", key.c_str(), c.get(key).c_str(), r.get(key).c_str());
            same = false;
        }
    }
    check(same, "from_file(to_file(c)) reproduces every typed key");
    check(r.hamiltonian_params == c.hamiltonian_params, "...and every Hamiltonian parameter bitwise");
    check(r.T_start == c.T_start && r.pump_amplitude == c.pump_amplitude, "doubles survive bitwise");

    // The default configuration round-trips too (no empty-list artefacts).
    d.to_file(path);
    r = SpinConfig::from_file(path, false);
    same = true;
    for (const std::string& key : SpinConfig::typed_keys()) same = same && r.get(key) == d.get(key);
    check(same && r.sweep_starts.empty() && r.sweep_parameters.empty(), "the default configuration round-trips");
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    check(ss.str().find("\nsimulation_mode = ") != std::string::npos,
          "to_file writes simulation_mode (it wrote 'simulation', which did not parse)");
    fs::remove_all(dir);
}

void test_example_configs() {
    const fs::path root(CLASSICAL_SPIN_SOURCE_DIR);
    size_t n = 0, bad = 0;
    for (const char* sub : {"example_configs", "examples", "tests/data"}) {
        if (!fs::exists(root / sub)) continue;
        for (const auto& e : fs::recursive_directory_iterator(root / sub)) {
            if (!e.is_regular_file() || e.path().extension() != ".param") continue;
            ++n;
            try {
                const SpinConfig c = SpinConfig::from_file(e.path().string(), false);
                const auto errors = c.validation_errors();
                for (const auto& m : errors) std::printf("      %s: %s\n", e.path().c_str(), m.c_str());
                if (!errors.empty()) ++bad;
            } catch (const std::exception& ex) {
                std::printf("      %s\n", ex.what());
                ++bad;
            }
        }
    }
    check(n > 0 && bad == 0, "all " + std::to_string(n) + " example configurations parse and validate");
}

void test_registry_covers_sources() {
    const fs::path root(CLASSICAL_SPIN_SOURCE_DIR);
    const std::regex call(R"re((?:get_param|has_param|was_set)\(\s*"([A-Za-z0-9_]+)"\s*[,)])re");
    size_t n = 0;
    std::set<std::string> missing;
    for (const char* sub : {"src", "include"}) {
        for (const auto& e : fs::recursive_directory_iterator(root / sub)) {
            const std::string ext = e.path().extension().string();
            if (!e.is_regular_file() || (ext != ".cpp" && ext != ".h" && ext != ".cu" && ext != ".cuh")) continue;
            std::ifstream in(e.path());
            std::stringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();
            for (std::sregex_iterator it(text.begin(), text.end(), call), end; it != end; ++it) {
                ++n;
                const std::string key = (*it)[1];
                if (!SpinConfig::is_known_key(key)) missing.insert(key + " (" + e.path().filename().string() + ")");
            }
        }
    }
    for (const auto& m : missing) std::printf("      unregistered: %s\n", m.c_str());
    check(n > 100 && missing.empty(),
          "every literal get_param/has_param/was_set key in src/ and include/ is registered (" +
              std::to_string(n) + " reads)");
}

}  // namespace

int main() {
    test_numbers();
    test_unknown_keys_and_registry();
    test_set_and_aliases();
    test_sweep_grid();
    test_validation();
    test_round_trip();
    test_example_configs();
    test_registry_covers_sources();
    return phys_test::finish("test_config");
}
