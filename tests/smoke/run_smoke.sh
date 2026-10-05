#!/usr/bin/env bash
# run_smoke.sh — end-to-end smoke tests of the spin_solver driver.
#
# Runs every simulation mode on tiny systems (seconds each) and checks that
# the run exits with status 0 and writes its expected outputs. The physics
# itself is validated by the tests/test_*.cpp suites; this catches driver,
# config-parsing, MPI and I/O regressions that unit tests cannot see.
#
# Usage: run_smoke.sh <path/to/spin_solver> [mpiexec]
set -u
SOLVER="$(realpath "$1")"
MPIEXEC="${2:-mpiexec}"
MPIFLAGS="--oversubscribe --allow-run-as-root"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
export OMP_NUM_THREADS=1
failures=0

run_case() {  # name nranks expected_glob config_text
    local name="$1" np="$2" expect="$3" cfg="$4"
    mkdir -p "$name" && printf '%s\n' "$cfg" > "$name/run.param"
    local log="$name/log.txt" status
    if [ "$np" -gt 1 ]; then
        (cd "$name" && timeout 300 "$MPIEXEC" $MPIFLAGS -np "$np" "$SOLVER" run.param) > "$log" 2>&1
    else
        (cd "$name" && timeout 300 "$SOLVER" run.param) > "$log" 2>&1
    fi
    status=$?
    if [ $status -ne 0 ]; then
        echo "[FAIL] $name: exit status $status"; tail -n 25 "$log"; failures=$((failures + 1)); return
    fi
    if ! compgen -G "$name/out/$expect" > /dev/null; then
        echo "[FAIL] $name: missing output $expect"; ls -R "$name/out" 2>/dev/null | head -40
        failures=$((failures + 1)); return
    fi
    echo "[ OK ] $name"
}

KITAEV='system = honeycomb_kitaev
lattice_size = 4,4,1
K = -1.0
Gamma = 0.25
Gammap = -0.02
J = 0.0
field_strength = 0.2
field_direction = 0.577,0.577,0.577
seed = 7
output_dir = out'

run_case sa_kitaev 2 "sample_1/*" "$KITAEV
simulation_mode = simulated_annealing
num_trials = 2
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 40
overrelaxation_rate = 2
gaussian_move = true
T_zero = true
n_deterministics = 200
save_observables = true"

run_case sa_heat_bath 1 "sample_0/*" "system = pyrochlore
lattice_size = 2,2,2
Jxx = 1.0
Jyy = 1.0
Jzz = 1.0
seed = 3
output_dir = out
simulation_mode = simulated_annealing
local_update = heat_bath
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 40
T_zero = true
n_deterministics = 100"

run_case pt_kitaev 4 "*" "$KITAEV
simulation_mode = parallel_tempering
T_start = 2.0
T_end = 0.1
annealing_steps = 400
overrelaxation_rate = 2
pt_exchange_frequency = 5
probe_rate = 10
pt_optimize_temperatures = false
ranks_to_write = 0,1,2,3
save_observables = true"

run_case pt_kitaev_tuned 4 "*" "$KITAEV
simulation_mode = parallel_tempering
T_start = 2.0
T_end = 0.1
annealing_steps = 300
pt_exchange_frequency = 2
probe_rate = 10
pt_optimize_temperatures = true
pt_optimization_warmup = 50
pt_optimization_sweeps = 50
pt_optimization_iterations = 3
ranks_to_write = 0
save_observables = true"

for integ in spherical_midpoint color_split dopri5 rk4; do
run_case "md_${integ}" 1 "*" "$KITAEV
simulation_mode = molecular_dynamics
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 30
md_time_start = 0.0
md_time_end = 4.0
md_timestep = 0.05
md_save_interval = 4
md_integrator = $integ
save_observables = true"
done

run_case pump_probe 1 "*" "$KITAEV
simulation_mode = pump_probe
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 30
T_zero = true
n_deterministics = 100
pump_direction = 0,0,1
pump_amplitude = 0.1
pump_width = 0.5
pump_frequency = 0.5
pump_time = 0.0
md_time_start = -2.0
md_time_end = 6.0
md_timestep = 0.05
md_integrator = dopri5"

run_case twodcs 2 "*" "$KITAEV
simulation_mode = 2dcs
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 30
T_zero = true
n_deterministics = 100
pump_direction = 0,0,1
pump_amplitude = 0.1
pump_width = 0.5
pump_frequency = 0.5
tau_start = -1.0
tau_end = 1.0
tau_step = 0.5
md_time_start = -3.0
md_time_end = 5.0
md_timestep = 0.05
md_integrator = dopri5
parallel_tau = true"

run_case sweep_field 2 "*" "$KITAEV
simulation_mode = parameter_sweep
sweep_base_simulation = simulated_annealing
sweep_parameter = field_strength
sweep_start = 0.0
sweep_end = 0.4
sweep_step = 0.2
T_start = 2.0
T_end = 0.1
cooling_rate = 0.7
annealing_steps = 20"

run_case tmfeo3_sa 1 "*" "system = tmfeo3
lattice_size = 2,2,2
seed = 5
output_dir = out
simulation_mode = simulated_annealing
T_start = 2.0
T_end = 0.05
cooling_rate = 0.7
annealing_steps = 20
T_zero = true
n_deterministics = 50"

run_case ncto_sa 1 "*" "system = ncto
lattice_size = 4,4,1
seed = 11
output_dir = out
simulation_mode = simulated_annealing
T_start = 1.0
T_end = 0.05
cooling_rate = 0.7
annealing_steps = 20"

if [ $failures -ne 0 ]; then
    echo "smoke: $failures case(s) FAILED"; exit 1
fi
echo "smoke: all cases passed"
