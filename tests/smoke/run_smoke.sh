#!/usr/bin/env bash
# run_smoke.sh — end-to-end smoke tests of the spin_solver driver.
#
# Runs every simulation mode on tiny systems (seconds each) and checks that
# the run exits with status 0 and writes its expected outputs. The physics
# itself is validated by the tests/test_*.cpp suites; this catches driver,
# config-parsing, MPI and I/O regressions that unit tests cannot see.
#
# Usage: run_smoke.sh <path/to/spin_solver> [mpiexec]
# Registered in CTest as smoke_spin_solver (label smoke).
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

run_mpi() {  # nranks config -> runs in the current directory, prints the exit status
    local np="$1" cfg="$2"
    if [ "$np" -gt 1 ]; then
        timeout 300 "$MPIEXEC" $MPIFLAGS -np "$np" "$SOLVER" "$cfg"
    else
        timeout 300 "$SOLVER" "$cfg"
    fi
}

check() {  # name description condition...
    local name="$1" what="$2"; shift 2
    if "$@"; then echo "[ OK ] $name: $what"; else echo "[FAIL] $name: $what"; failures=$((failures + 1)); fi
}

expect_failure() {  # name nranks expected_log_regex config_text
    local name="$1" np="$2" regex="$3" cfg="$4" status
    mkdir -p "$name" && printf '%s\n' "$cfg" > "$name/run.param"
    (cd "$name" && run_mpi "$np" run.param) > "$name/log.txt" 2>&1
    status=$?
    if [ $status -eq 0 ] || [ $status -eq 124 ]; then
        echo "[FAIL] $name: exit status $status (expected a clean failure)"; tail -n 15 "$name/log.txt"
        failures=$((failures + 1)); return
    fi
    if ! grep -Eq "$regex" "$name/log.txt"; then
        echo "[FAIL] $name: log lacks /$regex/"; tail -n 15 "$name/log.txt"; failures=$((failures + 1)); return
    fi
    echo "[ OK ] $name (exit $status)"
}

# energies (column 3) of a trial_summary.txt, one per line
energies() { grep -v '^#' "$1" | awk '{print $3}'; }

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

# Provenance: run_info.txt is a complete configuration; rerunning it repeats the run.
check sa_kitaev "run_info.txt records seed, MPI size, git and flags" \
    grep -q "^# git: " sa_kitaev/out/run_info.txt
check sa_kitaev "every trial (both ranks) has a final energy and the summary lists 2 trials" \
    test "$(energies sa_kitaev/out/trial_summary.txt | wc -l)" -eq 2 -a -f sa_kitaev/out/sample_1/final_energy.txt
cp sa_kitaev/out/trial_summary.txt sa_kitaev/first_summary.txt
(cd sa_kitaev && run_mpi 2 out/run_info.txt) > sa_kitaev/rerun_log.txt 2>&1
check sa_kitaev "rerunning out/run_info.txt reproduces the trial energies bitwise" \
    cmp -s <(energies sa_kitaev/first_summary.txt) <(energies sa_kitaev/out/trial_summary.txt)

# A loaded configuration is the start of EVERY trial (trials > 0 used to start
# from random spins); annealing_steps = 0 evaluates it, so all energies agree.
run_case sa_seeded_trials 2 "sample_2/final_energy.txt" "$KITAEV
simulation_mode = simulated_annealing
num_trials = 3
annealing_steps = 0
initial_spin_config = $WORK/sa_kitaev/out/sample_0/spins_final.txt"
check sa_seeded_trials "all 3 trials start from the loaded configuration" \
    test "$(energies sa_seeded_trials/out/trial_summary.txt | sort -u | wc -l)" -eq 1

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

# A one-point sweep builds exactly the system of the direct run (one factory):
# same seed -> bitwise the same annealed energy.
run_case sa_direct 1 "trial_summary.txt" "$KITAEV
simulation_mode = simulated_annealing
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 30"
run_case sa_one_point_sweep 1 "field_strength_2.000000e-01/trial_summary.txt" "$KITAEV
simulation_mode = parameter_sweep
sweep_base_simulation = simulated_annealing
sweep_parameter = field_strength
sweep_start = 0.2
sweep_end = 0.2
sweep_step = 0.1
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 30"
check sa_one_point_sweep "sweep point == direct run (Lattice)" \
    cmp -s <(energies sa_direct/out/trial_summary.txt) \
           <(energies sa_one_point_sweep/out/field_strength_2.000000e-01/trial_summary.txt)
NCTO='system = ncto
lattice_size = 2,2,1
seed = 13
output_dir = out
T_start = 1.0
T_end = 0.05
cooling_rate = 0.7
annealing_steps = 10
field_direction = 0,0,1'
run_case ncto_direct 1 "trial_summary.txt" "$NCTO
simulation_mode = simulated_annealing
field_strength = 0.1"
run_case ncto_one_point_sweep 1 "field_strength_1.000000e-01/trial_summary.txt" "$NCTO
simulation_mode = parameter_sweep
sweep_base_simulation = simulated_annealing
sweep_parameter = field_strength
sweep_start = 0.1
sweep_end = 0.1
sweep_step = 0.1"
check ncto_one_point_sweep "sweep point == direct run (PhononLattice)" \
    cmp -s <(energies ncto_direct/out/trial_summary.txt) \
           <(energies ncto_one_point_sweep/out/field_strength_1.000000e-01/trial_summary.txt)

# Sweep of a TYPED key (pump_amplitude; sweeps used to only fill the Hamiltonian
# map, so every point ran the same pulse), 3 points on 2 ranks (the uneven split
# deadlocked on a COMM_WORLD barrier inside the runners).
run_case sweep_pump_amplitude_uneven 2 "pump_amplitude_3.000000e-01/sample_0/pump_probe_trajectory.txt" "$KITAEV
simulation_mode = parameter_sweep
sweep_base_simulation = pump_probe
sweep_parameter = pump_amplitude
sweep_start = 0.1
sweep_end = 0.3
sweep_step = 0.1
T_start = 2.0
T_end = 0.05
cooling_rate = 0.8
annealing_steps = 20
T_zero = true
n_deterministics = 50
pump_direction = 0,0,1
pump_width = 0.5
pump_frequency = 0.5
probe_amplitude = 0
md_time_start = -2.0
md_time_end = 3.0
md_timestep = 0.05
md_integrator = rk4"
P=sweep_pump_amplitude_uneven/out
check sweep_pump_amplitude_uneven "each point records its pump_amplitude" \
    grep -q "^pump_amplitude = 0.30000000000000004$" $P/pump_amplitude_3.000000e-01/run_info.txt
check sweep_pump_amplitude_uneven "different amplitudes give different trajectories" \
    bash -c "! cmp -s $P/pump_amplitude_1.000000e-01/sample_0/pump_probe_trajectory.txt $P/pump_amplitude_3.000000e-01/sample_0/pump_probe_trajectory.txt"

# Parallel tempering in a sweep: equal replica groups on sub-communicators.
run_case sweep_pt_groups 4 "field_strength_2.000000e-01/sample_0/*" "$KITAEV
simulation_mode = parameter_sweep
sweep_base_simulation = parallel_tempering
sweep_parameter = field_strength
sweep_start = 0.0
sweep_end = 0.2
sweep_step = 0.2
pt_ranks_per_point = 2
T_start = 2.0
T_end = 0.1
annealing_steps = 100
pt_exchange_frequency = 5
probe_rate = 10
pt_optimize_temperatures = false"

# TmFeO3 MD sweep, 3 points on 2 ranks (the mixed runners used COMM_WORLD barriers).
run_case tmfeo3_md_sweep 2 "*/sample_0/trajectory.h5" "system = tmfeo3
lattice_size = 1,1,1
seed = 9
output_dir = out
simulation_mode = parameter_sweep
sweep_base_simulation = molecular_dynamics
sweep_parameter = Kminus_2y
sweep_start = 0.0
sweep_end = 0.1
sweep_step = 0.05
T_start = 1.0
T_end = 0.05
cooling_rate = 0.5
annealing_steps = 5
md_time_end = 0.5
md_timestep = 0.05
md_integrator = rk4"

# Errors: one message, non-zero exit, no hang.
expect_failure bad_key 2 "did you mean 'annealing_steps'" "$KITAEV
simulation_mode = simulated_annealing
anealing_steps = 10"
expect_failure unsupported_mode 1 "not supported for system 'ncto'" "system = ncto
lattice_size = 2,2,1
output_dir = out
simulation_mode = parallel_tempering"
expect_failure missing_seed_file 2 "\[rank [0-9]+\] error: .*no_such_seed" "$KITAEV
simulation_mode = molecular_dynamics
initial_spin_config = no_such_seed.txt
md_time_end = 1.0
md_timestep = 0.05"
expect_failure endless_schedule 1 "cooling_rate must lie in" "$KITAEV
simulation_mode = simulated_annealing
cooling_rate = 1.0"

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

# Loaded TmFeO3 configuration (prefix of the _SU2/_SU3 files) starts every trial.
run_case tmfeo3_seeded_trials 2 "sample_1/final_energy.txt" "system = tmfeo3
lattice_size = 2,2,2
seed = 5
output_dir = out
simulation_mode = simulated_annealing
num_trials = 2
annealing_steps = 0
initial_spin_config = $WORK/tmfeo3_sa/out/sample_0/spins_final"
check tmfeo3_seeded_trials "both trials start from the loaded configuration" \
    test "$(energies tmfeo3_seeded_trials/out/trial_summary.txt | sort -u | wc -l)" -eq 1

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
