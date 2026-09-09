# ---- thermal variant: classical Langevin spins at 6 K (k_B T = 0.517 meV), single pulse.
#      CAUTION: classical spins overstate thermal fluctuations at T/T_N = 0.22; check that the
#      3Q order survives 30 ps of Langevin dynamics BEFORE the pulse (pre-time-zero window).
simulation_mode = pump_probe
output_dir = OUTDIR
langevin_temperature = 0.517
langevin_seed = 12345
pump_amplitude = E0VALUE
pump_time = 55.0                    # 30 ps of thermalisation before the pulse
probe_amplitude = 0.0
md_time_start = 0.0
md_time_end = 815.0                 # 500 ps after the pulse
md_save_interval = 300
