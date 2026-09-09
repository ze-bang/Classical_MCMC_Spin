# ---- protocol: single THz pulse at 6 K, magnetisation window -100 .. 500 ps (Fig. 3b)
simulation_mode = pump_probe
output_dir = OUTDIR
pump_amplitude = E0VALUE
probe_amplitude = 0.0
md_time_start = 0.0
md_time_end = 770.0                 # 500 ps after the pulse centre (t0 = 10 = 6.6 ps)
md_save_interval = 300              # every 1.0 ps: 770 frames of the full spin configuration
