# ---- protocol: phonon window 0 .. 40 ps after the pulse (Fig. 2b,c), fine sampling of Q(t)
simulation_mode = pump_probe
output_dir = OUTDIR
pump_amplitude = E0VALUE
probe_amplitude = 0.0
md_time_start = 0.0
md_time_end = 71.0                  # 40 ps after t0
md_save_interval = 4                # every 0.013 ps (18 samples per 4.2 THz period)
