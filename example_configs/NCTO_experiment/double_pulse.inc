# ---- protocol: double-THz coherent control (Fig. 4): pulse 1 at t0, pulse 2 at t0 + tau,
#      E2 = 0.6 E1, same shape.  Three runs per tau (both / pulse 1 only / pulse 2 only),
#      M_NL = M12 - M1 - M2.  TAUVALUE in code units (tau_ps / 0.658212); P1ON/P2ON in {0,1}.
simulation_mode = pump_probe
output_dir = OUTDIR
pump_amplitude = P1AMP              # = E0VALUE * P1ON
probe_amplitude = P2AMP             # = 0.6 * E0VALUE * P2ON
probe_frequency = 12.41
probe_width = 0.273
probe_phase = 0.0
probe_polarization = 0.0
probe_time = P2TIME                 # = 10 + TAUVALUE
md_time_start = 0.0
md_time_end = 920.0                 # 600 ps after t0
md_save_interval = 300              # every 1.0 ps
