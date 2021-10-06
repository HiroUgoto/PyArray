import numpy as np
import Array

observed_pv_file = "phase_velocity/KYT013.vel"
observed_hv_file = "hvsr/KYT013/hvsr.dat"
model_file = "model/KYT013.dat"

model_pv_file = "model/KYT013.vel"
model_hv_file = "model/KYT013.hv"

freq_obs,vel_obs = Array.io.read_pv_file(observed_pv_file,fmax=16)
freq_hv_obs,hv_obs = Array.io.read_pv_file(observed_hv_file,fmax=16)
model = Array.io.read_model_file(model_file)

freq_sim,vel_sim,hv_sim = Array.analysis.model_phase_velocity_py(model,fmax=16,print_flag=True,plot_flag=False)

Array.analysis.compare_phase_velocity(freq_obs,vel_obs,freq_sim,vel_sim)
Array.analysis.compare_hvsr(freq_hv_obs,hv_obs,freq_sim,hv_sim)

Array.io.output_pv_file(model_pv_file,freq_sim,vel_sim)
Array.io.output_hv_file(model_hv_file,freq_sim,hv_sim)
