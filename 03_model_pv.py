import numpy as np
import Array

observed_pv_file = "phase_velocity/KYT013.vel"
model_file = "model/KYT013.dat"
model_pv_file = "model/KYT013.vel"

freq_obs,vel_obs = Array.io.read_pv_file(observed_pv_file,fmax=16)
model = Array.io.read_model_file(model_file)

freq_sim,vel_sim,_ = Array.analysis.model_phase_velocity(model,fmax=16,print_flag=True,plot_flag=False)
Array.analysis.compare_phase_velocity(freq_obs,vel_obs,freq_sim,vel_sim)
Array.io.output_pv_file(model_pv_file,freq_sim,vel_sim)
