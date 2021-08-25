import numpy as np
import Array

input_pv_dir = "phase_velocity/KYT013/"
file_list = ["1m/spac.vel","2m/spac.vel","5m/spac.vel","10m/spac.vel"]
output_pv_file = "phase_velocity/KYT013.vel"

fmax = 20.0
fmin_list = [12,8,4,2.5]

input_pv_files = [input_pv_dir+s for s in file_list]

ns = len(fmin_list)
fr = fmin_list + [fmax] + fmin_list[0:-1]
freqency_range =  list(zip(*[fr[i:i+ns] for i in range(0,2*ns,ns)]))

print(input_pv_files)
print(freqency_range)

freq_list,pv_list = Array.io.read_pv_files(input_pv_files)
freq,vel = Array.analysis.connect_phase_velocity(freq_list,pv_list,freqency_range,plot_flag=True)
Array.io.output_pv_file(output_pv_file,freq,vel)
