import numpy as np
import Array

ctl_dir = "input/KYT013/"
ctl_list = ["01.ctl","02.ctl","05.ctl","10.ctl"]

fmin_list = []
for file in ctl_list:
    control_file = ctl_dir + file
    param,data = Array.io.read_control_file(control_file)
    segment_data = Array.analysis.segment_selection(param,data)

    freq,spac_coeff = Array.analysis.spac_coeff(param,segment_data,plot_flag=True)
    freq,cca_coeff,ns_ratio,fmin = Array.analysis.cca_coeff(param,segment_data,plot_flag=False)

    freq_spac,vel_spac = Array.analysis.spac_phase_velocity(param,freq,spac_coeff,fmin=0.5*fmin,plot_flag=True)
    Array.io.output_data_file("spac.vel",param,freq_spac,vel_spac)

    freq_cca,vel_cca = Array.analysis.cca_phase_velocity(param,freq,cca_coeff,plot_flag=False)
    Array.io.output_data_file("cca.vel",param,freq_cca,vel_cca)

    fmin_list += [fmin]

print(fmin_list)
