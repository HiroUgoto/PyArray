import numpy as np
import Array

ctl_dir = "input/KYT013/"
ctl_file = "hv.ctl"

control_file = ctl_dir + ctl_file
param,data = Array.io.read_control_file(control_file,single_flag=True)
segment_data = Array.analysis.segment_selection_3d(param,data)

freq,hvsr = Array.analysis.hv_spactra(param,segment_data)

Array.io.output_data_file("hvsr.dat",param,freq,hvsr)
