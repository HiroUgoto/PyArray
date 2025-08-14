import numpy as np
import Array

model_file = "model/Test.dat"
# output_file = "model/Test.resp"
output_file = "model/Test.tf"

model = Array.io.read_model_file(model_file)

# freq_sim,resp_sim = Array.analysis.model_medium_response_py(model,fmax=3.5,nmode=5,print_flag=True,plot_flag=True)
freq_sim,resp_sim = Array.analysis.model_transfer_function_py(model,fmax=1.5,abs=True,print_flag=True,plot_flag=True)

Array.io.output_pv_file(output_file,freq_sim,resp_sim)
