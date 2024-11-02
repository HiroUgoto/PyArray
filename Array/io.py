import numpy as np
import os,glob
from . import Ludwig1970

#-----------------------------------------------------------------#
def read_data_file(file_name):
    ud = np.loadtxt(file_name,usecols=(0,),unpack=True)
    return ud

def read_data_file_3d(file_name):
    ud,ew,ns = np.loadtxt(file_name,usecols=(0,1,2),unpack=True)
    return ud,ew,ns

def output_data_file(file_name,param,freq,vel):
    if not os.path.isdir(param["output_dir"]):
        os.makedirs(param["output_dir"])
    output_file_name = param["output_dir"] + file_name
    output_line = np.c_[freq,vel]
    np.savetxt(output_file_name,output_line)

#-----------------------------------------------------------------#
def read_control_file(file_name,print_flag=True,single_flag=False):
    if print_flag:
        print("------------------------------------")
        print("control file =>",file_name)

    raw_lines = open(file_name).readlines()
    lines = [s for s in raw_lines if not s.startswith("#")]

    param = {}
    data_dir = lines[0].strip()
    param["output_dir"] = lines[1].strip()
    param["sampling_frequency"] = int(lines[2])
    param["segment_length"] = float(lines[3].strip().split()[0])
    param["max_number_segment"] = int(lines[3].strip().split()[1])
    param["band_width"] = float(lines[4])

    n = int(lines[5])

    param["site"] = []
    data = []
    r = 0.0
    for i in range(0,n):
        list = lines[i+6].strip().split()
        file = data_dir+list[2]

        if int(list[3]) == 1:
            site_center = {"r":float(list[0]),"theta":float(list[1]),"file":file,"center":int(list[3])}
            if not single_flag:
                data_center = read_data_file(file)
            else:
                data_center = read_data_file_3d(file)
        else:
            param["site"] += [{"r":float(list[0]),"theta":float(list[1]),"file":file,"center":int(list[3])}]
            if not single_flag:
                data += [read_data_file(file)]
            else:
                data += [read_data_file_3d(file)]
            r += float(list[0])

    param["site"].insert(0,site_center)
    if not single_flag:
        param["r"] = r/(n-1)
    data.insert(0,data_center)

    if print_flag:
        print("+ output directory       :",param["output_dir"])
        print("+ sampling frequency [Hz]:",param["sampling_frequency"])
        print("+ segment length [s]     :",param["segment_length"])
        print("+ max number of segment  :",param["max_number_segment"])
        print("+ band width [Hz]        :",param["band_width"])

        print("+ number of sensors      :",len(param["site"]))
        if not single_flag:
            print("  (r[m], theta[deg], file)  ")
            for s in param["site"]:
                print("  ",s["r"],"  ",s["theta"],"  ",s["file"])
                print("+ average radius [m]     :",param["r"])
        else:
            print("+ data file              :",param["site"][0]["file"])

        print("------------------------------------")

    return param, data

#-----------------------------------------------------------------#
def read_model_file(file_name,rho_vp=False,print_flag=True):
    if print_flag:
        print("------------------------------------")
        print("model file =>",file_name)

    raw_lines = open(file_name).readlines()
    lines = [s for s in raw_lines if not s.startswith("#")]

    nlay = int(lines[0])

    vs = np.empty(nlay)
    vp = np.empty(nlay)
    rho = np.empty(nlay)
    depth = np.empty(nlay-1)
    thick = np.empty(nlay-1)

    if not rho_vp:
        for i in range(0,nlay):
            list = lines[i+1].strip().split()
            if i == nlay-1:
                vs[i],vp[i],rho[i] = list
            else:
                vs[i],vp[i],rho[i],depth[i] = list
    else:
        for i in range(0,nlay):
            list = lines[i+1].strip().split()
            if i == nlay-1:
                vs[i],vp[i] = list
            else:
                vs[i],vp[i],depth[i] = list
        rho = Ludwig1970.rho(vp*0.001)*1000

    thick[0] = depth[0]
    thick[1:nlay-1] = np.diff(depth)

    model = {"nlay":nlay, "vs":vs, "vp":vp, "density":rho, "thick":thick, "depth":depth}

    if print_flag:
        print("+ number of layers   :",model["nlay"])
        print("  (Vs[m/s],Vp[m/s],rho[kg/m3],depth[m])")
        for i in range(0,nlay-1):
            print("  {:4.0f},  {:4.0f},  {:4.0f},  {:5.1f}".format(vs[i],vp[i],rho[i],depth[i]))
        print("  {:4.0f},  {:4.0f},  {:4.0f},   ----".format(vs[-1],vp[-1],rho[-1]))
        print("------------------------------------")

    return model

def parse_model(model):
    nlay = model["nlay"]
    vs = model["vs"]
    vp = model["vp"]
    rho = model["density"]
    thick = model["thick"]
    return nlay,vs,vp,rho,thick

def output_model_file(file_name,model,gp_flag=False):
    if gp_flag:
        output_line = "{:.2f} {:.2f} {:.2f} {:.1f} \n".format(model["vs"][0],model["vp"][0],model["density"][0],0.0)
        for i in range(model["nlay"]-1):
            output_line += "{:.2f} {:.2f} {:.2f} {:.1f} \n".format(model["vs"][i],model["vp"][i],model["density"][i],model["depth"][i])
            output_line += "{:.2f} {:.2f} {:.2f} {:.1f} \n".format(model["vs"][i+1],model["vp"][i+1],model["density"][i+1],model["depth"][i])
        output_line += "{:.2f} {:.2f} {:.2f} {:.1f} \n".format(model["vs"][-1],model["vp"][-1],model["density"][-1],9999)

        print(output_line)

        with open(file_name,"w") as f:
            f.write(output_line)

        return

    with open(file_name,"w") as f:
        f.write("{} \n".format(model["nlay"]))
        for i in range(model["nlay"]-1):
            output_line = "{:.2f} {:.2f} {:.2f} {:.1f} \n".format(model["vs"][i],model["vp"][i],model["density"][i],model["depth"][i])
            f.write(output_line)
        output_line = "{:.2f} {:.2f} {:.2f} \n".format(model["vs"][-1],model["vp"][-1],model["density"][-1])
        f.write(output_line)

#-----------------------------------------------------------------#
def read_pv_file(file_name,fmax=10):
    freq,vel = np.loadtxt(file_name,usecols=(0,1),unpack=True)
    return freq[freq<fmax],vel[freq<fmax]

def read_pv_files(file_lists):
    freq_list = []
    pv_list = []
    for file in file_lists:
        freq,vel = np.loadtxt(file,usecols=(0,1),unpack=True)
        pv_list += [vel]
        freq_list += [freq]
    return freq_list, pv_list

def output_pv_file(file_name,freq,vel):
    output_file_name = file_name
    output_line = np.c_[freq,vel]
    np.savetxt(output_file_name,output_line)

def output_hv_file(file_name,freq,hv):
    output_file_name = file_name
    output_line = np.c_[freq,hv]
    np.savetxt(output_file_name,output_line)

#-----------------------------------------------------------------#
def output_file(file_name,freq,val):
    output_file_name = file_name
    output_line = np.c_[freq,val]
    np.savetxt(output_file_name,output_line)