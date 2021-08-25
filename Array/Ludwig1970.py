import numpy as np

#------------------------------------------------------#
def rho(vp):
    rho0 = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 \
            - 0.0043*vp**4 + 0.000106*vp**5
    return np.clip(rho0,1.635,None)
