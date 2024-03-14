import numpy as np
from numpy import sin,cos,sqrt,exp,pi

def fitfun_Lorentzian(x, power_in, lambda_1, FWHM_1):
    # Lorentzian lineshape based on behavior parameters
    denom_1 = 1j*(x-lambda_1)/(FWHM_1/2)+1
    denom_2 = 1j*(x-lambda_1)/(FWHM_1/2)+1
    f=0.0
    ampl=0.5
    retval=power_in*np.abs(1-ampl*((1+f)**2/denom_1+(1-f)**2/denom_2))**2
    if np.any(np.isnan(retval)):
        raise ValueError('NaN in fitting function')
    return retval

def fitfun_thru(x, r1, r2a, lam, EL):
    # Ring thru port based on physical parameters
    retval = (r1**2+r2a**2-2*r1*r2a*cos(EL*(1/x-1/lam)))/(1+r1**2*r2a**2-2*r1*r2a*cos(EL*(1/x-1/lam)))
    if np.any(np.isnan(retval)):
        raise ValueError('NaN in fitting function')
    return retval

def fitfun_thru_FWHM(x, *data):
    r1, r2a, lam, EL = data
    retval = 10*np.log10((r1**2+r2a**2-2*r1*r2a*cos(EL*(1/x-1/lam)))/(1+r1**2*r2a**2-2*r1*r2a*cos(EL*(1/x-1/lam))))+3
    if np.any(np.isnan(retval)):
        raise ValueError('NaN in fitting function')
    return retval

def fitfun_drop(x, A, r1r2a, lam, EL):
    # Ring drop port based on physical parameters
    retval = 1-A/(1+r1r2a**2-2*r1r2a*cos(EL*(1/x-1/lam)))
    if np.any(np.isnan(retval)):
        raise ValueError('NaN in fitting function')
    return retval

def fitfun_drop_FWHM(x, *data):
    A, r1r2a, lam, EL = data
    retval = 10*np.log10(1-A/(1+r1r2a**2-2*r1r2a*cos(EL*(1/x-1/lam))))+3
    if np.any(np.isnan(retval)):
        raise ValueError('NaN in fitting function')
    return retval