import pandas as pd
import numpy as np
from scipy.stats import weibull_min

def format_array(x):
    '''
        Given an object (list, np.array, pd.Series), makes sure it gets converted into a numpy array
    '''
    # If pandas object, call its "to_numpy" function
    if( type(x) in [type(pd.Series([])), type(pd.DataFrame())] ):
        x = x.to_numpy()
    # If it is a list, convert it to np.array
    if( type(x) in [list] ):
        x = np.array(x)
    # If it is not any known list type, it must be a single value, so create a singleton
    if( type(x) not in [type(np.array([])), list] ):
        x = np.array([x])
    return x

def pdf(y, phi1_, phi2_):
    '''
        Probability density function for the Weibull distribution with parametrization by Cancho (2021)
    '''
    y = format_array(y)
    phi1_ = format_array(phi1_)
    phi2_ = format_array(phi2_)
    k_ = phi1_
    lambda_ = np.exp(-phi2_ / phi1_)
    f_y = weibull_min.pdf(y, c = k_, scale = lambda_)
    return f_y

def cdf(y, phi1_, phi2_, lower_tail = True):
    '''
        Cumulative density function for the Weibull distribution with parametrization by Cancho (2021)
    '''
    y = format_array(y)
    phi1_ = format_array(phi1_)
    phi2_ = format_array(phi2_)
    k_ = phi1_
    lambda_ = np.exp(-phi2_ / phi1_)
    F_y = weibull_min.cdf(y, c = k_, scale = lambda_)
    if(lower_tail):
        return F_y
    return 1-F_y

def ppf(q, phi1_, phi2_):
    '''
        Quantile function for the Weibull distribution with parametrization by Cancho (2021)
    '''
    q = format_array(q)
    phi1_ = format_array(phi1_)
    phi2_ = format_array(phi2_)
    k_ = phi1_
    lambda_ = np.exp(-phi2_ / phi1_)
    Fy = weibull_min.ppf(q, c = k_, scale = lambda_)
    return y

def rvs(phi1_, phi2_, size = 1, random_state = None):
    '''
        Sample generator for the Weibull distribution with parametrization by Cancho (2021)
    '''
    k_ = phi1_
    lambda_ = np.exp(-phi2_ / phi1_)
    y = weibull_min.rvs(size = size, c = k_, scale = lambda_, random_state = random_state)
    return y
