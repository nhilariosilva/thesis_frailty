import pandas as pd
import numpy as np

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
# import tensorflow as tf
# import tensorflow_probability as tfp

from scipy.stats import levy_stable, uniform
from scipy.special import loggamma

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

def pdf_hougaard(w, alpha_, delta_, theta_, numerical_norm = True):
    '''
        Probability density function for the PVF distribution P(alpha, delta, theta) by Hougaard (1986)
    '''
    w = format_array(w)
    alpha_ = format_array(alpha_)
    delta_ = format_array(delta_)
    theta_ = format_array(theta_)
    
    # f_Y(y) where Y ~ P(alpha, delta, 0) - The series function is replaced by levy_stable with this specific parameters set
    c = (delta_ / alpha_)**(1/alpha_) / (1+np.tan(np.pi*alpha_/2))
    f_y = levy_stable.pdf(w, alpha = alpha_, beta = 1.0, loc = 0.0, scale = c)
    # The exponent associated to the laplace transform
    laplace_exponent = delta_ /alpha_ * theta_**alpha_
    # Density kernel for W ~ P(alpha, delta, theta)
    f_w = f_y * np.exp(-theta_*w)
    if(numerical_norm):
        # Normalize the density numerically. More stable, but slower
        norm_const = np.trapz(f_w, w, axis = 0)
        f_w /= norm_const
    else:
        # If desired, use the exact normalizing constant. That might be numerically unstable!
        f_w *= np.exp( delta_ /alpha_ * theta_**alpha_ )
    return f_w

def pdf(w, alpha_, mu_, gamma_, numerical_norm = True):
    '''
        Probability density function for the PVF distribution P(alpha, mu, gamma) by Cancho et al. (2021)
    '''
    w = format_array(w)
    alpha_ = format_array(alpha_)
    mu_ = format_array(mu_)
    gamma_ = format_array(gamma_)
    
    lambda_ = mu_**2 / gamma_
    theta_ = lambda_/mu_*(1-alpha_)
    delta_ = mu_*theta_**(1-alpha_)
    return pdf_hougaard(w, alpha_, delta_, theta_, numerical_norm = numerical_norm)

def rvs_hougaard(alpha_, delta_, theta_, size = 1, random_state = None):
    '''
        Sampler for the PVF distribution P(alpha, delta, theta) by Hougaard (1986)
    '''
    # Set the random_state for every sampler
    rng = np.random.default_rng(random_state)
    
    finished = False
    # Value to obtain the number of iid samples m
    psi_theta = np.exp(-theta_**alpha_ * delta_/alpha_)
    m = int( np.max([1, np.round(-np.log(psi_theta))]) )
    c = (delta_ / (alpha_ * m))**(1/alpha_) / (1+np.tan(np.pi*alpha_/2))
    # Generate all initial candidates S_k and U
    s_k = levy_stable.rvs(size = (size, m), alpha = alpha_, beta = 1.0, loc = 0.0, scale = c, random_state = rng)
    U = uniform.rvs(size = (size, m), loc = 0.0, scale = 1.0, random_state = rng)
    ind = np.exp(-theta_ * s_k)
    # Samples that got rejected receive None
    s_k[U > ind] = None
    # While there are rejected samples, continue resampling them
    while(np.any(np.isnan(s_k))):
        # Number of rejected samples in the previous step
        nan_count = np.sum(np.isnan(s_k))
        # Resample the values that got rejected
        s_k_resample = levy_stable.rvs(size = nan_count, alpha = alpha_, beta = 1.0, loc = 0.0, scale = c, random_state = rng)
        U_resample = uniform.rvs(size = nan_count, loc = 0.0, scale = 1.0, random_state = rng)
        ind_resample = np.exp(-theta_ * s_k_resample)
        s_k_resample[U_resample > ind_resample] = None
        s_k[np.isnan(s_k)] = s_k_resample
    return np.sum(s_k, axis = 1)

def rvs(alpha_, mu_, gamma_, size = 1, random_state = None):
    '''
        Probability density function for the PVF distribution P(alpha, mu, gamma) by Cancho et al. (2021)
    '''    
    lambda_ = mu_**2 / gamma_
    theta_ = lambda_/mu_*(1-alpha_)
    delta_ = mu_*theta_**(1-alpha_)
    return rvs_hougaard(alpha_, delta_, theta_, size = size, random_state = random_state)

# First version of sampling for the hougaard PVF distribution (proof of concept).
# def sample_hougaard1(alpha_, delta_, theta_):
#     # Value to obtain the number of iid samples m
#     psi_theta = np.exp(-theta_**alpha_ * delta_/alpha_)
#     m = int( np.max([1, np.round(-np.log(psi_theta))]) )
#     c = (delta_ / (alpha_ * m) )**(1/alpha_) / (1+np.tan(np.pi*alpha_/2))
#     S = []
#     for k in range(m):
#         while(True):
#             s_k = levy_stable.rvs(size = 1, alpha = alpha_, beta = 1.0, loc = 0.0, scale = c)
#             u = uniform.rvs(loc = 0.0, scale = 1.0)
#             if(u <= np.exp(-theta_ * s_k)):
#                 break
#         S.append( s_k )
#     return np.sum( S )

# def sample_hougaard(alpha_, delta_, theta_, size = 1):
#     sample = []
#     for i in range(size):
#         sample.append( sample_hougaard1(alpha_, delta_, theta_) )
#     return np.array(sample)

