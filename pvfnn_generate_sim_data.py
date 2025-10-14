import pandas as pd
import numpy as np

import scipy
import scipy.stats as ss
from scipy.special import loggamma
from scipy.integrate import quad

from matplotlib import pyplot as plt
import seaborn as sns

import lifelines

import pvf
import weibull

from sklearn.model_selection import train_test_split

import itertools
from time import time

import os, shutil
from pathlib import Path

from tqdm import tqdm


def sample_structured1(theta_function, alpha, mu, gamma, phi1, phi2, size = 1, cens_loc = 0.0, cens_scale = 4.0, random_state = None):
    '''
        Sample n subject times and censor indicators based on a single U(0,1) generated (or given) covariate
    '''
    rng = np.random.default_rng(random_state)

    # Sample covariates and compute theta 
    x = ss.uniform.rvs(size = size, loc = 0, scale = 1, random_state = rng)
    theta_x = theta_function(x)
    # Sample individual, random frailty values
    w = pvf.rvs(alpha, mu, gamma, size = size, random_state = rng)

    # Number of latent causes for each subject
    z = ss.poisson.rvs(mu = w*theta_x, size = size, random_state = rng)
        
    # Generate all the competing lifetimes. For each individual, generates M_i. That's why we take np.sum(m_x) in total
    t_times = weibull.rvs(phi1, phi2, size = np.sum(z), random_state = rng)
    
    # Observed times
    obs_times = np.zeros(size)
    # Censored times
    cens_times = ss.uniform.rvs(size = size, loc = cens_loc, scale = cens_scale, random_state = rng)
    # Indicator of censors
    delta = np.zeros(size)
    
    last_i = 0
    # Runs through the entire vector of Z_i's
    for i in range(len(z)):
        # Take the minimum from all the competing times for the ith subject, if any
        if(z[i] > 0):
            # From the latest location accessed take the next Z_i times
            t_s = t_times[ last_i:(last_i+z[i]) ]
            obs_times[i] = np.min(t_s)
            last_i += z[i]
            if(obs_times[i] <= cens_times[i]):
                delta[i] = 1
    
        # If subject is cured or if censor time greater than subject time, censor the observation
        if(z[i] == 0 or obs_times[i] > cens_times[i]):
            obs_times[i] = cens_times[i]
    results = {
        "time": obs_times,
        "delta": delta,
        "x": x,
        "theta": theta_x,
        "w": w,
        "z": z
    }
    return results

def sample_train_test1(theta_function, alpha, mu, gamma, phi1, phi2, n_train, n_test, cens_loc = 0.0, cens_scale = 6.0, random_state = 1):
    n = n_train + n_test
    model_sample = sample_structured1(size = n, theta_function = theta_function, alpha = alpha, mu = mu, gamma = gamma, phi1 = phi1, phi2 = phi2, cens_loc = 0.0, cens_scale = cens_scale, random_state = random_state)
    y = model_sample["time"]
    delta = model_sample["delta"]
    x = model_sample["x"]
    theta = model_sample["theta"]
    w = model_sample["w"]
    z = model_sample["z"]
    
    df = pd.DataFrame({"x": x, "y": y, "delta": delta, "theta": theta, "w": w, "z": z})
    df_train = df[:n_train].copy()
    df_test = df[n_train:].copy()
    
    x_train = df_train["x"].to_numpy()
    y_train = df_train["y"].to_numpy()
    delta_train = df_train["delta"].to_numpy()
    theta_train = df_train["theta"].to_numpy()
    w_train = df_train["w"].to_numpy()
    z_train = df_train["z"].to_numpy()
    
    x_test = df_test["x"].to_numpy()
    y_test = df_test["y"].to_numpy()
    delta_test = df_test["delta"].to_numpy()
    theta_test = df_test["theta"].to_numpy()
    w_test = df_test["w"].to_numpy()
    z_test = df_test["z"].to_numpy()

    df_train.loc[:,"set"] = "train"
    df_test.loc[:,"set"] = "test"
    df = pd.concat([df_train, df_test])
    
    return y_train, delta_train, x_train, theta_train, w_train, z_train, \
           y_test, delta_test, x_test, theta_test, w_test, z_test, df

def sample_structured2(theta_function, alpha, mu, gamma, phi1, phi2, size = 1, cens_loc = 0.0, cens_scale = 4.0, random_state = None):
    '''
        Sample n subject times and censor indicators based on a single U(0,1) generated (or given) covariate
    '''
    rng = np.random.default_rng(random_state)

    # Sample covariates and compute theta 
    x1 = ss.uniform.rvs(size = size, loc = -1, scale = 2, random_state = rng)
    x2 = ss.uniform.rvs(size = size, loc = -1, scale = 2, random_state = rng)
    
    theta_x = theta_function(x1, x2)
    # Sample individual, random frailty values
    w = pvf.rvs(alpha, mu, gamma, size = size, random_state = rng)

    # Number of latent causes for each subject
    z = ss.poisson.rvs(mu = w*theta_x, size = size, random_state = rng)
        
    # Generate all the competing lifetimes. For each individual, generates M_i. That's why we take np.sum(m_x) in total
    t_times = weibull.rvs(phi1, phi2, size = np.sum(z), random_state = rng)
    
    # Observed times
    obs_times = np.zeros(size)
    # Censored times
    cens_times = ss.uniform.rvs(size = size, loc = cens_loc, scale = cens_scale, random_state = rng)
    # Indicator of censors
    delta = np.zeros(size)
    
    last_i = 0
    # Runs through the entire vector of Z_i's
    for i in range(len(z)):
        # Take the minimum from all the competing times for the ith subject, if any
        if(z[i] > 0):
            # From the latest location accessed take the next Z_i times
            t_s = t_times[ last_i:(last_i+z[i]) ]
            obs_times[i] = np.min(t_s)
            last_i += z[i]
            if(obs_times[i] <= cens_times[i]):
                delta[i] = 1
    
        # If subject is cured or if censor time greater than subject time, censor the observation
        if(z[i] == 0 or obs_times[i] > cens_times[i]):
            obs_times[i] = cens_times[i]
    results = {
        "time": obs_times,
        "delta": delta,
        "x1": x1,
        "x2": x2,
        "theta": theta_x,
        "w": w,
        "z": z
    }
    return results

def sample_train_test2(theta_function, alpha, mu, gamma, phi1, phi2, n_train, n_test, cens_loc = 0.0, cens_scale = 6.0, random_state = 1):
    n = n_train + n_test
    model_sample = sample_structured2(size = n, theta_function = theta_function, alpha = alpha, mu = mu, gamma = gamma, phi1 = phi1, phi2 = phi2, cens_loc = 0.0, cens_scale = cens_scale, random_state = random_state)
    y = model_sample["time"]
    delta = model_sample["delta"]
    x1 = model_sample["x1"]
    x2 = model_sample["x2"]
    theta = model_sample["theta"]
    w = model_sample["w"]
    z = model_sample["z"]
    
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y, "delta": delta, "theta": theta, "w": w, "z": z})
    df_train = df[:n_train].copy()
    df_test = df[n_train:].copy()
    
    x1_train = df_train["x1"].to_numpy()
    x2_train = df_train["x2"].to_numpy()
    y_train = df_train["y"].to_numpy()
    delta_train = df_train["delta"].to_numpy()
    theta_train = df_train["theta"].to_numpy()
    w_train = df_train["w"].to_numpy()
    z_train = df_train["z"].to_numpy()
    
    x1_test = df_test["x1"].to_numpy()
    x2_test = df_test["x2"].to_numpy()
    y_test = df_test["y"].to_numpy()
    delta_test = df_test["delta"].to_numpy()
    theta_test = df_test["theta"].to_numpy()
    w_test = df_test["w"].to_numpy()
    z_test = df_test["z"].to_numpy()

    df_train.loc[:,"set"] = "train"
    df_test.loc[:,"set"] = "test"
    df = pd.concat([df_train, df_test])
    
    return y_train, delta_train, x1_train, x2_train, theta_train, w_train, z_train, \
           y_test, delta_test, x1_test, x2_test, theta_test, w_test, z_test, df

def sample_structured10(theta_function, alpha, mu, gamma, phi1, phi2, size = 1, cens_loc = 0.0, cens_scale = 4.0, random_state = None):
    '''
        Sample n subject times and censor indicators based on a single U(0,1) generated (or given) covariate
    '''
    rng = np.random.default_rng(random_state)

    # Sample covariates and compute theta 
    Sigma = np.array([[1, 0.8, 0.5, 0.2, 0],
                      [0.8, 1, 0.2, 0.6, 0],
                      [0.5, 0.2, 1, 0.3, 0],
                      [0.2, 0.6, 0.3, 1, 0],
                      [0, 0, 0, 0, 1]])
    x1_x5 = ss.multivariate_normal.rvs(mean = np.repeat(0.0, 5), cov = Sigma, size = size, random_state = rng)
    x1 = x1_x5[:,0]
    x2 = x1_x5[:,1]
    x3 = x1_x5[:,2]
    x4 = x1_x5[:,3]
    x5 = x1_x5[:,4]
    x6 = ss.norm.rvs(loc = 0, scale = 1, size = size, random_state = rng)
    x7 = ss.norm.rvs(loc = 0, scale = 1, size = size, random_state = rng)
    x8 = ss.norm.rvs(loc = 0, scale = 1, size = size, random_state = rng)
    x9 = ss.norm.rvs(loc = 0, scale = 1, size = size, random_state = rng)
    x10 = ss.norm.rvs(loc = 0, scale = 1, size = size, random_state = rng)
    
    theta_x = theta_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
    # Sample individual, random frailty values
    w = pvf.rvs(alpha, mu, gamma, size = size, random_state = rng)

    # Number of latent causes for each subject
    z = ss.poisson.rvs(mu = w*theta_x, size = size, random_state = rng)
        
    # Generate all the competing lifetimes. For each individual, generates M_i. That's why we take np.sum(m_x) in total
    t_times = weibull.rvs(phi1, phi2, size = np.sum(z), random_state = rng)
    
    # Observed times
    obs_times = np.zeros(size)
    # Censored times
    cens_times = ss.uniform.rvs(size = size, loc = cens_loc, scale = cens_scale, random_state = rng)
    # Indicator of censors
    delta = np.zeros(size)
    
    last_i = 0
    # Runs through the entire vector of Z_i's
    for i in range(len(z)):
        # Take the minimum from all the competing times for the ith subject, if any
        if(z[i] > 0):
            # From the latest location accessed take the next Z_i times
            t_s = t_times[ last_i:(last_i+z[i]) ]
            obs_times[i] = np.min(t_s)
            last_i += z[i]
            if(obs_times[i] <= cens_times[i]):
                delta[i] = 1
    
        # If subject is cured or if censor time greater than subject time, censor the observation
        if(z[i] == 0 or obs_times[i] > cens_times[i]):
            obs_times[i] = cens_times[i]
    results = {
        "time": obs_times,
        "delta": delta,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "x6": x6,
        "x7": x7,
        "x8": x8,
        "x9": x9,
        "x10": x10,
        "theta": theta_x,
        "w": w,
        "z": z
    }
    return results

def sample_train_test10(theta_function, alpha, mu, gamma, phi1, phi2, n_train, n_test, cens_loc = 0.0, cens_scale = 6.0, random_state = 1):
    n = n_train + n_test
    model_sample = sample_structured10(size = n, theta_function = theta_function, alpha = alpha_, mu = mu_, gamma = gamma_, phi1 = phi1_, phi2 = phi2_, cens_loc = 0.0, cens_scale = cens_scale, random_state = random_state)
    y = model_sample["time"]
    delta = model_sample["delta"]
    x1 = model_sample["x1"]
    x2 = model_sample["x2"]
    x3 = model_sample["x3"]
    x4 = model_sample["x4"]
    x5 = model_sample["x5"]
    x6 = model_sample["x6"]
    x7 = model_sample["x7"]
    x8 = model_sample["x8"]
    x9 = model_sample["x9"]
    x10 = model_sample["x10"]
    
    theta = model_sample["theta"]
    w = model_sample["w"]
    z = model_sample["z"]
    
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6, "x7": x7, "x8": x8, "x9": x9, "x10": x10,
                       "y": y, "delta": delta, "theta": theta, "w": w, "z": z})
    df_train = df[:n_train].copy()
    df_test = df[n_train:].copy()
    
    x1_train = df_train["x1"].to_numpy()
    x2_train = df_train["x2"].to_numpy()
    x3_train = df_train["x3"].to_numpy()
    x4_train = df_train["x4"].to_numpy()
    x5_train = df_train["x5"].to_numpy()
    x6_train = df_train["x6"].to_numpy()
    x7_train = df_train["x7"].to_numpy()
    x8_train = df_train["x8"].to_numpy()
    x9_train = df_train["x9"].to_numpy()
    x10_train = df_train["x10"].to_numpy()
    y_train = df_train["y"].to_numpy()
    delta_train = df_train["delta"].to_numpy()
    theta_train = df_train["theta"].to_numpy()
    w_train = df_train["w"].to_numpy()
    z_train = df_train["z"].to_numpy()
    
    x1_test = df_test["x1"].to_numpy()
    x2_test = df_test["x2"].to_numpy()
    x3_test = df_test["x3"].to_numpy()
    x4_test = df_test["x4"].to_numpy()
    x5_test = df_test["x5"].to_numpy()
    x6_test = df_test["x6"].to_numpy()
    x7_test = df_test["x7"].to_numpy()
    x8_test = df_test["x8"].to_numpy()
    x9_test = df_test["x9"].to_numpy()
    x10_test = df_test["x10"].to_numpy()
    y_test = df_test["y"].to_numpy()
    delta_test = df_test["delta"].to_numpy()
    theta_test = df_test["theta"].to_numpy()
    w_test = df_test["w"].to_numpy()
    z_test = df_test["z"].to_numpy()

    df_train.loc[:,"set"] = "train"
    df_test.loc[:,"set"] = "test"
    df = pd.concat([df_train, df_test])
    
    return y_train, delta_train, x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train, x10_train, theta_train, w_train, z_train, \
           y_test, delta_test, x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test, x9_test, x10_test, theta_test, w_test, z_test, df