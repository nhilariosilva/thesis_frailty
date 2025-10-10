
import numpy as np

import frailty_model
import frailty_model

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import models, layers, initializers, optimizers, losses

if(__name__ == "__main__"):
    
    # Number of knots in the piecewise exponential base model
    G = 5

    def update_alpha(alpha, s, t, delta, z_r):
        # Convert z_r to a numpy object
        z_r = z_r.numpy().flatten()
        # New alpha vector
        new_alpha = alpha.copy()
        # Get the indices of the intervals whose each observed time belongs
        ind_t_g = np.searchsorted(s, t)-1
        for g in range(len(alpha)-1):
            i = (ind_t_g == g)
            num = np.sum( delta[i] )
            den = np.sum( m_r[i] * (t[i] - s[g]) ) + np.sum( z_r[ind_t_g > g] * (s[g+1] - s[g]) )
            if(den == 0.0):
                new_alpha[g] = alpha[g]
            else:
                # Obtém numerador e denominador para a atualização do parâmetro alpha_g
                new_alpha[g] = num / den
        return new_alpha

    # As an example, let us reproduce the model given by Xie & Yu (2021), that is, the standard promotion time cure model with proportional hazards
    # with a neural network component in the Poisson mean parameter, theta.
    parameters = {
        # Unknown parameter modeled as the output of the neural network
        "theta": {"domain": [0, tf.constant(-np.inf)], "link": lambda x : tf.math.exp(x), "par_type": "nn", "update_func": None, "shape": None},
        # Treated as a constant, known parameter
        "q": {"domain": [0, tf.constant(np.inf)], "link": None, "par_type": "fixed", "update_func": None, "shape": None},
        # Unknown vector of parameters, but independent of covariates
        "alpha": {"domain": [[0, tf.constant(np.inf)] for i in range(G+1)], "link": None, "par_type": "manual", "update_func": update_alpha, "shape": (G+1,)}
    }
    print(parameters)


