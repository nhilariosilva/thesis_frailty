import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import models, layers, initializers, optimizers, losses

class FrailtyModel(object):
    """
        This class summarises any frailty model framework.

        It has a very broad definition. Virtually any survival model with hazard given by h(t; x) = Z h_0(t; x)
        may be included as an object from this class.
    """

    def __init__(self,
                 discrete, em,
                 parameters,
                 loglikelihood,
                 neuralnet_structure = None,
                 zpdf = None, zpmf = None, zcdf = None, zppf = None, zrvs = None
                 ):
        """
            Init a frailty model by saving all its relevant functions.

            First, we distinct it between a continuous frailty and a discrete frailty. In both cases, we refer to the frailty as Z

            ----- Probability functions for the continuous frailty distribution (Optional) -----
                - zpdf: The probability density function
                - zcdf: The cumulative density function
                - zppf: The quantile function
                - zrvs: A sampler for this distribution

            ----- Probability functions for the discrete frailty distribution (Optional) -----
                - zppf: The probability mass function
                - zcdf: The cumulative probability function
                - zppf: The quantile function
                - zrvs: A sampler for this distribution

            Those functions do not have to be stricly specified, as some frailty distributions do not even have a closed form for such functions.
            Essentially, by providing these functions, some methods will be able to provide more insightful plots in the future.

            We list the essential inputs for every frailty model
                - discrete:
                    - True: Z is treated as a discrete distribution
                    - False: Z is treated as a continuous, positive distribution
                - em:
                    - True: The user expecte to estimate parameters using the EM algorithm;
                    - False: It is assumed that tensorflow will estimate all the parameters according to their specifications and derivatives, unless specified that some of them will be updated by closed forms.

                - parameters: a dict object specifying all model parameters and how they should be modeled. An example for a Poisson frailty model is given below;
                - loglikelihood: The log-likelihood function for the model;
        """
        
        
        # If the EM algorithm will be used, the neural network model will have to be executed several times.
        # That process takes a lot of memory, which is hard to clear out in a notebook environment. Because of that, this requires a call to an external
        # .py file as a subprocess.
        if(em):
            pass
        else:
            pass

        self.discrete = discrete
        
        self.parameters = parameters
        self.loglikelihood = loglikelihood


        # If any parameter is specified with nn, then neuralnet_structure can not be a None object, as it will bring information of how to link x to our parameters

