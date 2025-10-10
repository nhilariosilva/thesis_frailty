
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import layers, initializers, optimizers, losses

class FrailtyModelNN(keras.models.Model):

    def __init__(self, parameters, loglikelihood):
        pass

    def clone_architecture(self, model):
        pass
    
    def define_structure(self):

        # Goes through the list of parameters for the model and filter them by their classes:
        # - "nn" will be treated as an output from a given neural network that receives the variables x as input.
        # - "independet" will be treated an an individual tf.Variable, trainable object. It is still trained in tensorflow, but is constant for all subjects
        # - "fixed" will be treated as a non-trainable tf.Variable. Basically just a known constant.
        # - "manual" will be treated as a non-trainable tf.Variable, but its value will be eventually updated manually using user provided functions (useful in cases where closed forms can be obtained)
        # - "dependent" will be treated simply as a deterministic function of other parameters and will be updated after training

        nn_pars = []
        independent_pars = []
        fixed_pars = []
        manual_pars = []
        for parameter in self.parameters:
            if(parameter["par_type"] == "nn"):
                nn_pars.append( parameter )
            elif(parameter["par_type"] == "independent"):
                independent_pars.append( parameter )
            elif(parameter["par_type"] == "fixed"):
                fixed_pars.append( parameter )
            elif(parameter["par_type"] == "manual"):
                manual_pars.append( parameter )
            else:
                raise Exception("Invalid parameter {} type: {}".format(parameter, parameter["par_type"]))

        # If at least one parameter is to be modeled as a neural network output, define its architecture here
        if( len(nn_pars) > 0 ):
            # Try to implement a function that, given a model, clone its architecture
            # clone_architecture(given_model)

            # The user provided architecture must output an array with the exact same shape as the concatenation of all "nn" parameters.
            # For example, if theta (nn) is a single constant and alpha (nn) is an array with 3 values, the expected output for the neural network
            # component is 4. For that, these parameters must be at most one dimensional arrays. (No matrix, or more complex structured parameters!)

        # Dictionary with all parameters that are its individual weights
        model_variables = {}

        # Include variables that do not depend on the variables x, but are still trained by tensorflow
        for parameter in independent_pars:
            model_variables[parameter] = self.add_weight(
                name = parameter,
                shape = parameter["shape"],
                initializer = keras.initializers.Constant( parameter["initializer"] ),
                trainable = True,
                dtype = tf.float32
            )

        # Include variables that are not trained by tensorflow (known, fixed constants or manual trained variables)
        for parameter in np.concatenate([fixed_pars, manual_pars]):
            model_variables[parameter] = self.add_weight(
                name = "raw_" + parameter,
                shape = parameter["shape"],
                initializer = keras.initializers.Constant( parameter["initializer"] ),
                trainable = False,
                dtype = tf.float32
            )

    def loss_func(self):
        # ESSENTIALLY THE NEGATIVE LOGLIK. Should we pass this function as provided by the user? It must define its variables according to the variables dictionaries from define_Structures
        pass

    def train_step(self, data):
        """
            Called by each batch in order to evaluate the loglikelihood and accumulate the parameters gradients using training data.
        """
        pass

    def test_step(self, data):
        """
            Called by each batch in order to evaluate the loglikelihood and update the convergence metrics using test data.
        """
        pass

    

    def compile_model(self):
        """
            Defines the configuration for the model, such as batch size, training mode, early stopping.
        """
        pass

    def train_model(self):
        """
            This is the function that start the training.
        """
        pass
    
    def apply_accumulated_gradients(self):
        """
            Given the proper number of steps for the model to accumulate gradients over time, finally applies gradients to update the model weights.
        """

    