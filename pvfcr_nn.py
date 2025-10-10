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

# As we are considering only structured data in this project, we don't use the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import models, layers, initializers, optimizers, losses

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config = config)

from tensorflow.keras.callbacks import Callback
from tqdm.keras import TqdmCallback
from tqdm import tqdm

def sample_structured1(theta_function, alpha, mu, gamma, phi1, phi2, size = 1, cens_loc = 0.0, cens_scale = 6.0, random_state = None):
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

class PVFcrModelStable(keras.models.Model):    
    def __init__(self, seed = 1, input_dim = (None, 1),
                 fix_alpha = False, fix_gamma = False, fix_phi1 = False, fix_phi2 = False,
                 init_alpha = 0.5, init_gamma = 1.0, init_phi1 = 1.0, init_phi2 = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.seed = seed
        self.fix_alpha = fix_alpha
        self.fix_gamma = fix_gamma
        self.fix_phi1 = fix_phi1
        self.fix_phi2 = fix_phi2
        
        self.init_alpha = init_alpha
        self.init_gamma = init_gamma
        self.init_phi1 = init_phi1
        self.init_phi2 = init_phi2
        
        # Define the architecture
        self.define_structure()
        # Call the model with a dummy input to initialize its weights
        self.dummy_input = keras.layers.Input(shape = self.input_dim)
        self(self.dummy_input)

        # Number of accumulated gradients by batches
        self.n_acum_step = tf.Variable(0, dtype = tf.int32, trainable = False)

        # Number of parameters that were fixed for training
        self.fixed_params = fix_alpha + fix_gamma + fix_phi1 + fix_phi2
        # The number of independent parameters to be estimated directly by tensorflow
        self.trainable_params = 4 - self.fixed_params

        self.gradient_accumulation_alpha = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_gamma = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_phi1 = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_phi2 = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_other = [tf.Variable(tf.zeros_like(v, dtype = tf.float32), trainable = False) for v in self.trainable_variables[self.trainable_params:]]

    def define_structure(self):
        '''
            This method must contain all the layers from the neural networks model, preferrably 
        '''
        initializer = initializers.GlorotNormal(seed = self.seed)
        self.dense1 = keras.layers.Dense(units = 16, activation = "gelu", kernel_initializer = initializer, dtype = tf.float32, name = "dense1")
        self.dense2 = keras.layers.Dense(units = 1, kernel_initializer = initializer, dtype = tf.float32, activation = None, use_bias = False, name = "output")

        # Trainable parameters from the model - They are all unconstrained in R (raw)

        self.init_raw_alpha = tf.math.log( self.init_alpha/(1-self.init_alpha) )
        self.init_raw_gamma = tf.math.log( self.init_gamma )
        self.init_raw_phi1 = tf.math.log( self.init_phi1 )
        self.init_raw_phi2 = tf.identity( self.init_phi2 )

        self.raw_alpha = self.add_weight(name = 'alpha', shape = (), initializer = keras.initializers.Constant( self.init_raw_alpha ), trainable = not self.fix_alpha, dtype = tf.float32)
        self.raw_gamma = self.add_weight(name = 'gamma', shape = (), initializer = keras.initializers.Constant( self.init_raw_gamma ), trainable = not self.fix_gamma, dtype = tf.float32)
        self.raw_phi1 = self.add_weight(name = 'phi1', shape = (), initializer = keras.initializers.Constant( self.init_raw_phi1 ), trainable = not self.fix_phi1, dtype = tf.float32)
        self.raw_phi2 = self.add_weight(name = 'phi2', shape = (), initializer = keras.initializers.Constant( self.init_raw_phi2 ), trainable = not self.fix_phi2, dtype = tf.float32)

        self.alpha = 1/(1+np.exp(-self.raw_alpha))
        self.gamma = np.exp(self.raw_gamma)
        self.phi1 = np.exp(self.raw_phi1)
        self.phi2 = np.copy(self.raw_phi2)
        
    def call(self, x_input):
        x = self.dense1(x_input)
        return self.dense2(x)

    def save_model(self, filename):
        self.save_weights(filename)

    def load_model(self, filename):
        self.load_weights(filepath = filename)
        self.alpha = 1/(1+np.exp(-self.raw_alpha))
        self.gamma = np.exp(self.raw_gamma)
        self.phi1 = np.exp(self.raw_phi1)
        self.phi2 = np.copy(self.raw_phi2)
        return self
    
    def copy(self):
        '''
            Creates a new object of the same class as a copy.
        '''
        new_model = PVFcrModelStable(seed = self.seed, input_dim = self.input_dim,
                                     fix_alpha = self.fix_alpha, fix_gamma = self.fix_gamma, fix_phi1 = self.fix_phi1, fix_phi2 = self.fix_phi2,
                                     init_alpha = self.init_alpha, init_gamma = self.init_gamma, init_phi1 = self.init_phi1, init_phi2 = self.init_phi2)
        new_model.set_weights(self.get_weights())
        new_model.alpha = 1/(1+np.exp(-new_model.raw_alpha))
        new_model.gamma = np.exp(new_model.raw_gamma)
        new_model.phi1 = np.exp(new_model.raw_phi1)
        new_model.phi2 = np.copy(new_model.raw_phi2)
        return new_model
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # eta
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # y
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)   # delta
    ])
    def likelihood_loss(self, eta, y, delta):
        theta = tf.math.exp(eta)
        
        alpha = 1 / (1 + tf.math.exp(-self.raw_alpha))
        gamma = tf.math.exp(self.raw_gamma)
        phi1 = tf.math.exp(self.raw_phi1)
        phi2 = tf.identity(self.raw_phi2)
        
        log_f0 = tf.math.log(phi1) + (phi1-1)*tf.math.log(y) + phi2 - tf.math.exp(phi2) * tf.math.pow(y, phi1)
        F0 = 1 - tf.math.exp(-tf.math.pow(y, phi1) * tf.math.exp(phi2))
        laplace_transform_term = 1 + gamma*theta*F0/(1-alpha)

        loss_weights = delta*eta + delta*log_f0 + (1-alpha)/(alpha*gamma)*(1 - tf.math.pow(laplace_transform_term, alpha)) + (alpha-1)*delta*tf.math.log(laplace_transform_term)
        loss_weights_mean = -tf.math.reduce_mean(loss_weights)
        
        return loss_weights_mean

    def apply_accumulated_gradients(self):
        # ----------------------------------- Independent parameters component -----------------------------------
        # Apply the accumulated gradients to the trainable variables
        if(not self.fix_alpha):
            self.optimizer_alpha.apply_gradients( zip(self.gradient_accumulation_alpha, [self.raw_alpha]) )
            self.gradient_accumulation_alpha[0].assign(tf.zeros((), dtype = tf.float32))
        if(not self.fix_gamma):
            self.optimizer_gamma.apply_gradients( zip(self.gradient_accumulation_gamma, [self.raw_gamma]) )
            self.gradient_accumulation_gamma[0].assign(tf.zeros((), dtype = tf.float32))
        if(not self.fix_phi1):
            self.optimizer_phi1.apply_gradients( zip(self.gradient_accumulation_phi1, [self.raw_phi1]) )
            self.gradient_accumulation_phi1[0].assign(tf.zeros((), dtype = tf.float32))
        if(not self.fix_phi2):
            self.optimizer_phi2.apply_gradients( zip(self.gradient_accumulation_phi2, [self.raw_phi2]) )
            self.gradient_accumulation_phi2[0].assign(tf.zeros((), dtype = tf.float32))

        # Reset the gradient accumulation steps counter to zero
        self.n_acum_step.assign(0)

        # ----------------------------------- Neural network component -----------------------------------
        self.optimizer_other.apply_gradients( zip(self.gradient_accumulation_other, self.trainable_variables[self.trainable_params:]) )
        for i in range(len(self.gradient_accumulation_other)):
            self.gradient_accumulation_other[i].assign(tf.zeros_like(self.trainable_variables[self.trainable_params:][i], dtype = tf.float32))
    
    def train_step(self, data):
        '''
            Override the train_step method for custom training logic
        '''
        x, y, delta = data

        self.n_acum_step.assign_add(1)
        with tf.GradientTape() as tape:
            eta = self(x, training = True)
            likelihood_loss = self.likelihood_loss(eta = eta, y = y, delta = delta)

        gradients = tape.gradient(likelihood_loss, self.trainable_variables)

        i_grad = 0
        if(not self.fix_alpha):
            alpha_gradients = gradients[i_grad]
            self.gradient_accumulation_alpha[0].assign_add(alpha_gradients)
            i_grad += 1
        if(not self.fix_gamma):
            gamma_gradients = gradients[i_grad]
            self.gradient_accumulation_gamma[0].assign_add(gamma_gradients)
            i_grad += 1
        if(not self.fix_phi1):
            phi1_gradients = gradients[i_grad]
            self.gradient_accumulation_phi1[0].assign_add(phi1_gradients)
            i_grad += 1
        if(not self.fix_phi2):
            phi2_gradients = gradients[i_grad]
            self.gradient_accumulation_phi2[0].assign_add(phi2_gradients)
            i_grad += 1

        # other_gradients = gradients[self.trainable_params:]
        other_gradients = gradients[i_grad:]
        
        for i in range(len(self.gradient_accumulation_other)):
            self.gradient_accumulation_other[i].assign_add(other_gradients[i])
    
        tf.cond(tf.equal(self.n_acum_step, self.gradient_accumulation_steps), self.apply_accumulated_gradients, lambda: None)
        
        return {"likelihood_loss": likelihood_loss, "alpha": 1/(1+tf.math.exp(-self.raw_alpha)), "gamma": tf.math.exp(self.raw_gamma), "phi1": tf.math.exp(self.raw_phi1), "phi2": self.raw_phi2}

    def test_step(self, data):
        x, y, delta = data
        eta = self(x, training = False)
        likelihood_loss = self.likelihood_loss(eta = eta, y = y, delta = delta)
        return {"likelihood_loss": likelihood_loss}

    def likelihood_loss_predict(self, x, y, delta):
        x = tf.cast(x, dtype = tf.float32)
        y = tf.cast(y, dtype = tf.float32)
        delta = tf.cast(delta, dtype = tf.float32)
        if(len(x.shape) == 1):
            x = tf.reshape( x, shape = (len(x), 1) )
        if(len(y.shape) == 1):
            y = tf.reshape( y, shape = (len(y), 1) )
        if(len(delta.shape) == 1):
            delta = tf.reshape( delta, shape = (len(delta), 1) )

        eta = self(x, training = False)
        theta = tf.math.exp(eta)
        
        alpha = 1 / (1 + tf.math.exp(-self.raw_alpha))
        gamma = tf.math.exp(self.raw_gamma)
        phi1 = tf.math.exp(self.raw_phi1)
        phi2 = tf.identity(self.raw_phi2)
        
        log_f0 = tf.math.log(phi1) + (phi1-1)*tf.math.log(y) + phi2 - tf.math.exp(phi2) * tf.math.pow(y, phi1)
        F0 = 1 - tf.math.exp(-tf.math.pow(y, phi1) * tf.math.exp(phi2))
        laplace_transform_term = 1 + gamma*theta*F0/(1-alpha)

        loss_weights = delta*eta + delta*log_f0 + (1-alpha)/(alpha*gamma)*(1 - tf.math.pow(laplace_transform_term, alpha)) + (alpha-1)*delta*tf.math.log(laplace_transform_term)
        loss_weights_mean = -tf.math.reduce_mean(loss_weights)
        
        return loss_weights_mean
    
    def compile_model(self,
                      optimizer_alpha = optimizers.Adam(learning_rate = 0.001),
                      optimizer_gamma = optimizers.Adam(learning_rate = 0.1),
                      optimizer_phi1 = optimizers.Adam(learning_rate = 0.1),
                      optimizer_phi2 = optimizers.Adam(learning_rate = 0.1),
                      optimizer_other = optimizers.Adam(learning_rate = 0.1), run_eagerly = False):
        self.optimizer_alpha = optimizer_alpha
        self.optimizer_gamma = optimizer_gamma
        self.optimizer_phi1 = optimizer_phi1
        self.optimizer_phi2 = optimizer_phi2
        self.optimizer_other = optimizer_other
        self.compile(
            run_eagerly = run_eagerly
        )
        
    def compile_train_model(self, x, y, delta,
                            validation = False, val_prop = None, x_val = None, y_val = None, delta_val = None,
                            epochs = 100,
                            buffer_size = 4096, train_batch_size = None, val_batch_size = None,
                            optimizer_alpha = optimizers.Adam(learning_rate = 0.001),
                            optimizer_gamma = optimizers.Adam(learning_rate = 0.1),
                            optimizer_phi1 = optimizers.Adam(learning_rate = 0.1),
                            optimizer_phi2 = optimizers.Adam(learning_rate = 0.1),
                            optimizer_other = optimizers.Adam(learning_rate = 0.1),
                            run_eagerly = False, gradient_accumulation_steps = None,
                            early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10, early_stopping_warmup = 0,
                            shuffle = False,
                            verbose = 2):
        '''
            Organiza os conjunto de treino e validação e inicia o treinamento da rede neural
        '''
        self.validation = validation
        
        # Pass the input variables to tensorflow default types
        x = tf.cast(x, dtype = tf.float32)
        y = tf.cast(y, dtype = tf.float32)
        delta = tf.cast(delta, dtype = tf.float32)
        
        # If input is a vector, transform it into a column
        if(len(x.shape) == 1):
            x = tf.reshape( x, shape = (len(x), 1) )
        if(len(y.shape) == 1):
            y = tf.reshape( y, shape = (len(y), 1) )
        if(len(delta.shape) == 1):
            delta = tf.reshape( delta, shape = (len(delta), 1) )

        # Salva os dados originais
        self.x = x
        self.y = y
        self.delta = delta

        if(self.validation):
            if(x_val is not None and y_val is not None and delta_val is not None):
                # Proper validation data provided
                
                x_val = tf.cast(x_val, dtype = tf.float32)
                y_val = tf.cast(y_val, dtype = tf.float32)
                delta_val = tf.cast(delta_val, dtype = tf.float32)

                if(len(x_val.shape) == 1):
                    x_val = tf.reshape( x_val, shape = (len(x_val), 1) )
                if(len(y_val.shape) == 1):
                    y_val = tf.reshape( y_val, shape = (len(y_val), 1) )
                if(len(delta_val.shape) == 1):
                    delta_val = tf.reshape( delta_val, shape = (len(delta_val), 1) )
                
                self.x_val = x_val
                self.y_val = y_val
                self.delta_val = delta_val
                self.x_train, self.y_train, self.delta_train = self.x, self.y, self.delta
            else:
                # If validation is wanted, but no data was given, select val_prop * 100% observations as validation set
                
                self.indexes_train = np.arange(x.shape[0])
                if(shuffle):
                    self.indexes_train = tf.random.shuffle( self.indexes_train )
                    
                x_shuffled = tf.gather( x, self.indexes_train )
                y_shuffled = tf.gather( y, self.indexes_train )
                delta_shuffled = tf.gather( delta, self.indexes_train )

                if(val_prop is None):
                    raise Exception("Please, provide the size of the validation set (between 0 and 1).")
                # Selects the subsample as validation data
                val_size = int(x.shape[0] * val_prop)
                self.x_val, self.y_val, self.delta_val = x_shuffled[:val_size], y_shuffled[:val_size], delta_shuffled[:val_size]
                self.x_train, self.y_train, self.delta_train = x_shuffled[val_size:], y_shuffled[val_size:], delta_shuffled[val_size:]
        else:
            # If no validation step should be taken, training data is the same as validation data
            self.x_train, self.y_train, self.delta_train = self.x, self.y, self.delta
            self.x_val, self.y_val, self.delta_val = self.x, self.y, self.delta
        
        # Declara os callbacks do modelo
        self.callbacks = [ ]
        
        if(verbose >= 1):
            self.callbacks.append( TqdmCallback(verbose = 0, position = 0, leave = True) )
        
        if(early_stopping):
            # Avoids overfitting and speeds training
            if(self.validation):
                metric = "val_likelihood_loss"
            else:
                metric = "likelihood_loss"
            es = keras.callbacks.EarlyStopping(monitor = metric,
                                               mode = "min",
                                               start_from_epoch = early_stopping_warmup,
                                               min_delta = early_stopping_min_delta,
                                               patience = early_stopping_patience,
                                               restore_best_weights = True)
            self.callbacks.append(es)
        
        # If batch_size is unspecified, set it to be the training size. Note that decreasing the batch size to smaller values, such as 500 for example, has previously lead the
        # model to converge too early, leading to a lot of time of investigation. When dealing with neural networks in the statistical models context, we recommend to use a single
        # batch in training. Alternatives in the case that the sample is too big might be to consider a "gradient accumulation" approach.
        self.train_batch_size = self.x_train.shape[0]
        if(train_batch_size is not None):
            self.train_batch_size = train_batch_size

        self.val_batch_size = self.x_val.shape[0]
        if(val_batch_size is not None):
            self.val_batch_size = val_batch_size
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if(self.gradient_accumulation_steps is None):
            # The number of batches until the actual weights update (we ensure that the weights are updated only once per epoch, even though we might have multiple batches)
            self.gradient_accumulation_steps = int(np.ceil( self.x_train.shape[0] / self.train_batch_size ))

        self.compile_model(optimizer_alpha = optimizer_alpha, optimizer_gamma = optimizer_gamma, optimizer_phi1 = optimizer_phi1, optimizer_phi2 = optimizer_phi2, optimizer_other = optimizer_other, run_eagerly = run_eagerly)

        # Create the training dataset
        self.buffer_size = buffer_size
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train, self.delta_train))
        train_dataset = train_dataset.shuffle(buffer_size = self.buffer_size).batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = None
        if(validation):
            # Create the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val, self.delta_val))
            val_dataset = val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs = epochs,
            verbose = 0,
            callbacks = self.callbacks,
            batch_size = self.train_batch_size,
            shuffle = shuffle
        )

        self.alpha = 1/(1+np.exp(-self.raw_alpha))
        self.gamma = np.exp(self.raw_gamma)
        self.phi1 = np.exp(self.raw_phi1)
        self.phi2 = np.copy(self.raw_phi2)
        
        self.final_history = self.history.history

    
class PVFcrModelStableSteps(keras.models.Model):    
    def __init__(self, seed = 1, input_dim = (None, 1),
                 fix_alpha = False, fix_gamma = False, fix_phi1 = False, fix_phi2 = False,
                 init_alpha = 0.5, init_gamma = 1.0, init_phi1 = 1.0, init_phi2 = 0.0,
                 stat_stepsize = 100, neuralnet_stepsize = 500, stat_turn_init = False):
        super().__init__()
        self.input_dim = input_dim
        self.seed = seed
        self.fix_alpha = fix_alpha
        self.fix_gamma = fix_gamma
        self.fix_phi1 = fix_phi1
        self.fix_phi2 = fix_phi2
        
        self.init_alpha = init_alpha
        self.init_gamma = init_gamma
        self.init_phi1 = init_phi1
        self.init_phi2 = init_phi2

        # We shall update the weights in different moments, which alternate between themselves
        # Number of epochs where the only updated weights are the statistical parameters
        self.stat_stepsize = tf.Variable(stat_stepsize, dtype = tf.int32, trainable = False)
        # Number of epochs where the only updated weights are the neural network ones
        self.neuralnet_stepsize = tf.Variable(neuralnet_stepsize, dtype = tf.int32, trainable = False)
        # We start by tuning the neural network first, so the statistical parameters's turn False
        self.stat_turn = tf.Variable(stat_turn_init, dtype=tf.bool, trainable=False)
        
        # Define the architecture
        self.define_structure()
        # Call the model with a dummy input to initialize its weights
        self.dummy_input = keras.layers.Input(shape = self.input_dim)
        self(self.dummy_input)

        # Number of accumulated gradients by batches
        self.n_acum_step = tf.Variable(0, dtype = tf.int32, trainable = False)

        # Number of epochs spent training only statistical parameters
        self.n_stat_step = tf.Variable(0, dtype = tf.int32, trainable = False)
        # Number of epochs spent training only neural network weights
        self.n_neuralnet_step = tf.Variable(0, dtype = tf.int32, trainable = False)
        
        # Number of parameters that were fixed for training
        self.fixed_params = fix_alpha + fix_gamma + fix_phi1 + fix_phi2
        # The number of independent parameters to be estimated directly by tensorflow
        self.trainable_params = 4 - self.fixed_params

        self.gradient_accumulation_alpha = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_gamma = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_phi1 = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_phi2 = [tf.Variable(0.0, dtype = tf.float32, trainable = False)]
        self.gradient_accumulation_other = [tf.Variable(tf.zeros_like(v, dtype = tf.float32), trainable = False) for v in self.trainable_variables[self.trainable_params:]]

    def define_structure(self):
        '''
            This method must contain all the layers from the neural networks model, preferrably 
        '''
        initializer = initializers.GlorotNormal(seed = self.seed)
        self.dense1 = keras.layers.Dense(units = 16, activation = "gelu", kernel_initializer = initializer, dtype = tf.float32, name = "dense1")
        self.dense2 = keras.layers.Dense(units = 1, kernel_initializer = initializer, dtype = tf.float32, activation = None, use_bias = False, name = "output")

        # Trainable parameters from the model - They are all unconstrained in R (raw)

        self.init_raw_alpha = tf.math.log( self.init_alpha/(1-self.init_alpha) )
        self.init_raw_gamma = tf.math.log( self.init_gamma )
        self.init_raw_phi1 = tf.math.log( self.init_phi1 )
        self.init_raw_phi2 = tf.identity( self.init_phi2 )

        self.raw_alpha = self.add_weight(name = 'alpha', shape = (), initializer = keras.initializers.Constant( self.init_raw_alpha ), trainable = not self.fix_alpha, dtype = tf.float32)
        self.raw_gamma = self.add_weight(name = 'gamma', shape = (), initializer = keras.initializers.Constant( self.init_raw_gamma ), trainable = not self.fix_gamma, dtype = tf.float32)
        self.raw_phi1 = self.add_weight(name = 'phi1', shape = (), initializer = keras.initializers.Constant( self.init_raw_phi1 ), trainable = not self.fix_phi1, dtype = tf.float32)
        self.raw_phi2 = self.add_weight(name = 'phi2', shape = (), initializer = keras.initializers.Constant( self.init_raw_phi2 ), trainable = not self.fix_phi2, dtype = tf.float32)

        self.alpha = 1/(1+np.exp(-self.raw_alpha))
        self.gamma = np.exp(self.raw_gamma)
        self.phi1 = np.exp(self.raw_phi1)
        self.phi2 = np.copy(self.raw_phi2)
        
    def call(self, x_input):
        x = self.dense1(x_input)
        return self.dense2(x)

    def save_model(self, filename):
        self.save_weights(filename)

    def load_model(self, filename):
        self.load_weights(filepath = filename)
        self.alpha = 1/(1+np.exp(-self.raw_alpha))
        self.gamma = np.exp(self.raw_gamma)
        self.phi1 = np.exp(self.raw_phi1)
        self.phi2 = np.copy(self.raw_phi2)
        return self
    
    def copy(self):
        '''
            Creates a new object of the same class as a copy.
        '''
        new_model = PVFcrModelStable(seed = self.seed, input_dim = self.input_dim,
                                     fix_alpha = self.fix_alpha, fix_gamma = self.fix_gamma, fix_phi1 = self.fix_phi1, fix_phi2 = self.fix_phi2,
                                     init_alpha = self.init_alpha, init_gamma = self.init_gamma,
                                     init_phi1 = self.init_phi1, init_phi2 = self.init_phi2)
        new_model.set_weights(self.get_weights())
        new_model.alpha = 1/(1+np.exp(-new_model.raw_alpha))
        new_model.gamma = np.exp(new_model.raw_gamma)
        new_model.phi1 = np.exp(new_model.raw_phi1)
        new_model.phi2 = np.copy(new_model.raw_phi2)
        return new_model
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # eta
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # y
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)   # delta
    ])
    def likelihood_loss(self, eta, y, delta):
        theta = tf.math.exp(eta)
        
        alpha = 1 / (1 + tf.math.exp(-self.raw_alpha))
        gamma = tf.math.exp(self.raw_gamma)
        phi1 = tf.math.exp(self.raw_phi1)
        phi2 = tf.identity(self.raw_phi2)
        
        log_f0 = tf.math.log(phi1) + (phi1-1)*tf.math.log(y) + phi2 - tf.math.exp(phi2) * tf.math.pow(y, phi1)
        F0 = 1 - tf.math.exp(-tf.math.pow(y, phi1) * tf.math.exp(phi2))
        laplace_transform_term = 1 + gamma*theta*F0/(1-alpha)

        loss_weights = delta*eta + delta*log_f0 + (1-alpha)/(alpha*gamma)*(1 - tf.math.pow(laplace_transform_term, alpha)) + (alpha-1)*delta*tf.math.log(laplace_transform_term)
        loss_weights_mean = -tf.math.reduce_mean(loss_weights)
        
        return loss_weights_mean

    def switch_training_mode(self):
        # Switch the training mode
        self.stat_turn.assign( tf.logical_not(self.stat_turn) )
        # As it is switching training modes, it can set both counters to zero with no problems
        self.n_stat_step.assign(0)
        self.n_neuralnet_step.assign(0)
        tf.cond(self.stat_turn, lambda: tf.print("Now training statistical parameters"), lambda: tf.print("Now training neuralnet weights"))
    
    def apply_accumulated_gradients(self):
        # ----------------------------------- Independent parameters component -----------------------------------
        # Apply the accumulated gradients to the trainable variables
        if(not self.fix_alpha):
            self.optimizer_alpha.apply_gradients( zip(self.gradient_accumulation_alpha, [self.raw_alpha]) )
            self.gradient_accumulation_alpha[0].assign(tf.zeros((), dtype = tf.float32))
        if(not self.fix_gamma):
            self.optimizer_gamma.apply_gradients( zip(self.gradient_accumulation_gamma, [self.raw_gamma]) )
            self.gradient_accumulation_gamma[0].assign(tf.zeros((), dtype = tf.float32))
        if(not self.fix_phi1):
            self.optimizer_phi1.apply_gradients( zip(self.gradient_accumulation_phi1, [self.raw_phi1]) )
            self.gradient_accumulation_phi1[0].assign(tf.zeros((), dtype = tf.float32))
        if(not self.fix_phi2):
            self.optimizer_phi2.apply_gradients( zip(self.gradient_accumulation_phi2, [self.raw_phi2]) )
            self.gradient_accumulation_phi2[0].assign(tf.zeros((), dtype = tf.float32))

        # Reset the gradient accumulation steps counter to zero
        self.n_acum_step.assign(0)

        # ----------------------------------- Neural network component -----------------------------------
        self.optimizer_other.apply_gradients( zip(self.gradient_accumulation_other, self.trainable_variables[self.trainable_params:]) )
        for i in range(len(self.gradient_accumulation_other)):
            self.gradient_accumulation_other[i].assign(tf.zeros_like(self.trainable_variables[self.trainable_params:][i], dtype = tf.float32))

        # Update the stat counter if on stat mode, otherwise, update the neuranet counter
        tf.cond(self.stat_turn, lambda: self.n_stat_step.assign_add(1), lambda: self.n_neuralnet_step.assign_add(1))
        
        # If on stat turn, check if stat counter has reached its limit. If so, switch training mode to neuralnet turn, otherwise, do the same to neuralnet turn favoring stat turn
        tf.cond(self.stat_turn,
                lambda: tf.cond(tf.equal(self.n_stat_step, self.stat_stepsize), self.switch_training_mode, lambda: None),
                lambda: tf.cond(tf.equal(self.n_neuralnet_step, self.neuralnet_stepsize), self.switch_training_mode, lambda: None))

    def accumulate_stat_gradients(self, gradients):
        i_grad = 0
        if(not self.fix_alpha):
            alpha_gradients = gradients[i_grad]
            self.gradient_accumulation_alpha[0].assign_add(alpha_gradients)
            i_grad += 1
        if(not self.fix_gamma):
            gamma_gradients = gradients[i_grad]
            self.gradient_accumulation_gamma[0].assign_add(gamma_gradients)
            i_grad += 1
        if(not self.fix_phi1):
            phi1_gradients = gradients[i_grad]
            self.gradient_accumulation_phi1[0].assign_add(phi1_gradients)
            i_grad += 1
        if(not self.fix_phi2):
            phi2_gradients = gradients[i_grad]
            self.gradient_accumulation_phi2[0].assign_add(phi2_gradients)

    def accumulate_neuralnet_gradients(self, gradients):
        other_gradients = gradients[self.trainable_params:]
        for i in range(len(self.gradient_accumulation_other)):
            self.gradient_accumulation_other[i].assign_add(other_gradients[i])
    def train_step(self, data):
        '''
            Override the train_step method for custom training logic
        '''
        x, y, delta = data

        self.n_acum_step.assign_add(1)
        with tf.GradientTape() as tape:
            eta = self(x, training = True)
            likelihood_loss = self.likelihood_loss(eta = eta, y = y, delta = delta)

        gradients = tape.gradient(likelihood_loss, self.trainable_variables)

        # If on stat mode, accumulate stat gradients, otherwise, accumulate neuralnet gradients
        tf.cond(self.stat_turn, lambda: self.accumulate_stat_gradients(gradients), lambda: self.accumulate_neuralnet_gradients(gradients))

        # Always update all the weights, but depending on the turn, some weights have zero gradient to be applied:
        # --- If we're in the statistical parameters' turn, the neural network gradients are zero
        # --- If we're in the neuralnet's turn, the statistical parameters gradients are zero
        # If it the model has passed through all the batches, apply the gradients to the weights, otherwise, only keep accumulating gradients
        tf.cond(tf.equal(self.n_acum_step, self.gradient_accumulation_steps), self.apply_accumulated_gradients, lambda: None)
        
        return {"likelihood_loss": likelihood_loss, "alpha": 1/(1+tf.math.exp(-self.raw_alpha)), "gamma": tf.math.exp(self.raw_gamma), "phi1": tf.math.exp(self.raw_phi1), "phi2": self.raw_phi2}

    def test_step(self, data):
        x, y, delta = data
        eta = self(x, training = False)
        likelihood_loss = self.likelihood_loss(eta = eta, y = y, delta = delta)
        return {"likelihood_loss": likelihood_loss}

    def likelihood_loss_predict(self, x, y, delta):
        x = tf.cast(x, dtype = tf.float32)
        y = tf.cast(y, dtype = tf.float32)
        delta = tf.cast(delta, dtype = tf.float32)
        if(len(x.shape) == 1):
            x = tf.reshape( x, shape = (len(x), 1) )
        if(len(y.shape) == 1):
            y = tf.reshape( y, shape = (len(y), 1) )
        if(len(delta.shape) == 1):
            delta = tf.reshape( delta, shape = (len(delta), 1) )

        eta = self(x, training = False)
        theta = tf.math.exp(eta)
        
        alpha = 1 / (1 + tf.math.exp(-self.raw_alpha))
        gamma = tf.math.exp(self.raw_gamma)
        phi1 = tf.math.exp(self.raw_phi1)
        phi2 = tf.identity(self.raw_phi2)
        
        log_f0 = tf.math.log(phi1) + (phi1-1)*tf.math.log(y) + phi2 - tf.math.exp(phi2) * tf.math.pow(y, phi1)
        F0 = 1 - tf.math.exp(-tf.math.pow(y, phi1) * tf.math.exp(phi2))
        laplace_transform_term = 1 + gamma*theta*F0/(1-alpha)

        loss_weights = delta*eta + delta*log_f0 + (1-alpha)/(alpha*gamma)*(1 - tf.math.pow(laplace_transform_term, alpha)) + (alpha-1)*delta*tf.math.log(laplace_transform_term)
        loss_weights_mean = -tf.math.reduce_mean(loss_weights)
        
        return loss_weights_mean
    
    def compile_model(self,
                      optimizer_alpha = optimizers.Adam(learning_rate = 0.001),
                      optimizer_gamma = optimizers.Adam(learning_rate = 0.1),
                      optimizer_phi1 = optimizers.Adam(learning_rate = 0.1),
                      optimizer_phi2 = optimizers.Adam(learning_rate = 0.1),
                      optimizer_other = optimizers.Adam(learning_rate = 0.1), run_eagerly = False):
        self.optimizer_alpha = optimizer_alpha
        self.optimizer_gamma = optimizer_gamma
        self.optimizer_phi1 = optimizer_phi1
        self.optimizer_phi2 = optimizer_phi2
        self.optimizer_other = optimizer_other
        self.compile(
            run_eagerly = run_eagerly
        )
        
    def compile_train_model(self, x, y, delta,
                            validation = False, val_prop = None, x_val = None, y_val = None, delta_val = None,
                            epochs = 100,
                            buffer_size = 4096, train_batch_size = None, val_batch_size = None,
                            optimizer_alpha = optimizers.Adam(learning_rate = 0.001),
                            optimizer_gamma = optimizers.Adam(learning_rate = 0.1),
                            optimizer_phi1 = optimizers.Adam(learning_rate = 0.1),
                            optimizer_phi2 = optimizers.Adam(learning_rate = 0.1),
                            optimizer_other = optimizers.Adam(learning_rate = 0.1),
                            run_eagerly = False, gradient_accumulation_steps = None,
                            early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10, early_stopping_warmup = 0,
                            shuffle = False,
                            verbose = 2):
        '''
            Organiza os conjunto de treino e validação e inicia o treinamento da rede neural
        '''
        self.validation = validation
        
        # Pass the input variables to tensorflow default types
        x = tf.cast(x, dtype = tf.float32)
        y = tf.cast(y, dtype = tf.float32)
        delta = tf.cast(delta, dtype = tf.float32)
        
        # If input is a vector, transform it into a column
        if(len(x.shape) == 1):
            x = tf.reshape( x, shape = (len(x), 1) )
        if(len(y.shape) == 1):
            y = tf.reshape( y, shape = (len(y), 1) )
        if(len(delta.shape) == 1):
            delta = tf.reshape( delta, shape = (len(delta), 1) )

        # Salva os dados originais
        self.x = x
        self.y = y
        self.delta = delta

        if(self.validation):
            if(x_val is not None and y_val is not None and delta_val is not None):
                # Proper validation data provided
                
                x_val = tf.cast(x_val, dtype = tf.float32)
                y_val = tf.cast(y_val, dtype = tf.float32)
                delta_val = tf.cast(delta_val, dtype = tf.float32)

                if(len(x_val.shape) == 1):
                    x_val = tf.reshape( x_val, shape = (len(x_val), 1) )
                if(len(y_val.shape) == 1):
                    y_val = tf.reshape( y_val, shape = (len(y_val), 1) )
                if(len(delta_val.shape) == 1):
                    delta_val = tf.reshape( delta_val, shape = (len(delta_val), 1) )
                
                self.x_val = x_val
                self.y_val = y_val
                self.delta_val = delta_val
                self.x_train, self.y_train, self.delta_train = self.x, self.y, self.delta
            else:
                # If validation is wanted, but no data was given, select val_prop * 100% observations as validation set
                
                self.indexes_train = np.arange(x.shape[0])
                if(shuffle):
                    self.indexes_train = tf.random.shuffle( self.indexes_train )
                    
                x_shuffled = tf.gather( x, self.indexes_train )
                y_shuffled = tf.gather( y, self.indexes_train )
                delta_shuffled = tf.gather( delta, self.indexes_train )

                if(val_prop is None):
                    raise Exception("Please, provide the size of the validation set (between 0 and 1).")
                # Selects the subsample as validation data
                val_size = int(x.shape[0] * val_prop)
                self.x_val, self.y_val, self.delta_val = x_shuffled[:val_size], y_shuffled[:val_size], delta_shuffled[:val_size]
                self.x_train, self.y_train, self.delta_train = x_shuffled[val_size:], y_shuffled[val_size:], delta_shuffled[val_size:]
        else:
            # If no validation step should be taken, training data is the same as validation data
            self.x_train, self.y_train, self.delta_train = self.x, self.y, self.delta
            self.x_val, self.y_val, self.delta_val = self.x, self.y, self.delta
        
        # Declara os callbacks do modelo
        self.callbacks = [ ]
        
        if(verbose >= 1):
            self.callbacks.append( TqdmCallback(verbose = 0, position = 0, leave = True) )
        
        if(early_stopping):
            # Avoids overfitting and speeds training
            if(self.validation):
                metric = "val_likelihood_loss"
            else:
                metric = "likelihood_loss"
            es = keras.callbacks.EarlyStopping(monitor = metric,
                                               mode = "min",
                                               start_from_epoch = early_stopping_warmup,
                                               min_delta = early_stopping_min_delta,
                                               patience = early_stopping_patience,
                                               restore_best_weights = True)
            self.callbacks.append(es)
        
        # If batch_size is unspecified, set it to be the training size. Note that decreasing the batch size to smaller values, such as 500 for example, has previously lead the
        # model to converge too early, leading to a lot of time of investigation. When dealing with neural networks in the statistical models context, we recommend to use a single
        # batch in training. Alternatives in the case that the sample is too big might be to consider a "gradient accumulation" approach.
        self.train_batch_size = self.x_train.shape[0]
        if(train_batch_size is not None):
            self.train_batch_size = train_batch_size

        self.val_batch_size = self.x_val.shape[0]
        if(val_batch_size is not None):
            self.val_batch_size = val_batch_size
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if(self.gradient_accumulation_steps is None):
            # The number of batches until the actual weights update (we ensure that the weights are updated only once per epoch, even though we might have multiple batches)
            self.gradient_accumulation_steps = int(np.ceil( self.x_train.shape[0] / self.train_batch_size ))

        self.compile_model(optimizer_alpha = optimizer_alpha, optimizer_gamma = optimizer_gamma, optimizer_phi1 = optimizer_phi1, optimizer_phi2 = optimizer_phi2, optimizer_other = optimizer_other, run_eagerly = run_eagerly)

        # Create the training dataset
        self.buffer_size = buffer_size
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train, self.delta_train))
        train_dataset = train_dataset.shuffle(buffer_size = self.buffer_size).batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = None
        if(validation):
            # Create the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val, self.delta_val))
            val_dataset = val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs = epochs,
            verbose = 0,
            callbacks = self.callbacks,
            batch_size = self.train_batch_size,
            shuffle = shuffle
        )

        self.alpha = 1/(1+np.exp(-self.raw_alpha))
        self.gamma = np.exp(self.raw_gamma)
        self.phi1 = np.exp(self.raw_phi1)
        self.phi2 = np.copy(self.raw_phi2)
        
        self.final_history = self.history.history