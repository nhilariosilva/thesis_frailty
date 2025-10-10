
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import layers, initializers, optimizers, losses

from tensorflow.keras.callbacks import Callback
from tqdm.keras import TqdmCallback
from tqdm import tqdm


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
        x, y, delta = data

        self.n_acum_step.assign_add(1)
        with tf.GradientTape() as tape:
            pars = self(x, training = True)
            likelihood_loss = self.loss_func(pars = pars, y = y, delta = delta)

        gradients = tape.gradient(likelihood_loss, self.trainable_variables)

    def compile_model(self, learning_rate):
        """
            Defines the configuration for the model, such as batch size, training mode, early stopping.
        """
        self.compile(
            optimizer = optimizers.Adam(learning_rate = learning_rate, gradient_accumulation_steps = None),
            loss = self.loss_func,
            run_eagerly = True
        )

    def train_model(self, x, t, delta, verbose = 1):
        """
            This is the function that start the training.
        """
        
        # Pass the input variables to tensorflow default types
        x = tf.cast(x, dtype = tf.float32)
        t = tf.cast(t, dtype = tf.float32)
        delta = tf.cast(delta, dtype = tf.float32)
        
        # If input is a vector, transform it into a column
        if(len(x.shape) == 1):
            x = tf.reshape( x, shape = (len(x), 1) )
        if(len(y.shape) == 1):
            t = tf.reshape( t, shape = (len(t), 1) )
        if(len(delta.shape) == 1):
            delta = tf.reshape( delta, shape = (len(delta), 1) )

        # Salva os dados originais
        self.x = x
        self.y = y
        self.delta = delta

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
    
    def apply_accumulated_gradients(self):
        """
            Given the proper number of steps for the model to accumulate gradients over time, finally applies gradients to update the model weights.
        """

    