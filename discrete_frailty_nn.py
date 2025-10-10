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
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

