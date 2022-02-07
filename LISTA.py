from cProfile import label
import utils as utils
import examples as examples
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
tf.config.run_functions_eagerly(True)   ##Not sure what this does but prevents some errors related with "EagerTensors"...

##soft_threshold, should be turned into a @tf function before implemented in a keras layer
def soft_threshold(input, threshold):
    pos = keras.activations.relu(input-threshold)
    neg = keras.activations.relu(-input - threshold)
    return pos - neg


##ISTA_Layer: a class for a single layer in ISTA
#Computes linear combinations of observed signal and current estimated solution
#This linear combination then goes through a soft threshold (with threshold theta)
class ISTA_Layer(keras.layers.Layer):
    def __init__(self, signal_dim, sol_dim, rate=1e-2 ):
        super(ISTA_Layer, self).__init__()
        self.sol_dim = sol_dim
        self.signal_dim = signal_dim
        self.rate = rate

        self.W_1 = self.add_weight(shape = (sol_dim, signal_dim),
                                        initializer = "random_normal", trainable=True )
        self.W_2 = self.add_weight(shape = (sol_dim, sol_dim),
                                        initializer = "random_normal", trainable=True )
        
        self.theta = tf.Variable(0.1, trainable=True)

    def call(self, input_signal, input_sol):
        z = tf.linalg.matvec(self.W_1,input_signal) + tf.linalg.matvec(self.W_2, input_sol) #z = W_1 b +W_2x;
        pos_ret = keras.activations.relu(z - self.theta)       #Soft_max(z,theta)  = (Id-theta)1_{z>theta} - (Id+theta)1_{z<-theta} 
        neg_ret =  keras.activations.relu( -z - self.theta)    #                   = ReLU(z -theta) - ReLU(-z-theta)
        return  pos_ret - neg_ret

##LISTA_1_Layer : NN model with 1 layer
#The layer consists of a single ISTA layer
#The initial estimate for support x^0 is set as the 0 vector
class LISTA_1_Layer(tf.keras.Model):

  def __init__(self, signal_dim, sol_dim):
    super().__init__()
    self.dense1 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim)
    self.signal_dim = signal_dim
    self.sol_dim = sol_dim

  def call(self, input_signal):
    return self.dense1(input_signal, tf.zeros(self.sol_dim))

##setup_LISTA_1_Layer(): Initializes a 1 layer LISTA and appends an optimizer and a loss
def setup_LISTA_1_Layer(signal_dim, sol_dim):
    model = LISTA_1_Layer(signal_dim = signal_dim, sol_dim = sol_dim)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())

    return model

##train_LISTA_1_Layer(): Trains a model initialized by LISTA_1_Layer (or any compiled model)
def train_LISTA_1_Layer(model, array_obs, array_sols, batch_size=None, epochs=None):
    if batch_size is None and epochs is None:
        batch_size=1
        epochs = len(array_obs)
    elif epochs is None:
        epochs= int(len(array_obs)/batch_size)
    elif batch_size is None:
        batch_size = int(len(array_obs)/batch_size)

    history = model.fit(x = np.array(array_obs), y = np.array(array_sols), batch_size = batch_size,epochs = epochs)

    return history
