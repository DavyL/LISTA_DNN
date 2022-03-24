from ast import Not
from cProfile import label
from unicodedata import name
import utils as utils
import examples as examples
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
import os

#tf.config.run_functions_eagerly(True)   ##Not sure what this does but prevents some errors related with "EagerTensors"...

##soft_threshold, should be turned into a @tf function before implemented in a keras layer
@tf.function  # The decorator converts `add` into a `Function`.
def soft_threshold(input, threshold):
    pos = keras.activations.relu(input-threshold)
    neg = keras.activations.relu(-input - threshold)
    return pos - neg

class ProxMetric(keras.losses.Loss):
    def __init__(self, dict=None, regularizer=0.0, name="proximal_metric"):
        super().__init__(name=name)
        self.true_dictionary = tf.convert_to_tensor(dict, dtype=np.float32)
        self.regularizer=regularizer

    def call(self, x_true, x_pred):
        precision = tf.math.reduce_mean(tf.square(tf.linalg.matvec(self.true_dictionary,x_true - x_pred)))
        reg = self.regularizer*tf.math.reduce_mean(tf.abs(x_pred))
        return precision + reg 
##ISTA_Layer: a class for a single layer in ISTA
#Computes linear combinations of observed signal and current estimated solution
#This linear combination then goes through a soft threshold (with threshold theta)
class ISTA_Layer(keras.layers.Layer):
    def __init__(self, signal_dim, sol_dim, rate=1e-2, depth = "0" ):
        super(ISTA_Layer, self).__init__()
        self.sol_dim = sol_dim
        self.signal_dim = signal_dim
        self.rate = rate

        self.W_1 = self.add_weight(shape = (sol_dim, signal_dim), initializer = "random_normal", trainable=True, name="W_1 at depth" + depth)
        self.W_2 = self.add_weight(shape = (sol_dim, sol_dim), initializer = "random_normal", trainable=True, name="W_2 at depth" + depth)
        self.theta = tf.Variable(0.1, trainable=True, name="theta at depth" + depth)

    def call(self, input_signal, input_sol):
        z = tf.add(tf.linalg.matvec(self.W_1,input_signal), tf.linalg.matvec(self.W_2, input_sol)) #z = W_1 b +W_2x;
        return soft_threshold(z, self.theta)

        """
        pos_ret = keras.activations.relu(z - self.theta)       #Soft_max(z,theta)  = (Id-theta)1_{z>theta} - (Id+theta)1_{z<-theta} 
        neg_ret =  keras.activations.relu( -z - self.theta)    #                   = ReLU(z -theta) - ReLU(-z-theta)
        return  tf.add(pos_ret, - neg_ret)
        """

##ISTA_Layer_CP: a class for a single layer in ISTA
#Computes linear combinations partially coupled of observed signal and current estimated solution
#This linear combination then goes through a soft threshold (with threshold theta)
class ISTA_Layer_CP(keras.layers.Layer):
    def __init__(self, signal_dim, sol_dim, dict, rate=1e-2, depth = "0" ):
        super(ISTA_Layer_CP, self).__init__()
        self.sol_dim = sol_dim
        self.signal_dim = signal_dim
        self.rate = rate
        
        ## The transpose of W_1 in non coupled ISTA
        self.W_1 = self.add_weight(shape = (sol_dim, signal_dim), initializer = "random_normal", trainable=True, name="W_1_T at depth" + depth)
        #self.W_2 = self.add_weight(shape = (sol_dim, sol_dim), initializer = "random_normal", trainable=True, name="W_2 at depth" + depth)
        self.theta = tf.Variable(0.1, trainable=True, name="theta at depth" + depth)
        self.dict = tf.cast(dict, dtype = tf.float32)

    def call(self, input_signal, input_sol):
        interm = input_signal - tf.linalg.matvec(self.dict, input_sol)
        z = input_sol + tf.linalg.matvec(self.W_1, interm)
        return soft_threshold(z, self.theta)

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
def setup_LISTA_1_Layer(signal_dim, sol_dim, dict = None):
    model = LISTA_1_Layer(signal_dim = signal_dim, sol_dim = sol_dim)
    if dict:
        custom_metric = ProxMetric(dict=dict, regularizer=0.1)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(), metrics=custom_metric)

    return model

##train_LISTA_1_Layer(): Trains a model initialized by LISTA_1_Layer (or any compiled model)
def train_LISTA_1_Layer(model, array_obs, array_sols, batch_size=1, epochs=10):
    """if batch_size is None and epochs is None:
        batch_size=1
        epochs = len(array_obs)
    elif epochs is None:
        epochs= int(len(array_obs)/batch_size)
        #return  model.fit(x = np.array(array_obs), y = np.array(array_sols), batch_size = batch_size)
    elif batch_size is None:
        batch_size = int(len(array_obs)/epochs)
        #return model.fit(x = np.array(array_obs), y = np.array(array_sols), epochs = epochs)"""
    print("Starting to train LISTA 1 layer with batchsize" + str(batch_size) + "and epochs" + str(epochs))
    history = model.fit(x = np.array(array_obs), y = np.array(array_sols), batch_size = batch_size,epochs = epochs)

    return history


class LISTA_4_Layer(tf.keras.Model):
    def __init__(self, signal_dim, sol_dim):
        super().__init__()
        self.signal_dim = signal_dim
        self.sol_dim = sol_dim
        self.dense1 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth ="1")
        self.dense2 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="2")
        self.dense3 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="3")
        self.dense4 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="4")


    def call(self, input_signal):
        x1 = self.dense1(input_signal, tf.zeros(self.sol_dim))
        x2 = self.dense2(input_signal, x1)
        x3 = self.dense3(input_signal, x2)
        x4 = self.dense4(input_signal, x3)

        return [x1,x2,x3,x4]

##setup_LISTA_4_Layer(): Initializes a 4 layer LISTA and appends a loss
#By design, our LISTA_4_Layer outputs all the estimates of each layer, 
#by choosing the loss_weights, we can choose which layer to train

def setup_LISTA_4_Layer(signal_dim, sol_dim, dict):
    model = LISTA_4_Layer(signal_dim = signal_dim, sol_dim = sol_dim)
    model.compile(optimizer=keras.optimizers.Adam(), 
        loss=[keras.losses.MeanSquaredError() for i in range(4)],
        loss_weights = [0.0,0.0,0.0,1.0]) ##By default trains only last output

    return model

##train_LISTA_4_Layer(): Trains a model initialized by LISTA_4_Layer (or any compiled model)
def train_LISTA_4_Layer(model, array_obs, array_sols, batch_size=1, epochs=10):
    decay = 0.2
    j=0
    historylist = [[] for i in range(5)]
    history = dict()
    
    print("Starting to train LISTA 4 layer with batchsize" + str(batch_size) + "and epochs" + str(epochs))
    weights = np.zeros(4)
    for i in range(4):
        print("In train_LISTA_4_Layer() : starting to learn layer " + str(i))
        weights *= decay    ##decay multiplication MUST BE BEFORE setting i-th weight to 1
        weights[i] = 1.0
        model.compile(optimizer=keras.optimizers.Adam(), 
            loss=[keras.losses.MeanSquaredError() for i in range(4)],
            loss_weights = weights)

        new_hist = model.fit(x = np.array(array_obs), y = [np.array(array_sols) for i in range(4)], batch_size = batch_size,epochs = epochs).history

        ##Everything below is to return a single dictionnary with all losses through layers and epochs
        #I hope the order of a dict with constant names is consistent (it seems to be the case), oth nonsense might happen
        for l in new_hist:
            historylist[j].extend(new_hist[l])
            j+=1
        j = 0


    for l in new_hist:
        history.update({l:historylist[j]})  #We rebuild a dictionary but with the whole list of errors
        j+=1

    return history








class LISTA_16_Layer(tf.keras.Model):
    def __init__(self, signal_dim, sol_dim):
        super().__init__()
        self.signal_dim = signal_dim
        self.sol_dim = sol_dim
        self.dense1 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth ="1")
        self.dense2 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="2")
        self.dense3 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="3")
        self.dense4 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="4")
        self.dense5 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth ="5")
        self.dense6 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="6")
        self.dense7 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="7")
        self.dense8 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="8")        
        self.dense9 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth ="9")
        self.dense10 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="10")
        self.dense11 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="11")
        self.dense12 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="12")        
        self.dense13 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth ="13")
        self.dense14 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="14")
        self.dense15 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="15")
        self.dense16 = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim, depth="16")

    def call(self, input_signal):
        x1 = self.dense1(input_signal, tf.zeros(self.sol_dim))
        x2 = self.dense2(input_signal, x1)
        x3 = self.dense3(input_signal, x2)
        x4 = self.dense4(input_signal, x3)       
        x5 = self.dense5(input_signal, x4)
        x6 = self.dense6(input_signal, x5)
        x7 = self.dense7(input_signal, x6)
        x8 = self.dense8(input_signal, x7)        
        x9 = self.dense9(input_signal, x8)
        x10 = self.dense10(input_signal, x9)
        x11 = self.dense11(input_signal, x10)
        x12 = self.dense12(input_signal, x11)
        x13 = self.dense13(input_signal, x12)
        x14 = self.dense14(input_signal, x13)
        x15 = self.dense15(input_signal, x14)
        x16 = self.dense16(input_signal, x15)
        return [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16]



class LISTA_16_Layer_CP(tf.keras.Model):
    def __init__(self, signal_dim, sol_dim, dict):
        super().__init__()
        self.signal_dim = signal_dim
        self.sol_dim = sol_dim
        self.dense1 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="1", dict = dict)
        self.dense2 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="2", dict = dict)
        self.dense3 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="3", dict = dict)
        self.dense4 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="4", dict = dict)
        self.dense5 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="5", dict = dict)
        self.dense6 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="6", dict = dict)
        self.dense7 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="7", dict = dict)
        self.dense8 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="8"  , dict = dict )     
        self.dense9 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="9", dict = dict)
        self.dense10 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="10", dict = dict)
        self.dense11 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="11", dict = dict)
        self.dense12 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="12", dict = dict  )      
        self.dense13 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="13", dict = dict)
        self.dense14 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="14", dict = dict)
        self.dense15 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="15", dict = dict)
        self.dense16 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="16", dict = dict)

    def call(self, input_signal):
        x1 = self.dense1(input_signal, tf.zeros(self.sol_dim, dtype=tf.float32))
        x2 = self.dense2(input_signal, x1)
        x3 = self.dense3(input_signal, x2)
        x4 = self.dense4(input_signal, x3)       
        x5 = self.dense5(input_signal, x4)
        x6 = self.dense6(input_signal, x5)
        x7 = self.dense7(input_signal, x6)
        x8 = self.dense8(input_signal, x7)        
        x9 = self.dense9(input_signal, x8)
        x10 = self.dense10(input_signal, x9)
        x11 = self.dense11(input_signal, x10)
        x12 = self.dense12(input_signal, x11)
        x13 = self.dense13(input_signal, x12)
        x14 = self.dense14(input_signal, x13)
        x15 = self.dense15(input_signal, x14)
        x16 = self.dense16(input_signal, x15)
        return x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16

class LISTA_16_Layer_CP_AAO(tf.keras.Model):
    def __init__(self, signal_dim, sol_dim, dict):
        super().__init__()
        self.signal_dim = signal_dim
        self.sol_dim = sol_dim
        self.dense1 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="1", dict = dict)
        self.dense2 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="2", dict = dict)
        self.dense3 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="3", dict = dict)
        self.dense4 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="4", dict = dict)
        self.dense5 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="5", dict = dict)
        self.dense6 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="6", dict = dict)
        self.dense7 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="7", dict = dict)
        self.dense8 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="8"  , dict = dict )     
        self.dense9 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="9", dict = dict)
        self.dense10 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="10", dict = dict)
        self.dense11 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="11", dict = dict)
        self.dense12 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="12", dict = dict  )      
        self.dense13 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth ="13", dict = dict)
        self.dense14 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="14", dict = dict)
        self.dense15 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="15", dict = dict)
        self.dense16 = ISTA_Layer_CP(signal_dim=signal_dim, sol_dim=sol_dim, depth="16", dict = dict)

    def call(self, input_signal):
        x1 = self.dense1(input_signal, tf.zeros(self.sol_dim, dtype=tf.float32))
        x2 = self.dense2(input_signal, x1)
        x3 = self.dense3(input_signal, x2)
        x4 = self.dense4(input_signal, x3)       
        x5 = self.dense5(input_signal, x4)
        x6 = self.dense6(input_signal, x5)
        x7 = self.dense7(input_signal, x6)
        x8 = self.dense8(input_signal, x7)        
        x9 = self.dense9(input_signal, x8)
        x10 = self.dense10(input_signal, x9)
        x11 = self.dense11(input_signal, x10)
        x12 = self.dense12(input_signal, x11)
        x13 = self.dense13(input_signal, x12)
        x14 = self.dense14(input_signal, x13)
        x15 = self.dense15(input_signal, x14)
        x16 = self.dense16(input_signal, x15)
        return x16

    def call_by_layers(self, input_signal):
        x1 = self.dense1(input_signal, tf.zeros(self.sol_dim, dtype=tf.float32))
        x2 = self.dense2(input_signal, x1)
        x3 = self.dense3(input_signal, x2)
        x4 = self.dense4(input_signal, x3)       
        x5 = self.dense5(input_signal, x4)
        x6 = self.dense6(input_signal, x5)
        x7 = self.dense7(input_signal, x6)
        x8 = self.dense8(input_signal, x7)        
        x9 = self.dense9(input_signal, x8)
        x10 = self.dense10(input_signal, x9)
        x11 = self.dense11(input_signal, x10)
        x12 = self.dense12(input_signal, x11)
        x13 = self.dense13(input_signal, x12)
        x14 = self.dense14(input_signal, x13)
        x15 = self.dense15(input_signal, x14)
        x16 = self.dense16(input_signal, x15)
        return x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16


##setup_LISTA_L_Layer(): Initializes a L layer LISTA and appends a loss
#By design, our LISTA_L_Layer outputs all the estimates of each layer, 
#by choosing the loss_weights, we can choose which layer to train

def setup_LISTA_L_Layer(signal_dim, sol_dim, dict, L=16, CP = False, AAO = False):

    if CP is False:
        model = LISTA_16_Layer(signal_dim = signal_dim, sol_dim = sol_dim)
    else:
        model = LISTA_16_Layer_CP(signal_dim = signal_dim, sol_dim = sol_dim, dict = dict)
    

    weights = np.zeros(16)
    weights[-1] =1
    model.compile(optimizer=keras.optimizers.Adam(), 
        loss=[keras.losses.MeanSquaredError() for i in range(L)],
        loss_weights = weights) ##By default trains only last output

    return model

##train_LISTA_L_Layer(): Trains a model initialized by LISTA_L_Layer (or any compiled model)
def train_LISTA_L_Layer(model, array_obs, array_sols, batch_size=1, epochs=10, L = 16, weight_decay = False, AAO = False, double_pass = True):
    
    decay = 0.2
    j=0
    historylist = [[] for i in range(L+1)] #First index corresponds to the weighted loss
    history = dict()
    
    print("Starting to train LISTA "+str(L)+" layer with batchsize" + str(batch_size) + "and epochs" + str(epochs))
    weights = [0 for i in range(L)]
    first_weights = np.zeros(L)

    layers = model.layers
    first_pass = True
    for layer in layers:
        layer.trainable = False

    #######All at once training
    if AAO:
        weights[-1] = 1.0
        for layer in layers:
            layer.trainable = True
        layers[-1].trainable = True
        lw = {'output_'+str(i):0.0 for i in range(1,L+1)}
        lw['output_16'] = 1.0
        model.compile(optimizer=keras.optimizers.Adam(), 
            loss={'output_'+str(i) : keras.losses.MeanSquaredError() for i in range(1,L+1)},
            loss_weights = lw)
        new_hist = model.fit(x = np.array(array_obs), y = [np.array(array_sols) for i in range(L)], 
                                batch_size = batch_size,epochs = epochs).history
        print(new_hist)
                                
        return new_hist

    #######Layer by layer training
    for i in range(L):
        if double_pass:
            ########Locking previous layers on first pass
        
            print("In train_LISTA_"+str(L)+"_Layer() : starting to learn layer " + str(i))
            for layer in layers:
                layer.trainable = False
            first_weights[i-1] = 0.0
            first_weights[i] = 1.0
            layers[i].trainable = True
            model.compile(optimizer=keras.optimizers.Adam(), 
                loss=[keras.losses.MeanSquaredError() for l in range(L)],
                loss_weights = first_weights)

            new_hist = model.fit(x = np.array(array_obs), y = [np.array(array_sols) for l in range(L)], 
                                    batch_size = batch_size,epochs = epochs).history
            ##Everything below is to return a single dictionnary with all losses through layers and epochs
            #I hope the order of a dict with constant names is consistent (it seems to be the case), oth nonsense might happen
            for l in new_hist:
                historylist[j].extend(new_hist[l])
                j+=1
            j=0
 
            
        ##Training without fixed layers
        
        
        if weight_decay:
            weights *= decay    ##decay multiplication MUST BE BEFORE setting i-th weight to 1
        else:
            weights[i-1] = 0.0
        
        weights[-1]=0.2 ###Forcing training on last layer
        weights[i] = 1.0
        for k in range(L):
            layers[k].trainable = True
        layers[i].trainable = True
        model.compile(optimizer=keras.optimizers.Adam(), 
            loss=[keras.losses.MeanSquaredError() for l in range(L)],
            loss_weights = weights)
        new_hist = model.fit(x = np.array(array_obs), y = [np.array(array_sols) for l in range(L)], 
                                batch_size = batch_size,epochs = epochs).history
        ##Everything below is to return a single dictionnary with all losses through layers and epochs
        #I hope the order of a dict with constant names is consistent (it seems to be the case), oth nonsense might happen
        for l in new_hist:
            historylist[j].extend(new_hist[l])
            j+=1
        j=0




        for l in new_hist:
            history.update({l:historylist[j]})  #We rebuild a dictionary but with the whole list of errors
            j+=1
        j=0
        
    return history


def compute_parameter_decay(model, dict):
    mat_dif_list = []
    eig_max_list = []
    theta_list = []
    layers = model.layers
    for l in range(len(layers)):
        theta_list.append(layers[l].theta)
    for l in range(len(layers)):
        mat_dif = layers[l].W_2 -np.identity(n=len(dict[0])) + np.matmul(layers[l].W_1, dict)
        mat_dif_list.append(mat_dif)
        eig_max_list.append(max(abs(np.linalg.eigvals(np.matmul(tf.transpose(mat_dif), mat_dif))) ))
    return mat_dif_list,eig_max_list, theta_list











##LISTA_rec_Layer(): Constructs a LISTA network as a recurrent network
#Constructs a model of LISTA of given depth in the following way
#Each step consists of an iteration of a ISTA Layer
#The recurrence is tracked by the current depth
class LISTA_rec_Layer(tf.keras.Model):
    def __init__(self, signal_dim, sol_dim, depth=1):
            super().__init__()
            self.ista = ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim)
            self.signal_dim = signal_dim
            self.sol_dim = sol_dim 
            self.depth = depth
            if(depth>=1):
                self.next_layer = LISTA_rec_Layer(signal_dim, sol_dim=sol_dim, depth = self.depth - 1)
            else:
                self.next_layer=None

            
    def call(self, input_signal, input_sol=None, depth=None):
        if depth is None:
            depth=self.depth
        if input_sol is None:
            input_sol = tf.zeros(self.sol_dim)
        new_est = self.ista(input_signal = input_signal, input_sol = input_sol)
        if ((self.next_layer is None) or (depth <= 1)):
            return new_est
        return self.next_layer(input_signal = input_signal, input_sol = input_sol, depth=depth-1)

def setup_LISTA_rec_Layer(signal_dim, sol_dim, network_depth):
    model = LISTA_rec_Layer(signal_dim = signal_dim, sol_dim = sol_dim, depth=network_depth)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())

    """
    submodel = model
    while(submodel is not None):
        submodel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
        submodel = submodel.next_layer
    """
    return model

def train_LISTA_rec_Layer(model, array_obs, array_sols, batch_size=None, epochs=None):
    return train_LISTA_1_Layer(model = model, array_obs = array_obs, array_sols = array_sols, batch_size=batch_size, epochs=epochs)




##LISTA_rec_Layer(): Constructs a LISTA network as a recurrent network
#Constructs a model of LISTA of given depth in the following way
#Each step consists of an iteration of a ISTA Layer
#The recurrence is tracked by the current depth
class LISTA_iter_model(tf.keras.Model):
    def __init__(self, signal_dim, sol_dim, depth=1):
            super().__init__()
            self.ista_layers = []
            
            self.signal_dim = signal_dim
            self.sol_dim = sol_dim 
            self.depth = depth
            for i in range(depth):
                self.ista_layers.append(ISTA_Layer(signal_dim=signal_dim, sol_dim=sol_dim))
           

            
    def call(self, input_signal, input_sol=None, depth=None):
        if depth is None:
            depth=self.depth
        if input_sol is None:
            input_sol = tf.zeros(self.sol_dim)
        for i in range(depth):
            input_sol = self.ista_layers[i](input_signal = input_signal, input_sol = input_sol)
        return input_sol

def setup_LISTA_iter(signal_dim, sol_dim, network_depth):
    model = LISTA_iter_model(signal_dim = signal_dim, sol_dim = sol_dim, depth=network_depth)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    return model

def train_LISTA_iter(model, array_obs, array_sols, batch_size=None, epochs=None):
    network_depth = model.depth
    loss=[]
    for i in range(1, network_depth):
        print("Training iterative LISTA up to depth " + str(i))
        model.depth=i
        history = model.fit(x = np.array(array_obs), y = np.array(array_sols), batch_size = batch_size,epochs = epochs)
        loss.append(history.history['loss'])
    return loss
