from cProfile import label
import utils as utils
import examples as examples
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
import os
#tf.config.run_functions_eagerly(True)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'       ##Turning off CUDA, seems slower with CUDA turned on...

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

#examples.Test_example_LISTA_rec_Layer(show_plots=True, depth=1, train_size=5000, batch_size = 10, epochs = 10)
#examples.Test_example_LISTA_iter(show_plots=True, depth=5, train_size=5000, batch_size = 1, epochs = 1)
#examples.Test_example_LISTA_4_Layer(show_plots=True, train_size=5000, batch_size = 10, epochs = 5)
examples.Test_example_LISTA_16_Layer(show_plots=True, train_size=5000, batch_size = 10, epochs = 3)

#examples.Test_example_LISTA_1_Layer(show_plots=True, train_size=5000, epochs = 5, batch_size=10)
