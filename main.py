from cProfile import label
import utils as utils
import examples as examples
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
tf.config.run_functions_eagerly(True)

examples.Test_example_LISTA_1_Layer(show_plots=True, train_size=64*64, batch_size = 64)

