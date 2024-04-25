import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LeakyRelU, MaxPooling1D, Dropout, concatenate, UpSampling1D
import tensorflow as tf

#Creating the U-Net Model
def unetSeparator():