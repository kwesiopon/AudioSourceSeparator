import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LeakyRelU, MaxPool1D, Dropout, concatenate, UpSampling1D
import tensorflow as tf
'''
Slightly modified  U-Net Model for Audio Source Extraction
We utilize causal padding to account for temporal nature of the data
additionally we use dilated convulation layers to investigate the 
impact of performance.

TODO:Need to determine suitable size of filters and input size
'''
def conv_layer(inputs,num_filters):
    '''

    :param inputs:
    :param num_filters:
    :return:
    '''
    conv1d= tf.keras.layers.Conv1D(num_filters,10,activation='relu',padding='causal')(inputs)
    return conv1d

def encoder(inputs,num_filters):
    encode = conv_layer(inputs,num_filters)
    pool_layer = tf.keras.layers.MaxPool1D(5)(encode)
    return encode,pool_layer

def decoder(inputs,skip,num_filters):
    decode = tf.keras.layers.Conv1DTranspose(num_filters,10,padding='causal')(inputs)
    decode = tf.keras.layers.concatenate()([decode,skip])
    decode = conv_layer(inputs,num_filters)
    return  decode

def build_unet(input_shape):
    '''
    Combines the encoder, decoder and bridge to form the WaveNet Architecture
    Every convolutional later uses ReLU apart from the output layer which uses tanh
    :param input_shape:
    :return:
    '''
    inputs = tf.keras.layers.Input(input_shape)

    #Encoder
    e1, p1 = encoder(inputs,32)
    e2, p2 = encoder(p1,64)
    e3, p3 = encoder(p2,128)
    e4, p4 = encoder(p3, 256)

    #Bridge
    bridge = conv_layer(p4,512)

    #Decoder
    d1 = decoder(bridge,e4,512)
    d2 = decoder(d1,e3,256)
    d3 = decoder(d2, e2,128)
    d4 = decoder(d3,e1,64)

    output_layer = tf.keras.layers.Conv1D(1,1,activation='tanh',padding='causal')(d4)
    model = tf.keras.models.Model(inputs,output_layer,name="Base U-NET")
    return model
