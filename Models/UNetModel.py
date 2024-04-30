import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D, Dropout, concatenate, UpSampling1D, BatchNormalization,Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


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
    conv1d= tf.keras.layers.Conv1D(num_filters,1,activation='relu',padding='same')(inputs)
    #print(conv1d.shape)
    activation = layers.ReLU()(conv1d)
    return activation

def encoder(inputs,num_filters):
    encode = conv_layer(inputs,num_filters)
    pool_layer = layers.MaxPool1D(pool_size=2,padding='same')(encode)
    return encode,pool_layer

def decoder(inputs,skip,num_filters):
    decode = layers.UpSampling1D(2)(inputs)
    # Get shape information for skip and decode tensors

    decode = layers.Concatenate(axis=1)([decode,skip])
    #decode = tf.keras.layers.concatenate([inputs,decode,skip],axis=1)
    decode = conv_layer(decode,num_filters)
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
    e1, p1 = encoder(inputs,1024)
    e2, p2 = encoder(p1,1024)
    e3, p3 = encoder(p2,1024)
    e4, p4 = encoder(p3, 1024)

    #Bridge
    bridge = conv_layer(p4,1024)

    #Decoder
    d1 = decoder(bridge,e4,1024)
    d2 = decoder(d1,e3,1024)
    d3 = decoder(d2, e2,1024)
    d4 = decoder(d3,e1,1024)


    output_layer = tf.keras.layers.Conv1D(1,32,activation='tanh',padding='same')(d4)
    model = tf.keras.models.Model(inputs,output_layer,name="U-NET")
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
