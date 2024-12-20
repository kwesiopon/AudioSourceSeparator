import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Dropout, UpSampling1D, BatchNormalization,Activation, Cropping1D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda


'''
Slightly modified  U-Net Model for Audio Source Extraction
We utilize causal padding to account for temporal nature of the data
additionally we use dilated convolution layers to investigate the 
impact of performance.

TODO:Need to determine suitable size of filters and input size
'''
def conv_layer(inputs,num_filters):
    '''
    :param inputs:
    :param num_filters:
    :return:
    '''
    conv1d= tf.keras.layers.Conv1D(num_filters,1, padding='same')(inputs)
    #print(conv1d.shape)
    activation = layers.ReLU(max_value=100)(conv1d)
    return activation

def adjust_shape_for_concat(decode, skip):
    if decode.shape[1] != skip.shape[1]:
        # Trim or pad the decode output
        if decode.shape[1] > skip.shape[1]:
            decode = decode[:, :skip.shape[1], :]
        else:
            pad_size = skip.shape[1] - decode.shape[1]
            decode = tf.pad(decode, [[0, 0], [0, pad_size], [0, 0]])
    return decode


def encoder(inputs,num_filters):
    encode = conv_layer(inputs,num_filters)
    pool_layer = layers.MaxPool1D(pool_size=2,strides=2,padding='same')(encode)
    return encode,pool_layer

def decoder(inputs,skip,num_filters):
    decode = layers.UpSampling1D(2)(inputs)

    # Get shape information for skip and decode tensors
    decode = adjust_shape_for_concat(decode, skip)
    decode = layers.Concatenate(axis=1)([decode,skip])
    decode = conv_layer(decode,num_filters)
    return  decode

def build_unet(input_shape):
    '''
    Combines the encoder, decoder and bridge to form the WaveNet Architecture
    Every convolutional later uses ReLU apart from the output layer which uses tanh
    :param input_shape:
    :return:
    '''
    inputs = tf.keras.layers.Input(shape=input_shape,batch_size=4)

    #Encoder
    e1, p1 = encoder(inputs,512)
    e2, p2 = encoder(p1,512)
    e3, p3 = encoder(p2,512)
    e4, p4 = encoder(p3, 512)

    #Bridge
    bridge = conv_layer(p4,512)

    #Decoder
    d1 = decoder(bridge,e4,512)
    d2 = decoder(d1,e3,512)
    d3 = decoder(d2, e2,512)
    #d4 = decoder(d3,e1,512)


    output_layer = tf.keras.layers.Conv1D(1,1,activation='tanh',padding='same')(d3)
    model = tf.keras.models.Model(inputs,output_layer,name="U-NET")

    return model
