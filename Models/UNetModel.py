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
    conv1d= tf.keras.layers.Conv1D(num_filters,2,padding='causal')(inputs)
    #print(conv1d.shape)
    activation = layers.ReLU()(conv1d)
    return activation

def encoder(inputs,num_filters):
    encode = conv_layer(inputs,num_filters)
    pool_layer = layers.MaxPool1D(pool_size=2,padding='same')(encode)
    return encode,pool_layer

def decoder(inputs,skip,num_filters):
    decode = layers.Conv1DTranspose(num_filters,10,strides=2,padding='same')(inputs)
    # Get shape information for skip and decode tensors
    '''
        skip_shape = K.int_shape(skip)
    decode_shape = K.int_shape(decode)
    if skip_shape is None or decode_shape is None:
        raise ValueError("Shapes of skip or decode tensors are None.")
     # Check if shapes match along the second axis
    if skip_shape[1] != decode_shape[1]:
       # Perform cropping or padding on skip tensor to match decode tensor
        crop_size = skip_shape[1] - decode_shape[1]
        if crop_size > 0:
            skip = skip[:, :decode_shape[1], :]
        else:
            pad_size = abs(crop_size) // 2
            skip = K.temporal_padding(skip, padding=(pad_size, pad_size))

    '''

    decode = layers.Concatenate(axis=1)([decode,skip])
    #decode = tf.keras.layers.concatenate([input,decode,skip],axis=1)
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
    e1, p1 = encoder(inputs,32)
    #print(e1.shape)
    #print(p1.shape)
    e2, p2 = encoder(p1,64)
    e3, p3 = encoder(p2,128)
    e4, p4 = encoder(p3, 256)

    #Bridge
    bridge = conv_layer(p4,512)

    #Decoder
    d1 = decoder(bridge,e4,256)
    d2 = decoder(d1,e3,128)

    d3 = decoder(d2, e2,64)
    d4 = decoder(d3,e1,32)


    output_layer = tf.keras.layers.Conv1D(1,1,activation='tanh',padding='same')(d4)
    model = tf.keras.models.Model(inputs,output_layer,name="U-NET")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
