import tensorflow as tf
import keras
from keras import Model
from keras import layers
def unetmod2(input_shape):

    input = keras.layers.Input(shape=input_shape,batch_size=4)

    #Downsampling Block
    #First Conv Layer with Max Pooling
    conv1d_1 = keras.layers.Conv1D(512,activation="relu",padding="same",kernel_size=2)(input)
    maxpool_1 =  keras.layers.MaxPool1D(2)(conv1d_1)

    #Second Conv Layer w Max Pooling
    conv1d_2 = keras.layers.Conv1D(512, activation='relu',padding="same",kernel_size=2)(maxpool_1)
    maxpool_2 = keras.layers.MaxPool1D(2)(conv1d_2)

    #Third Conv Layer w Max Pooling
    conv1d_3 = keras.layers.Conv1D(512,activation='relu', padding="same",kernel_size=2)(maxpool_2)
    maxpool_3 = keras.layers.MaxPool1D(2)(conv1d_3)

    #Fourth Conv Layer w Max Pooling
    conv1d_4 = keras.layers.Conv1D(512,activation='relu',padding='same',kernel_size=2)(maxpool_3)
    maxpool_4 = keras.layers.MaxPool1D(2)(conv1d_4)

    #Bridge
    bridge = keras.layers.Conv1D(512,activation='relu', padding='same',kernel_size=2)(conv1d_4)

    #Upsampling Block

    #First Layer with Concat
    upsample_1 = keras.layers.UpSampling1D(2)(bridge)
    concat_1 = keras.layers.Concatenate(axis=1)([upsample_1,bridge])
    conv1d_up1 = keras.layers.Conv1D(512,activation='relu',padding='same',kernel_size=2)(concat_1)

    #Second Layer
    upsample_2 = keras.layers.UpSampling1D(2)(conv1d_4)
    concat_2 = keras.layers.Concatenate()([upsample_2,conv1d_3])
    conv1d_up2 = keras.layers.Conv1D(512, activation='relu', padding='same', kernel_size=2)(concat_2)

    #Third Layer
    upsample_3 =  keras.layers.UpSampling1D(2)(conv1d_3)
    concat_3 = keras.layers.Concatenate()([upsample_3,conv1d_4])
    conv1d_up3 = keras.layers.Conv1D(512,activation='relu',padding='same',kernel_size=2)(concat_3)

    #Fourth Layer
    upsample_4 = keras.layers.UpSampling1D(2)(conv1d_2)
    concat_4 = keras.layers.Concatenate()([upsample_4,conv1d_4])
    conv1d_up4 = keras.layers.Conv1D(512,activation='relu', padding='same', kernel_size=2)(concat_4)

    #Output Layer
    output = keras.layers.Conv1D(1,activation='tanh',padding='same',kernel_size=2) (conv1d_up4)

    model2 = keras.models.Model(input,output, name="U-Net 2")

def check_paddding(input1,input2):
    if(input1.shape > input2.shape):
        keras.layers.ZeroPadding1D()
