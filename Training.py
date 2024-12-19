import numpy as np

import Util
from Models import UNetModel
import tensorflow as tf

def train(data_path):
    #get dataset batch for x_train
    sr = 44100
    audio_data_batch = Util.getBatchFromDataSet(dataset_path=data_path,batch_size=16)
    pre_processed_data = Util.getProcessedAudio(batch_path=audio_data_batch,target_sr=sr)

    #audio conversion for y_train
    target_audio_path = '/Users/Kwesi/Downloads/musdb18hq/train/Training Isolated Vocals'
    target_audio = Util.getBatchFromDataSet(dataset_path=target_audio_path, batch_size=16)
    processed_target_data = Util.getProcessedAudio(batch_path=target_audio,target_sr=sr)

    #Standardizing/Normalizing Data
    min_len = 100000
    standardized_data = Util.padding_output(pre_processed_data,min_len)
    y_train_normal = Util.padding_output(processed_target_data,min_len)

    #Build Model
    input_shape = (min_len,1)
    model = UNetModel.build_unet(input_shape)

    #Training Data Conversion to Tensor
    x_train = np.array(standardized_data)
    x_train_reshape = x_train.reshape(-1, min_len, 1)
    x_train_tensor = tf.convert_to_tensor(x_train_reshape,dtype=tf.float32)

    #Target Data Conversion to Tensor
    y_train = np.array(y_train_normal)
    y_train_reshape = y_train.reshape(-1,min_len, 1)
    y_train_tensor = tf.convert_to_tensor(y_train_reshape, dtype= tf.float32)

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x=x_train_tensor,y=y_train_tensor, epochs=20, verbose=2,batch_size=4)

    model.save("/Users/Kwesi/PycharmProjects/AudioSourceSeparator/train_modelDOS.keras")
    print("Model Trained!")
    return model
