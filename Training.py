import numpy as np

import Util
from Models import UNetModel
import tensorflow as tf

def train(data_path,input_shape):
    #get dataset batch
    audio_data_batch = Util.getBatchFromDataSet(dataset_path=data_path,batch_size=16)

    sr=48000
    pre_processed_data = Util.getProcessedAudio(batch_path=audio_data_batch,target_sr=sr)

    for val in pre_processed_data:
        print(type(val), val.shape)

    max_length = max(arr.shape[0] for arr in pre_processed_data)
    standardized_data = []
    for arr in pre_processed_data:
        if arr.shape[0] < max_length:
            # Pad with zeros to match max_length
            padded_arr = np.pad(arr, [(0, max_length - arr.shape[0]), (0, 0)], mode='constant')
        else:
            # Truncate to max_length
            padded_arr = arr[:max_length, :]
        standardized_data.append(padded_arr)

    for val in standardized_data:
        print(type(val), val.shape)
    #build Model

    input_shape = (max_length,1)
    model = UNetModel.build_unet(input_shape)
    x_train = np.array(standardized_data)
    x_train_reshape = x_train.reshape(-1,max_length,1)
    x_train_tensor = tf.convert_to_tensor(x_train_reshape,dtype=tf.float32)
    print(x_train_tensor.shape)
    print(model.summary())


    history =  model.fit(x=x_train_tensor,batch_size=64, epochs=100, verbose=2)


    #model.save("/Users/Kwesi/PycharmProjects/AudioSourceSeparator/train_modelUNO.keras")
    print("Model Trained!")
    return history
