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

    max_length = min(arr.shape[0] for arr in pre_processed_data)
    standardized_data = Util.padding_output(pre_processed_data,max_length)
    #build Model

    input_shape = (max_length,1)
    model = UNetModel.build_unet(input_shape)
    x_train = np.array(standardized_data)
    x_train_reshape = x_train.reshape(-1,max_length,1)
    x_train_tensor = tf.convert_to_tensor(x_train_reshape,dtype=tf.float32)

    print("Tensor Info:", x_train_tensor)
    print("Any NaN in tensor:", tf.math.is_nan(x_train_tensor).numpy().any())
    print("Tensor Shape and Type:", x_train_tensor.shape, x_train_tensor.dtype)
    print(x_train_tensor.shape)
    print(model.summary())

    model.compile(loss='mean_squared_error', optimizer='adam')

    history =  model.fit(x=x_train_tensor, epochs=100, verbose=2)

    #model.save("/Users/Kwesi/PycharmProjects/AudioSourceSeparator/train_modelUNO.keras")
    print("Model Trained!")
    return history
