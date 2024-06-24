import numpy as np

import Util
from Models import UNetModel,UNetModel2
import tensorflow as tf

def train(data_path):
    #get dataset batch
    audio_data_batch = Util.getBatchFromDataSet(dataset_path=data_path,batch_size=16)
    sr=48000
    pre_processed_data = Util.getProcessedAudio(batch_path=audio_data_batch,target_sr=sr)

    #max_length = min(arr.shape[0] for arr in pre_processed_data)
    min_len = 100000
    standardized_data = Util.padding_output(pre_processed_data,min_len)

    #build Model

    input_shape = (min_len,1)
    model = UNetModel.build_unet(input_shape)
    x_train = np.array(standardized_data)
    x_train_reshape = x_train.reshape(-1, min_len, 1)
    x_train_tensor = tf.convert_to_tensor(x_train_reshape,dtype=tf.float32)

    print("Tensor Info:", x_train_tensor)
    print("Any NaN in tensor:", tf.math.is_nan(x_train_tensor).numpy().any())
    print("Tensor Shape and Type:", x_train_tensor.shape, x_train_tensor.dtype)

    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(x=x_train_tensor,y=x_train_tensor, epochs=3, verbose=2,batch_size=4)

    model.save("/Users/Kwesi/PycharmProjects/AudioSourceSeparator/train_modelUNO.keras")
    print("Model Trained!")
    return history
