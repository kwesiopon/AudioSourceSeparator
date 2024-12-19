import Util
import numpy as np
import tensorflow as tf
from TrialRun import model_train
from sklearn.metrics import balanced_accuracy_score

def model_test(dataset_path):
    #Convert test audio to NumPy array
    test_audio_processing = Util.getBatchFromDataSet(dataset_path=dataset_path,batch_size=16)
    test_processed_data = Util.getProcessedAudio(batch_path=test_audio_processing,target_sr=48000)
    normalized_test_data = Util.padding_output(test_processed_data,100000)

    #Convert NumPy array to Tensor
    test_array = np.array(normalized_test_data)
    test_array_reshape = test_array.reshape(-1,100000,1)
    test_array_tensor = tf.convert_to_tensor(test_array_reshape, dtype=tf.float32)

    predictions = model_train.predict(test_array_tensor)

    #Metrics
    bacc = balanced_accuracy_score(test_array,predictions)

    print("Predictions", predictions.shape)
    print("Balanced accuracy:", bacc)