import tensorflow as tf
import numpy as np
import os
import librosa

def resample(audio,orig_sr,new_sr):
    return librosa.resample(audio,orig_sr,new_sr)

def load(path,sr):
    y, sr = librosa.load(path,sr)
    return y,sr

def getBatchFromDataSet(dataset_path,batch_size):
    #Get all audio file paths in the dataset path
    audio_files = [os.path.join(root,file) for root,dir,files in os.walk(dataset_path) for file in files]

    #Shuffle the audio files for randomness
    np.random.shuffle(audio_files)

    #Yield batches of audio data
    for i in range(0,len(audio_files).batch_size):
        batch_paths = audio_files[i:i + batch_size]

    return batch_paths

def getProcessedAudio(batch_path,target_sr):
    batch_data = []
    for file_path in batch_path:
        orig_sr = librosa.get_samplerate(file_path)

        #Load and preprocess each audio file
        y, sr = load(file_path,target_sr)

        #Resample Audio if lower than target sample rate
        if orig_sr > target_sr:
           res_y = resample(y,orig_sr,target_sr)
           batch_data.append(res_y)
        else:
            batch_data.append(y)

    return batch_data

