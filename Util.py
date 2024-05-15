import tensorflow as tf
import numpy as np
import os
import librosa

def resample(audio,orig_sr,new_sr):
    return librosa.resample(audio,orig_sr,new_sr)

def load(path):
    #Outputs audio (n_frames,n_channels)
    y, sr = librosa.load(path)
    if len(y.shape) == 1:
        y = np.expand_dims(y,axis=0)
    return y.T,sr

def getBatchFromDataSet(dataset_path,batch_size):
    #Get all audio file paths in the dataset path
    audio_files = [os.path.join(root,file) for root,dir,files in os.walk(dataset_path) for file in files]

    #Shuffle the audio files for randomness
    np.random.shuffle(audio_files)

    #Yield batches of audio data
    for i in range(0,len(audio_files),batch_size):
        batch_paths = audio_files[i:i + batch_size]

    return batch_paths

def getProcessedAudio(batch_path,target_sr):
    '''
    Converts all audio in the batch to numpy array format for analysis by the model

    :param batch_path:
    :param target_sr:
    :return:
    '''
    batch_data = []
    for file_path in batch_path:
        orig_sr = librosa.get_samplerate(file_path)

        #Load and preprocess each audio file
        y, sr = load(file_path)

        #Resample Audio if lower than target sample rate
        if orig_sr > target_sr:
           res_y = resample(y,orig_sr,target_sr)
           batch_data.append(res_y)
        else:
            batch_data.append(y)

    audio_array = np.asarray(batch_data,dtype="object")
    return audio_array

def padding_output(pre_processed_data,max_length):
    standardized_data = []
    if (max_length % 2) != 0:
        max_length = max_length + 1
    for arr in pre_processed_data:
        if arr.shape[0] < max_length:
            # Pad with zeros to match max_length
            padded_arr = np.pad(arr, [(0, max_length - arr.shape[0]), (0, 0)], mode='constant')
            print(padded_arr)
        else:
            # Truncate to max_length
            padded_arr = arr[:max_length, :]
            print(padded_arr)
        standardized_data.append(padded_arr)

    for val in standardized_data:
        print(type(val), val.shape)
        print(val)
    return standardized_data

def audio_numpy_to_spectogram(audio_data):
    spectogram = librosa.stft( audio_data)
    stft_magnitude, stft_phase = librosa.magphase(spectogram)

    magnitude_dB = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    return magnitude_dB, stft_phase


def magnitude_db_phase_to_audio(magnitude_db,phase):
    mag_reverse = librosa.db_to_amplitude(magnitude_db,ref=1.0)

    audio_reverse_stft = mag_reverse * phase
    reconstructed_audio = librosa.core.istft(audio_reverse_stft)

    return reconstructed_audio