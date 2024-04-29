import os
import numpy as np
import librosa

def resample(audio, orig_sr, new_sr):
    return librosa.resample(audio, orig_sr, new_sr)

def load(path):
    y, sr = librosa.load(path)  # Load audio file without specifying the sample rate (let librosa handle it)
    return y, sr

def getBatchFromDataSet(dataset_path, batch_size):
    audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_path) for file in files if file.endswith(('.wav', '.mp3', '.mp4'))]
    np.random.shuffle(audio_files)
    for i in range(0, len(audio_files), batch_size):
        batch_paths = audio_files[i:i + batch_size]
        yield batch_paths

def getProcessedAudio(batch_paths, target_sr, max_length=None, frame_length_ms=25):
    batch_data = []
    for file_path in batch_paths:
        orig_sr = librosa.get_samplerate(file_path)

        # Load and preprocess each audio file
        y, sr = load(file_path)
        y_resampled = resample(y, orig_sr, target_sr) if orig_sr != target_sr else y

        # Determine frame length in samples
        frame_length = int(sr * frame_length_ms / 1000)

        # Calculate number of time steps (frames) in the audio data
        time_steps = len(y_resampled) // frame_length

        # Optionally truncate or zero-pad to a fixed length
        if max_length is not None:
            # Ensure y_resampled is a 1D array (mono audio)
            y_resampled = y_resampled[:max_length] if len(y_resampled) > max_length else np.pad(y_resampled, (0, max_length - len(y_resampled)), mode='constant')

        batch_data.append(y_resampled)

    # Convert batch_data to a NumPy array
    batch_data = np.array(batch_data)
    return batch_data, time_steps

