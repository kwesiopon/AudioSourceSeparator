import tensorflow as tf
import numpy as np
import librosa

# Load the trained model
model = tf.keras.models.load_model('/Users/emb/downloads/AudioSourceSeparator/trainedmodels/out.keras')

def preprocess_audio(file_path, target_sr, max_length=None):
    # Load and preprocess the audio file
    y, sr = librosa.load(file_path, sr=target_sr)  # Load with target sample rate
    if max_length is not None:
        y = y[:max_length] if len(y) > max_length else np.pad(y, (0, max_length - len(y)), mode='constant')
    return y

def predict_audio(model, audio_data):
    # Reshape input audio data (assuming model input shape is compatible)
    input_data = audio_data.reshape(1, -1, 1)  # Reshape for model input (batch_size, samples, channels)
    
    # Make predictions using the model
    predicted_data = model.predict(input_data)
    
    return predicted_data

# Path to the new audio file
new_audio_path = '/Users/emb/downloads/AudioSourceSeparator/AllSaintsNeverEver(BookerTsUpNorthdub).flac'

# Load and preprocess the new audio file
target_sr = 44100  # Target sample rate
max_length = 44100  # Max length of audio data (adjust as needed)
input_audio = preprocess_audio(new_audio_path, target_sr=target_sr, max_length=max_length)

# Make predictions using the loaded model
predicted_audio = predict_audio(model, input_audio)

# Process the predicted output as needed (e.g., save to file)
# Example: Save the predicted audio to a new file
output_audio_path = '/path/to/save/predicted/audio/file.wav'
librosa.output.write_wav(output_audio_path, predicted_audio.flatten(), sr=target_sr)

