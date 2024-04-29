import numpy as np
from Util import getBatchFromDataSet, getProcessedAudio
from UNetSeparator import build_unet
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Concatenate, Layer

# Define dataset path and batch size
dataset_path = '/Users/emb/downloads/musdb18/train'
batch_size =10

# Specify number of epochs and validation split
epochs = 10
validation_split = 0.1  # Fraction of training data to use as validation

# Build the U-Net model
batch_paths = next(getBatchFromDataSet(dataset_path, batch_size))
batch_data, max_time_steps = getProcessedAudio(batch_paths, target_sr=44100, max_length=44100)
batch_data_reshaped = np.expand_dims(batch_data, axis=-1)  # Add channel dimension
batch_data_reshaped = batch_data_reshaped[:, :max_time_steps, :]  # Trim to desired time steps if needed
# Assuming batch_data_reshaped has shape (batch_size, original_time_steps, 1)
# Trim or pad to desired max_time_steps
desired_time_steps = max_time_steps
if batch_data_reshaped.shape[1] > desired_time_steps:
    batch_data_reshaped = batch_data_reshaped[:, :desired_time_steps, :]
elif batch_data_reshaped.shape[1] < desired_time_steps:
    batch_data_reshaped = np.pad(batch_data_reshaped, ((0, 0), (0, desired_time_steps - batch_data_reshaped.shape[1]), (0, 0)), mode='constant')

# Ensure target data (batch_data_reshaped) matches input data shape
target_data_reshaped = batch_data_reshaped  # Example: Should be the same as input
print("Input shape:", batch_data_reshaped.shape)
print("Output shape:", target_data_reshaped.shape) 
# Build the U-Net model
input_shape = (max_time_steps, 1)  # Specify input shape based on your data
inputs = tf.keras.layers.Input(shape=input_shape)
model = build_unet(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model on the current batch
model.fit(x=target_data_reshaped, y=target_data_reshaped, epochs=epochs, validation_split=validation_split)

# Optionally save the trained model
model.save('/Users/emb/downloads/AudioSourceSeparator/trainedmodels/out.keras')

