import numpy as np
from Util import getBatchFromDataSet, getProcessedAudio
from UNetSeparator import build_unet

# Define dataset path and batch size
dataset_path = '/Users/emb/downloads/musdb18'
batch_size = 16

# Specify number of epochs and validation split
epochs = 10
validation_split = 0.1  # Fraction of training data to use as validation

# Build the U-Net model
input_shape = (None, 1)  # Specify input shape based on your data
model = build_unet(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Get batch paths from the dataset and process audio data
for batch_paths in getBatchFromDataSet(dataset_path, batch_size):
    batch_data = getProcessedAudio(batch_paths, target_sr=44100, max_length=44100)  # Adjust target_sr and max_length

    # Train the model on the current batch
    model.fit(batch_data, batch_data, epochs=epochs, validation_split=validation_split)

# Optionally save the trained model
model.save('/Users/emb/downloads/AudioSourceSeparator/trainedmodels')

