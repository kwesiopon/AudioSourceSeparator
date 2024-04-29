import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Concatenate, Layer
from tensorflow.keras import backend as K

def conv_layer(inputs, num_filters, kernel_size=3, activation='relu', padding='same'):
    return Conv1D(num_filters, kernel_size, activation=activation, padding=padding)(inputs)

def encoder(inputs, num_filters):
    encode = conv_layer(inputs, num_filters)
    pool_layer = tf.keras.layers.MaxPool1D(pool_size=2, padding='same')(encode)
    return encode, pool_layer

def decoder(inputs, skip, num_filters):
    # Transpose convolution to upsample the input
    decode = Conv1DTranspose(num_filters, 10, padding='same')(inputs)

    # Get shape information for skip and decode tensors
    skip_shape = K.int_shape(skip)
    decode_shape = K.int_shape(decode)

    if skip_shape is None or decode_shape is None:
        raise ValueError("Shapes of skip or decode tensors are None.")

    # Check if shapes match along the second axis
    if skip_shape[1] != decode_shape[1]:
        # Perform cropping or padding on skip tensor to match decode tensor
        crop_size = skip_shape[1] - decode_shape[1]
        if crop_size > 0:
            skip = skip[:, :decode_shape[1], :]
        else:
            pad_size = abs(crop_size) // 2
            skip = K.temporal_padding(skip, padding=(pad_size, pad_size))
    # Reshape skip tensor to match decode tensor's time dimension
    #skip = tf.keras.layers.Reshape((70, -1))(skip)
    # Crop skip tensor to match decode tensor's time dimension
    #skip = skip[:, :70, :]


    # Concatenate the decoded output with the adjusted skip connection
    decode = tf.keras.layers.Concatenate(axis=1)([decode, skip])

    # Apply convolutional layer to the concatenated output
    decode = conv_layer(decode, num_filters)
    return decode

def build_unet(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    e1, p1 = encoder(inputs, 32)
    e2, p2 = encoder(p1, 64)
    e3, p3 = encoder(p2, 128)
    e4, p4 = encoder(p3, 256)

    # Bridge
    bridge = conv_layer(p4, 512)

    # Decoder
    d1 = decoder(bridge, e4, 256)
    d2 = decoder(d1, e3, 128)
    d3 = decoder(d2, e2, 64)
    d4 = decoder(d3, e1, 32)

    output_layer = Conv1D(1, 1, activation='tanh', padding='same')(d4)
    model = tf.keras.models.Model(inputs, output_layer, name="Base_UNET")
    return model
