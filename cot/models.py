import tensorflow as tf
from tensorflow.keras import models, layers

# As per Co-training, all leaky Relu has slope of 0.01
lrelu = lambda features: tf.nn.leaky_relu(features, alpha=0.01)


def CNN(shape, num_classes: int):
    """Returns a CNN model used in cot
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=shape))
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(512, (3, 3), activation=lrelu, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.AveragePooling2D(pool_size=1, padding="valid"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(num_classes, activation=tf.nn.softmax))
    return model


def mobilenetv1(shape, num_classes, activation="relu", alpha=1.0):
    """Obtain a mobilenet V1 model
    Args:
        shape: the shape of the input tensor
        num_classes: the number of outputs
        activation: the activation function of each later
        alpha: hyper parameter for adjusting the width of convolution layers
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=shape))

    def add_conv_block(channels, strides, kernel_size=3):
        channels = int(channels * alpha)
        model.add(
            layers.Conv2D(
                channels, kernel_size=kernel_size, use_bias=False, padding="same"
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))

    def add_dw_sep_block(channels, strides):
        channels = int(channels * alpha)
        model.add(
            layers.DepthwiseConv2D(
                kernel_size=3, strides=strides, use_bias=False, padding="same"
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        add_conv_block(channels, strides=1, kernel_size=1)

    add_conv_block(32, 2)

    model_shapes_channel_strides = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        *[(512, 1) for _ in range(5)],
        (1024, 2),
        (1024, 2),
    ]

    for c, s in model_shapes_channel_strides:
        add_dw_sep_block(c, s)

    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dense(1000, activation=activation))
    model.add(layers.Dense(num_classes, activation="softmax", name="softmax"))

    return model
