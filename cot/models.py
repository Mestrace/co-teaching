import tensorflow as tf
from tensorflow.keras import models, layers
# As per Co-training, all leaky Relu has slope of 0.01
lrelu = lambda features : tf.nn.leaky_relu(features, alpha=0.01)

def CNN():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, input_shape=(32, 32, 3), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(512, (3, 3), activation=lrelu, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.AveragePooling2D(pool_size=1, padding="valid"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(10, activation=tf.nn.softmax))
    return model