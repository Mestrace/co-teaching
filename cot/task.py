from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
from time import time

import tensorflow as tf
from tensorflow.keras import datasets

tf.enable_eager_execution()

from cot import models as cot_models
from cot import noise as cot_noise


def selective_remember(model, X, y, loss_func, remember_rate):
    """selective_remember selects a subset with small losses from the given training samples. Returns the selected X and y.
    """
    assert 0 < remember_rate < 1
    # loss function reduction must be 'none'
    # compute loss
    loss_value = loss_func(y, model(X))

    sorted_index = tf.argsort(
        loss_value, axis=-1, direction="ASCENDING", stable=False, name=None
    )
    num_remembered = int(len(X) * remember_rate)
    assert num_remembered > 0, "Number of remembered elements is smaller than 1."

    return sorted_index[:num_remembered]
    # return tf.gather(X, sorted_index[:num_remembered]), tf.gather(y, sorted_index[:num_remembered])


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="local or GCS location for writing checkpoints and exporting models",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
    )
    args, _ = parser.parse_known_args()
    return args


def gather_remembered(model, X, y, remember_rate, to_be_gathered):
    """Wrapper over selective_remember
    As we evaluate per-instance loss in selective_remember, we use the non-reduction version of the loss function compiled into the model
    Then we map the selected indices to the tensor to be gathered.
    """
    loss_func = model.loss.__class__(reduction="none")
    selected_indices = selective_remember(model, X, y, loss_func, remember_rate)

    return tuple(
        map(lambda tensor: tf.gather(tensor, selected_indices), to_be_gathered)
    )


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.
    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.
    Args:
        args: dictionary of arguments - see get_args() for details
    """
    (
        (train_images, train_labels),
        (test_images, test_labels,),
    ) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = (
        tf.cast(train_images / 255.0, dtype=tf.float32),
        tf.cast(test_images / 255.0, dtype=tf.float32),
    )
    num_classes = 10

    # Hyper parameters
    noise_rate = 0.45
    batch_size = 128
    num_epochs = 200
    max_remember_epoch = 50
    forget_rate = noise_rate

    # Prepare model
    model1 = cot_models.CNN(shape=train_images.shape[1:], num_classes=10)
    model2 = cot_models.mobilenetv1(shape=train_images.shape[1:], num_classes=10)

    model1.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer="adam",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model2.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer="adam",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Prepare datasets
    noisy_data = cot_noise.label_random_flip(train_labels, noise_rate, num_classes)

    train_dataset1 = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels, noisy_data))
        .batch(batch_size)
        .shuffle(30000)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    train_dataset2 = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels, noisy_data))
        .batch(batch_size)
        .shuffle(30000)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    # Epoch statistics from evaluation
    model1_evaluate_history = []
    model2_evaluate_history = []

    # Training
    for current_epoch in range(num_epochs):
        remember_rate = 1 - forget_rate * min(
            (current_epoch + 1) / max_remember_epoch, 1
        )
        print("Epoch %d, remember_rate %.4f" % (current_epoch, remember_rate))

        train_start_time = time()
        for _, ((X1, y1, ny1), (X2, y2, ny2)) in enumerate(
            zip(train_dataset1, train_dataset2)
        ):
            X1, y1, ny1 = gather_remembered(
                model1, X1, ny1, remember_rate, (X1, y1, ny1)
            )
            X2, y2, ny2 = gather_remembered(
                model2, X2, ny2, remember_rate, (X2, y2, ny2)
            )

            model1.train_on_batch(X2, ny2, reset_metrics=False)
            model2.train_on_batch(X1, ny1, reset_metrics=False)
        train_end_time = time()
        print(
            "Epoch %d finished, time %d s"
            % (current_epoch, train_end_time - train_start_time)
        )

        model1_evaluate_history.append(model1.evaluate(test_dataset))
        model2_evaluate_history.append(model2.evaluate(test_dataset))

        print(model1_evaluate_history[-1])
        print(model2_evaluate_history[-1])

        model1.reset_metrics()
        model2.reset_metrics()

    print(model1_evaluate_history)
    print(model2_evaluate_history)


if __name__ == "__main__":
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
