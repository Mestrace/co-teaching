import tensorflow as tf
from utils import randomSelectKFromN, convertCorrectLabelToCorruptedLabel
from tensorflow.examples.tutorials.mnist import input_data


# Add random noise to MNIST training set
def addRandomNoiseToTrainingSet(data, noise_level):
    '''
    :param data: data structure that follow tensorflow MNIST demo
    :param noise_level: a percentage from 0 to 1, indicate how many percentage of labels are wrong
    :return: train dataset with added noise label
    '''
    # train data set x
    x_data_set = data.train.images

    # train data set label y,[1,2,0,5,...,9]
    label_data_set = data.train.labels
    label_data_revise = label_data_set.copy()

    totalNum = label_data_set.shape[0]
    corruptedIdxList = randomSelectKFromN(int(noise_level*totalNum), totalNum)

    for cIdx in corruptedIdxList:

        correctLabel = label_data_set[cIdx]
        wrongLabel = convertCorrectLabelToCorruptedLabel(correctLabel)
        label_data_revise[cIdx] = wrongLabel

    return x_data_set, label_data_revise


def label_random_flip(labels, prob, num_classes):
    '''Random flipping of the labels
    :param labels: input labels
    :param prob: the probability of random flip
    :param num_classes: the number of classes in the label. Defaults to max(labels) if not specified.
    :return: a label tensor with random flipping-noise added
    '''
    assert 0 < prob <= 0.5, "The flip probablity must between (0, 0.5] for the learning to be effective"

    # flip is a binary tensor specifies whether label at location i is flipped or not, shape = labels.shape
    flip = tf.random.stateless_categorical(tf.math.log([[1 - prob, prob]]),
                                           num_samples=max(labels.shape),
                                           seed=tf.random.uniform(
                                               (2,), maxval=tf.int32.max, dtype=tf.int32),
                                           dtype=tf.int32)
    flip = tf.reshape(flip, (-1, 1))

    # random noisy labels of equivalent shape of labels
    noisy_labels = tf.random.uniform(
        labels.shape, minval=0, maxval=num_classes, dtype=tf.int32)

    # returns the final noisy labels, element-wise multiplication and addition
    # Computations: labels[i] * (1 - flip[i]) + noisy[i] * flip[i]
    #               if flip[i] == 1, label[i] * 0 + noisy[i] * 1 = noisy[i]  ==>  flipped
    #               if flip[i] == 0, label[i] * 1 + noisy[i] * 0 = label[i]  ==>  original
    return labels * (1 - flip) + noisy_labels * flip


if __name__ == '__main__':
    data = input_data.read_data_sets("MNIST_data/", one_hot=False)
    noise_level = 0.3
    train_x, train_y = addRandomNoiseToTrainingSet(data, noise_level)
