import tensorflow as tf

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