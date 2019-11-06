import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt


def get(data_name: str,prob: float):
    """
    Returns:
        train_dataset : tf.data.Dataset <= (train_X, noisy, train_label_y)
        test_dataset : tf.data.Dataset <= (test_X, test_label_y)
    Usage:
        train_images,train_labels_noise,train_labels,test_images,test_labels = data.get("cifar10")
    """
    if data_name =='cifar10':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    elif data_name == 'cifar100':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = tf.cast(train_images / 255.0, dtype=tf.float32), tf.cast(test_images / 255.0,dtype=tf.float32)
    train_noisy_labels = label_random_flip(tf.convert_to_tensor(train_labels, dtype=tf.int32),prob)

    return train_images,train_noisy_labels,train_labels,test_images,test_labels



def label_random_flip(labels, prob, num_classes=None):
    '''Random flipping of the labels
    :param labels: input labels
    :param prob: the probability of random flip
    :param num_classes: the number of classes in the label. Defaults to max(labels) if not specified.
    :return: a label tensor with random flipping-noise added
    '''
    assert 0 < prob <= 0.5, "The flip probablity must between (0, 0.5] for the learning to be effective"
    if num_classes == None:
        num_classes = tf.math.reduce_max(labels) + 1

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

def sub_plot(images,labels,class_names,row_num,col_num):
    '''

    :param images: data x
    :param labels: label y
    :param class_names: label names
    :param row_num: the number of rows
    :param col_num: the number of cloumns
    :return: plot the number of row_num * col_num images
    '''
    plt.figure(figsize=(row_num,col_num))
    for i in range(row_num*col_num):
        plt.subplot(row_num,col_num,i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i][0]])
    plt.show()


if __name__ == '__main__':
    train_images, train_noisy_labels, train_labels, test_images, test_labels =  get("cifar10", prob = 0.3)
    meta_lt = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    sub_plot(train_images, train_labels, meta_lt, row_num = 6, col_num = 6)
    sub_plot(train_images, train_noisy_labels, meta_lt, row_num=6, col_num=6)
