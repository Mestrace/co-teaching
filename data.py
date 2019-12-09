import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal


def get_data(data_name, prob, flip,nb_class):

    """
    :param data_name: three datasets: mnist, cifar 10, cifar 100
    :param prob: the noise ration to flip the label
    :param flip: three flip methods: random, pair, symmtry
    :param nb_class: mnsit: 10 cifar10: 10 cifar100: 100

    :returns: train_images, train_noisy_labels, train_labels, test_images, test_labels
    """
   
    
    if data_name == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    elif data_name =='cifar10':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    elif data_name == 'cifar100':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = tf.cast(train_images / 255.0, dtype=tf.float32), tf.cast(test_images / 255.0,dtype=tf.float32)

    # Flip the data using different methods
    if flip == 'random':
      train_noisy_labels = label_random_flip(tf.convert_to_tensor(train_labels, dtype=tf.int32),prob)
    elif flip == 'pair':
      train_noisy_labels = label_pair_flip(train_labels,prob,nb_classes=nb_class)
    elif flip == 'symmetry':
      train_noisy_labels = label_symmetry_flip(train_labels,prob,nb_classes=nb_class)

    return train_images,train_noisy_labels,tf.convert_to_tensor(train_labels),test_images,tf.convert_to_tensor(test_labels)


def multi_label_flip(label, T, random_state=0):
    """
    Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.

    :param label: train labels (array)
    :param T: the original transition probability matrix
    :param random_state: a fixed set of random value

    
    :return: flipped label (array)
    
    """
    # print(np.max(label), T.shape[0])

    # row stochastic matrix
    assert_array_almost_equal(T.sum(axis=1), np.ones(T.shape[1]))

    m = label.shape[0]

    new_label = label.copy()
    flipper = np.random.RandomState(random_state)
    print(T.shape)
    
    for idx in np.arange(m):
      i = label[idx]
      # draw a vector with only an 1
      # print(i)
      flipped = flipper.multinomial(1, T[i, :][0], 1)[0]
      # print(flipped)
      new_label[idx] = np.where(flipped == 1)[0]

    return new_label

def label_pair_flip(y_train,prob,nb_classes,random_state=None):

    """
    Flip the label using pair methods. Assume the adjust labels are pairs.eg: 0 (airplane) and 1 (automobile).

    :param y_train: y label before flipping (array)
    :param prob: the noise ratio to flip the label
    :param nb_classes: the type number of labels 

    :param return: y_train, labels after flipping
         actual_noise, the percent of noise labels 
    """

    T = np.eye(nb_classes)
    n = prob

    T[0, 0], T[0, 1] = 1. - n, n
    for i in range(1, nb_classes-1):
      T[i, i], T[i, i + 1] = 1. - n, n

    T[nb_classes-1, nb_classes-1], T[nb_classes-1, 0] = 1. - n, n
    
    y_train_noisy = multi_label_flip(y_train, T=T,random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()

    assert actual_noise > 0.0

    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy
    print("the transition probability matrix is {}".format(T))
    return y_train, actual_noise

def label_symmetry_flip(y_train, prob,nb_classes,random_state=None):

    """
    Flip the label using symmetric methods. 

    :param y_train: y label before flipping (array)
    :param prob: the noise ratio to flip the label
    :param nb_classes: the type number of labels 

    :return: y_train, labels after symmtric flipping
         actual_noise, the percent of noise labels 
    """

    T = np.ones((nb_classes, nb_classes))
    n = prob
    T = (n / (nb_classes - 1)) * T

    T[0, 0] = 1. - n
    for i in range(1, nb_classes-1):
        T[i, i] = 1. - n
    T[nb_classes-1, nb_classes-1] = 1. - n
    
    y_train_noisy = multi_label_flip(y_train, T=T,random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()

    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy
    print("the transition probability matrix is {}".format(T))

    return y_train, actual_noise

def label_random_flip(labels, prob, num_classes=None):

    '''
    Random flipping of the labels

    :param labels: input labels
    :param prob: the probability of random flip
    :param num_classes: the number of classes in the label. Defaults to max(labels) if not specified.
    :return: a label tensor with random flipping-noise added
    '''
    assert 0 < prob <= 0.5, "The flip probablity must between (0, 0.5] for the learning to be effective"
    if num_classes == None:
        num_classes = tf.math.reduce_max(labels) + 1

    # flip is a binary tensor specifies whether label at location i is flipped or not, shape = labels.shape
    flip = tf.random.stateless_categorical(tf.math.log([[1 - prob, prob]]),num_samples=max(labels.shape),seed=tf.random.uniform((2,), maxval=tf.int32.max, dtype=tf.int32),dtype=tf.int32)
    flip = tf.reshape(flip, (-1, 1))

    # random noisy labels of equivalent shape of labels
    noisy_labels = tf.random.uniform(
        labels.shape, minval=0, maxval=num_classes, dtype=tf.int32)

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
    '''
    usages: get_data(data_name = "cifar100", prob = 0.4,flip = 'pair',nb_class=100)
    
    :param data_name: 'mnist','cifar10','cifar100'
    :param prob: 0.4,0.5,0.6,...
    :param flip: 'pair','symmetry','random'
    :param nb_class: mnist/cifar10: 10 cifar100: 100

    :return tensor format of train_x, train_noisy_y, train_y, test_x, test_y

    '''
    train_x, train_noisy_y, train_y, test_x, test_y =  get_data("cifar100", prob = 0.4,flip = 'pair',nb_class=100)
    
    # meta_lt = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    # sub_plot(train_images, train_labels, meta_lt, row_num = 6, col_num = 6)
    # sub_plot(train_images, train_noisy_labels, meta_lt, row_num=6, col_num=6)
