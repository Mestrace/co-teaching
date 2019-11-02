from utils import randomSelectKFromN,convertCorrectLabelToCorruptedLabel
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
    corruptedIdxList = randomSelectKFromN(int(noise_level*totalNum),totalNum)

    for cIdx in corruptedIdxList:

        correctLabel = label_data_set[cIdx]
        wrongLabel = convertCorrectLabelToCorruptedLabel(correctLabel)
        label_data_revise[cIdx] = wrongLabel

    return x_data_set,label_data_revise

if __name__ == '__main__':
    data = input_data.read_data_sets("MNIST_data/", one_hot=False)
    noise_level = 0.3
    train_x,train_y = addRandomNoiseToTrainingSet(data, noise_level)


