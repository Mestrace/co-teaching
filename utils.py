import numpy as np


# uniform randomly select K integers from range [0,N-1]
def randomSelectKFromN(K, N):
    '''

    :param K: select K integers randomly
    :param N: the number of total labels
    :return: the list of K random integers to add noise label
    '''

    resultList =[]
    seqList = list(range(N))
    while (len(resultList) < K):
        index = (int)(np.random.rand(1)[0] * len(seqList))
        resultList.append(seqList[index])
        seqList.remove(seqList[index])

    return resultList

# Convert correct ont-hot vector label to a wrong label, the error pattern is randomly selected, i.e. not considering the content of image
def convertCorrectLabelToCorruptedLabel(correctLabel):
    '''

    :param correctLabel: the value of true label
    :return: the value of noise label
    '''
    target_value = int(np.random.rand(1)[0]*10)%10
    if target_value == correctLabel:
        target_value = ((target_value+1) % 10)

    return target_value